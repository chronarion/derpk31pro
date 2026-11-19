#!/bin/bash
# The Grace pool's bench: a load-window measurement. Run it with the engine and the band sidecar down: it spins the
# pool's 48 cores (24-71) plus one poster core (23) for about two minutes, at normal priority.
#   engine/pool/bench_pool.sh [LOG]            (LOG default: build/pool/bench-<date>.log)
# Env: POOL_CORE0 (24) and POOL_THREADS (48): the pool's cores; POSTER (23): the thread that posts each job and spins
# on its done tag, as the GB300's send and landing do; HI2: a record file to bench on real records (default synthetic:
# the timing does not depend on the values); N (1000): jobs per case; RECORDS (96: 892 MB, so that a job's records
# are cold, as in a 388 GB bank); IMG.
# It runs, in the vLLM image with --cpuset-cpus POSTER,POOL_CORE0-...:
#   1. the self-test with pools up to 48 threads (bit-exact; ~1 min)
#   2. --bench:      on-time jobs, 1-4 records x 1/2/4/6 rows, 0.3 ms apart (gate: tag to done
#                    <= 40 us for 1 record, <= 70 us for 2, at 1-2 rows per record)
#   3. --bench-late: late jobs of 1/3/5/7/8 records x 5 rows (1-2 rows per record), back to back and in pairs
#                    (gate: <= 28 us per record in jobs of 5 -> 5 late slots; <= 40 -> 3; above: none)
#   4. --bench-late at 32 and 24 threads (scaling with cores), and the SDOT build at 48 (the same bits)
# Read: "tag -> done" columns (what the GB300 would wait, minus its own C2C read of the rows, ~9 us for 72 KB) and
# "us/record"; "pool us/job" is the pool's own time from taking the job to its done tag.
set -u
D=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$D/../.." && pwd)
IMG=${IMG:-vllm/vllm-openai:kimi-k31}
C0=${POOL_CORE0:-24}
NT=${POOL_THREADS:-48}
POSTER=${POSTER:-$((C0 - 1))}
N=${N:-1000}
RECORDS=${RECORDS:-96}
LOG=${1:-$REPO/build/pool/bench-$(date +%Y%m%d-%H%M%S).log}
mkdir -p "$(dirname "$LOG")"
MNT=()
HARG=""
if [ -n "${HI2:-}" ]; then
  MNT=(-v "$(dirname "$HI2")":/hi2:ro)
  HARG="--hi2 /hi2/$(basename "$HI2")"
fi
if docker ps --format '{{.Names}}' | grep -q .; then
  echo "note: containers are running (the bench wants a quiet host):"
  docker ps --format '  {{.Names}} {{.Status}}'
fi
echo "Grace pool bench: pool on cores $C0-$((C0 + NT - 1)), poster on $POSTER, $N jobs per case, log $LOG"
docker run --rm --network none --cpuset-cpus "$POSTER,$C0-$((C0 + NT - 1))" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$REPO":/repo "${MNT[@]}" -w /repo --entrypoint bash "$IMG" -c "
set -u
engine/pool/build.sh /repo/build/pool > /dev/null || exit 2
B=/repo/build/pool
echo '== 1. self-test'; date
\$B/cpustage_test $HARG | grep -E 'threads|PASS|FAIL|CPU set'
echo '== 2. on-time jobs, $NT threads'; date
\$B/cpustage_test --bench $NT $C0 $N --poster $POSTER --records $RECORDS $HARG
echo '== 3. late jobs, $NT threads'; date
\$B/cpustage_test --bench-late $NT $C0 $N --poster $POSTER --records $RECORDS $HARG
for nt in 32 24; do
  [ \$nt -lt $NT ] || continue
  echo \"== 4. late jobs, \$nt threads\"; date
  \$B/cpustage_test --bench-late \$nt $C0 $N --poster $POSTER --records $RECORDS $HARG
done
echo '== 4. late jobs, $NT threads, SDOT build'; date
\$B/cpustage_test_sdot --bench-late $NT $C0 $N --poster $POSTER --records $RECORDS $HARG
date" 2>&1 | tee "$LOG"
