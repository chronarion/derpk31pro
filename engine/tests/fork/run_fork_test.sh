#!/bin/bash
# derpk31pro fork tests: one test script in one fresh container of the kimi-k31 image on the GB300 alone.
# usage: run_fork_test.sh TAG MODE "ENV=VAL ..." SCRIPT [ARGS...]
#   MODE=stock   the image's own DeepGEMM headers
#   MODE=fork    the overlay's two headers bind-mounted over the image's files (the engine's deployment form)
#   MODE=copy    fork headers laid over a copy of the include tree inside the container (FORK_VIA=copy; for DEFINES)
#   MODE=sched   the image's own kernel header with only the overlay's scheduler mounted over the image's (what every
#                other MegaMoE kernel of the image sees once the overlay is deployed)
#   SCRIPT       a file of this directory (mounted read-only at /tests)
# Output: $OUT/TAG.out (default OUT=$PWD/out; mounted at /out), one line per run in
# $OUT/gpu_runs.log (command, seconds, exit code, peak GPU memory of the container's process by nvidia-smi).
# JITCACHE=dir keeps DeepGEMM's JIT cache (deep_jit: $HOME/.dj) across runs; unset, every run compiles afresh.
# Flags as the project's rules ask: GB300 only, --rm, no core dumps, cores 48-51, 48 GB of host memory, no network,
# uid 1001, at most TIMEOUT seconds (default 300; the container is killed on a hang).
set -u
TAG=$1; MODE=$2; ENVS=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
OVERLAY=${OVERLAY:-$HERE/../../overlay/megamoe_k31}
OUT=${OUT:-$PWD/out}
IMG=${IMG:-vllm/vllm-openai:kimi-k31}
GB=${GB:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "GB300") {print $1; exit}')}
DERPSEEK=${DERPSEEK:?set DERPSEEK to a Derpseek checkout (https://github.com/chronarion/derpseek)}
DG=/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm
K=include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh
S=include/deep_gemm/scheduler/mega_moe.cuh
mkdir -p "$OUT"
EARGS=""
for kv in $ENVS; do EARGS="$EARGS -e $kv"; done
MOUNTS=""
case "$MODE" in
  stock) EARGS="$EARGS -e MODE=stock" ;;
  fork)  EARGS="$EARGS -e MODE=fork -e FORK_VIA=mount"; MOUNTS="-v $OVERLAY/$K:$DG/$K:ro -v $OVERLAY/$S:$DG/$S:ro" ;;
  copy)  EARGS="$EARGS -e MODE=fork -e FORK_VIA=copy" ;;
  sched) EARGS="$EARGS -e MODE=stock -e SCHED_OVERLAY=1 -e ARM=sched"; MOUNTS="-v $OVERLAY/$S:$DG/$S:ro" ;;
  *) echo "MODE must be stock, fork, copy or sched"; exit 2 ;;
esac
START=$(date +%s)
PEAKF=$(mktemp); echo 0 > "$PEAKF"
(
  sleep 2
  while docker inspect r5-$TAG >/dev/null 2>&1; do
    PID=$(docker inspect -f '{{.State.Pid}}' r5-$TAG 2>/dev/null)
    [ -n "$PID" ] && [ "$PID" != 0 ] || break
    KIDS=$(ps -eo pid=,ppid= | awk -v r=$PID '{ p[$1] = $2 } END { for (c in p) { x = c; while (x in p && x != r) x = p[x]; if (x == r) printf "%s ", c } printf "%s", r }')
    M=$(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null |
        awk -F', ' -v g=$GB -v k=" $KIDS " '$3 == g && index(k, " " $1 " ") { s += $2 } END { print s + 0 }')
    [ "$M" -gt "$(cat "$PEAKF")" ] && echo "$M" > "$PEAKF"
    sleep 1
  done
) &
SAMPLER=$!
timeout ${TIMEOUT:-300} docker run --rm --name r5-$TAG --device nvidia.com/gpu=$GB \
  --ulimit core=0 --cpuset-cpus 48-51 --cpus 4 --memory ${MEM:-48g} --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 -e FORK=/overlay -e SF6_PY=/d/engine/derpseek_engine $EARGS \
  -v "$HERE":/tests:ro -v "$OVERLAY":/overlay:ro -v "$OUT":/out -v "$DERPSEEK":/d:ro $MOUNTS ${JITCACHE:+-v $JITCACHE:/tmp/.dj} \
  --entrypoint python3 $IMG /tests/"$@" > "$OUT/$TAG.out" 2>&1
RC=$?
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then docker kill r5-$TAG >/dev/null 2>&1; fi
wait $SAMPLER 2>/dev/null
DUR=$(( $(date +%s) - START ))
PEAK=$(cat "$PEAKF"); rm -f "$PEAKF"
echo "EXIT $RC after $DUR s; peak GPU memory of the container $PEAK MiB" >> "$OUT/$TAG.out"
echo "$(date '+%F %T') | r5-$TAG | $MODE | env: $ENVS | $* | $DUR s | exit $RC | peak $PEAK MiB" >> "$OUT/gpu_runs.log"
echo "$TAG: exit $RC after $DUR s, peak $PEAK MiB -> $OUT/$TAG.out"
