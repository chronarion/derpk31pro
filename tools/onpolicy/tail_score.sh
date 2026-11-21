#!/bin/bash
# On-policy tail benchmark, step 2 (where the official model runs): score one capture.
#   tail_score.sh CAPTURE.jsonl OUT_DIR
# 1. build_sets.py: the capture's prompts + sampled tokens as a set for the reference runner (CPU)
# 2. the reference runner (the official model) over the set: its probability of every sampled token, its 64 most
#    likely tokens and its final hidden states at every position (one GPU)
# 3. tailfloor.py: the exact official floor from the hidden states (same GPU), which are then deleted
# 4. tailscore.py: OUT_DIR/NAME.json and NAME.txt (gap, excess surprise, out of nucleus, the tail curve beside its
#    floor, confident errors, per kind and index bin, bootstrap intervals over prompts)
# Compare two scores of the same prompts with: tailscore.py compare A.json B.json
# Containers are named after the capture file: score one capture of a given name at a time.
# Environment (config.env is read first):
#   NATIVE       the official checkpoint
#   GPU          the GPU for the reference pass (CDI device name or UUID), or GPU_ARGS = the docker arguments that
#                attach it (default: --device nvidia.com/gpu=$GPU)
#   RUNNER_IMG   the runner image (default derpk31pro-runner); RUNNER / REF: the runner and the official model code
#                directories (default tools/runner and third_party/moonshot-k3-ref of this repository). This runner
#                reads sequences up to 2,048 tokens: the frozen set's prompts + 1,024 tokens fit
#   SET          the frozen set's manifest (default sets/tail-v1.json beside this script); only its prompts are scored
#   GEN_CAP      tokens scored per answer (default 1024)
#   MAX_POWER_W  if set: refuse to start a GPU step unless the GPU's power limit reads at most this (nothing is
#                changed)
#   RUNNER_CACHE the runner image's kernel cache directory (default OUT_DIR/cache)
#   CAP_S        seconds before the reference pass is killed (default 5400)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
CAPF=$(readlink -f "${1:?capture.jsonl}"); OUT=${2:?output directory}; mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
: "${NATIVE:?set NATIVE (the official checkpoint) in config.env}"
NAME=$(basename "$CAPF" .jsonl); IMG=${RUNNER_IMG:-derpk31pro-runner}
RUNNER=$(readlink -f "${RUNNER:-$REPO/tools/runner}"); REF=$(readlink -f "${REF:-$REPO/third_party/moonshot-k3-ref}")
SET=$(readlink -f "${SET:-$HERE/sets/tail-v1.json}"); GEN=${GEN_CAP:-1024}
GPU_ARGS=${GPU_ARGS:---device nvidia.com/gpu=${GPU:?set GPU or GPU_ARGS}}
FAIL='Traceback|RuntimeError|EXITED|timed out|out of memory|CUDA error|Killed|AssertionError'
S=$OUT/sets/$NAME; L=$OUT/score.log
log() { echo "$(date '+%F %T') $NAME: $*" | tee -a "$L"; }
power_ok() {
  [ -z "${MAX_POWER_W:-}" ] && return 0
  local lim; lim=$(nvidia-smi ${GPU:+-i $GPU} --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
  awk -v l="$lim" -v m="$MAX_POWER_W" 'BEGIN { exit !(l + 0 > 0 && l + 0 <= m + 0) }' \
    || { log "GPU power limit reads '$lim' W (allowed <= $MAX_POWER_W): not started"; return 1; }
}
CPU="--ulimit core=0 --network none --cpus 4 --memory 16g --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hfm"
mkdir -p "${RUNNER_CACHE:=$OUT/cache}"
MNT="-v $RUNNER_CACHE:/cache -v $HERE:/code:ro -v $OUT:/out -v $NATIVE:/model:ro -v $RUNNER:/runner:ro -v $REF:/ref:ro -v $(dirname "$CAPF"):/cap:ro -v $(dirname "$SET"):/set:ro"
rm -rf "$S"; mkdir -p "$S/ref" "$OUT/state"
docker rm -f tail-build-$NAME > /dev/null 2>&1
timeout 900 docker run --rm --name tail-build-$NAME $CPU $MNT --entrypoint python3 $IMG /code/build_sets.py /model /out/sets/$NAME \
  /cap/$NAME.jsonl --main-gen $GEN --sets main --gpus 1 --suffix=-$NAME > "$S/build.log" 2>&1 \
  || { docker kill tail-build-$NAME > /dev/null 2>&1; log "build_sets failed: $(tail -1 "$S/build.log" | cut -c1-200)"; exit 1; }
BUCKETS=$(python3 -I -c "import json,sys; print(' '.join(b['name'] for b in json.load(open(sys.argv[1]))['main']['buckets']))" "$S/sets.json")
[ -n "$BUCKETS" ] || { log "build_sets wrote no bucket"; exit 1; }
for B in $BUCKETS; do
  power_ok || exit 1
  C=tail-ref-$B; t0=$(date +%s); mkdir -p "$S/ref/$B"
  docker rm -f $C > /dev/null 2>&1
  docker run -d --name $C $GPU_ARGS --ulimit core=0 --memory 56g --cpus 8 --user $(id -u):$(id -g) -e HOME=/tmp \
    -e OMP_NUM_THREADS=8 -e PYTHONUNBUFFERED=1 $MNT -w /runner --entrypoint python3 $IMG -m k31.stream /model \
    /out/sets/$NAME/$B.pt /out/sets/$NAME/ref/$B --ref /ref --universes /runner/universes/reference.json \
    --state-dir /out/state/$C --readers 4 --no-record --dump-hidden /out/sets/$NAME/ref/$B/hidden.npy > /dev/null \
    || { log "docker run failed"; exit 1; }
  rc=$(timeout ${CAP_S:-5400} docker wait $C) || { docker kill $C > /dev/null 2>&1; rc="timed out"; }
  docker logs $C > "$S/ref/$B.log" 2>&1; docker rm $C > /dev/null 2>&1; rm -rf "$OUT/state/$C"
  [ "$rc" = 0 ] && [ -s "$S/ref/$B/kld.pt" ] || { rm -f "$S/ref/$B/hidden.npy"
    log "reference pass FAILED (rc $rc): $(grep -E "$FAIL" "$S/ref/$B.log" | tail -1 | cut -c1-200)"; exit 1; }
  log "reference pass $B done in $(( $(date +%s) - t0 )) s ($(grep -cE "$FAIL" "$S/ref/$B.log") failure lines)"
done
power_ok || exit 1
docker rm -f tail-floor-$NAME > /dev/null 2>&1
timeout 1800 docker run --rm --name tail-floor-$NAME $GPU_ARGS --ulimit core=0 --memory 56g --cpus 8 --user $(id -u):$(id -g) \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 $MNT --entrypoint python3 $IMG /code/tailfloor.py /model /runner /out/sets/$NAME \
  /out/sets/$NAME/ref --gen-cap $GEN > "$S/floor.log" 2>&1 \
  || { docker kill tail-floor-$NAME > /dev/null 2>&1; log "tailfloor did not finish (the score falls back to the 64-token bracket): $(tail -1 "$S/floor.log" | cut -c1-200)"; }
rm -f "$S"/ref/*/hidden.npy
docker rm -f tail-score-$NAME > /dev/null 2>&1
timeout 900 docker run --rm --name tail-score-$NAME $CPU $MNT --entrypoint python3 $IMG /code/tailscore.py score /out/sets/$NAME \
  /out/sets/$NAME/ref /cap/$NAME.jsonl --ids /set/$(basename "$SET") --gen-cap $GEN --label $NAME --out /out/$NAME.json \
  > "$OUT/$NAME.txt" 2> "$S/score.err" \
  || { docker kill tail-score-$NAME > /dev/null 2>&1; log "tailscore failed: $(tail -1 "$S/score.err" | cut -c1-200)"; exit 1; }
log "SCORE $(sed -n 3p "$OUT/$NAME.txt" | tr -s ' ' | cut -c1-400)"
