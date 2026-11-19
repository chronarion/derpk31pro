#!/bin/bash
# Launch a reference-runner pass (k31.stream) in a detached container on one GPU, reading the official checkpoint
# (NATIVE in config.env, read-only); the universes' hidden states go to memmaps under OUT/state.
# usage: run_pass.sh NAME GPU_UUID TOKENS UNIVERSES OUT [extra k31.stream args]
#   paths as seen in the container: /scratch = $STATE/runner (OUT is relative to it or under it), /proj = the repo
#   (read-only). Follow it with: docker logs -f NAME
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
: "${NATIVE:?set NATIVE in config.env}"
STATE=${STATE:-$REPO/state}; mkdir -p "$STATE/runner"
NAME=$1 GPU=$2 TOKENS=$3 UNIV=$4 OUT=$5
shift 5
docker run -d --name "$NAME" --device "nvidia.com/gpu=$GPU" --ulimit core=0 --memory "${MEM:-100g}" --cpus 4 \
  ${CPUS:+--cpuset-cpus "$CPUS"} --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=4 -e PYTHONUNBUFFERED=1 \
  -v "$NATIVE":/model:ro -v "$REPO":/proj:ro -v "$STATE/runner":/scratch -w /proj/tools/runner \
  --entrypoint python3 "${RUNNER_IMG:-derpk31pro-runner}" -m k31.stream /model "$TOKENS" "$OUT" \
  --ref /proj/third_party/moonshot-k3-ref --universes "$UNIV" --state-dir "$OUT/state" "$@"
