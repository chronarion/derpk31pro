#!/bin/bash
# Local Inference Lab's standard benchmark (llm-inference-bench, https://github.com/local-inference-lab/llm-inference-bench)
# against the running server. It runs inside the server's container: the benchmark reads the GPU through NVML and the
# serving process's arguments (max_model_len, max_num_seqs) there. Results go to $STATE/lil-bench/<time>/.
#
# usage: tools/bench/lil_bench.sh [lil_bench options]     (default: --profile standard --no-upload --no-p2pmark)
#   LIL_SRC   a checkout of llm-inference-bench (default: $STATE/llm-inference-bench, cloned on first use)
#   LIL_REF   the commit to check out on first use (default: 52216d7, lil-bench 1.7.1)
set -eu
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
STATE=${STATE:-$REPO/state}
case $STATE in /*) ;; *) STATE=$REPO/$STATE ;; esac
NAME=${NAME:-derpk31pro}
PORT=${PORT:-8113}
SRC=${LIL_SRC:-$STATE/llm-inference-bench}
if [ ! -d "$SRC" ]; then
  git clone -q https://github.com/local-inference-lab/llm-inference-bench.git "$SRC"
  git -C "$SRC" checkout -q "${LIL_REF:-52216d7}"
fi
docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || { echo "lil_bench.sh: server container $NAME is not running" >&2; exit 1; }
OUT=$STATE/lil-bench/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
docker exec "$NAME" rm -rf /tmp/lil-bench
docker cp -q "$SRC" "$NAME:/tmp/lil-bench"
docker exec "$NAME" mkdir -p /tmp/lil-bench/out
[ $# -gt 0 ] || set -- --profile standard --no-upload --no-p2pmark
docker exec -w /tmp/lil-bench/out -e HOME=/tmp/lil-bench/out -e PYTHONPATH=/tmp/lil-bench -e LLM_BENCH_NO_UPDATE_CHECK=1 "$NAME" \
  python3 -m lil_bench run --url "http://127.0.0.1:$PORT" "$@" 2>&1 | tee "$OUT/lil-bench.log"
rc=${PIPESTATUS[0]}
docker cp -q "$NAME:/tmp/lil-bench/out/." "$OUT/"
echo "lil_bench.sh: results in $OUT"
exit $rc
