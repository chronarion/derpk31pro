#!/bin/bash
# derpk31pro gate (a), second reading: every arm's dumps against the stock arm's, bit for bit, in a container without
# a GPU. usage: run_cmp_dumps.sh [DUMP_SUBDIR (default dump)] [ARM ...]
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/out}
SUB=${1:-dump}; shift
docker run --rm --name r5-cmp-dumps --ulimit core=0 --cpuset-cpus 48-51 --cpus 4 --memory 16g --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HERE":/tests:ro -v "$OUT":/out:ro --entrypoint python3 \
  ${IMG:-vllm/vllm-openai:kimi-k31} /tests/cmp_dumps.py /out/$SUB "$@" 2>&1 | grep -v "^W1117\|torch/_" | tee "$OUT/cmp_dumps-$SUB.out"
