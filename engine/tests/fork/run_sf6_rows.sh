#!/bin/bash
# derpk31pro: 6-bit scale rows (K31_SF6) on the GB300: tests/fork/sf6_rows_test.py in one fresh container of the
# kimi-k31 image, the fork's two headers bind-mounted over the image's (the engine's deployment form).
# usage: run_sf6_rows.sh [TAG] ["ENV=VAL ..."]      e.g. run_sf6_rows.sh sf6rows; run_sf6_rows.sh sf6rows-pdl "PDL=1"
# Output: $OUT/TAG.out (default OUT=$PWD/out). GB300 only, --rm, no core dumps, cores 48-51, no network, at most
# TIMEOUT seconds (default 600; the container is killed on a hang). Needs ~12 GB of GPU memory.
set -u
TAG=${1:-sf6rows}; ENVS=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
ENGINE=$(cd "$HERE/../.." && pwd)
OVERLAY=${OVERLAY:-$ENGINE/overlay/megamoe_k31}
OUT=${OUT:-$PWD/out}
IMG=${IMG:-vllm/vllm-openai:kimi-k31}
GB=${GB:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "GB300") {print $1; exit}')}
DG=/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm
K=include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh
S=include/deep_gemm/scheduler/mega_moe.cuh
mkdir -p "$OUT" "$OUT/torch_ext"
EARGS=""
for kv in $ENVS; do EARGS="$EARGS -e $kv"; done
START=$(date +%s)
timeout ${TIMEOUT:-600} docker run --rm --name k31-$TAG --device nvidia.com/gpu=$GB \
  --ulimit core=0 --cpuset-cpus 48-51 --cpus 4 --memory ${MEM:-48g} --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 -e FORK=/overlay -e MODE=fork -e FORK_VIA=mount -e ENGINE=/engine \
  -e TORCH_EXTENSIONS_DIR=/out/torch_ext $EARGS \
  -v "$ENGINE":/engine:ro -v "$HERE":/tests:ro -v "$OVERLAY":/overlay:ro -v "$OUT":/out \
  -v "$OVERLAY/$K":"$DG/$K":ro -v "$OVERLAY/$S":"$DG/$S":ro ${JITCACHE:+-v $JITCACHE:/tmp/.dj} \
  --entrypoint python3 $IMG /tests/sf6_rows_test.py > "$OUT/$TAG.out" 2>&1
RC=$?
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then docker kill k31-$TAG >/dev/null 2>&1; fi
echo "EXIT $RC after $(( $(date +%s) - START )) s" >> "$OUT/$TAG.out"
echo "$TAG: exit $RC -> $OUT/$TAG.out"
grep -h "^1\. \|^2\. \|^3\. \|^done\|DIFFERS\|WRONG\|Traceback\|Error" "$OUT/$TAG.out" | cut -c1-240
exit $RC
