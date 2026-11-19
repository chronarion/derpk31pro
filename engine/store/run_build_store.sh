#!/bin/bash
# Build the record store HI2_DIR from the official checkpoint NATIVE (config.env): each expert's 2-bit hi2 record and
# its lo2 complement, per MoE layer, in MegaMoE's kernel layout (build_store.py). Runs in the vLLM image on the GB300
# (vLLM's own loader lays the weights out). ~1.4 TB written; hours, bound by the drive.
# Arguments go to build_store.py: e.g. `--format hi2-k2` for records with packed scales (8.2% smaller hi2 records, the
# bank maps *-k2.json; an existing hi2-k1 store converts in minutes with store/convert_store.py instead).
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
: "${NATIVE:?set NATIVE in config.env}" "${HI2_DIR:?set HI2_DIR in config.env}"
GB=${GB:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "GB300") {print $1; exit}')}
mkdir -p "$HI2_DIR"
docker run --rm --name k31-build-store --network none --device "nvidia.com/gpu=$GB" --ipc host --ulimit core=0 \
  -v "$REPO/engine":/engine:ro -v "$NATIVE":/native:ro -v "$HI2_DIR":/out -e K31_ENGINE_DIR=/engine \
  --entrypoint python3 "${IMG:-vllm/vllm-openai:kimi-k31}" /engine/store/build_store.py /native /out \
  --parts hi2,lo2 "$@"
