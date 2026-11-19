#!/bin/bash
# Write the stripped checkpoint MODEL from the official checkpoint NATIVE (config.env): every tensor except the routed
# experts, one file per layer, with Tomas Cupr's kstrip.py (third_party/tomcupr). Runs in the runner image
# (docker build -t derpk31pro-runner tools/runner/docker). Log: $STATE/kstrip.log.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
: "${NATIVE:?set NATIVE in config.env}" "${MODEL:?set MODEL in config.env}"
STATE=${STATE:-$REPO/state}; mkdir -p "$STATE" "$MODEL"
docker run --rm --name k31-kstrip --network none --cpus 8 --memory 96g --ulimit core=0 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$REPO":/proj:ro -v "$NATIVE":/native:ro -v "$MODEL":/out \
  --entrypoint python3 "${RUNNER_IMG:-derpk31pro-runner}" /proj/third_party/tomcupr/kstrip.py /native /out --workers 8 \
  > "$STATE/kstrip.log" 2>&1 && echo "stripped: $(du -sh "$MODEL" | cut -f1) in $MODEL" || { tail -20 "$STATE/kstrip.log"; exit 1; }
