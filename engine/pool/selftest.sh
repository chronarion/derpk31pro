#!/bin/bash
# Build the Grace pool and run its bit-exact self-test (both builds: SMMLA and SDOT) in the vLLM image, CPU only.
#   engine/pool/selftest.sh CPUS [HI2_FILE]
# CPUS: the container's CPU set (e.g. 67-69; pools of every size that fits it are tested, up to 48 threads).
# HI2_FILE: a record file (e.g. /models/k31-hi2/L01.hi2) for the checks on three real records as well.
# Exit code 0 = both builds PASS. Runs at nice 19; no GPU, no network.
set -u
D=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$D/../.." && pwd)
CPUS=${1:?usage: selftest.sh CPUS [HI2_FILE]}
HI2=${2:-}
IMG=${IMG:-vllm/vllm-openai:kimi-k31}
MNT=()
ARG=""
if [ -n "$HI2" ]; then
  MNT=(-v "$(dirname "$HI2")":/hi2:ro)
  ARG="--hi2 /hi2/$(basename "$HI2")"
fi
nice -n 19 docker run --rm --network none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$REPO":/repo "${MNT[@]}" -w /repo --entrypoint bash "$IMG" -c "
set -u
nice -n 19 engine/pool/build.sh /repo/build/pool || exit 2
rc=0
for b in cpustage_test cpustage_test_sdot; do
  echo \"== \$b\"
  nice -n 19 /repo/build/pool/\$b $ARG || rc=1
done
exit \$rc"
