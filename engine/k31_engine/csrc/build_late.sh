#!/bin/bash
# The late tier's GB300 kernels (late.cu) inside the vLLM image (build.sh calls this):
#   build_late.sh [OUT]        liblate.so for the GB300 (sm_103a), default engine/band/liblate.so
#   build_late.sh emu [OUT]    the same kernels for the host (CPU tests: late_emu.h), default build/late/liblate_emu.so
set -e
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
if [ "${1:-}" = emu ]; then
  out="${2:-$repo/build/late/liblate_emu.so}"; mkdir -p "$(dirname "$out")"
  g++ -O1 -std=c++20 -ffp-contract=off -DLATE_EMU -shared -fPIC -pthread -Wall -Wno-unknown-pragmas -Wno-enum-compare -Wno-deprecated-enum-enum-conversion \
    -x c++ -I "$repo/engine/pool" -I /usr/local/cuda/include "$here/late.cu" -o "$out"
  exit 0
fi
out="${1:-$repo/engine/band/liblate.so}"; mkdir -p "$(dirname "$out")"
nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC -Xcompiler -Wno-enum-compare \
  -I "$repo/engine/pool" -o "$out" "$here/late.cu"
