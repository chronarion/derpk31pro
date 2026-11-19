#!/bin/bash
# Build the band libraries (inside a container with nvcc from CUDA 13+).
#   ./build.sh [out.so] [-DBAND_PF=2 ...]      the launched kernels only (librtxband.so)
#   ./build.sh all [dir]                       librtxband.so, librtxband_svc.so (RTX, sm_120a; and *_k2.so for hi2-k2
#                                              record stores), libbandhost.so, and librtxband_gb300.so (GB300,
#                                              sm_103a) into dir (default /work/build)
set -e
here="$(cd "$(dirname "$0")" && pwd)"
RTX="-gencode arch=compute_120a,code=sm_120a"
if [ "$1" = all ]; then
  dir="${2:-/work/build}"; mkdir -p "$dir"
  nvcc -O3 -std=c++17 $RTX -shared -Xcompiler -fPIC -o "$dir/librtxband.so" "$here/band/rtxband.cu"
  nvcc -O3 -std=c++17 $RTX -shared -Xcompiler -fPIC --resource-usage -o "$dir/librtxband_svc.so" "$here/band/rtxband_svc.cu"
  # the same for hi2-k2 record stores (2-bit scale codes from per-row bases; band/rtxband.py picks by K31_REC_FMT)
  nvcc -O3 -std=c++17 $RTX -DBAND_K2=1 -shared -Xcompiler -fPIC -o "$dir/librtxband_k2.so" "$here/band/rtxband.cu"
  nvcc -O3 -std=c++17 $RTX -DBAND_K2=1 -shared -Xcompiler -fPIC --resource-usage -o "$dir/librtxband_svc_k2.so" "$here/band/rtxband_svc.cu"
  g++ -O2 -std=c++17 -shared -fPIC -o "$dir/libbandhost.so" "$here/band/band_host.cpp"
  if [ -f "$here/band/rtxband_gb300.cu" ]; then
    nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC --resource-usage \
      -o "$dir/librtxband_gb300.so" "$here/band/rtxband_gb300.cu"
  fi
  exit 0
fi
out="${1:-/work/build/librtxband.so}"; shift || true
mkdir -p "$(dirname "$out")"
nvcc -O3 -std=c++17 $RTX -shared -Xcompiler -fPIC --resource-usage "$@" -o "$out" "$here/band/rtxband.cu"
