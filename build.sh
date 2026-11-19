#!/bin/bash
# Build every native library from source inside the vLLM kimi-k31 image (its CUDA 13 nvcc); nothing is installed on
# the host. Outputs (all git-ignored):
#   engine/band/    the engine's libraries (GB300, sm_103a): librtxband_gb300.so, libbandhost.so, libbandbulk.so,
#                   librtxband_fused_v4.so (the fused band send / finish), libk31route.so, k31i8.so (int8 GEMV),
#                   libcpustage.so (the Grace pool, g++; build/pool/ gets its self-test and benches),
#                   liblate.so (the late tier's send / landing, K31_LATE)
#   build/sidecar/  the RTX band sidecar's libraries: librtxband.so, librtxband_svc.so (RTX PRO 6000, sm_120a; and
#                   librtxband_k2.so, librtxband_svc_k2.so for hi2-k2 record stores), libbandhost.so,
#                   librtxband_gb300.so
# usage: ./build.sh            (IMG overrides the image)
set -eu
REPO=$(cd "$(dirname "$0")" && pwd)
IMG=${IMG:-vllm/vllm-openai:kimi-k31}
mkdir -p "$REPO/engine/band" "$REPO/build/sidecar" "$REPO/build/k31i8" "$REPO/build/inc"
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$REPO":/repo -w /repo \
  --entrypoint bash "$IMG" -c '
set -eu
GB="-gencode arch=compute_103a,code=sm_103a"
echo "sidecar libraries"
sidecar/build.sh all /repo/build/sidecar
cp build/sidecar/librtxband_gb300.so build/sidecar/libbandhost.so engine/band/
echo "fused band kernels"
nvcc -O3 -std=c++17 $GB -shared -Xcompiler -fPIC -o engine/band/librtxband_fused_v4.so sidecar/band/rtxband_fused.cu
echo "router"
nvcc -O3 -std=c++17 $GB -shared -Xcompiler -fPIC -o engine/band/libk31route.so engine/k31_engine/csrc/k31route.cu
echo "late tier (K31_LATE)"
engine/k31_engine/csrc/build_late.sh engine/band/liblate.so
nvcc -O3 -std=c++17 $GB -shared -Xcompiler -fPIC -o engine/band/libbandbulk.so engine/k31_engine/csrc/bandbulk.cu
echo "int8 GEMV (torch extension)"
# torch'"'"'s CUDA headers include cusparse.h / cusolverDn.h: take only those from pip'"'"'s CUDA include directory
cp /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/cusparse*.h \
   /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/cusolver*.h /repo/build/inc/
K31I8_INC=/repo/build/inc python3 engine/csrc/k31i8/build.py /repo/build/k31i8 > /repo/build/k31i8/build.log 2>&1 \
  || { tail -20 /repo/build/k31i8/build.log; exit 1; }
cp build/k31i8/k31i8.so engine/band/
echo "Grace pool (g++, Neoverse V2)"
engine/pool/build.sh /repo/build/pool
ls -la engine/band build/sidecar
'
echo "built"
