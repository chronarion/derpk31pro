#!/bin/bash
# Build the Grace pool with g++ (run inside the vLLM kimi-k31 image: the top-level build.sh calls it; selftest.sh and
# bench_pool.sh too). Neoverse V2 code: SMMLA (i8mm) and SDOT; no SVE.
#   engine/pool/build.sh [OUT]
# Outputs (git-ignored): engine/band/libcpustage.so (the engine's, loaded by k31_engine/cpupool.py), OUT/cpustage_test
# (self-test and benches) and OUT/cpustage_test_sdot (the same with the SDOT form of the integer block: the same bits).
# OUT defaults to build/pool.
set -eu
D=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$D/../.." && pwd)
OUT=${1:-$REPO/build/pool}
CXX=${CXX:-g++}
FLAGS="-O3 -mcpu=neoverse-v2 -std=c++17 -ffp-contract=off -Wall -Wextra -pthread"
mkdir -p "$OUT" "$REPO/engine/band"
$CXX $FLAGS -fPIC -shared "$D/cpustage.cpp" -o "$REPO/engine/band/libcpustage.so"
$CXX $FLAGS "$D/cpustage.cpp" "$D/cpustage_test.cpp" -o "$OUT/cpustage_test"
$CXX $FLAGS -DCPUSTAGE_NO_I8MM "$D/cpustage.cpp" "$D/cpustage_test.cpp" -o "$OUT/cpustage_test_sdot"
