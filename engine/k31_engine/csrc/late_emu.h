// CPU tests only: late.cu's kernels compiled for the host (g++ -DLATE_EMU), one CTA = THREADS host threads.
// __syncthreads is a barrier of the block's threads, a warp shuffle an exchange through a per-warp buffer between two
// warp barriers, __shared__ a function-static (one block runs at a time), and the PTX memory operations become host
// atomics (acquire loads, release stores, a full fence; "volatile" 16-byte loads as four volatile words). The
// arithmetic intrinsics are the IEEE operations they name (rsqrtf is exact here; the GPU's is within 2 ulp).
// Build (engine/k31_engine/csrc/build_late.sh emu): g++ -O1 -std=c++20 -ffp-contract=off -DLATE_EMU -shared -fPIC
//   -pthread -x c++ -I engine/pool -I /usr/local/cuda/include engine/k31_engine/csrc/late.cu -o liblate_emu.so
#pragma once
#include <cuda_bf16.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <thread>
#include <vector>

#undef __global__
#undef __device__
#undef __host__
#undef __forceinline__
#undef __launch_bounds__
#undef __shared__
#undef __align__
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

using std::max;
using std::min;

namespace late_emu {
struct Dim3 {
  unsigned x, y, z;
};
inline thread_local Dim3 thread_idx{0, 0, 0};
inline std::barrier<>* block_bar = nullptr;
inline std::barrier<>* warp_bar[32] = {};
inline float xchg[32][32];

// One CTA of n threads running fn() each, with threadIdx.x set.
template <typename F>
void run_block(unsigned n, F fn) {
  std::barrier<> bb((std::ptrdiff_t)n);
  std::vector<std::unique_ptr<std::barrier<>>> wb;
  for (unsigned w = 0; w < (n + 31) / 32; w++) wb.emplace_back(new std::barrier<>((std::ptrdiff_t)std::min(32u, n - 32 * w)));
  block_bar = &bb;
  for (unsigned w = 0; w < wb.size(); w++) warp_bar[w] = wb[w].get();
  std::vector<std::thread> th;
  th.reserve(n);
  for (unsigned t = 0; t < n; t++)
    th.emplace_back([t, &fn] {
      thread_idx = Dim3{t, 0, 0};
      fn();
    });
  for (auto& x : th) x.join();
  block_bar = nullptr;
}
}  // namespace late_emu

#define threadIdx late_emu::thread_idx

inline void __syncthreads() { late_emu::block_bar->arrive_and_wait(); }

inline float __shfl_xor_sync(unsigned, float v, int off) {
  const unsigned t = threadIdx.x, w = t >> 5, l = t & 31;
  late_emu::xchg[w][l] = v;
  late_emu::warp_bar[w]->arrive_and_wait();
  const float r = late_emu::xchg[w][l ^ (unsigned)off];
  late_emu::warp_bar[w]->arrive_and_wait();
  return r;
}

inline uint32_t __float_as_uint(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float __uint_as_float(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline int __float2int_rn(float v) { return (int)std::lrintf(v); }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float rsqrtf(float x) { return 1.f / std::sqrt(x); }
