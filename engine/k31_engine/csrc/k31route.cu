// K3.1 MoE router for decode-size calls (T <= 8 rows) in one launch.
//
// Gate GEMV: BF16 weights [E, K] (E <= 1024, K % 8 == 0), BF16 rows [T, K], float32 logits. Four warps per expert,
// each over a quarter of K in 16-byte chunks (7 per lane at K = 7168, all in flight), summed in shared memory; two
// experts per CTA of 8 warps (448 CTAs for 896 experts).
// Selection (the last CTA to finish, one warp per row), as vLLM's grouped_topk for one group and sigmoid scoring:
//   s = sigmoid(logit); b = s + bias; the top-k of b in descending order (ties: lower expert id first);
//   weights = the chosen s / their sum (renormalize) x scale.
// PDL: waits for its predecessor at entry and releases its dependents at once (they wait for its completion).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int WPE = 2;                   // warps per expert
constexpr int EPC = WARPS / WPE;         // experts per CTA
constexpr int MAXT = 8;
constexpr int PER = 32;                  // experts per lane in the selection (E <= 1024)
constexpr int TOPK = 16;

struct RouteArgs {
  const __nv_bfloat16* x;   // [T, K]
  const __nv_bfloat16* w;   // [E, K]
  const float* bias;        // [E]
  float* logits;            // [MAXT, E] workspace
  unsigned* counter;        // zero between calls (the last CTA resets it)
  int* ids;                 // [T, TOPK]
  float* wt;                // [T, TOPK]
  int T, K, E, renorm;
  float scale;
};

__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void fma8(float& acc, const uint4& wv, const uint4& xv) {
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = __bfloat1622float2(w2[i]);
    const float2 b = __bfloat1622float2(x2[i]);
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
  }
}

template <int TT>
__device__ __forceinline__ void gemv(const RouteArgs& a, int e, int q, int lane, float (&acc)[MAXT]) {
  const int nchunk = a.K >> 3;
  const int per_q = (nchunk + WPE - 1) / WPE;
  const int c0 = q * per_q, c1 = min(nchunk, c0 + per_q);
  const uint4* wr = reinterpret_cast<const uint4*>(a.w + (size_t)e * a.K);
  constexpr int U = 8;                   // chunks per lane in flight
  for (int base = c0 + lane; base < c1; base += 32 * U) {
    uint4 wv[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = base + u * 32;
      if (c < c1) {
        wv[u] = __ldg(wr + c);
      } else {
        wv[u] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = base + u * 32;
      if (c < c1) {
#pragma unroll
        for (int t = 0; t < TT; ++t) {
          const uint4 xv = __ldg(reinterpret_cast<const uint4*>(a.x + (size_t)t * a.K) + c);
          fma8(acc[t], wv[u], xv);
        }
      }
    }
  }
}

template <int TT>
__global__ void __launch_bounds__(THREADS) route_kernel(const RouteArgs a) {
  pdl_enter();
  __shared__ float part[EPC][WPE][MAXT];
  __shared__ unsigned s_last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int le = warp / WPE, q = warp % WPE;
  const int e = blockIdx.x * EPC + le;
  float acc[MAXT];
#pragma unroll
  for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
  if (e < a.E) gemv<TT>(a, e, q, lane, acc);
#pragma unroll
  for (int t = 0; t < TT; ++t) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], off);
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TT; ++t) part[le][q][t] = acc[t];
  }
  __syncthreads();
  if (threadIdx.x < EPC * TT) {
    const int l2 = threadIdx.x / TT, t = threadIdx.x % TT;
    const int e2 = blockIdx.x * EPC + l2;
    if (e2 < a.E) {
      float s = 0.f;
#pragma unroll
      for (int qq = 0; qq < WPE; ++qq) s += part[l2][qq][t];   // fixed order: deterministic
      a.logits[t * a.E + e2] = s;
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(a.counter, 1u) == gridDim.x - 1) ? 1u : 0u;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (warp < a.T) {
    // one warp per row. Keys: order-preserving unsigned bits of the biased score (0 = taken / no expert); each lane
    // keeps its own best; a round is two warp reductions (redux.sync: max key, then the lowest expert id holding it)
    // and a rescan by the winning lane only.
    const int t = warp;
    float sc[PER];
    unsigned uk[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int ee = lane + 32 * i;
      sc[i] = 0.f;
      uk[i] = 0u;
      if (ee < a.E) {
        const float lg = __ldcg(a.logits + t * a.E + ee);
        const float s = 1.0f / (1.0f + expf(-lg));
        const unsigned u = __float_as_uint(s + __ldg(a.bias + ee));
        sc[i] = s;
        uk[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      }
    }
    unsigned bk = 0u;
    int bi = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if (uk[i] > bk) { bk = uk[i]; bi = i; }         // strict: the lower expert id keeps a tie
    }
    float myval = 0.f, sum = 0.f;
    int myid = 0;
#pragma unroll 1
    for (int k = 0; k < TOPK; ++k) {
      const unsigned mk = __reduce_max_sync(0xffffffffu, bk);
      const unsigned ix = __reduce_min_sync(0xffffffffu, (bk == mk && mk != 0u) ? (unsigned)(lane + 32 * bi)
                                                                               : 0xffffffffu);
      const int owner = (int)(ix & 31u);
      float s_own = 0.f;
      if (lane == owner) {
        bk = 0u;
        int nb = 0;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
          if (i == bi) { s_own = sc[i]; uk[i] = 0u; }
          else if (uk[i] > bk) { bk = uk[i]; nb = i; }
        }
        bi = nb;
      }
      const float s_win = __shfl_sync(0xffffffffu, s_own, owner);
      if (lane == k) { myval = s_win; myid = (int)ix; }
      sum += s_win;
    }
    if (lane < TOPK) {
      a.ids[t * TOPK + lane] = myid;
      a.wt[t * TOPK + lane] = (a.renorm ? myval / sum : myval) * a.scale;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) *a.counter = 0u;
}

int g_pdl = 1;

template <int TT>
int launch(const RouteArgs& a, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((a.E + EPC - 1) / EPC);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, route_kernel<TT>, a);
  return e != cudaSuccess ? (int)e : (int)cudaGetLastError();
}

}  // namespace

extern "C" {

struct K31Route {
  const void* x; const void* w; const void* bias; void* logits; void* counter; void* ids; void* wt;
  int T, K, E, renorm;
  float scale;
};
static_assert(sizeof(K31Route) == sizeof(RouteArgs), "argument block");

void k31_route_pdl(int on) { g_pdl = on; }

int k31_route(const K31Route* p, void* stream) {
  const RouteArgs& a = *reinterpret_cast<const RouteArgs*>(p);
  if (a.T < 1 || a.T > MAXT || a.E > 32 * PER || (a.K & 7)) return -1;
  cudaStream_t s = (cudaStream_t)stream;
  switch (a.T) {
    case 1: return launch<1>(a, s);
    case 2: return launch<2>(a, s);
    case 3: return launch<3>(a, s);
    case 4: return launch<4>(a, s);
    case 5: return launch<5>(a, s);
    case 6: return launch<6>(a, s);
    case 7: return launch<7>(a, s);
    default: return launch<8>(a, s);
  }
}

}  // extern "C"
