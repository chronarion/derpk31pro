// RTX band of derpk31pro: the kernel core shared by the launched kernels (rtxband.cu) and the persistent service
// kernel (rtxband_svc.cu). Layouts, the MMA and its operand expansion, one row tile's GEMM, SiTU, the two linears'
// CTA bodies, and the per-warp shares of the entry path (scale range, repack). Design notes: rtxband.cu.
//
// Templates on SVC: false = a launched kernel (plain global loads); true = the persistent kernel, where a kernel
// boundary never flushes L1, so every global load of data that changes between calls goes around L1 (ld.global.cg).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#ifndef BAND_PF
#define BAND_PF 4            // weight steps in flight per warp; must divide 28 and 24 (1, 2, 4)
#endif
#ifndef BAND_WARPS
#define BAND_WARPS 8         // warps (row tiles) per CTA: 4, 8 or 16
#endif
#ifndef BAND_WAIT_NS
#define BAND_WAIT_NS 0       // pause between two polls of a waiting CTA (0 = spin; 200 and 1000 measured no different)
#endif
#ifndef BAND_CG
#define BAND_CG 0            // 1: the launched kernels also load around L1 (a probe of what that costs)
#endif

namespace band {

constexpr int H = 3584;                 // routed-expert latent: K of w13, rows of w2
constexpr int I = 3072;                 // intermediate: gate / up rows of w13, K of w2
constexpr int MAXT = 8;                 // rows (tokens) per call = the MMA's N
constexpr int MAXE = 16;                // distinct experts per call
constexpr int CSTEP = 512;              // codes of one step: 32 lanes x 16 B
constexpr int SSTEP = 32;               // scale codes of one step: 16 rows x 4 K-blocks x 4 bits
constexpr int KQ1 = H / 128;            // 28 steps per w13 row tile
constexpr int KQ2 = I / 128;            // 24 steps per w2 row tile
constexpr int RT1 = 2 * I / 16;         // 384 row tiles of w13
constexpr int RT2 = H / 16;             // 224 row tiles of w2
// band record: w13 codes [tile][step][512], w13 scale codes [tile][step][32], w2 codes, w2 scale codes
constexpr long long W13C = (long long)RT1 * KQ1 * CSTEP;         // 5,505,024
constexpr long long W13S = (long long)RT1 * KQ1 * SSTEP;         //   344,064
constexpr long long W2C = (long long)RT2 * KQ2 * CSTEP;          // 2,752,512
constexpr long long W2S = (long long)RT2 * KQ2 * SSTEP;          //   172,032
constexpr long long REC_BYTES = W13C + W13S + W2C + W2S;         // 8,773,632
constexpr int BREC = 1056;              // activation step record: 32 lanes x 4 K-blocks x 8 B + 8 tokens x 4 scale B
constexpr int XF_BYTES = KQ1 * BREC;    // 29,568: the call's input
constexpr int AQ_BYTES = KQ2 * BREC;    // 25,344 per expert: linear 2's input
constexpr int W = BAND_WARPS;
constexpr int THREADS = W * 32;
constexpr int CTAS1 = RT1 / W;          // linear-1 units (CTAs) per expert
constexpr int CTAS2 = RT2 / W;          // linear-2 units (CTAs) per expert
constexpr int SMEM_BYTES = XF_BYTES + W * MAXT * 4;
// counters (int32, zero before the first call; every call leaves them zero):
constexpr int SYNC_L1 = 0;              // [MAXE]  linear-1 CTAs of expert e that have stored their activations
constexpr int SYNC_READ = MAXE;         // [MAXE]  linear-2 CTAs of expert e that have read them (fused launch)
constexpr int SYNC_PART = 2 * MAXE;     // [CTAS2] experts whose partial output of a row group is stored
constexpr int SYNC_ERR = 2 * MAXE + CTAS2;   // set if a wait gave up (a CTA it waited for never arrived): output invalid
constexpr int SYNC_WORDS = 2 * MAXE + CTAS2 + 1;
constexpr int SPIN_LIMIT = 1 << 21;     // polls before a wait gives up (of the order of a second): a broken launch
                                        // must not hang the GPU
static_assert(RT1 % W == 0 && RT2 % W == 0 && W % 4 == 0, "warps per CTA");
static_assert(KQ1 % BAND_PF == 0 && KQ2 % BAND_PF == 0, "lookahead must divide the steps of both linears");
static_assert(REC_BYTES == 8773632, "record size");

constexpr int FLAG_WEIGHT_AFTER = 1;    // routing weight applied to linear 2's output (the runner's order) instead of
                                        // to the activation before its FP8 cast (MegaMoE's order)
constexpr int FLAG_NO_BF16 = 2;         // do not round gate / up to bfloat16 before SiTU
constexpr int FLAG_ACT_BF16 = 4;        // round the activation to bfloat16 before the FP8 cast (the runner does)
constexpr int FLAG_PRECISE = 8;         // SiTU with expf / tanhf instead of MegaMoE's fast-math forms
constexpr int FLAG_OUT_BF16 = 16;       // round each expert's linear-2 output to bfloat16 before the sum over experts
                                        // (MegaMoE's combine and the runner both do)
constexpr int FLAG_STOP_FILL = 256;     // timing probes: return after the activation fill / after the GEMM
constexpr int FLAG_STOP_GEMM = 512;
constexpr int FLAG_PER_EXPERT = 1024;   // bulk units (prompt chunks, launched kernels only): expert slot e of the
                                        // call has its own 8 rows, so its input is xf + e XF_BYTES and its weighted
                                        // output rows go to y + e MAXT H, with no sum over experts

struct BandCall {                       // one call's work list (152 x 4 B)
  int n_exp;                            // distinct band experts (0..16)
  int n_rows;                           // rows T (1..8), informative: rows are masked by their weights
  int flags;
  int sf_min;                           // lowest UE8M0 code of the intermediate's block scales (DeepGEMM: 105)
  float alpha, beta;                    // SiTU: alpha tanh(gate / alpha) sigmoid(gate) * beta tanh(up / beta); beta 0 = up
  int pad0, pad1;
  int slot[MAXE];                       // band store slot of expert j
  float w[MAXE][MAXT];                  // routing weight of row t for expert j; 0 = the row does not use it
};
static_assert(sizeof(BandCall) == 4 * (8 + MAXE + MAXE * MAXT), "BandCall layout");

struct Bufs {                           // device buffers of a band, fixed for its lifetime
  const uint8_t* store;                 // band records by slot
  const uint8_t* sbase;                 // one byte per slot: the record's lowest scale code
  uint8_t* aq;                          // [MAXE][AQ_BYTES] linear 2's input, written by linear 1
  float* ypart;                         // [MAXE - 1][MAXT][H] partial outputs
  float* y;                             // [rows][H] the call's output (the service: the result rows in the mailbox)
  int* sync;                            // [SYNC_WORDS]
};

// Global loads. cg = L2 only.
template <bool SVC>
__device__ __forceinline__ uint4 ld_u4(const uint8_t* p) {
  if (SVC || BAND_CG) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
  }
  return *reinterpret_cast<const uint4*>(p);
}
template <bool SVC>
__device__ __forceinline__ uint32_t ld_u16(const uint8_t* p) {
  if (SVC || BAND_CG) {
    uint32_t v;
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
  }
  return *reinterpret_cast<const uint16_t*>(p);
}
template <bool SVC>
__device__ __forceinline__ uint32_t ld_u8(const uint8_t* p) {
  if (SVC || BAND_CG) {
    uint32_t v;
    asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
  }
  return *p;
}

// Wait until *counter == target; false (and the error flag set) if it does not get there.
__device__ __forceinline__ bool wait_for(int* counter, int target, int* err) {
  for (int i = 0; i < SPIN_LIMIT; i++) {
    if (atomicAdd(counter, 0) == target) return true;
#if BAND_WAIT_NS
    __nanosleep(BAND_WAIT_NS);                             // a waiting CTA should not take issue slots from working ones
#endif
  }
  atomicExch(err, 1);
  return false;
}

// One MMA: acc[16 weight rows x 8 tokens] += A[16 x 32, e2m1] . B[32 x 8, e4m3], block scales by byte BID of sa / sb
// (A row g <- lane 4g, row g + 8 <- lane 4g + 1; B column g <- lane 4g: probe/probe_mma.py).
#define BAND_MMA(BID)                                                                                                \
  asm volatile(                                                                                                      \
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "          \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {" #BID ", 0}, {%11}, {" #BID ", 0};\n"          \
      : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)                                                               \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb))

// 32 bits of 2-bit codes -> the four A registers of one MMA. Byte j of x holds, from bit 0 up, the pairs of
// (row g, k), (row g + 8, k), (row g, k + 16), (row g + 8, k + 16) for k = 4 t + j: the pair of register i's byte j.
// Container = nibble << 2 with nibble = (pair << 2) | 3 (bits 7:6 are ignored by the MMA).
#define BAND_EXPAND(x)                                                                                               \
  a0 = (((x) << 4) & 0x30303030u) | 0x0C0C0C0Cu;                                                                     \
  a1 = (((x) << 2) & 0x30303030u) | 0x0C0C0C0Cu;                                                                     \
  a2 = ((x) & 0x30303030u) | 0x0C0C0C0Cu;                                                                            \
  a3 = (((x) >> 2) & 0x30303030u) | 0x0C0C0C0Cu

// A warp's weight stream: BAND_PF steps (codes + scale codes) in flight.
struct Ahead {
  uint4 c[BAND_PF];
  uint32_t s[BAND_PF];
};

template <bool SVC>
__device__ __forceinline__ void gemm_begin(Ahead& q, const uint8_t* __restrict__ wc, const uint8_t* __restrict__ ws) {
#pragma unroll
  for (int i = 0; i < BAND_PF; i++) {
    q.c[i] = ld_u4<SVC>(wc + i * CSTEP);
    q.s[i] = ld_u16<SVC>(ws + i * SSTEP);
  }
}

// One row tile: NSTEP steps of weights from global memory (wc = this lane's codes, ws = its row's scale codes,
// base4 = the record's lowest scale code in every byte) against the activation step records in shared memory.
template <int NSTEP, bool SVC>
__device__ __forceinline__ void gemm_tile(Ahead& q, const uint8_t* __restrict__ wc, const uint8_t* __restrict__ ws,
                                          uint32_t base4, const uint8_t* __restrict__ bp, int lane, float& acc0,
                                          float& acc1, float& acc2, float& acc3) {
  const uint8_t* bx = bp + 32 * lane;                        // my 4 x 8 B of activations
  const uint8_t* bs = bp + 1024 + 4 * (lane >> 2);           // token g's scale word
#pragma unroll 1
  for (int s0 = 0; s0 < NSTEP; s0 += BAND_PF) {
    const bool more = s0 + BAND_PF < NSTEP;
#pragma unroll
    for (int i = 0; i < BAND_PF; i++) {
      const uint32_t s16 = q.s[i];
      uint32_t sa = (s16 | (s16 << 8)) & 0x00FF00FFu;        // four 4-bit codes -> four scale bytes
      sa = ((sa | (sa << 4)) & 0x0F0F0F0Fu) + base4;
      const uint4 x0 = *reinterpret_cast<const uint4*>(bx + i * BREC);
      const uint4 x1 = *reinterpret_cast<const uint4*>(bx + i * BREC + 16);
      const uint32_t sb = *reinterpret_cast<const uint32_t*>(bs + i * BREC);
      uint32_t a0, a1, a2, a3, b0, b1;
      BAND_EXPAND(q.c[i].x); b0 = x0.x; b1 = x0.y; BAND_MMA(0);
      BAND_EXPAND(q.c[i].y); b0 = x0.z; b1 = x0.w; BAND_MMA(1);
      BAND_EXPAND(q.c[i].z); b0 = x1.x; b1 = x1.y; BAND_MMA(2);
      BAND_EXPAND(q.c[i].w); b0 = x1.z; b1 = x1.w; BAND_MMA(3);
      if (more) {                                            // refill the slot: the step BAND_PF ahead
        q.c[i] = ld_u4<SVC>(wc + (BAND_PF + i) * CSTEP);
        q.s[i] = ld_u16<SVC>(ws + (BAND_PF + i) * SSTEP);
      }
    }
    wc += BAND_PF * CSTEP; ws += BAND_PF * SSTEP; bx += BAND_PF * BREC; bs += BAND_PF * BREC;
  }
}

// Straight copy of n uint4 from global to shared memory, the loads of a batch issued before its stores.
template <bool SVC>
__device__ __forceinline__ void copy_to_smem(uint4* __restrict__ dst, const uint8_t* __restrict__ src, int n, int tid) {
  constexpr int B = 4;
  for (int i = tid; i < n; i += B * THREADS) {
    uint4 v[B];
#pragma unroll
    for (int j = 0; j < B; j++) if (i + j * THREADS < n) v[j] = ld_u4<SVC>(src + 16LL * (i + j * THREADS));
#pragma unroll
    for (int j = 0; j < B; j++) if (i + j * THREADS < n) dst[i + j * THREADS] = v[j];
  }
}

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float rcp_fast(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// SiTU. Fast form = DeepGEMM's sm100_fp8_fp4_mega_moe_situ.cuh with fast_math: sigmoid(v) = rcp(1 + exp(-v)),
// tanh(v) = 2 sigmoid(2 v) - 1.
__device__ __forceinline__ float situ(float gate, float up, float alpha, float beta, bool precise) {
  if (precise) {
    const float sig = 1.0f / (1.0f + expf(-gate));
    const float gated = alpha * tanhf(gate / alpha) * sig;
    if (beta > 0.0f) up = beta * tanhf(up / beta);
    return gated * up;
  }
  const float sig = rcp_fast(1.0f + __expf(-gate));
  const float th = 2.0f * rcp_fast(1.0f + __expf(-2.0f * (gate * (1.0f / alpha)))) - 1.0f;
  const float gated = (th * alpha) * sig;
  if (beta > 0.0f) up = (2.0f * rcp_fast(1.0f + __expf(-2.0f * (up * (1.0f / beta)))) - 1.0f) * beta;
  return gated * up;
}

// UE8M0 code of the smallest power of two s with amax / s <= 448 (DeepGEMM's get_ue8m0_sf_exp), floored at sf_min.
__device__ __forceinline__ uint32_t sf_code(float amax, int sf_min) {
  const uint32_t r = (__float_as_uint(amax) + 0x7FFFFFu - 0x600000u) >> 23;
  return (uint32_t)max((int)r - 8, sf_min);
}

// ---------------------------------------------------------------------------------------------------------------
// Linear 1 + SiTU + FP8 cast. One unit = (expert slot e, W row tiles of 8 neurons), one tile per warp, run by a whole
// CTA. Four consecutive warps are one 32-neuron block = one K-block of linear 2; the unit writes its part of expert
// e's activation step records into aq. `call` may point to global or to shared memory. `fused`: count the unit in
// sync[SYNC_L1 + e] when its bytes are stored.
// ---------------------------------------------------------------------------------------------------------------
template <bool SVC>
__device__ __forceinline__ void l1_body(const BandCall* __restrict__ call, const Bufs& b, const uint8_t* __restrict__ xf,
                                        int e, int part, uint8_t* smem, bool fused) {
  uint8_t* xsm = smem;
  float* amax_s = reinterpret_cast<float*>(smem + XF_BYTES);          // [W][MAXT]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int flags = call->flags;
  const int tile = part * W + warp;                        // row tile of w13: neurons 8 tile .. 8 tile + 7
  const int slot = call->slot[e];
  const uint8_t* rec = b.store + (long long)slot * REC_BYTES;
  const uint8_t* wc = rec + (long long)tile * (KQ1 * CSTEP) + 16 * lane;
  const uint8_t* ws = rec + W13C + (long long)tile * (KQ1 * SSTEP) + 2 * (g + 8 * (t & 1));   // row g or g + 8
  const uint32_t base4 = ld_u8<SVC>(b.sbase + slot) * 0x01010101u;
  Ahead ahead;
  gemm_begin<SVC>(ahead, wc, ws);                          // the first DRAM misses overlap the fill

  const uint8_t* xfe = (!SVC && (flags & FLAG_PER_EXPERT)) ? xf + (long long)e * XF_BYTES : xf;
  copy_to_smem<SVC>(reinterpret_cast<uint4*>(xsm), xfe, XF_BYTES / 16, tid);
  __syncthreads();
  if (flags & FLAG_STOP_FILL) return;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;   // gate(n, 2t), gate(n, 2t+1), up(n, 2t), up(n, 2t+1)
  gemm_tile<KQ1, SVC>(ahead, wc, ws, base4, xsm, lane, acc0, acc1, acc2, acc3);
  if (flags & FLAG_STOP_GEMM) { if (acc0 == 1.2345f) b.aq[0] = 1; return; }

  if (!(flags & FLAG_NO_BF16)) {
    acc0 = round_bf16(acc0); acc1 = round_bf16(acc1); acc2 = round_bf16(acc2); acc3 = round_bf16(acc3);
  }
  const float alpha = call->alpha, beta = call->beta;
  const float w0 = call->w[e][2 * t], w1 = call->w[e][2 * t + 1];
  const bool precise = flags & FLAG_PRECISE;
  float v0 = situ(acc0, acc2, alpha, beta, precise), v1 = situ(acc1, acc3, alpha, beta, precise);
  if (flags & FLAG_ACT_BF16) { v0 = round_bf16(v0); v1 = round_bf16(v1); }
  if (flags & FLAG_WEIGHT_AFTER) {
    v0 = w0 != 0.f ? v0 : 0.f; v1 = w1 != 0.f ? v1 : 0.f;
  } else {
    v0 *= w0; v1 *= w1;
  }
  // block amax per token over the 32 neurons of four warps
  float m0 = fabsf(v0), m1 = fabsf(v1);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
  }
  if (g == 0) { amax_s[warp * MAXT + 2 * t] = m0; amax_s[warp * MAXT + 2 * t + 1] = m1; }
  __syncthreads();
  const int wb = warp & ~3;
#pragma unroll
  for (int r = 0; r < 4; r++) {
    m0 = fmaxf(m0, amax_s[(wb + r) * MAXT + 2 * t]);
    m1 = fmaxf(m1, amax_s[(wb + r) * MAXT + 2 * t + 1]);
  }
  const int sf_min = call->sf_min;
  const uint32_t s0 = sf_code(m0, sf_min), s1 = sf_code(m1, sf_min);
  const float2 scaled = make_float2(v0 * __uint_as_float((254u - s0) << 23), v1 * __uint_as_float((254u - s1) << 23));
  const uint32_t q8 = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
  // neuron n32 of K-block kap of linear 2, tokens 2t and 2t+1 -> the B fragment bytes of step kap / 4
  const int kap = tile >> 2, n32 = 8 * (tile & 3) + g;
  uint8_t* arec = b.aq + ((long long)e * KQ2 + (kap >> 2)) * BREC;
  const int off = ((n32 & 15) >> 2) * 32 + (kap & 3) * 8 + 4 * (n32 >> 4) + (n32 & 3);
  arec[(4 * (2 * t)) * 32 + off] = (uint8_t)(q8 & 0xFF);
  arec[(4 * (2 * t + 1)) * 32 + off] = (uint8_t)(q8 >> 8);
  if ((tile & 3) == 0 && g == 0) {
    arec[1024 + 4 * (2 * t) + (kap & 3)] = (uint8_t)s0;
    arec[1024 + 4 * (2 * t + 1) + (kap & 3)] = (uint8_t)s1;
  }
  if (fused) {
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(&b.sync[SYNC_L1 + e], 1);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Linear 2 and the weighted sum over experts. One unit = (expert slot e, W row tiles of 16 output rows), one tile
// per warp. Experts 0..n-2 store their (weighted) partial outputs and count themselves in sync[SYNC_PART + part];
// expert n-1's unit waits for them, adds everything in expert order and stores rows < n_out of b.y; it returns true.
// `fused`: first wait for expert e's linear 1.
// ---------------------------------------------------------------------------------------------------------------
template <bool SVC>
__device__ __forceinline__ bool l2_body(const BandCall* __restrict__ call, const Bufs& b, int e, int part, int n_exp,
                                        int n_out, uint8_t* smem, bool fused) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = part * W + warp;
  const int row = tile * 16 + g;                           // output row of acc0 / acc1; acc2 / acc3 are row + 8
  float* y0 = b.y + (2 * t) * H + row;
  float* y1 = b.y + (2 * t + 1) * H + row;
  const int flags = call->flags;
  const int slot = call->slot[e];
  const uint8_t* rec = b.store + (long long)slot * REC_BYTES + W13C + W13S;
  const uint8_t* wc = rec + (long long)tile * (KQ2 * CSTEP) + 16 * lane;
  const uint8_t* ws = rec + W2C + (long long)tile * (KQ2 * SSTEP) + 2 * (g + 8 * (t & 1));
  const uint32_t base4 = ld_u8<SVC>(b.sbase + slot) * 0x01010101u;
  Ahead ahead;
  gemm_begin<SVC>(ahead, wc, ws);
  if (fused) {                                             // expert e's activations must be complete
    if (tid == 0) wait_for(&b.sync[SYNC_L1 + e], CTAS1, &b.sync[SYNC_ERR]);
    __syncthreads();
  }
  copy_to_smem<SVC>(reinterpret_cast<uint4*>(smem), b.aq + (long long)e * AQ_BYTES, AQ_BYTES / 16, tid);
  __syncthreads();
  if (fused && tid == 0) {                                 // the last reader of expert e's activations resets its counters
    if (atomicAdd(&b.sync[SYNC_READ + e], 1) == CTAS2 - 1) {
      atomicExch(&b.sync[SYNC_L1 + e], 0);
      atomicExch(&b.sync[SYNC_READ + e], 0);
    }
  }
  if (flags & FLAG_STOP_FILL) return false;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;   // y(row, 2t), y(row, 2t+1), y(row+8, 2t), y(row+8, 2t+1)
  gemm_tile<KQ2, SVC>(ahead, wc, ws, base4, smem, lane, acc0, acc1, acc2, acc3);
  if (flags & FLAG_STOP_GEMM) { if (acc0 == 1.2345f) y0[0] = 1; return false; }
  if (flags & FLAG_OUT_BF16) {
    acc0 = round_bf16(acc0); acc1 = round_bf16(acc1); acc2 = round_bf16(acc2); acc3 = round_bf16(acc3);
  }
  if (flags & FLAG_WEIGHT_AFTER) {
    const float w0 = call->w[e][2 * t], w1 = call->w[e][2 * t + 1];
    acc0 *= w0; acc1 *= w1; acc2 *= w0; acc3 *= w1;
  }
  if (!SVC && (flags & FLAG_PER_EXPERT)) {                 // bulk unit: its own 8 rows, no sum over experts
    float* q0 = b.y + (long long)(e * MAXT + 2 * t) * H + row;
    float* q1 = b.y + (long long)(e * MAXT + 2 * t + 1) * H + row;
    q0[0] = acc0; q1[0] = acc1; q0[8] = acc2; q1[8] = acc3;
    return true;
  }
  int* done = b.sync + SYNC_PART + part;
  if (e < n_exp - 1) {
    float* p0 = b.ypart + (long long)(e * MAXT + 2 * t) * H + row;
    float* p1 = b.ypart + (long long)(e * MAXT + 2 * t + 1) * H + row;
    p0[0] = acc0; p1[0] = acc1; p0[8] = acc2; p1[8] = acc3;
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(done, 1);
    return false;
  }
  if (n_exp > 1) {
    __syncthreads();
    if (tid == 0) {
      wait_for(done, n_exp - 1, &b.sync[SYNC_ERR]);
      atomicExch(done, 0);
    }
    __syncthreads();
    const volatile float* p0 = b.ypart + (long long)(2 * t) * H + row;
    const volatile float* p1 = b.ypart + (long long)(2 * t + 1) * H + row;
    float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
    constexpr int B = 8;                                   // experts whose partials are loaded before they are added
    for (int j = 0; j < n_exp - 1; j += B) {
      float v[B][4];
#pragma unroll
      for (int k = 0; k < B; k++) {
        if (j + k < n_exp - 1) {
          const long long o = (long long)(j + k) * MAXT * H;
          v[k][0] = p0[o]; v[k][1] = p1[o]; v[k][2] = p0[o + 8]; v[k][3] = p1[o + 8];
        }
      }
#pragma unroll
      for (int k = 0; k < B; k++) {
        if (j + k < n_exp - 1) { r0 += v[k][0]; r1 += v[k][1]; r2 += v[k][2]; r3 += v[k][3]; }
      }
    }
    acc0 += r0; acc1 += r1; acc2 += r2; acc3 += r3;
  }
  if (!SVC) {
    if (2 * t < n_out) { y0[0] = acc0; y0[8] = acc2; }
    if (2 * t + 1 < n_out) { y1[0] = acc1; y1[8] = acc3; }
    return true;
  }
  // The service's rows go to host memory: whole lines at once. The unit's 128 values of every row are collected in
  // shared memory (the area behind the activations, unused by linear 2), then warp k stores row k's 512 B with one
  // instruction. Scattered 4-byte stores of the same data reach the host up to 20 us later.
  static_assert(XF_BYTES - AQ_BYTES >= MAXT * W * 16 * 4 && MAXT <= W, "room for a unit's result rows");
  float* sm = reinterpret_cast<float*>(smem + AQ_BYTES);
  const int col = warp * 16 + g;
  if (2 * t < n_out) { sm[(2 * t) * (W * 16) + col] = acc0; sm[(2 * t) * (W * 16) + col + 8] = acc2; }
  if (2 * t + 1 < n_out) { sm[(2 * t + 1) * (W * 16) + col] = acc1; sm[(2 * t + 1) * (W * 16) + col + 8] = acc3; }
  __syncthreads();
  if (warp < n_out) {
    const uint4 v = *reinterpret_cast<const uint4*>(smem + AQ_BYTES + warp * (W * 64) + 16 * lane);
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(b.y + (long long)warp * H + part * (W * 16)) + 16 * lane) = v;
  }
  return true;
}

// an empty call still defines its output
__device__ __forceinline__ void zero_y(float* y, int part) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int row = (part * W + warp) * 16 + g;
  float* y0 = y + (2 * t) * H + row;
  float* y1 = y + (2 * t + 1) * H + row;
  y0[0] = 0.f; y1[0] = 0.f; y0[8] = 0.f; y1[8] = 0.f;
}

// ---------------------------------------------------------------------------------------------------------------
// The L2 prefetch hint: a record in 64 KB pieces for cp.async.bulk.prefetch.L2 (pieces of 1 MB do nothing; many
// threads issuing at once lose pieces: bench/probe_hint.py).
// ---------------------------------------------------------------------------------------------------------------
#ifndef BAND_PREFETCH_PIECE
#define BAND_PREFETCH_PIECE 65536
#endif
constexpr long long PIECE = BAND_PREFETCH_PIECE;
constexpr int PIECES = (int)((REC_BYTES + PIECE - 1) / PIECE);       // 134 per record

__device__ __forceinline__ void prefetch_piece(const uint8_t* store, int slot, int j) {
  const uint8_t* a = store + (long long)slot * REC_BYTES + (long long)j * PIECE;
  const uint32_t bytes = (uint32_t)min(PIECE, REC_BYTES - (long long)j * PIECE);
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(bytes) : "memory");
}

// ---------------------------------------------------------------------------------------------------------------
// hi2 record -> band record (what runs when a record enters the band), as per-thread / per-warp shares:
//   scale_range_share: bytewise min / max over every `stride`-th uint4 of the record's scale bytes from `first`
//   repack_part:       one step (16-row tile x 128 inputs) by one warp: permutes the 2-bit codes into fragment
//                      order and writes the scale codes as 4-bit offsets from base
// hi2 layout: engine/store/build_store.py; band layout: band/rtxband.py.
// ---------------------------------------------------------------------------------------------------------------
constexpr long long HI2_C13 = 6144LL * 896, HI2_C2 = 3584LL * 768, HI2_S13 = 28LL * 6144 * 4, HI2_S2 = 24LL * 3584 * 4;
constexpr long long HI2_BYTES = HI2_C13 + HI2_C2 + HI2_S13 + HI2_S2;
constexpr int HI2_SCALE_U4 = (int)((HI2_S13 + HI2_S2) / 16);
static_assert(HI2_BYTES == 9289728, "hi2 record size");

template <bool SVC>
__device__ __forceinline__ void scale_range_share(const uint8_t* __restrict__ hi2, int first, int stride, uint32_t& lo,
                                                  uint32_t& hi) {
  const uint8_t* sf = hi2 + HI2_C13 + HI2_C2;
  for (int i = first; i < HI2_SCALE_U4; i += stride) {
    const uint4 v = ld_u4<SVC>(sf + 16LL * i);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const uint32_t c = (w[k] >> (8 * b)) & 0xFF;
        lo = min(lo, c); hi = max(hi, c);
      }
    }
  }
}

// pairs of one hi2 code byte (4 weights of one row) -> the low two bits of the 4 bytes of a word
__device__ __forceinline__ uint32_t spread4(uint32_t b) {
  return (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u;
}

template <bool SVC>
__device__ __forceinline__ uint32_t ld_u32(const uint8_t* p) {
  if (SVC || BAND_CG) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
  }
  return *reinterpret_cast<const uint32_t*>(p);
}

// One warp per step. Lane = 4 g' + kap for the codes: it reads the four hi2 words of rows g', g' + 8 and both
// 16-input halves of K-block kap, and writes that K-block's word of lanes 4 g' + 0..3.
template <int ROWS, int KQ, bool SVC>
__device__ __forceinline__ void repack_part(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                            uint8_t* __restrict__ out_c, uint8_t* __restrict__ out_s, int step,
                                            uint32_t base4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, kap = lane & 3;
  const int tile = step / KQ, kq = step % KQ;
  const int rowbytes = KQ * 32;                            // hi2 code bytes per weight row
  const uint8_t* r0 = codes + (long long)(16 * tile + g) * rowbytes + 32 * kq + 8 * kap;
  const uint8_t* r1 = r0 + 8LL * rowbytes;
  const uint32_t w00 = ld_u32<SVC>(r0), w01 = ld_u32<SVC>(r0 + 4);
  const uint32_t w10 = ld_u32<SVC>(r1), w11 = ld_u32<SVC>(r1 + 4);
  uint32_t* dst = reinterpret_cast<uint32_t*>(out_c + (long long)step * CSTEP + (4 * g) * 16 + kap * 4);
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const int sh = 8 * t;
    dst[4 * t] = spread4((w00 >> sh) & 0xFF) | (spread4((w10 >> sh) & 0xFF) << 2) |
                 (spread4((w01 >> sh) & 0xFF) << 4) | (spread4((w11 >> sh) & 0xFF) << 6);
  }
  if (lane < 16) {                                         // scale codes of row 16 tile + lane
    const int row = 16 * tile + lane;
    const int srow = 128 * (row / 128) + 4 * (row % 32) + (row % 128) / 32;      // the kernel layout's scale row
    const uint32_t v = ld_u32<SVC>(scales + ((long long)kq * ROWS + srow) * 4) - base4;
    const uint32_t n = (v | (v >> 4)) & 0x00FF00FFu;       // bytes 0, 2 = nibble pairs (0, 1), (2, 3)
    *reinterpret_cast<uint16_t*>(out_s + (long long)step * SSTEP + 2 * lane) = (uint16_t)(n | (n >> 8));
  }
}

// every step of a record for warp `warp` of `warps`
template <bool SVC>
__device__ __forceinline__ void repack_share(const uint8_t* __restrict__ hi2, uint8_t* __restrict__ rec, int warp,
                                             int warps, uint32_t base4) {
  for (int s = warp; s < RT1 * KQ1; s += warps)
    repack_part<6144, KQ1, SVC>(hi2, hi2 + HI2_C13 + HI2_C2, rec, rec + W13C, s, base4);
  for (int s = warp; s < RT2 * KQ2; s += warps)
    repack_part<3584, KQ2, SVC>(hi2 + HI2_C13, hi2 + HI2_C13 + HI2_C2 + HI2_S13, rec + W13C + W13S,
                                rec + W13C + W13S + W2C, s, base4);
}

}  // namespace band
