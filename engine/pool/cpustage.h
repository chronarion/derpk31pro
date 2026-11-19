// derpk31pro Grace pool: the kernel that computes 2-bit expert records (format hi2-k1) where they lie, on the
// Neoverse V2 cores. Header-only core, shared by the pool (cpustage.cpp) and the tests (cpustage_test.cpp).
//
// A job: P records (picks), T rows; for each record e and each row t that uses it (routing weight w[e][t] != 0)
//     gate, up = x[t] . W13_e^T        (int8 x with a power-of-two scale per 32 inputs; 2-bit weights, block scales)
//     a        = SiTU(bf16(gate), bf16(up)) * w[e][t]
//     y_e[t]   = bf16( int8(a) . W2_e^T )
//     out[t]   = sum over e of y_e[t]  (float32, record order)
// The record is read in place (the layer's pinned bank): weights in MegaMoE's kernel layout as build_store.py wrote
// them (w13 rows interleave gate / up by 8; one scale byte per 32 inputs in the UTCCP row order).
//
// Arithmetic (fixed, so that a scalar reference can match it bit for bit; cpustage_test.cpp holds that reference):
//   weights      code pair c -> 1, 4, -1, -4 (the staged nibble (c << 2) | 3 = 1.5, 6, -1.5, -6), times the block
//                scale 1.5 * 2^(sb - 127) (sb = the record's scale byte; 0 = an empty block, above 254 read as 254)
//   activations  int8 q and one exponent e per 32 inputs: v ~ q * 2^e (the rule below the includes); the GB300 kernel
//                quantizes x, this kernel the intermediate
//   a dot        per (weight row, row, 32-block) the exact integer D = sum c * q (SMMLA: a 2 x 2 block of two weight
//                rows x two rows at once), then acc = fma(float(D), sw * sx, acc) over the blocks in ascending order,
//                float32. Both products are exact (powers of two times 1.5 and |D| <= 16,256), so the result does
//                not depend on fused or separate multiply-add
//   SiTU         MegaMoE's form (sigmoid = 1 / (1 + exp(-v)), tanh(v) = 2 sigmoid(2 v) - 1) with IEEE division and
//                the single-precision exp below instead of the GPU's approximate ones
//   rounding     gate, up and each record's output rounded to bfloat16 (nearest even) as MegaMoE does
//
// Build: g++ -O3 -mcpu=neoverse-v2 -std=c++17 -ffp-contract=off ... (needs dotprod; i8mm unless -DCPUSTAGE_NO_I8MM)
#pragma once
#include <arm_neon.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// The segment (kept byte for byte; its one static_assert compares two enums, which C++ warns about).
// What it leaves implicit, as this pool and the GB300's send read it:
//   activation slot s (CPUSEG2_OFF_X + s * CPUSEG2_X_STRIDE): int8 [CPUSEG_MAXT][CPUSEG_H] row-planar rows at +0, then
//   uint8 [CPUSEG_MAXT][CPUSEG_KB] exponent bytes at +CPUSEG_MAXT * CPUSEG_H (28,672); result slot s: float32
//   [T][CPUSEG_H] (rows no pick uses are 0); done line s: u64 tag, u64 status (CPUSEG_ST_ERROR = refused);
//   state block: u64 words written by the pool (+0 running, see cpustage.cpp for the rest).
// Quantized activations: block b of a row = inputs 32 b .. 32 b + 31; amax = their largest
// magnitude; e = ((float_bits(amax) + 0x1FFFF) >> 23) - 133 clamped to [e_min, 120] (the smallest e with
// amax * 2^-e <= 127); q[k] = round-to-nearest-even(v[k] * 2^-e) clamped to [-127, 127]; exponent byte e + 128.
// Row-planar order: input k = 32 b + 4 i + p (p = 0..3, i = 0..7) is byte 32 b + 8 p + i of the row (cpuseg_planar).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
#include "cpustage_seg2.h"
#pragma GCC diagnostic pop

namespace cpustage {

constexpr int H = 3584;                  // latent: K of linear 1, rows of linear 2
constexpr int I = 3072;                  // intermediate: gate / up rows of linear 1, K of linear 2
constexpr int R1 = 2 * I, R2 = H;        // weight rows of w13 (kernel layout) and w2
constexpr int KB1 = H / 32, KB2 = I / 32;   // 32-wide K-blocks: 112, 96
constexpr int RB1 = H / 4, RB2 = I / 4;     // code bytes per weight row: 896, 768
constexpr long long C13 = (long long)R1 * RB1;            // 5,505,024
constexpr long long C2 = (long long)R2 * RB2;             // 2,752,512
constexpr long long S13 = (long long)(KB1 / 4) * R1 * 4;  //   688,128
constexpr long long S2 = (long long)(KB2 / 4) * R2 * 4;   //   344,064
constexpr long long REC_BYTES = C13 + C2 + S13 + S2;
static_assert(REC_BYTES == 9289728, "hi2-k1 record size");
static_assert(H == CPUSEG_H && KB1 == CPUSEG_KB, "segment constants");
static_assert(KB1 % 4 == 0 && KB2 % 4 == 0, "scale words hold four K-blocks");
constexpr int MAXT = CPUSEG_MAXT, MAXPAIR = CPUSEG_MAXT / 2, PMAX = CPUSEG2_PMAX, NSLOT = CPUSEG2_NSLOT;
constexpr int L1_BLOCKS = R1 / 64;       // 96 units of linear 1: 64 weight rows = 32 neurons = one K-block of linear 2
constexpr int L2_PAIRS = R2 / 2;         // 1792 row pairs of linear 2

// The slots of the segment (cpustage_seg2.h): job header, activation rows and their exponent bytes, result rows, done
// line (u64 tag, u64 status).
constexpr size_t X_OFF_E = (size_t)MAXT * H;   // exponent bytes after the int8 rows inside an activation slot
static_assert(X_OFF_E + (size_t)MAXT * KB1 == CPUSEG2_X_BYTES, "activation slot layout");
static inline uint8_t* seg_job(uint8_t* seg, int s) { return seg + CPUSEG2_OFF_JOB + (size_t)s * CPUSEG2_JOB_BYTES; }
static inline uint8_t* seg_xq(uint8_t* seg, int s) { return seg + CPUSEG2_OFF_X + (size_t)s * CPUSEG2_X_STRIDE; }
static inline uint8_t* seg_xe(uint8_t* seg, int s) { return seg_xq(seg, s) + X_OFF_E; }
static inline uint8_t* seg_res(uint8_t* seg, int s) { return seg + CPUSEG2_OFF_RES + (size_t)s * CPUSEG2_RES_BYTES; }
static inline uint8_t* seg_done(uint8_t* seg, int s) { return seg + CPUSEG2_OFF_DONE + 128 * (size_t)s; }

// One linear of a record.
struct Lin {
  const uint8_t* codes;    // [rows][kb * 8]: 2 bits per weight, k ascending from bit 0 of each byte
  const uint8_t* scales;   // [kb / 4][rows, scale-row order][4]: byte j of a word = K-block 4 q + j
  int rows, kb, rowbytes;
};
static inline Lin lin1(const uint8_t* rec) { return Lin{rec, rec + C13 + C2, R1, KB1, RB1}; }
static inline Lin lin2(const uint8_t* rec) { return Lin{rec + C13, rec + C13 + C2 + S13, R2, KB2, RB2}; }

// Scale-tensor row of weight row r (the kernel layout's order; vLLM DeepGemmMegaMoEExperts._kernel_layout_shard).
static inline int scale_row(int r) { return 128 * (r / 128) + 4 * (r % 32) + (r % 128) / 32; }

static inline float f32_from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t bits_from_f32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// Weight block scale 1.5 * 2^(sb - 127); sb 0 = an empty block (0).
static inline float wscale(uint32_t sb) {
  if (sb == 0) return 0.0f;
  if (sb > 254) sb = 254;
  return f32_from_bits((sb << 23) | 0x00400000u);
}
// Activation block scale 2^e from the stored exponent byte e + 128.
static inline float xscale(uint32_t eb) { return f32_from_bits(((eb - 1u) & 0xFFu) << 23); }
// Block exponent of a magnitude: the smallest e with amax * 2^-e <= 127, clamped to [emin, 120].
static inline int qexp(float amax, int emin) {
  const int e = (int)((bits_from_f32(amax) + 0x1FFFFu) >> 23) - 133;
  return e < emin ? emin : (e > 120 ? 120 : e);
}

// ---------------------------------------------------------------------------------------------------------------
// float32 pieces (NEON). The scalar twins in cpustage_test.cpp use the same operations in the same order.
// ---------------------------------------------------------------------------------------------------------------
static inline float32x4_t round_bf16(float32x4_t v) {            // nearest even, the result as float32
  uint32x4_t u = vreinterpretq_u32_f32(v);
  const uint32x4_t odd = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
  u = vaddq_u32(u, vaddq_u32(odd, vdupq_n_u32(0x7FFFu)));
  return vreinterpretq_f32_u32(vandq_u32(u, vdupq_n_u32(0xFFFF0000u)));
}

// exp for |x| <= 87 (clamped): n = rint(x log2 e), r = x - n ln 2 (two fused steps), a degree-5 polynomial (Cephes'
// coefficients) by fused Horner, scaled by 2^n. About 1 ulp.
static inline float32x4_t exp_ps(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(87.0f));
  const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
  r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
  float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
  const float32x4_t r2 = vmulq_f32(r, r);
  float32x4_t y = vfmaq_f32(r, p, r2);
  y = vaddq_f32(y, vdupq_n_f32(1.0f));
  const int32x4_t ni = vcvtq_s32_f32(n);
  const float32x4_t p2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
  return vmulq_f32(y, p2);
}

struct Situ {
  float alpha, beta, inv_alpha, inv_beta;      // inv = 1.0f / x (float32 division), as MegaMoE's constants
};
static inline Situ make_situ(float alpha, float beta) {
  Situ s;
  s.alpha = alpha; s.beta = beta;
  s.inv_alpha = 1.0f / alpha;
  s.inv_beta = beta > 0.0f ? 1.0f / beta : 0.0f;
  return s;
}

// alpha tanh(g / alpha) sigmoid(g) * beta tanh(u / beta)   (beta 0: * u), in MegaMoE's order of operations.
static inline float32x4_t situ_ps(float32x4_t g, float32x4_t u, const Situ& s) {
  const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f), m2 = vdupq_n_f32(-2.0f);
  const float32x4_t sig = vdivq_f32(one, vaddq_f32(one, exp_ps(vnegq_f32(g))));
  const float32x4_t eg = exp_ps(vmulq_f32(m2, vmulq_n_f32(g, s.inv_alpha)));
  const float32x4_t th = vsubq_f32(vmulq_f32(two, vdivq_f32(one, vaddq_f32(one, eg))), one);
  const float32x4_t gated = vmulq_f32(vmulq_n_f32(th, s.alpha), sig);
  if (s.beta > 0.0f) {
    const float32x4_t eu = exp_ps(vmulq_f32(m2, vmulq_n_f32(u, s.inv_beta)));
    const float32x4_t tu = vsubq_f32(vmulq_f32(two, vdivq_f32(one, vaddq_f32(one, eu))), one);
    u = vmulq_n_f32(tu, s.beta);
  }
  return vmulq_f32(gated, u);
}

// ---------------------------------------------------------------------------------------------------------------
// The integer core.
// ---------------------------------------------------------------------------------------------------------------
// r[2 i + j] += sum over k < 8 of a[8 i + k] * b[8 j + k]: two weight rows (a) x two rows (b), 8 inputs.
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(CPUSTAGE_NO_I8MM)
static inline int32x4_t mmla(int32x4_t r, int8x16_t a, int8x16_t b) { return vmmlaq_s32(r, a, b); }
#else
static inline int32x4_t mmla(int32x4_t r, int8x16_t a, int8x16_t b) {      // the same block from two SDOTs
  static const uint8_t perm[16] = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7};
  const int32x4_t z = vdupq_n_s32(0);
  const int32x4_t d = vdotq_s32(z, a, b);                       // [a0l.b0l, a0h.b0h, a1l.b1l, a1h.b1h]
  const int32x4_t e = vdotq_s32(z, a, vextq_s8(b, b, 8));       // [a0l.b1l, a0h.b1h, a1l.b0l, a1h.b0h]
  const int32x4_t s = vpaddq_s32(d, e);                         // [a0.b0, a1.b1, a0.b1, a1.b0]
  return vaddq_s32(r, vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(s), vld1q_u8(perm))));
}
#endif

// Code values by nibble: a record byte holds four pairs c0..c3 from bit 0 (weights k = 4 i + 0..3); its low nibble is
// c0 | c1 << 2, its high nibble c2 | c3 << 2.
alignas(16) static const int8_t TBL_LO[16] = {1, 4, -1, -4, 1, 4, -1, -4, 1, 4, -1, -4, 1, 4, -1, -4};   // low pair
alignas(16) static const int8_t TBL_HI[16] = {1, 1, 1, 1, 4, 4, 4, 4, -1, -1, -1, -1, -4, -4, -4, -4};   // high pair

// 8 code bytes of each of two weight rows (one K-block: 32 weights per row) -> four planes; plane p holds, for
// i = 0..7, the weight of input 4 i + p of row 0 in byte i and of row 1 in byte 8 + i (an SMMLA operand).
static inline void decode_planes(const uint8_t* c0, const uint8_t* c1, int8x16_t& w0, int8x16_t& w1, int8x16_t& w2,
                                 int8x16_t& w3) {
  const int8x16_t tlo = vld1q_s8(TBL_LO), thi = vld1q_s8(TBL_HI);
  const uint8x16_t v = vcombine_u8(vld1_u8(c0), vld1_u8(c1));
  const uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0F)), hi = vshrq_n_u8(v, 4);
  w0 = vqtbl1q_s8(tlo, lo); w1 = vqtbl1q_s8(thi, lo); w2 = vqtbl1q_s8(tlo, hi); w3 = vqtbl1q_s8(thi, hi);
}

// The block scales of two weight rows for the four K-blocks of one scale word each: a (row r0), b (row r0 + 1) ->
// sw[j] = {s(a_j), s(a_j), s(b_j), s(b_j)} as float32 (the lane order of an SMMLA result), s = wscale. Built on the
// bytes: the float bits of wscale(sb) are, low byte first, [0, 0, (sb & 1) << 7 | (sb != 0) << 6, sb >> 1] with sb
// clamped to 254 (0 for sb 0), so one table lookup per output vector places them.
alignas(16) static const uint8_t SCALE_IDX[4][16] = {
    {255, 255, 0, 8, 255, 255, 0, 8, 255, 255, 4, 12, 255, 255, 4, 12},
    {255, 255, 1, 9, 255, 255, 1, 9, 255, 255, 5, 13, 255, 255, 5, 13},
    {255, 255, 2, 10, 255, 255, 2, 10, 255, 255, 6, 14, 255, 255, 6, 14},
    {255, 255, 3, 11, 255, 255, 3, 11, 255, 255, 7, 15, 255, 255, 7, 15}};
static inline __attribute__((always_inline)) void scale_vectors(uint32_t a, uint32_t b, float32x4_t* sw) {
  const uint8x8_t v = vcreate_u8((uint64_t)a | ((uint64_t)b << 32));
  const uint8x8_t c = vmin_u8(v, vdup_n_u8(254));
  const uint8x8_t lo = vorr_u8(vshl_n_u8(c, 7), vand_u8(vtst_u8(v, v), vdup_n_u8(0x40)));
  const uint8x16_t t = vcombine_u8(lo, vshr_n_u8(c, 1));
  for (int j = 0; j < 4; ++j) sw[j] = vreinterpretq_f32_u8(vqtbl1q_u8(t, vld1q_u8(SCALE_IDX[j])));
}

// The block scales of weight rows r0 and r0 + 1 as sw[kb][4] (tests; the GEMV builds them inline).
static inline void pair_scales(const Lin& L, int r0, float* sw) {
  const uint8_t* s0 = L.scales + 4 * (size_t)scale_row(r0);
  const uint8_t* s1 = L.scales + 4 * (size_t)scale_row(r0 + 1);
  const size_t sq = (size_t)L.rows * 4;
  for (int q = 0; q < L.kb / 4; ++q) {
    uint32_t a, b;
    std::memcpy(&a, s0 + q * sq, 4);
    std::memcpy(&b, s1 + q * sq, 4);
    float32x4_t v[4];
    scale_vectors(a, b, v);
    for (int j = 0; j < 4; ++j) vst1q_f32(sw + 4 * (4 * q + j), v[j]);
  }
}

// Weight rows r0, r0 + 1 (r0 even) of linear L against NP row pairs. xop[p]: [kb][4 planes][16 B] (bytes 0..7 the
// pair's first row, 8..15 its second), xs4[p]: [kb][4] = {sa, sb, sa, sb}. acc[p] = {r0.a, r0.b, r1.a, r1.b}.
// pf: bytes ahead for the software prefetch of the codes (the next row pair of the same thread).
template <int NP>
static inline __attribute__((always_inline)) void gemv_pair(const Lin& L, int r0, int pf, const int8_t* const* xop,
                                                            const float* const* xs4, float32x4_t* acc) {
  const uint8_t* c0 = L.codes + (size_t)r0 * (size_t)L.rowbytes;
  const uint8_t* c1 = c0 + L.rowbytes;
  const uint8_t* s0 = L.scales + 4 * (size_t)scale_row(r0);
  const uint8_t* s1 = L.scales + 4 * (size_t)scale_row(r0 + 1);
  const size_t sq = (size_t)L.rows * 4;
  float32x4_t a[NP];
  for (int p = 0; p < NP; ++p) a[p] = vdupq_n_f32(0.0f);
  for (int q = 0; q < L.kb / 4; ++q) {
    uint32_t wa, wb;
    std::memcpy(&wa, s0 + q * sq, 4);
    std::memcpy(&wb, s1 + q * sq, 4);
    float32x4_t sw[4];
    scale_vectors(wa, wb, sw);
    for (int j = 0; j < 4; ++j) {
      const int b = 4 * q + j;
      if ((b & 7) == 0) {
        __builtin_prefetch(c0 + 8 * b + pf, 0, 3);
        __builtin_prefetch(c1 + 8 * b + pf, 0, 3);
      }
      int8x16_t w0, w1, w2, w3;
      decode_planes(c0 + 8 * b, c1 + 8 * b, w0, w1, w2, w3);
      for (int p = 0; p < NP; ++p) {
        const int8_t* x = xop[p] + 64 * b;
        int32x4_t d = mmla(vdupq_n_s32(0), w0, vld1q_s8(x));
        d = mmla(d, w1, vld1q_s8(x + 16));
        d = mmla(d, w2, vld1q_s8(x + 32));
        d = mmla(d, w3, vld1q_s8(x + 48));
        a[p] = vfmaq_f32(a[p], vcvtq_f32_s32(d), vmulq_f32(sw[j], vld1q_f32(xs4[p] + 4 * b)));
      }
    }
  }
  for (int p = 0; p < NP; ++p) acc[p] = a[p];
}

static inline void gemv(const Lin& L, int r0, int pf, const int8_t* const* xop, const float* const* xs4, int np,
                        float32x4_t* acc) {
  switch (np) {
    case 1: gemv_pair<1>(L, r0, pf, xop, xs4, acc); break;
    case 2: gemv_pair<2>(L, r0, pf, xop, xs4, acc); break;
    case 3: gemv_pair<3>(L, r0, pf, xop, xs4, acc); break;
    default: gemv_pair<4>(L, r0, pf, xop, xs4, acc); break;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// A job.
// ---------------------------------------------------------------------------------------------------------------
struct JobBufs {      // operands of one job (the pool owns one; ~530 KB)
  alignas(64) int8_t xop[PMAX * MAXPAIR][KB1 * 64];    // linear 1's inputs: one array per distinct row pair
  alignas(64) float xs4[PMAX * MAXPAIR][KB1 * 4];
  alignas(64) int8_t aop[PMAX][MAXPAIR][KB2 * 64];     // linear 2's inputs (the quantized intermediate) per pick
  alignas(64) float as4[PMAX][MAXPAIR][KB2 * 4];
};

struct Job {
  int T = 0, P = 0;
  uint32_t flags = CPUJOB_OUT_BF16;
  int emin = -100;
  Situ situ{4.0f, 25.0f, 0.25f, 0.04f};
  const uint8_t* rec[PMAX] = {};
  int nrow[PMAX] = {};               // rows that use pick p
  int rows[PMAX][MAXT];              // their indices, ascending
  float w[PMAX][MAXT];               // routing weight by row index (0 = unused)
  const int8_t* xop[PMAX][MAXPAIR];  // linear 1's operand of each pick's row pair (shared between picks with the same
  const float* xs4[PMAX][MAXPAIR];   //   pair of rows)
  JobBufs* buf = nullptr;
  float* out = nullptr;              // [T][H] float32
};

// v [K] (K % 32 == 0) -> row-planar int8 q [K] and exponent bytes [K / 32] (the rule at the top). Scalar: the
// GB300 kernel does this for x in production; tests and the one-shot path use it.
static inline void quant_row(const float* v, int K, int emin, int8_t* q, uint8_t* eb) {
  for (int b = 0; b < K / 32; ++b) {
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) {
      const float m = std::fabs(v[32 * b + k]);
      if (m > amax) amax = m;
    }
    const int e = qexp(amax, emin);
    const float inv = f32_from_bits((uint32_t)(127 - e) << 23);
    for (int k = 0; k < 32; ++k) {
      long r = std::lrint(v[32 * b + k] * inv);
      if (r > 127) r = 127;
      if (r < -127) r = -127;
      q[32 * b + 8 * (k & 3) + (k >> 2)] = (int8_t)r;
    }
    eb[b] = (uint8_t)(e + 128);
  }
}

// Row-planar rows xa, xb (xb null: the pair has one row) -> an operand array op [kb][4][16] and scales s4 [kb][4].
static inline void build_pair(const int8_t* xa, const uint8_t* ea, const int8_t* xb, const uint8_t* eb, int kb,
                              int8_t* op, float* s4) {
  const int64x2_t z = vdupq_n_s64(0);
  for (int b = 0; b < kb; ++b) {
    const int64x2_t a0 = vreinterpretq_s64_s8(vld1q_s8(xa + 32 * b));
    const int64x2_t a1 = vreinterpretq_s64_s8(vld1q_s8(xa + 32 * b + 16));
    const int64x2_t b0 = xb != nullptr ? vreinterpretq_s64_s8(vld1q_s8(xb + 32 * b)) : z;
    const int64x2_t b1 = xb != nullptr ? vreinterpretq_s64_s8(vld1q_s8(xb + 32 * b + 16)) : z;
    vst1q_s8(op + 64 * b, vreinterpretq_s8_s64(vzip1q_s64(a0, b0)));          // plane 0: bytes 0..7 of both rows
    vst1q_s8(op + 64 * b + 16, vreinterpretq_s8_s64(vzip2q_s64(a0, b0)));     // plane 1: bytes 8..15
    vst1q_s8(op + 64 * b + 32, vreinterpretq_s8_s64(vzip1q_s64(a1, b1)));     // plane 2: bytes 16..23
    vst1q_s8(op + 64 * b + 48, vreinterpretq_s8_s64(vzip2q_s64(a1, b1)));     // plane 3: bytes 24..31
    const float sa = xscale(ea[b]), sb = xb != nullptr ? xscale(eb[b]) : 0.0f;
    s4[4 * b] = sa; s4[4 * b + 1] = sb; s4[4 * b + 2] = sa; s4[4 * b + 3] = sb;
  }
}

// The rows of each pick and linear 1's operands (one array per distinct pair of rows). xq [T][H] row-planar,
// xe [T][KB1]. J.T, J.P, J.w, J.buf are set.
static inline void prepare(Job& J, const int8_t* xq, const uint8_t* xe) {
  int key[PMAX * MAXPAIR], nb = 0;
  for (int p = 0; p < J.P; ++p) {
    int n = 0;
    for (int t = 0; t < J.T; ++t)
      if (J.w[p][t] != 0.0f) J.rows[p][n++] = t;
    J.nrow[p] = n;
    for (int q = 0; q < (n + 1) / 2; ++q) {
      const int ta = J.rows[p][2 * q], tb = 2 * q + 1 < n ? J.rows[p][2 * q + 1] : -1;
      const int k = 16 * ta + tb + 1;
      int i = 0;
      while (i < nb && key[i] != k) ++i;
      if (i == nb) {
        key[nb++] = k;
        build_pair(xq + (size_t)ta * H, xe + (size_t)ta * KB1, tb >= 0 ? xq + (size_t)tb * H : nullptr,
                   tb >= 0 ? xe + (size_t)tb * KB1 : nullptr, KB1, J.buf->xop[i], J.buf->xs4[i]);
      }
      J.xop[p][q] = J.buf->xop[i];
      J.xs4[p][q] = J.buf->xs4[i];
    }
  }
}

// Linear 1 + SiTU + routing weight + quantization for unit u (0..95): weight rows 64 u .. 64 u + 63 = gate and up of
// neurons 32 u .. 32 u + 31 = K-block u of linear 2, for every pick. Writes block u of the picks' linear-2 operands.
// act (optional, tests): [P][MAXT][32] the activations before quantization, by the pick's row position.
static inline void l1_block(const Job& J, int u, float* act = nullptr) {
  alignas(64) float g[MAXT][32], up[MAXT][32];
  for (int p = 0; p < J.P; ++p) {
    const int nr = J.nrow[p], np = (nr + 1) / 2;
    if (nr == 0) continue;
    const Lin L = lin1(J.rec[p]);
    for (int rp = 0; rp < 32; ++rp) {
      const int r0 = 64 * u + 2 * rp;             // kernel rows r0, r0 + 1: gate (h < 4) or up of two neighbours
      float32x4_t acc[MAXPAIR];
      gemv(L, r0, 2 * RB1, J.xop[p], J.xs4[p], np, acc);
      const int h = rp & 7, n32 = 8 * (rp >> 3) + 2 * (h & 3);
      float (*dst)[32] = (h & 4) ? up : g;
      for (int q = 0; q < np; ++q) {
        dst[2 * q][n32] = vgetq_lane_f32(acc[q], 0);
        dst[2 * q + 1][n32] = vgetq_lane_f32(acc[q], 1);
        dst[2 * q][n32 + 1] = vgetq_lane_f32(acc[q], 2);
        dst[2 * q + 1][n32 + 1] = vgetq_lane_f32(acc[q], 3);
      }
    }
    for (int j = 0; j < nr; ++j) {
      const float wt = J.w[p][J.rows[p][j]];
      alignas(16) float a[32];
      float32x4_t mx = vdupq_n_f32(0.0f);
      for (int n = 0; n < 32; n += 4) {
        float32x4_t gv = vld1q_f32(&g[j][n]), uv = vld1q_f32(&up[j][n]);
        if (!(J.flags & CPUJOB_NO_BF16)) { gv = round_bf16(gv); uv = round_bf16(uv); }
        const float32x4_t av = vmulq_n_f32(situ_ps(gv, uv, J.situ), wt);
        vst1q_f32(a + n, av);
        mx = vmaxq_f32(mx, vabsq_f32(av));
      }
      if (act != nullptr) std::memcpy(act + ((size_t)p * MAXT + j) * 32, a, sizeof(a));
      const int e = qexp(vmaxvq_f32(mx), J.emin);
      const float inv = f32_from_bits((uint32_t)(127 - e) << 23);
      alignas(16) int32_t q32[32];
      for (int n = 0; n < 32; n += 4) {
        int32x4_t qi = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(a + n), inv));
        qi = vminq_s32(vmaxq_s32(qi, vdupq_n_s32(-127)), vdupq_n_s32(127));
        vst1q_s32(q32 + n, qi);
      }
      int8_t* op = J.buf->aop[p][j >> 1] + 64 * u;
      const int half = 8 * (j & 1);
      for (int n = 0; n < 32; ++n) op[16 * (n & 3) + (n >> 2) + half] = (int8_t)q32[n];
      const float s = xscale((uint32_t)(e + 128));
      float* s4 = J.buf->as4[p][j >> 1] + 4 * u;
      s4[j & 1] = s;
      s4[2 + (j & 1)] = s;
    }
    if (nr & 1) {                                 // the last pair has one row: its second half reads as zero
      int8_t* op = J.buf->aop[p][nr >> 1] + 64 * u;
      for (int pl = 0; pl < 4; ++pl) std::memset(op + 16 * pl + 8, 0, 8);
      float* s4 = J.buf->as4[p][nr >> 1] + 4 * u;
      s4[1] = 0.0f;
      s4[3] = 0.0f;
    }
  }
}

// Linear 2 for weight row pairs [rp0, rp1) (output elements 2 rp, 2 rp + 1), every pick, the sum over picks, and the
// store of all T rows' elements.
static inline void l2_pairs(const Job& J, int rp0, int rp1) {
  for (int rp = rp0; rp < rp1; ++rp) {
    const int r0 = 2 * rp;
    float tmp[MAXT][2];
    for (int t = 0; t < J.T; ++t) { tmp[t][0] = 0.0f; tmp[t][1] = 0.0f; }
    for (int p = 0; p < J.P; ++p) {
      const int nr = J.nrow[p], np = (nr + 1) / 2;
      if (nr == 0) continue;
      const int8_t* xo[MAXPAIR];
      const float* xs[MAXPAIR];
      for (int q = 0; q < np; ++q) { xo[q] = J.buf->aop[p][q]; xs[q] = J.buf->as4[p][q]; }
      float32x4_t acc[MAXPAIR];
      gemv(lin2(J.rec[p]), r0, 2 * RB2, xo, xs, np, acc);
      for (int q = 0; q < np; ++q) {
        float32x4_t y = acc[q];
        if (J.flags & CPUJOB_OUT_BF16) y = round_bf16(y);
        const int ta = J.rows[p][2 * q];
        tmp[ta][0] += vgetq_lane_f32(y, 0);
        tmp[ta][1] += vgetq_lane_f32(y, 2);
        if (2 * q + 1 < nr) {
          const int tb = J.rows[p][2 * q + 1];
          tmp[tb][0] += vgetq_lane_f32(y, 1);
          tmp[tb][1] += vgetq_lane_f32(y, 3);
        }
      }
    }
    for (int t = 0; t < J.T; ++t) {
      J.out[(size_t)t * H + r0] = tmp[t][0];
      J.out[(size_t)t * H + r0 + 1] = tmp[t][1];
    }
  }
}

// The whole job on the calling thread (tests, the bind-time self-check).
static inline void run_serial(const Job& J) {
  for (int u = 0; u < L1_BLOCKS; ++u) l1_block(J, u);
  l2_pairs(J, 0, L2_PAIRS);
}

// Software prefetch of the scale words a thread will read for weight rows [r0, r1) of one linear (they sit in kb / 4
// separate planes of the record; rows of one 128-row group share 512 B per plane).
static inline void prefetch_scales(const Lin& L, int r0, int r1) {
  const int g0 = r0 / 128, g1 = (r1 + 127) / 128;
  for (int q = 0; q < L.kb / 4; ++q) {
    const uint8_t* p = L.scales + (size_t)q * (size_t)L.rows * 4;
    for (int g = g0; g < g1; ++g)
      for (int o = 0; o < 512; o += 64) __builtin_prefetch(p + (size_t)g * 512 + o, 0, 3);
  }
}

}  // namespace cpustage
