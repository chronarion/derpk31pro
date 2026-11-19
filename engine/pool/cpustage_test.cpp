// derpk31pro Grace pool: self-checking tests of cpustage.h / cpustage.cpp, and the pool's benches. Exit code 0 = PASS.
//
//   ./cpustage_test                       every check: single thread + pools of every size the process's CPU set holds
//                                         (1, 2, 3, 8, 24, 32, 48 threads; a few seconds per size, ~350 MB)
//   ./cpustage_test --hi2 /hi2/L01.hi2    the decode / GEMV / job checks also on real records (experts 0, 1, 2)
//   ./cpustage_test --bench 48 24 [N]     on-time jobs through the pool: 48 threads on cores 24..71, N jobs per case
//   ./cpustage_test --bench-late 48 24 [N]   late jobs (3 / 5 / 7 / 8 records), back to back and queued in pairs
//   bench options: --poster CORE (the thread that posts and waits, as the GB300 does; default core0 - 1),
//                  --records N (bank size, default 96: 892 MB, the records are cold when a job reads them)
//   The benches are load-window measurements: they spin all the pool's cores (engine/pool/bench_pool.sh).
//
// What is checked, bit for bit unless a tolerance is named:
//   1  record decode: the kernel's planes and block scales against an independent decode of the hi2-k1 layout
//      (build_store.py: 4 pairs per byte from bit 0; stager.py: nibble = (pair << 2) | 3; hook.FP4: 1.5, 6, -1.5, -6;
//      scale byte = native - 1; vLLM's kernel layout for the scale rows); the block-scale builder for all 256 bytes
//   2  the 2 x 2 integer block (SMMLA or its SDOT form) against its definition
//   3  bfloat16 rounding against a nearest-neighbour search; the activation quantizer against a scalar copy of the rule
//   4  exp and SiTU: NEON against scalar twins (same operations, same order), and against libm (exp within 4e-7;
//      SiTU within the absolute error of its 2 sigmoid(2 v) - 1 form)
//   5  one weight-row pair x row pairs (the GEMV) against scalar dots in the documented order
//   6  whole jobs (linear 1, SiTU, routing weight, quantization, linear 2, bf16, sum over picks) against a scalar
//      reference pipeline, and against float64 formulas on the same quantized inputs (relative L2 < 5e-3: float32
//      against float64 sums flip a few roundings); 10: a job of 8 records x 5 rows
//   7  the pool: rows equal the single-thread ones for 1-3 and 8 records and every pool size; refused jobs (bad slot,
//      unknown layer, 9 records, a header that disagrees with its tag, a non-finite weight, alpha 0) publish an error
//      under their own tag; an on-time job is counted as one
//   8  three jobs posted back to back while the pool is held, then released: all three run in order (FIFO), each
//      slot holds its own job's rows
//   9  NSLOT + 2 jobs posted while held: the oldest three are counted stale and publish nothing, the newest three run;
//      a slot whose header carries another sequence number is skipped; a long job overtaken by NSLOT - 1 newer jobs
//      is abandoned (or, if it finished first, its rows are right) and the newer ones run
//   11 the door reset to a small sequence number (the engine restarted with the pool kept): the next job runs
#include "cpustage.h"

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
int cpustage_init(void* seg, int threads, int core0, int nice_val);
int cpustage_set_layer(int layer, const void* bank, long long nrec);
int cpustage_start();
void cpustage_stop();
void cpustage_stats(unsigned long long* out);
int cpustage_stats_n(unsigned long long* out, int n);
long long cpustage_record_bytes();
int cpustage_layout(long long* out, int n);
int cpustage_run_single(const void* const* recs, int n_pick, const float* x, const float* w, int T, float alpha,
                        float beta, unsigned flags, int e_min, float* out, signed char* xq_out,
                        unsigned char* xe_out);
unsigned long long cpustage_test_post(void* seg, unsigned long long seq, int layer, const int* slots, int n_pick,
                                      const float* x, const float* w, int T, float alpha, float beta, unsigned flags,
                                      int e_min);
int cpustage_test_wait(void* seg, unsigned long long tag, unsigned long long timeout_ns, unsigned long long* status);
void cpustage_test_land(void* seg, unsigned long long tag, const float* rows, int T, unsigned long long status);
void cpustage_test_hold(int on);
}

using namespace cpustage;

enum { S_RUNNING, S_DONE, S_ABANDONED, S_REFUSED, S_JOB_NS, S_PICKS, S_MAX_NS, S_STALE, S_RESETS, S_ONTIME, S_QUEUED,
       S_QMAX, S_WORDS };

static int g_fail = 0;
#define CHECK(cond, ...)                                 \
  do {                                                   \
    if (!(cond)) {                                       \
      if (g_fail < 40) { std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } \
      ++g_fail;                                          \
    }                                                    \
  } while (0)

// ------------------------------------------------------------------------------------------------ random numbers
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd64() {
  g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
  return g_rng * 0x2545F4914F6CDD1Dull;
}
static inline float rndu() { return (float)(rnd64() >> 40) * (1.0f / 16777216.0f); }       // [0, 1)
static inline float rndn() {                                                             // about normal
  float s = 0.0f;
  for (int i = 0; i < 12; ++i) s += rndu();
  return s - 6.0f;
}

// ------------------------------------------------------------------- independent reference: layout and arithmetic
static const double FP4[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};   // hook.py FP4

struct RefLin {
  const uint8_t* codes;
  const uint8_t* scales;
  int rows, K;
};
// record: w13 codes [6144][896], w2 codes [3584][768], w13 scales int32 [28][6144], w2 scales int32 [24][3584]
static RefLin ref_lin(const uint8_t* rec, int which) {
  const long long c13 = 6144LL * 896, c2 = 3584LL * 768, s13 = 28LL * 6144 * 4;
  if (which == 1) return RefLin{rec, rec + c13 + c2, 6144, 3584};
  return RefLin{rec + c13, rec + c13 + c2 + s13, 3584, 3072};
}
static inline int ref_pair(const RefLin& L, int r, int k) {
  return (L.codes[(size_t)r * (size_t)(L.K / 4) + (size_t)(k / 4)] >> (2 * (k % 4))) & 3;
}
static inline int ref_sb(const RefLin& L, int r, int kb) {
  const int sr = 128 * (r / 128) + 4 * (r % 32) + (r % 128) / 32;       // _kernel_layout_shard's scale row
  return L.scales[((size_t)(kb / 4) * (size_t)L.rows + (size_t)sr) * 4 + (size_t)(kb % 4)];
}
// the staged weight as MegaMoE reads it: e2m1 value of nibble (pair << 2) | 3 times 2^(scale byte - 127); this tier
// reads scale byte 0 as an empty block (MegaMoE: 2^-127)
static inline double ref_value(const RefLin& L, int r, int k) {
  const int sb = ref_sb(L, r, k / 32);
  return sb == 0 ? 0.0 : FP4[(ref_pair(L, r, k) << 2) | 3] * std::ldexp(1.0, sb - 127);
}
static inline int ref_code(const RefLin& L, int r, int k) {              // the value in units of 1.5 x the block scale
  return (int)std::lround(FP4[(ref_pair(L, r, k) << 2) | 3] / 1.5);
}
static inline float ref_wscale(int sb) { return sb == 0 ? 0.0f : 1.5f * std::ldexp(1.0f, std::min(sb, 254) - 127); }
static inline float ref_xscale(int eb) { return std::ldexp(1.0f, eb - 128); }

static inline float ref_bf16(float v) {
  uint32_t u = bits_from_f32(v);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  return f32_from_bits(u);
}
static float ref_exp(float x) {
  x = x < -87.0f ? -87.0f : (x > 87.0f ? 87.0f : x);
  const float n = std::nearbyint(x * 1.44269504088896341f);
  float r = __builtin_fmaf(-n, 0.693359375f, x);
  r = __builtin_fmaf(-n, -2.12194440e-4f, r);
  float p = 1.9875691500e-4f;
  p = __builtin_fmaf(p, r, 1.3981999507e-3f);
  p = __builtin_fmaf(p, r, 8.3334519073e-3f);
  p = __builtin_fmaf(p, r, 4.1665795894e-2f);
  p = __builtin_fmaf(p, r, 1.6666665459e-1f);
  p = __builtin_fmaf(p, r, 5.0000001201e-1f);
  const float r2 = r * r;
  float y = __builtin_fmaf(p, r2, r);
  y = y + 1.0f;
  return y * f32_from_bits((uint32_t)((int)n + 127) << 23);
}
static float ref_situ(float g, float u, float alpha, float beta) {
  const float inv_alpha = 1.0f / alpha;
  const float sig = 1.0f / (1.0f + ref_exp(-g));
  const float eg = ref_exp(-2.0f * (g * inv_alpha));
  const float th = 2.0f * (1.0f / (1.0f + eg)) - 1.0f;
  const float gated = (th * alpha) * sig;
  if (beta > 0.0f) {
    const float inv_beta = 1.0f / beta;
    const float eu = ref_exp(-2.0f * (u * inv_beta));
    const float tu = 2.0f * (1.0f / (1.0f + eu)) - 1.0f;
    u = tu * beta;
  }
  return gated * u;
}
// v [K] -> int8 in natural input order and exponent bytes: a scalar copy of the rule in cpustage.h
static void ref_quant(const float* v, int K, int emin, int8_t* q, uint8_t* eb) {
  for (int b = 0; b < K / 32; ++b) {
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(v[32 * b + k]));
    int e = (int)((bits_from_f32(amax) + 0x1FFFFu) >> 23) - 133;
    e = std::max(emin, std::min(120, e));
    for (int k = 0; k < 32; ++k) {
      long r = std::lrint(std::ldexp(v[32 * b + k], -e));
      q[32 * b + k] = (int8_t)std::max(-127L, std::min(127L, r));
    }
    eb[b] = (uint8_t)(e + 128);
  }
}
// one weight row decoded: codes [K] and block scales [K / 32]
static void ref_row(const RefLin& L, int r, int8_t* c, float* s) {
  for (int k = 0; k < L.K; ++k) c[k] = (int8_t)ref_code(L, r, k);
  for (int b = 0; b < L.K / 32; ++b) s[b] = ref_wscale(ref_sb(L, r, b));
}
// the dot of a decoded weight row with a quantized row: blocks ascending, float32
static float ref_dot(const int8_t* c, const float* s, int K, const int8_t* q, const uint8_t* eb) {
  float acc = 0.0f;
  for (int b = 0; b < K / 32; ++b) {
    int D = 0;
    for (int k = 0; k < 32; ++k) D += (int)c[32 * b + k] * (int)q[32 * b + k];
    const float sc = s[b] * ref_xscale(eb[b]);
    acc = acc + (float)D * sc;
  }
  return acc;
}

struct JobSpec {
  int P, T;
  const uint8_t* rec[PMAX];
  float w[PMAX][MAXT];
  float alpha, beta;
  unsigned flags;
  int emin;
};

// The scalar reference pipeline: out [T][H].
static void ref_job(const JobSpec& S, const float* x, float* out) {
  std::vector<int8_t> xq((size_t)S.T * H), c(H);
  std::vector<uint8_t> xe((size_t)S.T * KB1);
  std::vector<float> s(KB1), gate((size_t)S.T * I), up((size_t)S.T * I), a(I);
  for (int t = 0; t < S.T; ++t)
    ref_quant(x + (size_t)t * H, H, S.emin, xq.data() + (size_t)t * H, xe.data() + (size_t)t * KB1);
  std::fill(out, out + (size_t)S.T * H, 0.0f);
  for (int p = 0; p < S.P; ++p) {
    const RefLin L1 = ref_lin(S.rec[p], 1), L2 = ref_lin(S.rec[p], 2);
    for (int n = 0; n < I; ++n) {
      for (int half = 0; half < 2; ++half) {
        ref_row(L1, 16 * (n / 8) + 8 * half + n % 8, c.data(), s.data());        // gate (w1) / up (w3) row of neuron n
        for (int t = 0; t < S.T; ++t) {
          if (S.w[p][t] == 0.0f) continue;
          const float v = ref_dot(c.data(), s.data(), H, xq.data() + (size_t)t * H, xe.data() + (size_t)t * KB1);
          (half ? up : gate)[(size_t)t * I + n] = v;
        }
      }
    }
    std::vector<int8_t> aqs((size_t)S.T * I);
    std::vector<uint8_t> aes((size_t)S.T * KB2);
    for (int t = 0; t < S.T; ++t) {
      if (S.w[p][t] == 0.0f) continue;
      for (int n = 0; n < I; ++n) {
        float g = gate[(size_t)t * I + n], u = up[(size_t)t * I + n];
        if (!(S.flags & CPUJOB_NO_BF16)) { g = ref_bf16(g); u = ref_bf16(u); }
        a[n] = ref_situ(g, u, S.alpha, S.beta) * S.w[p][t];
      }
      ref_quant(a.data(), I, S.emin, aqs.data() + (size_t)t * I, aes.data() + (size_t)t * KB2);
    }
    for (int j = 0; j < H; ++j) {
      ref_row(L2, j, c.data(), s.data());
      for (int t = 0; t < S.T; ++t) {
        if (S.w[p][t] == 0.0f) continue;
        float y = ref_dot(c.data(), s.data(), I, aqs.data() + (size_t)t * I, aes.data() + (size_t)t * KB2);
        if (S.flags & CPUJOB_OUT_BF16) y = ref_bf16(y);
        out[(size_t)t * H + j] += y;
      }
    }
  }
}

// float64 formulas on the same quantized inputs (hook._selfcheck's form: gate = x @ w1^T, up = x @ w3^T,
// a = alpha tanh(gate / alpha) sigmoid(gate) * beta tanh(up / beta) * w, y = a @ w2^T), libm functions.
static void formula_job(const JobSpec& S, const float* x, std::vector<double>& out) {
  std::vector<int8_t> xq((size_t)S.T * H), aq(I);
  std::vector<uint8_t> xe((size_t)S.T * KB1), ae(KB2);
  std::vector<double> xd((size_t)S.T * H), wrow(H), ad((size_t)S.T * I);
  std::vector<float> gate((size_t)S.T * I), up((size_t)S.T * I), a(I);
  for (int t = 0; t < S.T; ++t) {
    ref_quant(x + (size_t)t * H, H, S.emin, xq.data() + (size_t)t * H, xe.data() + (size_t)t * KB1);
    for (int k = 0; k < H; ++k)
      xd[(size_t)t * H + k] = std::ldexp((double)xq[(size_t)t * H + k], (int)xe[(size_t)t * KB1 + k / 32] - 128);
  }
  out.assign((size_t)S.T * H, 0.0);
  for (int p = 0; p < S.P; ++p) {
    const RefLin L1 = ref_lin(S.rec[p], 1), L2 = ref_lin(S.rec[p], 2);
    for (int n = 0; n < I; ++n)
      for (int half = 0; half < 2; ++half) {
        const int r = 16 * (n / 8) + 8 * half + n % 8;
        for (int k = 0; k < H; ++k) wrow[k] = ref_value(L1, r, k);
        for (int t = 0; t < S.T; ++t) {
          if (S.w[p][t] == 0.0f) continue;
          double acc = 0.0;
          for (int k = 0; k < H; ++k) acc += wrow[k] * xd[(size_t)t * H + k];
          (half ? up : gate)[(size_t)t * I + n] = (float)acc;
        }
      }
    for (int t = 0; t < S.T; ++t) {
      if (S.w[p][t] == 0.0f) continue;
      for (int n = 0; n < I; ++n) {
        double g = gate[(size_t)t * I + n], u = up[(size_t)t * I + n];
        if (!(S.flags & CPUJOB_NO_BF16)) { g = ref_bf16((float)g); u = ref_bf16((float)u); }
        double v = S.alpha * std::tanh(g / S.alpha) / (1.0 + std::exp(-g));
        v *= S.beta > 0.0f ? S.beta * std::tanh(u / S.beta) : u;
        a[n] = (float)(v * S.w[p][t]);
      }
      ref_quant(a.data(), I, S.emin, aq.data(), ae.data());
      for (int n = 0; n < I; ++n) ad[(size_t)t * I + n] = std::ldexp((double)aq[n], (int)ae[n / 32] - 128);
    }
    for (int j = 0; j < H; ++j) {
      for (int k = 0; k < I; ++k) wrow[k] = ref_value(L2, j, k);
      for (int t = 0; t < S.T; ++t) {
        if (S.w[p][t] == 0.0f) continue;
        double acc = 0.0;
        for (int k = 0; k < I; ++k) acc += wrow[k] * ad[(size_t)t * I + k];
        if (S.flags & CPUJOB_OUT_BF16) acc = ref_bf16((float)acc);
        out[(size_t)t * H + j] += acc;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------- test data
// A synthetic record: random codes; scale bytes around a per-row level (range 12), a few empty blocks (0).
static void make_record(uint8_t* rec, uint64_t seed) {
  g_rng = seed * 0x9E3779B97F4A7C15ull + 0x1234567ull;
  const long long codes = C13 + C2;
  for (long long i = 0; i + 8 <= codes; i += 8) { const uint64_t v = rnd64(); std::memcpy(rec + i, &v, 8); }
  for (int which = 1; which <= 2; ++which) {
    const RefLin L = ref_lin(rec, which);
    uint8_t* sc = const_cast<uint8_t*>(L.scales);
    for (int r = 0; r < L.rows; ++r) {
      const int level = 110 + (int)(rnd64() % 8);
      const int sr = 128 * (r / 128) + 4 * (r % 32) + (r % 128) / 32;
      for (int kb = 0; kb < L.K / 32; ++kb) {
        int v = level + (int)(rnd64() % 5);
        if (rnd64() % 997 == 0) v = 0;
        sc[((size_t)(kb / 4) * (size_t)L.rows + (size_t)sr) * 4 + (size_t)(kb % 4)] = (uint8_t)v;
      }
    }
  }
}
static void make_x(float* x, int T, uint64_t seed) {
  g_rng = seed * 0xD1342543DE82EF95ull + 77ull;
  for (int i = 0; i < T * H; ++i) {
    float v = rndn() * 0.5f;
    if (rnd64() % 211 == 0) v *= 9.0f;                    // a few outliers
    if (rnd64() % 500 == 0) v = 0.0f;
    x[i] = ref_bf16(v);                                    // the engine's rows are bfloat16
  }
  for (int k = 64; k < 96; ++k) x[k] = 0.0f;               // one all-zero block (row 0)
}

static bool same_bits(float a, float b) { return bits_from_f32(a) == bits_from_f32(b) || (a == 0.0f && b == 0.0f); }

// --------------------------------------------------------------------------------------------------- the checks
static void test_decode(const uint8_t* rec) {
  std::printf("1  record decode\n");
  for (int which = 1; which <= 2; ++which) {
    const RefLin R = ref_lin(rec, which);
    const Lin L = which == 1 ? lin1(rec) : lin2(rec);
    CHECK(L.rows == R.rows && L.kb * 32 == R.K && L.rowbytes == R.K / 4, "linear %d geometry", which);
    const int rows[] = {0, 2, 30, 62, 126, 128, 1000, R.rows / 2 - 2, R.rows - 130, R.rows - 2};
    alignas(64) float sw[KB1 * 4];
    for (int r0 : rows) {
      const uint8_t* c0 = L.codes + (size_t)r0 * L.rowbytes;
      for (int b = 0; b < L.kb; ++b) {
        int8x16_t w[4];
        decode_planes(c0 + 8 * b, c0 + L.rowbytes + 8 * b, w[0], w[1], w[2], w[3]);
        for (int p = 0; p < 4; ++p) {
          int8_t v[16];
          vst1q_s8(v, w[p]);
          for (int i = 0; i < 16; ++i) {
            const int want = ref_code(R, r0 + (i >> 3), 32 * b + 4 * (i & 7) + p);
            CHECK(v[i] == want, "linear %d rows %d block %d plane %d byte %d: %d, want %d", which, r0, b, p, i, v[i],
                  want);
          }
        }
      }
      pair_scales(L, r0, sw);
      for (int b = 0; b < L.kb; ++b)
        for (int l = 0; l < 4; ++l) {
          const float want = ref_wscale(ref_sb(R, r0 + (l >> 1), b));
          CHECK(same_bits(sw[4 * b + l], want), "linear %d rows %d block %d lane %d scale %a, want %a", which, r0, b, l,
                (double)sw[4 * b + l], (double)want);
          // and the decoded weight equals the staged value of the record
          const double val = (double)ref_code(R, r0 + (l >> 1), 32 * b) * (double)sw[4 * b + l];
          CHECK(val == ref_value(R, r0 + (l >> 1), 32 * b), "linear %d rows %d block %d: value", which, r0, b);
        }
    }
  }
  // the block-scale builder on every byte value, in every position of the two words
  for (int sb = 0; sb < 256; ++sb)
    for (int j = 0; j < 4; ++j) {
      const uint32_t a = (uint32_t)sb << (8 * j), b = (uint32_t)(255 - sb) << (8 * ((j + 1) & 3));
      float32x4_t v[4];
      scale_vectors(a, b, v);
      float o[4];
      vst1q_f32(o, v[j]);
      CHECK(same_bits(o[0], wscale((uint32_t)sb)) && same_bits(o[1], wscale((uint32_t)sb)),
            "scale byte %d lane a%d", sb, j);
      const int bj = (int)((b >> (8 * j)) & 0xFF);
      CHECK(same_bits(o[2], wscale((uint32_t)bj)) && same_bits(o[3], wscale((uint32_t)bj)),
            "scale byte %d lane b%d", bj, j);
      CHECK(bits_from_f32(o[0]) == bits_from_f32(ref_wscale(sb)) || (sb == 0 && o[0] == 0.0f), "wscale(%d)", sb);
    }
  CHECK(wscale(0) == 0.0f && wscale(127) == 1.5f && wscale(126) == 0.75f && wscale(255) == wscale(254), "wscale");
  CHECK(xscale(128) == 1.0f && xscale(129) == 2.0f && xscale(28) == std::ldexp(1.0f, -100), "xscale");
}

static void test_mmla() {
  std::printf("2  the 2 x 2 integer block\n");
  g_rng = 4242;
  for (int it = 0; it < 2000; ++it) {
    int8_t a[16], b[16];
    int32_t r0[4], r[4];
    for (int i = 0; i < 16; ++i) {
      a[i] = (int8_t)(rnd64() & 0xFF);
      b[i] = (int8_t)(rnd64() & 0xFF);
      if (it < 4) { a[i] = (it & 1) ? 127 : -128; b[i] = (it & 2) ? 127 : -128; }
    }
    for (int i = 0; i < 4; ++i) r0[i] = (int32_t)(rnd64() & 0xFFFF) - 32768;
    vst1q_s32(r, mmla(vld1q_s32(r0), vld1q_s8(a), vld1q_s8(b)));
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        int32_t want = r0[2 * i + j];
        for (int k = 0; k < 8; ++k) want += (int32_t)a[8 * i + k] * (int32_t)b[8 * j + k];
        CHECK(r[2 * i + j] == want, "mmla lane %d: %d, want %d", 2 * i + j, r[2 * i + j], want);
      }
  }
}

static void test_round_quant() {
  std::printf("3  bfloat16 rounding and the quantizer\n");
  g_rng = 99;
  for (int it = 0; it < 200000; ++it) {
    float v[4], o[4];
    for (int i = 0; i < 4; ++i) {
      const uint32_t u = (uint32_t)rnd64();
      v[i] = it < 64 ? f32_from_bits((u & 0xFFFF0000u) | (it & 1 ? 0x8000u : 0x7FFFu + (uint32_t)(it & 2)))
                     : f32_from_bits(u);
      if (!std::isfinite(v[i]) || std::fabs(v[i]) > 1e30f) v[i] = 1.0f + rndu();
    }
    vst1q_f32(o, round_bf16(vld1q_f32(v)));
    for (int i = 0; i < 4; ++i) {
      CHECK(same_bits(o[i], ref_bf16(v[i])), "round_bf16(%a) = %a, twin %a", (double)v[i], (double)o[i],
            (double)ref_bf16(v[i]));
      // nearest of the two neighbours, ties to the even one
      const uint32_t u = bits_from_f32(v[i]), lo = u & 0xFFFF0000u, hi = lo + 0x10000u;
      const double x = v[i], dl = std::fabs(x - (double)f32_from_bits(lo));
      const double dh = std::fabs((double)f32_from_bits(hi) - x);
      const uint32_t want = dl < dh ? lo : (dh < dl ? hi : ((lo >> 16) & 1u ? hi : lo));
      CHECK(bits_from_f32(o[i]) == want, "round_bf16(%a): %08x, nearest %08x", (double)v[i], bits_from_f32(o[i]), want);
    }
  }
  std::vector<float> v(H);
  std::vector<int8_t> qp(H), qn(H);
  std::vector<uint8_t> e1(KB1), e2(KB1);
  for (int it = 0; it < 20; ++it) {
    make_x(v.data(), 1, 1000 + it);
    if (it == 3) for (int k = 0; k < 32; ++k) v[k] = (k == 5) ? 127.0f : 1.0f;            // amax exactly 127
    if (it == 4) for (int k = 0; k < 32; ++k) v[k] = (k == 5) ? 127.5f : 1.0f;            // just above
    if (it == 5) for (int k = 0; k < 32; ++k) v[k] = 1e-37f;                              // under the floor
    quant_row(v.data(), H, -100, qp.data(), e1.data());
    ref_quant(v.data(), H, -100, qn.data(), e2.data());
    for (int b = 0; b < KB1; ++b) {
      CHECK(e1[b] == e2[b], "quant exponent block %d: %d, want %d", b, e1[b], e2[b]);
      int mx = 0;
      for (int k = 0; k < 32; ++k) {
        CHECK(qp[cpuseg_planar(32 * b + k)] == qn[32 * b + k], "quant value %d", 32 * b + k);
        mx = std::max(mx, std::abs((int)qn[32 * b + k]));
      }
      CHECK(mx <= 127 && (mx >= 64 || e2[b] == 28), "quant range block %d: max %d exponent byte %d", b, mx, e2[b]);
    }
    if (it == 3) CHECK(e2[0] == 128 && qn[5] == 127, "amax 127 -> e 0");
    if (it == 4) CHECK(e2[0] == 129 && qn[5] == 64, "amax 127.5 -> e 1");
  }
}

static void test_situ() {
  std::printf("4  exp and SiTU\n");
  g_rng = 7;
  double worst_e = 0.0, worst_s = 0.0;
  for (int it = 0; it < 400000; ++it) {
    float x[4], g[4], u[4], o[4];
    for (int i = 0; i < 4; ++i) {
      x[i] = it < 2000 ? -90.0f + 0.0225f * (float)(4 * it + i) : (rndu() - 0.5f) * 60.0f;
      g[i] = rndn() * (it % 3 == 0 ? 12.0f : 2.0f);
      u[i] = rndn() * (it % 5 == 0 ? 60.0f : 3.0f);
    }
    vst1q_f32(o, exp_ps(vld1q_f32(x)));
    for (int i = 0; i < 4; ++i) {
      CHECK(same_bits(o[i], ref_exp(x[i])), "exp(%a) = %a, twin %a", (double)x[i], (double)o[i], (double)ref_exp(x[i]));
      const double xc = std::max(-87.0, std::min(87.0, (double)x[i])), want = std::exp(xc);
      worst_e = std::max(worst_e, std::fabs((double)o[i] - want) / want);
    }
    for (int k = 0; k < 2; ++k) {
      const Situ s = make_situ(4.0f, k ? 0.0f : 25.0f);
      vst1q_f32(o, situ_ps(vld1q_f32(g), vld1q_f32(u), s));
      for (int i = 0; i < 4; ++i) {
        const float tw = ref_situ(g[i], u[i], s.alpha, s.beta);
        CHECK(same_bits(o[i], tw), "situ(%a, %a) = %a, twin %a", (double)g[i], (double)u[i], (double)o[i], (double)tw);
        // accuracy: tanh as 2 sigmoid(2 v) - 1 has an absolute error of a few 1e-7 (MegaMoE's form has it too), so
        // the bound is 2e-6 |value| + 3e-6 |up term| + 1.2e-5 |gate term| (the 25 x tanh of the up term)
        const double gd = g[i], ud = u[i];
        const double gated = 4.0 * std::tanh(gd / 4.0) / (1.0 + std::exp(-gd));
        const double upt = k ? ud : 25.0 * std::tanh(ud / 25.0);
        const double want = gated * upt;
        const double tol =
            2e-6 * std::fabs(want) + 3e-6 * std::fabs(upt) + (k ? 0.0 : 1.2e-5 * std::fabs(gated)) + 1e-12;
        worst_s = std::max(worst_s, std::fabs((double)o[i] - want) / tol);
      }
    }
  }
  std::printf("   exp: worst relative error against libm %.2e; SiTU: worst error / bound %.2f\n", worst_e, worst_s);
  CHECK(worst_e < 4e-7, "exp accuracy %.3e", worst_e);
  CHECK(worst_s <= 1.0, "SiTU error is %.2f of its bound", worst_s);
}

static void test_gemv(const uint8_t* rec) {
  std::printf("5  the row-pair GEMV\n");
  static JobBufs buf;
  std::vector<float> x((size_t)MAXT * H);
  make_x(x.data(), MAXT, 555);
  std::vector<int8_t> xq((size_t)MAXT * H), xn((size_t)MAXT * H), c0(H), c1(H);
  std::vector<uint8_t> xe((size_t)MAXT * KB1);
  std::vector<float> s0(KB1), s1(KB1);
  for (int t = 0; t < MAXT; ++t) {
    quant_row(x.data() + (size_t)t * H, H, -100, xq.data() + (size_t)t * H, xe.data() + (size_t)t * KB1);
    for (int k = 0; k < H; ++k) xn[(size_t)t * H + k] = xq[(size_t)t * H + cpuseg_planar(k)];
  }
  for (int which = 1; which <= 2; ++which) {
    const RefLin R = ref_lin(rec, which);
    const Lin L = which == 1 ? lin1(rec) : lin2(rec);
    // linear 2 reads K = 3072: use the first 3072 inputs of the same rows as its activations
    for (int nrow = 1; nrow <= MAXT; ++nrow) {
      const int np = (nrow + 1) / 2;
      const int8_t* xo[MAXPAIR];
      const float* xs[MAXPAIR];
      for (int q = 0; q < np; ++q) {
        const int ta = 2 * q, tb = 2 * q + 1 < nrow ? 2 * q + 1 : -1;
        build_pair(xq.data() + (size_t)ta * H, xe.data() + (size_t)ta * KB1,
                   tb >= 0 ? xq.data() + (size_t)tb * H : nullptr, tb >= 0 ? xe.data() + (size_t)tb * KB1 : nullptr,
                   L.kb, buf.xop[q], buf.xs4[q]);
        xo[q] = buf.xop[q];
        xs[q] = buf.xs4[q];
      }
      const int rows[] = {0, 126, 2 * (nrow * 131 % (R.rows / 2)), R.rows - 2};
      for (int r0 : rows) {
        float32x4_t acc[MAXPAIR];
        gemv(L, r0, 2 * L.rowbytes, xo, xs, np, acc);
        ref_row(R, r0, c0.data(), s0.data());
        ref_row(R, r0 + 1, c1.data(), s1.data());
        for (int q = 0; q < np; ++q) {
          float o[4];
          vst1q_f32(o, acc[q]);
          for (int l = 0; l < 4; ++l) {
            const int t = 2 * q + (l & 1);
            if (t >= nrow) continue;
            const float want = ref_dot(l < 2 ? c0.data() : c1.data(), l < 2 ? s0.data() : s1.data(), R.K,
                                       xn.data() + (size_t)t * H, xe.data() + (size_t)t * KB1);
            CHECK(same_bits(o[l], want), "linear %d rows %d, %d rows, pair %d lane %d: %a, want %a", which, r0, nrow,
                  q, l, (double)o[l], (double)want);
          }
        }
      }
    }
  }
}

static double rel_l2(const float* a, const double* b, int n) {
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n; ++i) { const double d = (double)a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return std::sqrt(num / (den + 1e-300));
}

static JobSpec make_spec(int P, int T, const uint8_t* const* recs, const float w[][MAXT],
                         unsigned flags = CPUJOB_OUT_BF16, float beta = 25.0f) {
  JobSpec S;
  S.P = P; S.T = T; S.alpha = 4.0f; S.beta = beta; S.flags = flags; S.emin = -100;
  for (int p = 0; p < PMAX; ++p) {
    S.rec[p] = p < P ? recs[p] : nullptr;
    for (int t = 0; t < MAXT; ++t) S.w[p][t] = (p < P && t < T) ? w[p][t] : 0.0f;
  }
  return S;
}

static void run_kernel(const JobSpec& S, const float* x, float* out) {
  const void* recs[PMAX];
  float w[PMAX * MAXT];
  for (int p = 0; p < PMAX; ++p) {
    recs[p] = S.rec[p];
    for (int t = 0; t < MAXT; ++t) w[p * MAXT + t] = S.w[p][t];
  }
  const int rc = cpustage_run_single(recs, S.P, x, w, S.T, S.alpha, S.beta, S.flags, S.emin, out, nullptr, nullptr);
  CHECK(rc == 0, "cpustage_run_single returned %d", rc);
}

static void check_job(const char* name, const JobSpec& S, const float* x, bool formulas) {
  std::vector<float> got((size_t)S.T * H, -1.0f), want((size_t)S.T * H);
  run_kernel(S, x, got.data());
  ref_job(S, x, want.data());
  long bad = 0;
  double norm = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!same_bits(got[i], want[i])) {
      if (bad < 3)
        CHECK(false, "%s: row %zu element %zu: %a, reference %a", name, i / H, i % H, (double)got[i], (double)want[i]);
      ++bad;
    }
    norm += (double)want[i] * want[i];
  }
  CHECK(bad == 0, "%s: %ld of %zu elements differ from the scalar reference", name, bad, got.size());
  std::printf("   %-34s %d picks x %d rows: %s the scalar reference (rms %.3e)", name, S.P, S.T,
              bad == 0 ? "bit-equal to" : "DIFFERS from", std::sqrt(norm / (double)got.size()));
  if (formulas) {
    std::vector<double> f;
    formula_job(S, x, f);
    double worst = 0.0;
    bool finite = true;
    for (int t = 0; t < S.T; ++t) {
      bool used = false;
      for (int p = 0; p < S.P; ++p) used = used || S.w[p][t] != 0.0f;
      if (!used) continue;
      const double r = rel_l2(got.data() + (size_t)t * H, f.data() + (size_t)t * H, H);
      finite = finite && std::isfinite(r);
      worst = std::max(worst, r);
    }
    std::printf("; float64 formulas: worst row relative L2 %.2e", worst);
    CHECK(finite && worst < 5e-3, "%s: relative L2 %.3e against the float64 formulas", name, worst);
  }
  std::printf("\n");
}

static void test_jobs(const uint8_t* const* recs, const char* label) {
  std::printf("6  whole jobs (%s)\n", label);
  std::vector<float> x((size_t)MAXT * H);
  make_x(x.data(), MAXT, 31337);
  {
    const float w[PMAX][MAXT] = {{0.7f}};
    check_job("one pick, one row", make_spec(1, 1, recs, w), x.data(), true);
  }
  {
    const float w[PMAX][MAXT] = {{0.31f, 0, 0, 0.12f, 0}, {0, 0.22f, 0.05f, 0, 0.4f}};
    check_job("two picks, five rows (2 + 3)", make_spec(2, 5, recs, w), x.data(), true);
  }
  {
    const float w[PMAX][MAXT] = {{0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.05f, 0.25f, 0.35f}, {0, 0, 0, 0, 0, 0, 0, 0},
                                 {0, 0, 0.11f, 0, 0, 0, 0, 0.09f}};
    check_job("three picks (one unused), 8 rows", make_spec(3, 8, recs, w), x.data(), false);
  }
  {
    const float w[PMAX][MAXT] = {{0.2f, 0.3f, 0.1f}};
    check_job("no bf16 on gate / up", make_spec(1, 3, recs, w, CPUJOB_OUT_BF16 | CPUJOB_NO_BF16), x.data(), true);
    check_job("no bf16 on the output, beta 0", make_spec(1, 3, recs, w, 0u, 0.0f), x.data(), true);
  }
  {
    // 10: eight records, five rows; picks share rows (operands built once per pair of rows), one pick per row only
    const float w[PMAX][MAXT] = {{0.30f, 0.21f}, {0.25f, 0, 0, 0.18f}, {0, 0.12f, 0.07f, 0, 0.33f}, {0.11f, 0.09f},
                                 {0, 0, 0, 0, 0.05f}, {0.02f, 0.03f, 0.04f, 0.05f, 0.06f}, {0, 0, 0.27f},
                                 {0.08f, 0, 0, 0, 0.01f}};
    const uint8_t* r8[PMAX];
    for (int p = 0; p < PMAX; ++p) r8[p] = recs[p % 3];
    check_job("eight picks, five rows", make_spec(8, 5, r8, w), x.data(), false);
  }
}

// ------------------------------------------------------------------------------------------------------ the pool
static void* seg_alloc() {
  void* p = nullptr;
  if (posix_memalign(&p, 65536, CPUSEG2_BYTES) != 0) return nullptr;
  std::memset(p, 0, CPUSEG2_BYTES);
  return p;
}
static int first_core(int n) {            // the first n consecutive cores this process may use, or -1
  cpu_set_t cs;
  CPU_ZERO(&cs);
  if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return -1;
  for (int c = CPU_SETSIZE - n; c >= 0; --c) {
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) ok = CPU_ISSET(c + i, &cs);
    if (ok) return c;                     // the highest such range (the engine's own threads sit on the low cores)
  }
  return -1;
}
static int count_cores() {
  cpu_set_t cs;
  CPU_ZERO(&cs);
  if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return 0;
  return CPU_COUNT(&cs);
}
static void stats(unsigned long long* s) { cpustage_stats_n(s, S_WORDS); }
static void sleep_us(long us) {
  struct timespec ts = {us / 1000000, (us % 1000000) * 1000};
  nanosleep(&ts, nullptr);
}
static uint64_t done_word(void* seg, int s) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(seg_done(static_cast<uint8_t*>(seg), s)), __ATOMIC_ACQUIRE);
}

// A job of the pool tests: records by bank slot, weights, the rows it reads (x), its expected rows.
struct PJob {
  int P = 1, T = 1, layer = 7;
  int slots[PMAX] = {};
  float w[PMAX * MAXT] = {};
  unsigned flags = CPUJOB_OUT_BF16;
  float alpha = 4.0f;
  std::vector<float> want;
};
static PJob make_pjob(int nrec, int P, int T, uint64_t seed, unsigned flags = CPUJOB_OUT_BF16) {
  PJob j;
  j.P = P;
  j.T = T;
  j.flags = flags;
  g_rng = seed * 0xA24BAED4963EE407ull + 3ull;
  for (int p = 0; p < P; ++p) {
    j.slots[p] = (int)(rnd64() % (uint64_t)nrec);
    const int used = 1 + (int)(rnd64() % (uint64_t)std::min(T, 3));
    for (int k = 0; k < used; ++k) j.w[p * MAXT + (int)(rnd64() % (uint64_t)T)] = 0.02f + 0.3f * rndu();
  }
  return j;
}
static void expect(PJob& j, const uint8_t* bank, const float* x) {
  const void* recs[PMAX] = {};
  for (int p = 0; p < j.P; ++p) recs[p] = bank + (size_t)j.slots[p] * (size_t)REC_BYTES;
  j.want.assign((size_t)j.T * H, 0.0f);
  const int rc =
      cpustage_run_single(recs, j.P, x, j.w, j.T, j.alpha, 25.0f, j.flags, -100, j.want.data(), nullptr, nullptr);
  CHECK(rc == 0, "run_single %d", rc);
}
static unsigned long long post(void* seg, uint64_t seq, const PJob& j, const float* x) {
  return cpustage_test_post(seg, seq, j.layer, j.slots, j.P, x, j.w, j.T, j.alpha, 25.0f, j.flags, -100);
}
static long diff_rows(void* seg, uint64_t seq, const PJob& j) {
  const float* res = reinterpret_cast<const float*>(seg_res(static_cast<uint8_t*>(seg), (int)(seq % NSLOT)));
  long bad = 0;
  for (size_t i = 0; i < (size_t)j.T * H; ++i) bad += !same_bits(res[i], j.want[i]);
  return bad;
}

static void test_pool(const uint8_t* bank, int nrec) {
  std::printf("7  the pool (and 8, 9, 11: the job ring)\n");
  std::vector<float> x((size_t)MAXT * H);
  make_x(x.data(), MAXT, 2024);
  const int counts[] = {1, 2, 3, 8, 24, 32, 48};
  const unsigned long long LONG_NS = 60000000000ull;
  int ran = 0;
  for (int nt : counts) {
    const int core0 = first_core(nt);
    if (core0 < 0) { std::printf("   %d threads: not enough cores in this process's set, skipped\n", nt); continue; }
    ++ran;
    void* seg = seg_alloc();
    CHECK(seg != nullptr, "segment allocation");
    if (seg == nullptr) return;
    CHECK(cpustage_init(seg, nt, core0, 0) == 0, "init");
    CHECK(cpustage_set_layer(7, bank, nrec) == 0, "set_layer");
    CHECK(cpustage_start() == 0, "start");
    unsigned long long s0[S_WORDS], s1[S_WORDS], st = 0;
    uint64_t seq = 0;
    // 7: jobs of 1, 2, 3 and 8 records, each against the single-thread result
    long bad = 0;
    int njobs = 0;
    for (int P : {1, 2, 3, 8}) {
      for (int T : {1, 4, 5}) {
        PJob j = make_pjob(nrec, P, T, 1000u * (unsigned)P + (unsigned)T);
        expect(j, bank, x.data());
        const unsigned long long tag = post(seg, ++seq, j, x.data());
        const int rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
        CHECK(rc == 0 && st == 0, "%d threads, %d picks x %d rows: wait %d status %llu", nt, P, T, rc, st);
        bad += diff_rows(seg, seq, j);
        ++njobs;
      }
    }
    CHECK(bad == 0, "%d threads: %ld elements differ from the single-thread result", nt, bad);
    // refused jobs: each publishes an error under its own tag
    stats(s0);
    {
      PJob j = make_pjob(nrec, 2, 4, 77);
      j.slots[1] = nrec + 5;                                                            // a slot outside the bank
      unsigned long long tag = post(seg, ++seq, j, x.data());
      int rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && (st & CPUSEG_ST_ERROR), "%d threads: a bad slot was not refused (wait %d status %llu)", nt, rc,
            st);
      j = make_pjob(nrec, 1, 4, 78);
      j.layer = 9;                                                                      // an unregistered layer
      tag = post(seg, ++seq, j, x.data());
      rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && (st & CPUSEG_ST_ERROR), "%d threads: an unknown layer was not refused", nt);
      j = make_pjob(nrec, PMAX, 4, 79);
      tag = cpustage_test_post(seg, ++seq, 7, j.slots, PMAX + 1, x.data(), j.w, 4, 4.0f, 25.0f, CPUJOB_OUT_BF16, -100);
      rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && (st & CPUSEG_ST_ERROR), "%d threads: 9 records were not refused", nt);
      j = make_pjob(nrec, 1, 4, 80);
      cpustage_test_hold(1);                                                            // the pool reads nothing ...
      sleep_us(2000);
      tag = post(seg, ++seq, j, x.data());                                              // ... of this 4-row job ...
      reinterpret_cast<CpuJob2*>(seg_job(static_cast<uint8_t*>(seg), (int)(seq % NSLOT)))->n_rows = 3;   // ... now 3
      cpustage_test_hold(0);
      rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && (st & CPUSEG_ST_ERROR), "%d threads: a header that disagrees with its tag was not refused", nt);
      j = make_pjob(nrec, 2, 2, 81);
      j.w[1 * MAXT + 1] = std::nanf("");                                                // a non-finite weight
      tag = post(seg, ++seq, j, x.data());
      rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && (st & CPUSEG_ST_ERROR), "%d threads: a NaN weight was not refused", nt);
      j = make_pjob(nrec, 1, 2, 82);
      j.alpha = 0.0f;                                                                   // SiTU alpha 0
      tag = post(seg, ++seq, j, x.data());
      rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && (st & CPUSEG_ST_ERROR), "%d threads: alpha 0 was not refused", nt);
    }
    stats(s1);
    CHECK(s1[S_REFUSED] - s0[S_REFUSED] == 6, "%d threads: %llu refused, want 6", nt, s1[S_REFUSED] - s0[S_REFUSED]);
    // an on-time job is counted as one
    {
      PJob j = make_pjob(nrec, 2, 5, 83, CPUJOB_OUT_BF16 | CPUJOB2_ONTIME);
      expect(j, bank, x.data());
      const unsigned long long tag = post(seg, ++seq, j, x.data());
      const int rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      stats(s0);
      CHECK(rc == 0 && st == 0 && diff_rows(seg, seq, j) == 0 && s0[S_ONTIME] - s1[S_ONTIME] == 1,
            "%d threads: the on-time job (wait %d status %llu, on-time count %llu)", nt, rc, st,
            s0[S_ONTIME] - s1[S_ONTIME]);
    }
    // 8: three jobs back to back, queued behind a held pool, then run in order; each slot keeps its own rows
    {
      stats(s0);
      PJob j[3] = {make_pjob(nrec, 3, 5, 90), make_pjob(nrec, 1, 2, 91), make_pjob(nrec, 5, 5, 92)};
      unsigned long long tags[3];
      cpustage_test_hold(1);
      sleep_us(2000);
      for (int k = 0; k < 3; ++k) { expect(j[k], bank, x.data()); tags[k] = post(seg, ++seq, j[k], x.data()); }
      cpustage_test_hold(0);
      long b8 = 0;
      for (int k = 0; k < 3; ++k) {
        const int rc = cpustage_test_wait(seg, tags[k], LONG_NS, &st);
        CHECK(rc == 0 && st == 0, "%d threads: queued job %d (wait %d status %llu)", nt, k, rc, st);
      }
      for (int k = 0; k < 3; ++k) b8 += diff_rows(seg, seq - 2 + (uint64_t)k, j[k]);
      stats(s1);
      CHECK(b8 == 0, "%d threads: queued jobs: %ld wrong elements", nt, b8);
      CHECK(s1[S_DONE] - s0[S_DONE] == 3 && s1[S_STALE] == s0[S_STALE] && s1[S_QMAX] >= 2,
            "%d threads: queued jobs: done %llu stale %llu max queue %llu", nt, s1[S_DONE] - s0[S_DONE],
            s1[S_STALE] - s0[S_STALE], s1[S_QMAX]);
      // and free-running: posted as fast as the host can, all three run
      for (int k = 0; k < 3; ++k) tags[k] = post(seg, ++seq, j[k], x.data());
      b8 = 0;
      for (int k = 0; k < 3; ++k) {
        const int rc = cpustage_test_wait(seg, tags[k], LONG_NS, &st);
        CHECK(rc == 0 && st == 0, "%d threads: back-to-back job %d (wait %d status %llu)", nt, k, rc, st);
        b8 += diff_rows(seg, seq - 2 + (uint64_t)k, j[k]);
      }
      CHECK(b8 == 0, "%d threads: back-to-back jobs: %ld wrong elements", nt, b8);
    }
    // 9a: NSLOT + 2 jobs while held: the oldest three are stale and publish nothing, the newest three run
    {
      stats(s0);
      PJob j[NSLOT + 2];
      unsigned long long tags[NSLOT + 2];
      cpustage_test_hold(1);
      sleep_us(2000);
      const uint64_t first = seq + 1;
      const uint64_t before = done_word(seg, (int)((first + 2) % NSLOT));
      for (int k = 0; k < NSLOT + 2; ++k) {
        j[k] = make_pjob(nrec, 1 + k % 3, 3, 100 + (uint64_t)k);
        expect(j[k], bank, x.data());
        tags[k] = post(seg, ++seq, j[k], x.data());
      }
      cpustage_test_hold(0);
      long b9 = 0;
      for (int k = 3; k < NSLOT + 2; ++k) {
        const int rc = cpustage_test_wait(seg, tags[k], LONG_NS, &st);
        CHECK(rc == 0 && st == 0, "%d threads: the newest jobs: job %d (wait %d status %llu)", nt, k, rc, st);
        b9 += diff_rows(seg, first + (uint64_t)k, j[k]);
      }
      sleep_us(1000);
      stats(s1);
      CHECK(b9 == 0, "%d threads: the newest jobs: %ld wrong elements", nt, b9);
      CHECK(s1[S_STALE] - s0[S_STALE] == 3 && s1[S_DONE] - s0[S_DONE] == 3,
            "%d threads: behind the door: stale %llu (want 3), done %llu (want 3)", nt, s1[S_STALE] - s0[S_STALE],
            s1[S_DONE] - s0[S_DONE]);
      // the slot of the third-oldest is no one else's: its done word did not move
      CHECK(done_word(seg, (int)((first + 2) % NSLOT)) == before, "%d threads: a skipped job published its done tag",
            nt);
    }
    // 9b: a slot whose header carries another sequence number (rewritten): skipped, nothing published
    {
      stats(s0);
      cpustage_test_hold(1);
      sleep_us(2000);
      PJob j = make_pjob(nrec, 1, 2, 120);
      expect(j, bank, x.data());
      const uint64_t sq = ++seq;
      post(seg, sq, j, x.data());
      const uint64_t before = done_word(seg, (int)(sq % NSLOT));
      CpuJob2* h = reinterpret_cast<CpuJob2*>(seg_job(static_cast<uint8_t*>(seg), (int)(sq % NSLOT)));
      h->tag = cpuseg_tag(sq - NSLOT, 7, 2);                                            // the job NSLOT earlier
      cpustage_test_hold(0);
      PJob k = make_pjob(nrec, 2, 3, 121);
      expect(k, bank, x.data());
      const unsigned long long tag = post(seg, ++seq, k, x.data());
      const int rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      stats(s1);
      CHECK(rc == 0 && st == 0 && diff_rows(seg, seq, k) == 0, "%d threads: the job after a rewritten slot", nt);
      CHECK(s1[S_STALE] - s0[S_STALE] == 1 && done_word(seg, (int)(sq % NSLOT)) == before,
            "%d threads: a rewritten slot: stale %llu (want 1)", nt, s1[S_STALE] - s0[S_STALE]);
    }
    // 9c: a long job overtaken by NSLOT - 1 newer ones is abandoned (or done first: then its rows are right)
    {
      stats(s0);
      PJob big = make_pjob(nrec, PMAX, MAXT, 130);
      for (int p = 0; p < PMAX; ++p)
        for (int t = 0; t < MAXT; ++t) big.w[p * MAXT + t] = 0.01f + 0.01f * (float)(p + t);
      expect(big, bank, x.data());
      const uint64_t sb = ++seq;
      const unsigned long long btag = post(seg, sb, big, x.data());
      sleep_us(1500);
      PJob j[NSLOT - 1];
      unsigned long long tags[NSLOT - 1];
      for (int k = 0; k < NSLOT - 1; ++k) {
        j[k] = make_pjob(nrec, 1, 2, 140 + (uint64_t)k);
        expect(j[k], bank, x.data());
        tags[k] = post(seg, ++seq, j[k], x.data());
      }
      long b9 = 0;
      for (int k = 0; k < NSLOT - 1; ++k) {
        const int rc = cpustage_test_wait(seg, tags[k], LONG_NS, &st);
        CHECK(rc == 0 && st == 0, "%d threads: after the long job: job %d (wait %d status %llu)", nt, k, rc, st);
        b9 += diff_rows(seg, sb + 1 + (uint64_t)k, j[k]);
      }
      stats(s1);
      const bool big_done = done_word(seg, (int)(sb % NSLOT)) == btag;
      const unsigned long long ab = s1[S_ABANDONED] - s0[S_ABANDONED], stl = s1[S_STALE] - s0[S_STALE];
      CHECK(b9 == 0, "%d threads: after the long job: %ld wrong elements", nt, b9);
      CHECK((big_done ? 1 : 0) + ab + stl == 1, "%d threads: the long job: done %d abandoned %llu stale %llu", nt,
            big_done ? 1 : 0, ab, stl);
      if (big_done) CHECK(diff_rows(seg, sb, big) == 0, "%d threads: the long job finished with wrong rows", nt);
      std::printf("   %2d threads: the overtaken long job was %s\n", nt,
                  big_done ? "done first" : (ab ? "abandoned" : "skipped"));
    }
    // 11: the door reset to a small sequence number (the engine restarted, the pool kept): the next job runs
    {
      stats(s0);
      PJob j = make_pjob(nrec, 2, 4, 150);
      expect(j, bank, x.data());
      seq = 1;
      unsigned long long tag = post(seg, seq, j, x.data());
      int rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      CHECK(rc == 0 && st == 0 && diff_rows(seg, seq, j) == 0, "%d threads: the first job after a door reset", nt);
      PJob k = make_pjob(nrec, 1, 3, 151);
      expect(k, bank, x.data());
      tag = post(seg, ++seq, k, x.data());
      rc = cpustage_test_wait(seg, tag, LONG_NS, &st);
      stats(s1);
      CHECK(rc == 0 && st == 0 && diff_rows(seg, seq, k) == 0, "%d threads: the second job after a door reset", nt);
      CHECK(s1[S_RESETS] - s0[S_RESETS] == 1, "%d threads: door resets %llu, want 1", nt, s1[S_RESETS] - s0[S_RESETS]);
    }
    cpustage_stop();
    stats(s1);
    std::printf("   %2d threads on cores %d..%d: %s; done %llu, refused %llu, stale %llu, abandoned %llu, resets %llu, "
                "on-time %llu, mean job %.0f us\n", nt, core0, core0 + nt - 1,
                g_fail == 0 ? "every job's rows bit-equal to the single-thread results" : "FAILURES above", s1[S_DONE],
                s1[S_REFUSED], s1[S_STALE], s1[S_ABANDONED], s1[S_RESETS], s1[S_ONTIME],
                s1[S_DONE] ? (double)s1[S_JOB_NS] / (double)s1[S_DONE] / 1000.0 : 0.0);
    CHECK(s1[S_RUNNING] == 0, "running word after stop");
    std::free(seg);
  }
  CHECK(ran > 0, "no pool size fits this process's CPU set");
  // a pool that cannot pin its threads does not start
  {
    void* seg = seg_alloc();
    cpu_set_t cs;
    CPU_ZERO(&cs);
    sched_getaffinity(0, sizeof(cs), &cs);
    int outside = -1;
    for (int c = 0; c < CPU_SETSIZE - 1 && outside < 0; ++c)
      if (!CPU_ISSET(c, &cs)) outside = c;
    if (outside >= 0) {
      CHECK(cpustage_init(seg, 1, outside, 0) == 0 && cpustage_start() == -3,
            "a pool on core %d (outside the set) started", outside);
    }
    std::free(seg);
  }
}

// ---------------------------------------------------------------------------------------------------- benches
static int g_poster = -1;
static void pin_self(int core) {
  if (core < 0) return;
  cpu_set_t cs;
  CPU_ZERO(&cs);
  CPU_SET(core, &cs);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
    std::printf("note: could not pin the poster to core %d\n", core);
}
static double tdiff_us(const struct timespec& a, const struct timespec& b) {
  return ((double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec)) / 1000.0;
}
struct Pct {
  double med, p10, p90, p99, mx;
};
static double pool_us(const unsigned long long* s0, const unsigned long long* s1) {   // the pool's mean us per job
  return s1[S_DONE] > s0[S_DONE] ? (double)(s1[S_JOB_NS] - s0[S_JOB_NS]) / (double)(s1[S_DONE] - s0[S_DONE]) / 1000.0
                                 : 0.0;
}
static Pct pct(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return Pct{v[n / 2], v[n / 10], v[n * 9 / 10], v[std::min(n - 1, n * 99 / 100)], v.back()};
}

// The header of slot s for the next job (records taken in turn from the bank: cold), then the door. The rows of all
// slots were written once (the GB300 writes them per job; their 29 KB are not what is measured here).
struct Poster {
  uint8_t* seg;
  uint64_t seq;
  int nrec, next = 0;
  unsigned long long post(int P, int T, const float* wflat, unsigned flags) {
    ++seq;
    const int s = (int)(seq % NSLOT);
    CpuJob2* h = reinterpret_cast<CpuJob2*>(seg_job(seg, s));
    h->tag = cpuseg_tag(seq, 7, (uint32_t)T);
    h->n_pick = (uint32_t)P;
    h->n_rows = (uint32_t)T;
    h->layer = 7;
    h->flags = flags;
    h->alpha = 4.0f;
    h->beta = 25.0f;
    h->e_min = -100;
    for (int p = 0; p < PMAX; ++p) {
      h->slot[p] = p < P ? next : 0;
      if (p < P) next = (next + 1) % nrec;
      for (int t = 0; t < MAXT; ++t) h->w[p][t] = p < P && t < T ? wflat[p * MAXT + t] : 0.0f;
    }
    __atomic_store_n(reinterpret_cast<uint64_t*>(seg + CPUSEG2_OFF_DOOR), (uint64_t)h->tag, __ATOMIC_RELEASE);
    return h->tag;
  }
};

static void* bench_start(const uint8_t* bank, int nrec, int nt, int core0, Poster& ps) {
  void* seg = seg_alloc();
  if (seg == nullptr || cpustage_init(seg, nt, core0, 0) != 0 || cpustage_set_layer(7, bank, nrec) != 0 ||
      cpustage_start() != 0) {
    std::printf("bench: the pool did not start (threads %d on cores %d..%d)\n", nt, core0, core0 + nt - 1);
    return nullptr;
  }
  std::vector<float> x((size_t)MAXT * H);
  make_x(x.data(), MAXT, 1);
  ps.seg = static_cast<uint8_t*>(seg);
  ps.seq = 0;
  ps.nrec = nrec;
  // the rows of every slot, once; a few jobs to fault in the result slots and warm the threads
  const int slots[PMAX] = {};
  float w[PMAX * MAXT] = {0.1f};
  unsigned long long st, tag = 0;
  for (int k = 0; k < 2 * NSLOT; ++k) {
    tag = cpustage_test_post(seg, ++ps.seq, 7, slots, 1, x.data(), w, MAXT, 4.0f, 25.0f, CPUJOB_OUT_BF16, -100);
    if (cpustage_test_wait(seg, tag, 10000000000ull, &st) != 0) {
      std::printf("bench: a warm-up job timed out\n");
      return nullptr;
    }
  }
  return seg;
}

// rows: how many of the job's T rows use each record (rows 0.., spread)
static void weights(float* wflat, int P, int T, int rows) {
  std::memset(wflat, 0, sizeof(float) * PMAX * MAXT);
  for (int p = 0; p < P; ++p)
    for (int k = 0; k < rows; ++k) wflat[p * MAXT + (p + k) % T] = 0.1f + 0.01f * (float)(p + k);
}

// On-time jobs: one job per call, jobs 0.3 ms apart; tag -> done as the poster sees it.
static void bench_ontime(const uint8_t* bank, int nrec, int nt, int core0, int n) {
  Poster ps{};
  void* seg = bench_start(bank, nrec, nt, core0, ps);
  if (seg == nullptr) return;
  unsigned long long s0[S_WORDS], s1[S_WORDS], st;
  std::printf("on-time jobs: pool of %d threads on cores %d..%d, poster on core %d, %d records of %.2f MB taken in "
              "turn (cold), jobs 0.3 ms apart, %d per case\n",
              nt, core0, core0 + nt - 1, g_poster, nrec, REC_BYTES / 1e6, n);
  std::printf("records rows/record | tag -> done us: median    p10    p90    p99    max | pool us/job | us/record\n");
  for (int P = 1; P <= 4; ++P)
    for (int rows : {1, 2, 4, 6}) {
      float wflat[PMAX * MAXT];
      weights(wflat, P, 6, rows);
      stats(s0);
      std::vector<double> us;
      for (int it = 0; it < n; ++it) {
        sleep_us(300);
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        const unsigned long long tag = ps.post(P, 6, wflat, CPUJOB_OUT_BF16 | CPUJOB2_ONTIME);
        if (cpustage_test_wait(seg, tag, 2000000000ull, &st) != 0) { std::printf("bench: a job timed out\n"); break; }
        clock_gettime(CLOCK_MONOTONIC, &b);
        us.push_back(tdiff_us(a, b));
      }
      if (us.empty()) continue;
      stats(s1);
      const Pct q = pct(us);
      const double pool = pool_us(s0, s1);
      std::printf("  %d        %d        | %24.1f %6.1f %6.1f %6.1f %6.1f | %11.1f | %9.1f\n", P, rows, q.med, q.p10,
                  q.p90, q.p99, q.mx, pool, q.med / P);
    }
  stats(s1);
  std::printf("pool: %llu jobs, %llu abandoned, %llu stale, longest %.1f us\n", s1[S_DONE], s1[S_ABANDONED],
              s1[S_STALE], (double)s1[S_MAX_NS] / 1000.0);
  std::printf("gate: 1 record <= 40 us, 2 records <= 70 us, tag to done, 1-2 rows per record\n");
  cpustage_stop();
  std::free(seg);
}

// Late jobs: 5 rows per call, each record used by 1 or 2 of them; back to back (the next job posted
// when the previous one is done) and in pairs (two jobs posted together: the second waits for the first, FIFO).
static void bench_late(const uint8_t* bank, int nrec, int nt, int core0, int n) {
  Poster ps{};
  void* seg = bench_start(bank, nrec, nt, core0, ps);
  if (seg == nullptr) return;
  unsigned long long s0[S_WORDS], s1[S_WORDS], st;
  std::printf("late jobs: pool of %d threads on cores %d..%d, poster on core %d, %d records of %.2f MB taken in turn "
              "(cold), 5 rows per job, %d jobs per case\n",
              nt, core0, core0 + nt - 1, g_poster, nrec, REC_BYTES / 1e6, n);
  std::printf("mode         records rows/record | tag -> done us: median    p10    p90    p99    max | pool us/job | "
              "us/record\n");
  for (int mode = 0; mode < 2; ++mode)
    for (int P : {1, 3, 5, 7, 8})
      for (int rows : {1, 2}) {
        if (rows == 1 && P != 5) continue;
        float wflat[PMAX * MAXT];
        weights(wflat, P, 5, rows);
        stats(s0);
        std::vector<double> us;
        for (int it = 0; it < n; ++it) {
          struct timespec a, b;
          if (mode == 0) {
            clock_gettime(CLOCK_MONOTONIC, &a);
            const unsigned long long tag = ps.post(P, 5, wflat, CPUJOB_OUT_BF16);
            if (cpustage_test_wait(seg, tag, 2000000000ull, &st) != 0) {
              std::printf("bench: a job timed out\n");
              break;
            }
            clock_gettime(CLOCK_MONOTONIC, &b);
            us.push_back(tdiff_us(a, b));
          } else {
            sleep_us(200);
            clock_gettime(CLOCK_MONOTONIC, &a);
            ps.post(P, 5, wflat, CPUJOB_OUT_BF16);
            const unsigned long long tag = ps.post(P, 5, wflat, CPUJOB_OUT_BF16);
            if (cpustage_test_wait(seg, tag, 2000000000ull, &st) != 0) {
              std::printf("bench: a job timed out\n");
              break;
            }
            clock_gettime(CLOCK_MONOTONIC, &b);
            us.push_back(tdiff_us(a, b) / 2.0);                       // per job of the pair
          }
        }
        if (us.empty()) continue;
        stats(s1);
        const Pct q = pct(us);
        const double pool = pool_us(s0, s1);
        std::printf("%-12s %d       %d           | %24.1f %6.1f %6.1f %6.1f %6.1f | %11.1f | %9.1f\n",
                    mode == 0 ? "back-to-back" : "pairs (/2)", P, rows, q.med, q.p10, q.p90, q.p99, q.mx, pool,
                    q.med / P);
      }
  stats(s1);
  std::printf("pool: %llu jobs, %llu abandoned, %llu stale, max queue %llu, longest %.1f us\n", s1[S_DONE],
              s1[S_ABANDONED], s1[S_STALE], s1[S_QMAX], (double)s1[S_MAX_NS] / 1000.0);
  std::printf("gate: <= 28 us per record in jobs of 5 -> 5 late records per KDA layer-step; <= 40 -> 3; "
              "above 40: build S5 instead\n");
  cpustage_stop();
  std::free(seg);
}

// A canary: with fused multiply-adds formed by the compiler the documented order would not hold everywhere. The
// kernel's own sums are written so that it does not matter (exact products, explicit fma); this only warns.
__attribute__((noinline)) static float canary(float a, float b, float c) { return a * b + c; }

static bool read_records(const std::string& path, uint8_t* dst, int n) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  bool ok = true;
  for (int e = 0; ok && e < n; ++e) {
    size_t got = 0;
    while (ok && got < (size_t)REC_BYTES) {
      const ssize_t r =
          pread(fd, dst + (size_t)e * REC_BYTES + got, (size_t)REC_BYTES - got, (off_t)e * REC_BYTES + (off_t)got);
      ok = r > 0;
      if (ok) got += (size_t)r;
    }
  }
  close(fd);
  return ok;
}

int main(int argc, char** argv) {
  std::string hi2;
  int bench_mode = 0, bench_nt = 0, bench_core = 0, bench_n = 200, nrec_bench = 96;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--hi2" && i + 1 < argc) hi2 = argv[++i];
    else if ((a == "--bench" || a == "--bench-late") && i + 2 < argc) {
      bench_mode = a == "--bench" ? 1 : 2;
      bench_nt = std::atoi(argv[++i]);
      bench_core = std::atoi(argv[++i]);
      if (i + 1 < argc && argv[i + 1][0] != '-') bench_n = std::atoi(argv[++i]);
    } else if (a == "--poster" && i + 1 < argc) g_poster = std::atoi(argv[++i]);
    else if (a == "--records" && i + 1 < argc) nrec_bench = std::max(PMAX, std::atoi(argv[++i]));
    else {
      std::printf("usage: %s [--hi2 FILE] [--bench | --bench-late THREADS CORE0 [N]] [--poster CORE] [--records N]\n",
                  argv[0]);
      return 2;
    }
  }
  if (cpustage_record_bytes() != REC_BYTES) { std::printf("library and header disagree\n"); return 2; }
  {
    long long lay[16];
    const int m = cpustage_layout(lay, 16);
    CHECK(m == 16 && lay[0] == CPUSEG2_BYTES && lay[8] == (long long)MAXT * H && lay[14] == (long long)sizeof(CpuJob2),
          "library layout");
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11: 0 with separate operations, 2^-24 when fused
    volatile float va = 1.0f + 0x1p-12f, vb = 1.0f + 0x1p-12f, vc = -(1.0f + 0x1p-11f);
    if (canary(va, vb, vc) != 0.0f)
      std::printf("note: the compiler fuses a * b + c here; build with -ffp-contract=off\n");
  }
  if (bench_mode != 0) {
    if (g_poster < 0) g_poster = bench_core > 0 ? bench_core - 1 : bench_core + bench_nt;
    pin_self(g_poster);
    std::vector<uint8_t> bank((size_t)nrec_bench * REC_BYTES);
    for (int e = 0; e < nrec_bench; ++e) make_record(bank.data() + (size_t)e * REC_BYTES, 1000 + (uint64_t)e);
    if (!hi2.empty() && !read_records(hi2, bank.data(), nrec_bench))   // real records time the same; allowed
      std::printf("note: could not read %d records from %s; synthetic records\n", nrec_bench, hi2.c_str());
    if (bench_mode == 1) bench_ontime(bank.data(), nrec_bench, bench_nt, bench_core, bench_n);
    else bench_late(bank.data(), nrec_bench, bench_nt, bench_core, bench_n);
    return g_fail == 0 ? 0 : 1;
  }
  std::printf("CPU set: %d cores; build: %s\n", count_cores(),
#if defined(__ARM_FEATURE_MATMUL_INT8) && !defined(CPUSTAGE_NO_I8MM)
              "SMMLA (i8mm)"
#else
              "SDOT (no i8mm)"
#endif
  );
  const int NREC = 12;
  std::vector<uint8_t> bank((size_t)NREC * REC_BYTES);
  if (!hi2.empty()) std::printf("synthetic records first, then %s\n", hi2.c_str());
  for (int e = 0; e < NREC; ++e) make_record(bank.data() + (size_t)e * REC_BYTES, 1000 + (uint64_t)e);
  const uint8_t* recs[PMAX];
  for (int p = 0; p < PMAX; ++p) recs[p] = bank.data() + (size_t)(p % 3) * REC_BYTES;
  test_decode(bank.data());
  test_decode(bank.data() + 2 * REC_BYTES);
  test_mmla();
  test_round_quant();
  test_situ();
  test_gemv(bank.data() + REC_BYTES);
  test_jobs(recs, "synthetic records");
  if (!hi2.empty()) {
    std::vector<uint8_t> real((size_t)3 * REC_BYTES);
    const bool ok = read_records(hi2, real.data(), 3);
    CHECK(ok, "could not read three records from %s", hi2.c_str());
    if (ok) {
      const uint8_t* rr[PMAX];
      for (int p = 0; p < PMAX; ++p) rr[p] = real.data() + (size_t)(p % 3) * REC_BYTES;
      test_decode(real.data());
      test_gemv(real.data() + REC_BYTES);
      test_jobs(rr, hi2.c_str());
    }
  }
  test_pool(bank.data(), NREC);
  std::printf(g_fail == 0 ? "CPUSTAGE: PASS\n" : "CPUSTAGE: FAIL (%d checks)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
