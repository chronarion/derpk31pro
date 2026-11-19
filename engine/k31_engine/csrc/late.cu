// The late tier's GB300 side (K31_LATE, k31_engine/latetier.py): two one-CTA kernels around a job of the Grace pool.
// The pool (engine/pool, libcpustage.so) and these kernels share one pinned segment laid out by
// engine/pool/cpustage_seg2.h: NSLOT job / result slots used round-robin by sequence number.
//
//   late_send   after the late plan (late.py `plan`, on the late stream): the job header (the call's late records and
//               each row's routing weight for them), then the call's latent rows as int8 with a power-of-two scale per
//               32 inputs in row-planar byte order (the rule of cpustage_seg2.h) into job slot seq % NSLOT, a
//               system-scope fence, and the door tag (release). A call whose plan named no record publishes nothing:
//               the state's tag is 0 and the landing finds no job.
//   late_land   after the layer's routed output transform (late stream): a bounded wait for the done tag of the job
//               the send published, then one pass over the T x 3584 late rows (float32 in the segment, host memory
//               read over C2C), with u = the routed latent sum the layer normalized and v = rows - mfix (mfix = the
//               mean sum the layer's on-time cut-mean already gave these picks, or none):
//                   A = sum u^2, C = sum u v, B = sum v^2 per row (fixed-order reductions: the output is deterministic)
//                   rb = rsqrt(A / H + eps), ra = rsqrt((A + 2 C + B) / H + eps) = rb / sqrt(1 + q),
//                   q = (2 C + B) / (A + H eps), cb = ra / rb - 1 = -q / (sqrt(1 + q) (1 + sqrt(1 + q)))
//                   xl = bf16((v * ra) * g)          the input of the layer's own up GEMV: z = up(xl)
//                   coef[t] = (has, cb, ra, rb)      has = 1 if the row has a late share (some v != 0)
//               The completion is then dst += z + cb * y (late.py `late_add`; y = the layer's on-time routed output, the
//               up GEMV's result before the residual add):
//                   up(norm(u + v)) - up(norm(u)) = up(ra g v) + (ra / rb - 1) y          (check_identity.py)
//               If the pool is late, down or refused the job, every row has has = 0 and xl = 0: the completion adds
//               exactly nothing, and the late picks keep the mean they got on time ("mean now, correction later").
//               With seg == null the rows come from device memory (v_dev, float32 or bf16): the GPU variant, and the
//               tests' way to check the arithmetic without a pool.
//
// The sequence: the state's `seq` is the last job's sequence number, `tag` the tag of the job the current call sent
// (0 = none). The send and the landing of a layer run on one stream (send(L), landing(L), send(L + 1), ...), so the
// landing finds its own job in `tag` and clears it.
//
// Build (build.sh, or engine/k31_engine/csrc/build_late.sh): nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a
//   -shared -Xcompiler -fPIC -I engine/pool -o engine/band/liblate.so engine/k31_engine/csrc/late.cu
// The CPU tests build the same kernels for the host with -DLATE_EMU (late_emu.h: one CTA = 512 host threads).
#if defined(LATE_EMU)
#include "late_emu.h"
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#endif

#include <time.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cpustage_seg2.h"

#if defined(LATE_EMU)
#define LATE_FN inline
#else
#define LATE_FN __host__ __device__ __forceinline__
#endif

namespace late {

constexpr int H = CPUSEG_H;                        // 3,584 latent inputs
constexpr int KB = CPUSEG_KB;                      // 112 blocks of 32
constexpr int MT = CPUSEG_MAXT;                    // 8 rows
constexpr int PM = CPUSEG2_PMAX;                   // 8 records per job
constexpr int NS = CPUSEG2_NSLOT;                  // 4 slots
constexpr int THREADS = 512;
constexpr int G8 = H / 8;                          // 448 groups of 8 values per row
constexpr int TURNS = MT * G8 / THREADS;           // 7 groups per thread at T = 8
constexpr int WARPS = THREADS / 32;
static_assert(MT * G8 == TURNS * THREADS, "whole turns");
static_assert(G8 % 32 == 0, "a warp's 32 groups of a turn lie in one row");
static_assert(sizeof(CpuJob2) == 328, "CpuJob2 layout (cpustage_seg2.h)");
static_assert(offsetof(CpuJob2, slot) == 40 && offsetof(CpuJob2, w) == 72, "CpuJob2 layout (cpustage_seg2.h)");

// Device state, 16 u64 words (latetier.py STATE_WORDS), zero at start but for `seq`.
struct LateState {
  unsigned long long seq;          // sequence number of the last job sent
  unsigned long long tag;          // tag of the job the current call sent; 0 = none (the landing reads and clears it)
  unsigned long long n_jobs;       // jobs published
  unsigned long long n_recs;       // records in them
  unsigned long long n_added;      // landings that found their rows
  unsigned long long n_timeout;    // landings that gave up waiting
  unsigned long long n_error;      // jobs the pool refused
  unsigned long long n_empty;      // landings of calls that sent no job
  unsigned long long n_down;       // landings that found the pool stopped
  unsigned long long wait_ns;      // sum of the landings' waits (from the kernel's start)
  unsigned long long wait_max;     // the longest
  unsigned long long t_pub;        // GB300 timer when the last tag was published
  unsigned long long n_rows;       // rows that received a late share
  unsigned long long pad[3];
};
static_assert(sizeof(LateState) == 128, "state block");

enum : unsigned long long { LT_ADDED = 0, LT_TIMEOUT = 1, LT_ERROR = 2, LT_EMPTY = 3, LT_DOWN = 4 };

// ------------------------------------------------------------------------------------------------ shared arithmetic
LATE_FN uint32_t f2u(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
#endif
}
LATE_FN float u2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}
LATE_FN int rint_even(float v) {                   // round to nearest, ties to even (the default rounding mode)
#if defined(__CUDA_ARCH__)
  return __float2int_rn(v);
#else
  return (int)std::lrintf(v);
#endif
}
LATE_FN float bf16_value(uint16_t b) { return u2f((uint32_t)b << 16); }

// One 32-input block of a row -> 32 int8 codes in row-planar order (input k = 4 i + p of the block is byte 8 p + i)
// as eight little-endian words, and the exponent byte e + 128 (cpustage_seg2.h / r12 cpustage_seg.h):
//   amax = max |v|; e = ((bits(amax) + 0x1FFFF) >> 23) - 133 clamped to [e_min, 120] (the smallest e with
//   amax 2^-e <= 127); q = round-to-nearest-even(v 2^-e) clamped to [-127, 127].
LATE_FN int quant_block(const float* v, int e_min, uint32_t* o) {
  float amax = 0.f;
  for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
  int e = (int)((f2u(amax) + 0x1FFFFu) >> 23) - 133;
  e = e < e_min ? e_min : (e > 120 ? 120 : e);
  const float inv = u2f((uint32_t)(127 - e) << 23);             // 2^-e, exact
  for (int w = 0; w < 8; w++) o[w] = 0u;
  for (int k = 0; k < 32; k++) {
    int q = rint_even(v[k] * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    const int idx = 8 * (k & 3) + (k >> 2);
    o[idx >> 2] |= (uint32_t)(q & 0xFF) << (8 * (idx & 3));
  }
  return e + 128;
}

// The job header the send writes (thread 0; the host twin for the tests uses the same function).
LATE_FN void fill_header(CpuJob2* h, unsigned long long tag, int n, int T, int layer, int flags, int e_min, float alpha,
                         float beta, const int* lslot, const float* lw) {
  h->tag = tag;
  h->n_pick = (uint32_t)n;
  h->n_rows = (uint32_t)T;
  h->layer = (uint32_t)layer;
  h->flags = (uint32_t)flags;
  h->alpha = alpha;
  h->beta = beta;
  h->e_min = e_min;
  h->pad = 0u;
  for (int p = 0; p < PM; p++) {
    h->slot[p] = p < n ? lslot[p] : -1;
    for (int t = 0; t < MT; t++) h->w[p][t] = (p < n && t < T) ? lw[p * MT + t] : 0.f;
  }
}

// A row's coefficients from its four sums (A = sum u^2, C = sum u v, B = sum v^2, N = sum |v|): (has, cb, ra, rb).
LATE_FN void row_coef(float A, float C, float B, float N, bool rows, float eps, float* coef) {
  const float hE = (float)H * eps, den = A + hE;
#if defined(__CUDA_ARCH__)
  const float rb = rsqrtf(A / (float)H + eps);
#else
  const float rb = 1.f / std::sqrt(A / (float)H + eps);
#endif
  const bool has = rows && N > 0.f;
  float ra = rb, cb = 0.f;
  if (has) {
    float q = (2.f * C + B) / den;
    float s = 1.f + q;
    const float smin = hE / den;                   // |u + v| = 0: the RMS is eps's alone (rounding can undershoot it)
    if (!(s > smin)) {
      s = smin;
      q = smin - 1.f;
    }
    const float sq = sqrtf(s);
    cb = -q / (sq * (1.f + sq));                  // ra / rb - 1 without the cancellation of a small share
    ra = rb / sq;
  }
  coef[0] = has ? 1.f : 0.f;
  coef[1] = cb;
  coef[2] = ra;
  coef[3] = rb;
}

#if defined(__CUDACC__) || defined(LATE_EMU)
// ------------------------------------------------------------------------------------------------ device helpers
#if defined(LATE_EMU)
inline unsigned long long globaltimer() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}
inline unsigned long long ld_acquire_sys(const void* p) {
  return __atomic_load_n(reinterpret_cast<const unsigned long long*>(p), __ATOMIC_ACQUIRE);
}
inline void st_release_sys(void* p, unsigned long long v) {
  __atomic_store_n(reinterpret_cast<unsigned long long*>(p), v, __ATOMIC_RELEASE);
}
inline void fence_sys() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline uint4 ld_volatile_u4(const void* p) {
  const volatile uint32_t* q = reinterpret_cast<const volatile uint32_t*>(p);
  return make_uint4(q[0], q[1], q[2], q[3]);
}
inline void pdl_wait() {}
inline void pdl_trigger() {}
#else
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long ld_acquire_sys(const void* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(void* p, unsigned long long v) {
  asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
// Loads that never hit a line this SM cached earlier (host memory another agent writes).
__device__ __forceinline__ uint4 ld_volatile_u4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
// Programmatic dependent launch: no-ops without the launch attribute.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
#endif

// ------------------------------------------------------------------------------------------------ late_send
struct SendArgs {
  uint8_t* seg;                    // the segment (device pointer of the pinned pages)
  LateState* st;
  const int* lslot;                // [PM]: bank record of late slot p, -1 past the last (the plan fills from 0)
  const float* lw;                 // [PM][MT]: routing weight of row t for slot p (0 = the row does not use it)
  const __nv_bfloat16* x;          // [T][H] the call's latent rows
  int T, layer, flags, e_min;      // flags: CPUJOB_* of cpustage_seg2.h
  float alpha, beta;               // SiTU constants
};

__global__ void __launch_bounds__(THREADS) late_send(const SendArgs a) {
  pdl_wait();
  __shared__ __align__(16) uint32_t hdr[CPUSEG2_JOB_BYTES / 4];
  __shared__ __align__(16) uint8_t ex[MT * KB];
  __shared__ int n_s;
  __shared__ unsigned long long seq_s;
  const int tid = threadIdx.x, T = min(max(a.T, 1), MT);
  if (tid < CPUSEG2_JOB_BYTES / 4) hdr[tid] = 0u;
  if (tid == 0) {
    int n = 0;
    while (n < PM && a.lslot[n] >= 0) n++;
    n_s = n;
    seq_s = a.st->seq + 1;
  }
  __syncthreads();
  const int n = n_s;
  if (n == 0) {                                     // nothing late in this call: no job, the landing finds tag 0
    if (tid == 0) a.st->tag = 0;
    pdl_trigger();
    return;
  }
  const unsigned long long seq = seq_s;
  const unsigned long long tag = cpuseg_tag(seq, (uint32_t)a.layer, (uint32_t)T);
  const int s = (int)(seq % NS);
  if (tid == 0)
    fill_header(reinterpret_cast<CpuJob2*>(hdr), tag, n, T, a.layer, a.flags, a.e_min, a.alpha, a.beta, a.lslot,
                a.lw);
  uint8_t* xq = a.seg + CPUSEG2_OFF_X + (size_t)s * CPUSEG2_X_STRIDE;    // int8 rows [MT][H], then exponents [MT][KB]
  for (int blk = tid; blk < T * KB; blk += THREADS) {   // one 32-input block per thread and turn
    const int g = blk / KB, b = blk % KB;
    const uint4* xp = reinterpret_cast<const uint4*>(a.x + (size_t)g * H + b * 32);
    float v[32];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 u = xp[q];
      const uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int r = 0; r < 4; r++) {
        v[q * 8 + 2 * r] = __uint_as_float(wd[r] << 16);
        v[q * 8 + 2 * r + 1] = __uint_as_float(wd[r] & 0xFFFF0000u);
      }
    }
    uint32_t o[8];
    ex[blk] = (uint8_t)quant_block(v, a.e_min, o);
    uint4* dst = reinterpret_cast<uint4*>(xq + (size_t)g * H + b * 32);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
  __syncthreads();                                  // the header and the exponents are in shared memory
  if (tid < CPUSEG2_JOB_BYTES / 16)
    reinterpret_cast<uint4*>(a.seg + CPUSEG2_OFF_JOB + (size_t)s * CPUSEG2_JOB_BYTES)[tid] =
        reinterpret_cast<const uint4*>(hdr)[tid];
  if (tid < T * KB / 16)                            // 7 x 16 B per row
    reinterpret_cast<uint4*>(xq + MT * H)[tid] = reinterpret_cast<const uint4*>(ex)[tid];
  fence_sys();                                      // every thread's writes are visible system-wide ...
  __syncthreads();
  if (tid == 0) {                                   // ... before the door says there is a job
    st_release_sys(a.seg + CPUSEG2_OFF_DOOR, tag);
    LateState* st = a.st;
    st->seq = seq;
    st->tag = tag;
    st->n_jobs += 1;
    st->n_recs += (unsigned long long)n;
    st->t_pub = globaltimer();
  }
  pdl_trigger();
}

// ------------------------------------------------------------------------------------------------ late_land
struct LandArgs {
  uint8_t* seg;                    // the segment, or null: the rows are on the device (v_dev)
  LateState* st;
  const __nv_bfloat16* u;          // [T][H] the routed latent sum the layer normalized (rows contiguous)
  const void* v_dev;               // seg == null: [T][H] float32, or bf16 when v_bf16
  const float* mfix;               // [T][H] float32 or null: the mean sum the on-time path gave the late picks
  const __nv_bfloat16* g;          // [H] the routed output transform's RMSNorm weight
  __nv_bfloat16* xl;               // [T][H] out
  float* coef;                     // [T][4] out: has, cb, ra, rb
  int T, v_bf16;
  float eps;
  int layer;                       // the job must be this layer's, with T rows (else: no job)
  unsigned long long timeout_ns;   // bound of the wait, from this kernel's start
};

__global__ void __launch_bounds__(THREADS) late_land(const LandArgs a) {
  pdl_wait();
  __shared__ unsigned long long sh[2];
  __shared__ float part[TURNS][WARPS][4];
  __shared__ float scale_s[MT];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int T = min(max(a.T, 1), MT), n8 = T * G8;
  if (tid == 0) {
    unsigned long long status = LT_ADDED, slot = 0;
    if (a.seg != nullptr) {
      LateState* st = a.st;
      const unsigned long long tag = st->tag;
      const unsigned long long t0 = globaltimer();
      if (tag == 0 || ((tag >> 8) & 0xFF) != (unsigned long long)(a.layer & 0xFF) || (tag & 0xFF) != (unsigned long long)T) {
        status = LT_EMPTY;                          // no job in this call (or not this layer's: never read its rows)
      } else {
        slot = cpuseg_seq(tag) % NS;
        const uint8_t* done = a.seg + CPUSEG2_OFF_DONE + 128 * (size_t)slot;
        if (ld_acquire_sys(a.seg + CPUSEG2_OFF_STATE) == 0 && ld_acquire_sys(done) != tag) {
          status = LT_DOWN;                         // no pool: nothing to wait for
        } else {
          status = LT_TIMEOUT;
          for (;;) {
            if (ld_acquire_sys(done) == tag) {      // the rows and the status word were written before the tag
              status = (ld_acquire_sys(done + 8) & CPUSEG_ST_ERROR) ? LT_ERROR : LT_ADDED;
              break;
            }
            if (globaltimer() - t0 > a.timeout_ns) break;
          }
          const unsigned long long w = globaltimer() - t0;
          st->wait_ns += w;
          if (w > st->wait_max) st->wait_max = w;
        }
      }
      st->tag = 0;                                  // a later landing without a send finds no job
    }
    sh[0] = status;
    sh[1] = slot;
  }
  __syncthreads();
  const unsigned long long status = sh[0];
  const bool rows = status == LT_ADDED;
  const uint8_t* res = (rows && a.seg != nullptr) ? a.seg + CPUSEG2_OFF_RES + (size_t)sh[1] * CPUSEG2_RES_BYTES : nullptr;
  // 1. v = rows - mfix for every group of 8 values the thread owns (all loads first: the rows are a C2C round trip)
  float v[TURNS][8];
  if (rows) {
    if (res != nullptr) {
      uint4 r0[TURNS], r1[TURNS];
#pragma unroll
      for (int j = 0; j < TURNS; j++) {
        const int i = tid + j * THREADS;
        if (i < n8) {
          r0[j] = ld_volatile_u4(res + 32 * (size_t)i);
          r1[j] = ld_volatile_u4(res + 32 * (size_t)i + 16);
        } else {
          r0[j] = make_uint4(0u, 0u, 0u, 0u);
          r1[j] = r0[j];
        }
      }
#pragma unroll
      for (int j = 0; j < TURNS; j++) {
        v[j][0] = __uint_as_float(r0[j].x); v[j][1] = __uint_as_float(r0[j].y);
        v[j][2] = __uint_as_float(r0[j].z); v[j][3] = __uint_as_float(r0[j].w);
        v[j][4] = __uint_as_float(r1[j].x); v[j][5] = __uint_as_float(r1[j].y);
        v[j][6] = __uint_as_float(r1[j].z); v[j][7] = __uint_as_float(r1[j].w);
      }
    } else {
#pragma unroll
      for (int j = 0; j < TURNS; j++) {
        const int i = tid + j * THREADS;
        if (i < n8 && a.v_bf16) {
          const uint4 r = reinterpret_cast<const uint4*>(a.v_dev)[i];
          const uint32_t rw[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int k = 0; k < 4; k++) {
            v[j][2 * k] = __uint_as_float(rw[k] << 16);
            v[j][2 * k + 1] = __uint_as_float(rw[k] & 0xFFFF0000u);
          }
        } else if (i < n8) {
          const float4 r0 = reinterpret_cast<const float4*>(a.v_dev)[2 * i];
          const float4 r1 = reinterpret_cast<const float4*>(a.v_dev)[2 * i + 1];
          v[j][0] = r0.x; v[j][1] = r0.y; v[j][2] = r0.z; v[j][3] = r0.w;
          v[j][4] = r1.x; v[j][5] = r1.y; v[j][6] = r1.z; v[j][7] = r1.w;
        } else {
#pragma unroll
          for (int k = 0; k < 8; k++) v[j][k] = 0.f;
        }
      }
    }
    if (a.mfix != nullptr) {
#pragma unroll
      for (int j = 0; j < TURNS; j++) {
        const int i = tid + j * THREADS;
        if (i < n8) {
          const float4 m0 = reinterpret_cast<const float4*>(a.mfix)[2 * i];
          const float4 m1 = reinterpret_cast<const float4*>(a.mfix)[2 * i + 1];
          v[j][0] = __fsub_rn(v[j][0], m0.x); v[j][1] = __fsub_rn(v[j][1], m0.y);
          v[j][2] = __fsub_rn(v[j][2], m0.z); v[j][3] = __fsub_rn(v[j][3], m0.w);
          v[j][4] = __fsub_rn(v[j][4], m1.x); v[j][5] = __fsub_rn(v[j][5], m1.y);
          v[j][6] = __fsub_rn(v[j][6], m1.z); v[j][7] = __fsub_rn(v[j][7], m1.w);
        }
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < TURNS; j++)
#pragma unroll
      for (int k = 0; k < 8; k++) v[j][k] = 0.f;
  }
  // 2. the four sums of each (turn, warp): one row each (a warp's 32 groups lie in one row), fixed-order shuffles
#pragma unroll
  for (int j = 0; j < TURNS; j++) {
    const int i = tid + j * THREADS;
    float sA = 0.f, sC = 0.f, sB = 0.f, sN = 0.f;
    if (i < n8) {
      const uint4 ur = reinterpret_cast<const uint4*>(a.u)[i];
      const uint32_t uw[4] = {ur.x, ur.y, ur.z, ur.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float u0 = __uint_as_float(uw[k] << 16), u1 = __uint_as_float(uw[k] & 0xFFFF0000u);
        const float v0 = v[j][2 * k], v1 = v[j][2 * k + 1];
        sA = __fadd_rn(sA, __fadd_rn(__fmul_rn(u0, u0), __fmul_rn(u1, u1)));
        sC = __fadd_rn(sC, __fadd_rn(__fmul_rn(u0, v0), __fmul_rn(u1, v1)));
        sB = __fadd_rn(sB, __fadd_rn(__fmul_rn(v0, v0), __fmul_rn(v1, v1)));
        sN = __fadd_rn(sN, __fadd_rn(fabsf(v0), fabsf(v1)));
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sA += __shfl_xor_sync(0xffffffffu, sA, off);
      sC += __shfl_xor_sync(0xffffffffu, sC, off);
      sB += __shfl_xor_sync(0xffffffffu, sB, off);
      sN += __shfl_xor_sync(0xffffffffu, sN, off);
    }
    if (lane == 0) {
      part[j][warp][0] = sA; part[j][warp][1] = sC; part[j][warp][2] = sB; part[j][warp][3] = sN;
    }
  }
  __syncthreads();
  // 3. per row, the (turn, warp) partial sums in a fixed order, then the coefficients
  if (tid < T) {
    float A = 0.f, C = 0.f, B = 0.f, N = 0.f;
    for (int j = 0; j < TURNS; j++)
      for (int w = 0; w < WARPS; w++) {
        const int i0 = 32 * w + THREADS * j;
        if (i0 < n8 && i0 / G8 == tid) {
          A += part[j][w][0]; C += part[j][w][1]; B += part[j][w][2]; N += part[j][w][3];
        }
      }
    float c[4];
    row_coef(A, C, B, N, rows, a.eps, c);
    reinterpret_cast<float4*>(a.coef)[tid] = make_float4(c[0], c[1], c[2], c[3]);
    scale_s[tid] = c[0] > 0.f ? c[2] : 0.f;         // xl = 0 for a row without a share
  }
  __syncthreads();
  // 4. xl = bf16((v * ra) * g)
#pragma unroll
  for (int j = 0; j < TURNS; j++) {
    const int i = tid + j * THREADS;
    if (i < n8) {
      const float sc = scale_s[i / G8];
      const uint4 gr = reinterpret_cast<const uint4*>(a.g)[i % G8];
      const uint32_t gw[4] = {gr.x, gr.y, gr.z, gr.w};
      uint32_t ow[4];
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float g0 = __uint_as_float(gw[k] << 16), g1 = __uint_as_float(gw[k] & 0xFFFF0000u);
        const __nv_bfloat162 p = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(v[j][2 * k], sc), g0),
                                                       __fmul_rn(__fmul_rn(v[j][2 * k + 1], sc), g1));
        ow[k] = *reinterpret_cast<const uint32_t*>(&p);
      }
      reinterpret_cast<uint4*>(a.xl)[i] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  }
  if (tid == 0 && a.seg != nullptr) {
    LateState* st = a.st;
    if (status == LT_ADDED) st->n_added += 1;
    else if (status == LT_TIMEOUT) st->n_timeout += 1;
    else if (status == LT_ERROR) st->n_error += 1;
    else if (status == LT_EMPTY) st->n_empty += 1;
    else st->n_down += 1;
  }
  __syncthreads();                                  // scale_s is final: count the rows that received a share
  if (tid == 0 && a.seg != nullptr) {
    unsigned long long nr = 0;
    for (int t = 0; t < T; t++) nr += scale_s[t] != 0.f ? 1ull : 0ull;
    a.st->n_rows += nr;
  }
  pdl_trigger();                                    // at the end: the up GEMV that follows must not hold SMs while
}                                                   // this kernel waits for the pool

#if defined(LATE_EMU)
template <typename Args>
int launch(void (*kernel)(const Args), const Args& args, void*, int) {
  late_emu::run_block(THREADS, [&] { kernel(args); });
  return 0;
}
#else
template <typename Args>
int launch(void (*kernel)(const Args), const Args& args, cudaStream_t stream, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, args);
  return e != cudaSuccess ? (int)e : (int)cudaGetLastError();
}
#endif
#endif  // __CUDACC__ || LATE_EMU

}  // namespace late

extern "C" {

// ctypes mirrors of the argument blocks (latetier.py _Send / _Land)
struct LateSendC {
  void* seg; void* st; const void* lslot; const void* lw; const void* x;
  int T, layer, flags, e_min;
  float alpha, beta;
};
struct LateLandC {
  void* seg; void* st; const void* u; const void* v_dev; const void* mfix; const void* g; void* xl; void* coef;
  int T, v_bf16;
  float eps;
  int layer;
  unsigned long long timeout_ns;
};

#if defined(LATE_EMU)
typedef void* late_stream_t;
#else
typedef cudaStream_t late_stream_t;
#endif
static_assert(sizeof(LateSendC) == sizeof(late::SendArgs), "send argument block");
static_assert(sizeof(LateLandC) == sizeof(late::LandArgs), "landing argument block");

int late_send(const LateSendC* a, void* stream, int pdl) {
  return late::launch(late::late_send, *reinterpret_cast<const late::SendArgs*>(a), (late_stream_t)stream, pdl);
}

int late_land(const LateLandC* a, void* stream, int pdl) {
  return late::launch(late::late_land, *reinterpret_cast<const late::LandArgs*>(a), (late_stream_t)stream, pdl);
}

#if defined(LATE_EMU)
int late_map(void* host, long long, void** dev) {          // host memory is the "device" memory here
  *dev = host;
  return 0;
}
int late_unmap(void*) { return 0; }
#else

// The segment's pages pinned and mapped for the GPU (anonymous mmap from the engine; the pool uses the host address).
int late_map(void* host, long long bytes, void** dev) {
  cudaError_t e = cudaHostRegister(host, (size_t)bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (e != cudaSuccess) return (int)e;
  e = cudaHostGetDevicePointer(dev, host, 0);
  return (int)e;
}

int late_unmap(void* host) { return (int)cudaHostUnregister(host); }
#endif

// What a Python caller needs of cpustage_seg2.h, so that it never restates an offset: [segment bytes, door, state, done,
// job, job bytes, x, x stride, result, result bytes, slots, records per job, rows, H, blocks, magic, sizeof(CpuJob2),
// offsetof slot, offsetof w, state block bytes, threads]
void late_layout(long long* out) {
  out[0] = CPUSEG2_BYTES; out[1] = CPUSEG2_OFF_DOOR; out[2] = CPUSEG2_OFF_STATE; out[3] = CPUSEG2_OFF_DONE;
  out[4] = CPUSEG2_OFF_JOB; out[5] = CPUSEG2_JOB_BYTES; out[6] = CPUSEG2_OFF_X; out[7] = CPUSEG2_X_STRIDE;
  out[8] = CPUSEG2_OFF_RES; out[9] = CPUSEG2_RES_BYTES; out[10] = CPUSEG2_NSLOT; out[11] = CPUSEG2_PMAX;
  out[12] = CPUSEG_MAXT; out[13] = CPUSEG_H; out[14] = CPUSEG_KB; out[15] = (long long)CPUSEG2_MAGIC;
  out[16] = (long long)sizeof(CpuJob2); out[17] = (long long)offsetof(CpuJob2, slot);
  out[18] = (long long)offsetof(CpuJob2, w); out[19] = (long long)sizeof(late::LateState); out[20] = late::THREADS;
}

// Host twins of the send's arithmetic (CPU tests; the kernel runs the same functions):
//   the T rows x [T][3584] (float values of bf16) -> row-planar int8 codes xq [T][3584] and exponent bytes xe [T][112]
void late_host_quant_rows(const float* x, int T, int e_min, uint8_t* xq, uint8_t* xe) {
  for (int g = 0; g < T; g++)
    for (int b = 0; b < late::KB; b++) {
      uint32_t o[8];
      xe[g * late::KB + b] = (uint8_t)late::quant_block(x + (size_t)g * late::H + 32 * b, e_min, o);
      std::memcpy(xq + (size_t)g * late::H + 32 * b, o, 32);
    }
}

//   the job header bytes (CPUSEG2_JOB_BYTES, zero-padded) the send writes
void late_host_header(unsigned long long seq, int layer, int T, const int* lslot, const float* lw, int flags, int e_min,
                      float alpha, float beta, uint8_t* out) {
  int n = 0;
  while (n < late::PM && lslot[n] >= 0) n++;
  std::memset(out, 0, CPUSEG2_JOB_BYTES);
  CpuJob2 h;
  std::memset(&h, 0, sizeof(h));
  late::fill_header(&h, cpuseg_tag(seq, (uint32_t)layer, (uint32_t)T), n, T, layer, flags, e_min, alpha, beta, lslot,
                    lw);
  std::memcpy(out, &h, sizeof(h));
}

//   a row's coefficients from its four sums (has, cb, ra, rb); rows = the job's rows exist
void late_host_coef(const float* sums, int rows, float eps, float* coef) {
  late::row_coef(sums[0], sums[1], sums[2], sums[3], rows != 0, eps, coef);
}

// The segment's control words from a CPU thread (the bind's self-check, the tests' stand-in pool): acquire loads and
// release stores, as the pool and the kernels use them.
unsigned long long late_host_get(const uint8_t* seg, long long off) {
  return __atomic_load_n(reinterpret_cast<const unsigned long long*>(seg + off), __ATOMIC_ACQUIRE);
}
void late_host_set(uint8_t* seg, long long off, unsigned long long v) {
  __atomic_store_n(reinterpret_cast<unsigned long long*>(seg + off), v, __ATOMIC_RELEASE);
}
// Spin until the u64 at off holds v (0) or timeout_ns passed (1).
int late_host_wait(const uint8_t* seg, long long off, unsigned long long v, long long timeout_ns) {
  timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (unsigned n = 1;; n++) {
    if (late_host_get(seg, off) == v) return 0;
    if ((n & 255) == 0) {
      clock_gettime(CLOCK_MONOTONIC, &t1);
      if ((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec) > timeout_ns) return 1;
    }
  }
}

}  // extern "C"
