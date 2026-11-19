// RTX band of derpk31pro: the engine's GB300 side fused into two kernels per MoE call (k31_engine/band.py, K31_BAND_FUSED):
//
//   band_qsend        the call's work list from the router's picks (distinct non-resident band experts by largest
//                     weight, ties by lower id, at most 16; each row's weight for them), the MXFP8 rows of the call's
//                     latent input (DeepGEMM's rule: smallest power-of-two scale with amax / scale <= 448, round to
//                     nearest even, saturating) written straight into the request slot in the band's byte order, then
//                     the tag (release). Replaces band.plan + band.quant + band_send.
//   band_finish_bf16  the bounded wait of band_finish, then the result rows added into the routed latent sum
//                     out [T][3584] bfloat16 in place (float32 add, one rounding). Replaces the memset, band_finish
//                     and the float32 -> bfloat16 add.
//
// Same state block (bandgb::GbState) and mailbox protocol as rtxband_gb300.cu; the two can be mixed call by call.
// K31_BAND_PRIOR (B1, band_prior.cuh): with prior_on, a call past its cap keeps the experts with the largest
// weight x the row's prior chance of being accepted (plan / hint kernel and send alike); prior_on = 0 is today's plan.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC -o librtxband_fused.so
//        rtxband_fused.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "band_gb300.cuh"
#include "band_prior.cuh"

namespace {
using namespace bandgb;

constexpr int THREADS = 512, NE = 16, MT = 8, NPMAX = MT * 16, CALL_INTS = 8 + NE + NE * MT;

// Programmatic dependent launch: launched with programmatic stream serialization (rtxband_fused_pdl), a kernel may
// start while the previous one in the stream ends; it waits for that kernel's writes before reading anything and lets
// the next kernel launch at once. Both instructions are no-ops without the launch attribute.
__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

struct FusedArgs {
  uint8_t* seg;
  GbState* st;
  const int* ids;                      // [T][16] int32 (the router's picks)
  const float* w;                      // [T][16] float32 (their weights)
  const int* row_map;                  // [896]: resident row or -1
  const int* band_map;                 // [896]: band store slot or -1
  const __nv_bfloat16* x;              // [T][3584] the call's latent input rows
  __nv_bfloat16* out;                  // [T][3584] the routed latent sum (finish adds here)
  int* stats;                          // [3]: picks served, past a call's 16 experts, all picks (or null)
  int T, layer, flags, sf_min;
  float alpha, beta;
  unsigned long long timeout_ns;
  int* served;                         // [T][16] or null: 1 where the call's send serves the pick (v3, the cut-mean's
                                       // test: band picks past the cap get the mean instead of nothing)
  const float* msum;                   // [T][3584] float32 or null (v4): the cut picks' mean sum, added with the rows
  float* rstd;                         // [T] or null (v4): rsqrt(mean(out^2) + eps) of the finished BF16 rows, for the
  float eps;                           //   routed up-projection's RMSNorm (w8a16 NORM 2: no statistic prologue)
  float prior[MT];                     // B1: row t's prior chance of being accepted (used when prior_on != 0)
  int prior_on;                        //   0: the cap keeps the largest weights (today's plan)
};

// B1: the kept set by weight x the row's prior, then today's work-list order among the kept experts, written into p_s
// (past the cap: NPMAX). Called by every thread of the CTA after p_s and n_first are final (it has its own barriers);
// the code after it reads p_s exactly as without a prior.
__device__ __forceinline__ void prior_places(const FusedArgs& a, int NP, const int* e_s, const int* b_s,
                                             const float* w_s, const int* f_s, const float* sc_s, int* p_s) {
  __shared__ float k_s[NPMAX], pr_s[MT];
  __shared__ int kp_s[NPMAX];
  const int tid = threadIdx.x;
  const int cap = (a.flags >> 16) & 0xFF;
  const int ne = (cap > 0 && cap < NE) ? cap : NE;
  if (tid == 0) {
#pragma unroll
    for (int r = 0; r < MT; r++) pr_s[r] = a.prior[r];    // constant offsets into the argument block
  }
  __syncthreads();
  if (tid < NPMAX) k_s[tid] = bandprior::key(tid, NP, e_s, b_s, w_s, pr_s);
  __syncthreads();
  if (tid < NPMAX) kp_s[tid] = bandprior::kept(tid, NP, e_s, b_s, f_s, k_s, ne);
  __syncthreads();
  if (tid < NPMAX && b_s[tid] >= 0) p_s[tid] = bandprior::place(tid, NP, e_s, f_s, kp_s, sc_s, NPMAX);
  __syncthreads();
}

// The plan / hint kernel (K31_BAND_HINT): the send's planning half, launched right after top-k on the
// band's stream, writes the served flags and stats and publishes the call's band slots as a hint, so the RTX pulls
// those records into its L2 while the latent down-projection still runs; band_qsend then recomputes the same plan
// (same inputs) with served = stats = null and sends the rows.
__global__ void __launch_bounds__(THREADS) band_plan_hint(const FusedArgs a) {
  pdl_enter();
  __shared__ int e_s[NPMAX], b_s[NPMAX], f_s[NPMAX], p_s[NPMAX];
  __shared__ float w_s[NPMAX], sc_s[NPMAX];
  __shared__ int n_first;
  const int tid = threadIdx.x, T = min(max(a.T, 1), MT), NP = T * 16;
  if (tid < NPMAX) {
    int e = -1, b = -1;
    float w = 0.f;
    if (tid < NP) {
      e = a.ids[tid];
      w = a.w[tid];
      if (e >= 0 && e < 896 && a.row_map[e] < 0) b = a.band_map[e];
    }
    e_s[tid] = e; w_s[tid] = w; b_s[tid] = b >= 0 ? b : -1;
  }
  __syncthreads();
  if (tid < NPMAX) {                   // first occurrence of the expert in the call, the expert's largest weight
    int first = 0;
    float sc = -INFINITY;
    if (b_s[tid] >= 0) {
      first = 1;
      const int e = e_s[tid];
      for (int j = 0; j < NP; j++)
        if (b_s[j] >= 0 && e_s[j] == e) {
          if (j < tid) first = 0;
          sc = fmaxf(sc, w_s[j]);
        }
    }
    f_s[tid] = first; sc_s[tid] = sc;
  }
  __syncthreads();
  if (tid < NPMAX) {                   // place among the distinct experts: by largest weight, ties by lower id
    int pos = -1;
    if (b_s[tid] >= 0) {
      pos = 0;
      const float sc = sc_s[tid];
      const int e = e_s[tid];
      for (int j = 0; j < NP; j++)
        if (f_s[j] && (sc_s[j] > sc || (sc_s[j] == sc && e_s[j] < e))) pos++;
    }
    p_s[tid] = pos;
  }
  if (tid == 0) {
    int n = 0;
    for (int j = 0; j < NP; j++) n += f_s[j];
    n_first = n;
  }
  __syncthreads();
  if (a.prior_on) prior_places(a, NP, e_s, b_s, w_s, f_s, sc_s, p_s);    // B1 (a uniform branch: a kernel argument)
  // a runtime cap on the call's experts (K31_BAND_MAXE: flags bits 16-23, 0 = the full NE): the RTX's time per call
  // grows with its experts; picks past the cap count as past-16 (cut), as past NE always did
  const int cap = (a.flags >> 16) & 0xFF;
  const int ne = (cap > 0 && cap < NE) ? cap : NE;
  const int n_exp = min(n_first, ne);
  if (a.served != nullptr && tid < NP) a.served[tid] = (b_s[tid] >= 0 && p_s[tid] < ne) ? 1 : 0;
  if (tid == 0 && a.stats != nullptr) {
    int served = 0, past = 0, all = 0;
    for (int j = 0; j < NP; j++) {
      all += e_s[j] >= 0;
      if (b_s[j] >= 0) { if (p_s[j] < ne) served++; else past++; }
    }
    atomicAdd(a.stats + 0, served); atomicAdd(a.stats + 1, past); atomicAdd(a.stats + 2, all);
  }
  __shared__ int hint_s[1 + NE];
  if (tid < 1 + NE) hint_s[tid] = 0;
  __syncthreads();
  if (tid < NP && b_s[tid] >= 0 && p_s[tid] < n_exp && f_s[tid]) hint_s[1 + p_s[tid]] = b_s[tid];
  if (tid == 0) hint_s[0] = n_exp;
  __syncthreads();
  if (n_exp > 0) bandgb::hint_body(a.seg, a.st, hint_s);   // uniform branch: n_exp is the same in every thread
}

__global__ void __launch_bounds__(THREADS) band_qsend(const FusedArgs a) {
  pdl_enter();
  __shared__ int e_s[NPMAX], b_s[NPMAX], f_s[NPMAX], p_s[NPMAX];
  __shared__ float w_s[NPMAX], sc_s[NPMAX];
  __shared__ __align__(16) int call_s[CALL_INTS];
  __shared__ int n_first;
  const int tid = threadIdx.x, T = min(max(a.T, 1), MT), NP = T * 16;
  if (tid < NPMAX) {
    int e = -1, b = -1;
    float w = 0.f;
    if (tid < NP) {
      e = a.ids[tid];
      w = a.w[tid];
      if (e >= 0 && e < 896 && a.row_map[e] < 0) b = a.band_map[e];
    }
    e_s[tid] = e; w_s[tid] = w; b_s[tid] = b >= 0 ? b : -1;
  }
  if (tid < CALL_INTS) call_s[tid] = 0;
  __syncthreads();
  if (tid < NPMAX) {                   // first occurrence of the expert in the call, the expert's largest weight
    int first = 0;
    float sc = -INFINITY;
    if (b_s[tid] >= 0) {
      first = 1;
      const int e = e_s[tid];
      for (int j = 0; j < NP; j++)
        if (b_s[j] >= 0 && e_s[j] == e) {
          if (j < tid) first = 0;
          sc = fmaxf(sc, w_s[j]);
        }
    }
    f_s[tid] = first; sc_s[tid] = sc;
  }
  __syncthreads();
  if (tid < NPMAX) {                   // place among the distinct experts: by largest weight, ties by lower id
    int pos = -1;
    if (b_s[tid] >= 0) {
      pos = 0;
      const float sc = sc_s[tid];
      const int e = e_s[tid];
      for (int j = 0; j < NP; j++)
        if (f_s[j] && (sc_s[j] > sc || (sc_s[j] == sc && e_s[j] < e))) pos++;
    }
    p_s[tid] = pos;
  }
  if (tid == 0) {
    int n = 0;
    for (int j = 0; j < NP; j++) n += f_s[j];
    n_first = n;
  }
  __syncthreads();
  if (a.prior_on) prior_places(a, NP, e_s, b_s, w_s, f_s, sc_s, p_s);    // B1 (a uniform branch: a kernel argument)
  // a runtime cap on the call's experts (K31_BAND_MAXE: flags bits 16-23, 0 = the full NE): the RTX's time per call
  // grows with its experts; picks past the cap count as past-16 (cut), as past NE always did
  const int cap = (a.flags >> 16) & 0xFF;
  const int ne = (cap > 0 && cap < NE) ? cap : NE;
  const int n_exp = min(n_first, ne);
  if (a.served != nullptr && tid < NP) a.served[tid] = (b_s[tid] >= 0 && p_s[tid] < ne) ? 1 : 0;
  if (tid == 0 && a.stats != nullptr) {
    int served = 0, past = 0, all = 0;
    for (int j = 0; j < NP; j++) {
      all += e_s[j] >= 0;
      if (b_s[j] >= 0) { if (p_s[j] < ne) served++; else past++; }
    }
    atomicAdd(a.stats + 0, served); atomicAdd(a.stats + 1, past); atomicAdd(a.stats + 2, all);
  }
  if (n_exp == 0) {                    // no band expert in this call: nothing goes out (the finish sees tag 0)
    if (tid == 0) a.st->tag = 0;
    return;
  }
  if (tid == 0) {
    call_s[0] = n_exp; call_s[1] = T; call_s[2] = a.flags & 0xFFFF; call_s[3] = a.sf_min;
    call_s[4] = __float_as_int(a.alpha); call_s[5] = __float_as_int(a.beta);
  }
  __syncthreads();
  if (tid < NP && b_s[tid] >= 0 && p_s[tid] < ne) {    // unused slots stay 0: the service checks all 16
    const int p = p_s[tid];
    if (f_s[tid]) call_s[8 + p] = b_s[tid];
    call_s[8 + NE + p * MT + tid / 16] = __float_as_int(w_s[tid]);
  }
  __syncthreads();
  const unsigned long long seq = a.st->seq + 1;
  uint8_t* rq = a.seg + OFF_REQ + (seq % NSLOT) * REQ_BYTES;
  if (tid < CALL_BYTES / 16)
    reinterpret_cast<uint4*>(rq + REQ_CALL)[tid] = reinterpret_cast<const uint4*>(call_s)[tid];
  for (int blk = tid; blk < T * (H / 32); blk += THREADS) {   // MXFP8 rows, 32 inputs per thread and turn
    const int g = blk / (H / 32), b = blk % (H / 32);
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
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    const int code = max((int)((__float_as_uint(amax) + 0x1FFFFFu) >> 23) - 8, 0);
    const float inv = __uint_as_float((uint32_t)(254 - code) << 23);
    uint8_t* row = rq + REQ_ROWS + (size_t)g * ROW_IN;
    const int k0 = b * 32, base = (k0 / 128) * 128 + ((k0 % 128) / 32) * 8;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int c = 0; c < 4; c++) {    // inputs i = 16h + 4c .. +3 -> bytes base + 32c + 4h .. +3
        uint32_t word = 0;
#pragma unroll
        for (int j = 0; j < 4; j++)
          word |= (uint32_t)__nv_cvt_float_to_fp8(v[16 * h + 4 * c + j] * inv, __NV_SATFINITE, __NV_E4M3) << (8 * j);
        *reinterpret_cast<uint32_t*>(row + base + 32 * c + 4 * h) = word;
      }
    row[H + b] = (uint8_t)code;
  }
  fence_sys();
  __syncthreads();
  if (tid == 0) {
    const unsigned long long tag = make_tag(seq, (uint32_t)a.layer, (uint32_t)T);
    st_release_sys_u64(a.seg + OFF_DOOR + DOOR_REQ + 8 * (seq % NSLOT), tag);
    a.st->seq = seq; a.st->tag = tag; a.st->t_pub = globaltimer();
  }
}

__global__ void __launch_bounds__(THREADS) band_finish_bf16(const FusedArgs a) {
  pdl_enter();
  __shared__ unsigned long long sh[4];
  const int tid = threadIdx.x;
  GbState* st = a.st;
  const unsigned long long tag = st->tag;
  uint8_t* res = a.seg + OFF_RES + (tag_seq(tag) % NSLOT) * RES_BYTES;
  if (tid == 0) {
    const unsigned long long t0 = globaltimer();
    unsigned long long status = ST_TIMEOUT;
    if (tag == 0) {
      status = ST_EMPTY;
    } else if (ld_acquire_sys_u64(a.seg + OFF_STATE + STATE_RUNNING) == 0 && ld_acquire_sys_u64(res + RES_TAG) != tag) {
      status = ST_DOWN;
    } else {
      for (;;) {
        if (ld_acquire_sys_u64(res + RES_TAG) == tag) {
          const unsigned long long rs = ld_acquire_sys_u64(res + RES_STATUS);
          status = (rs & RES_ERROR) ? ST_ERROR : (rs & RES_EMPTY) ? ST_EMPTY : ST_ADDED;
          break;
        }
        if (globaltimer() - t0 > a.timeout_ns) break;
      }
    }
    sh[0] = status;
  }
  __syncthreads();
  const unsigned long long status = sh[0];
  if (a.msum != nullptr || a.rstd != nullptr) {
    // v4: out = bf16(out + rows (if added) + msum) with one rounding, and each row's RMS statistic of the result.
    // Element groups of 8: i = tid + j * THREADS; a warp's 32 groups of a turn lie in one row (448 = 14 x 32)
    __shared__ float rsum[MT];
    constexpr int TURNS4 = (MT * (H / 8) + THREADS - 1) / THREADS;
    if (tid < MT) rsum[tid] = 0.f;
    __syncthreads();
    const bool added = status == ST_ADDED;
    const int T = min(max(a.T, 1), MT), n8 = T * (H / 8);
    constexpr int PASS = 4;              // turns per pass: 4 x (2 row + 1 out loads, 2 msum loads) fit the registers
    for (int base = 0; base < TURNS4; base += PASS) {
      uint4 r0[PASS], r1[PASS], o[PASS];
      float4 m0[PASS], m1[PASS];
#pragma unroll
      for (int j = 0; j < PASS; j++) {
        const int i = tid + (base + j) * THREADS;
        if (base + j < TURNS4 && i < n8) {
          if (added) {
            r0[j] = ld_volatile_u4(res + RES_ROWS + 32 * (size_t)i);
            r1[j] = ld_volatile_u4(res + RES_ROWS + 32 * (size_t)i + 16);
          }
          if (a.msum != nullptr) {
            m0[j] = reinterpret_cast<const float4*>(a.msum)[2 * i];
            m1[j] = reinterpret_cast<const float4*>(a.msum)[2 * i + 1];
          }
          o[j] = reinterpret_cast<const uint4*>(a.out)[i];
        }
      }
#pragma unroll
      for (int j = 0; j < PASS; j++) {
        const int i = tid + (base + j) * THREADS;
        float ss = 0.f;
        if (base + j < TURNS4 && i < n8) {
          float v[8];
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o[j]);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            v[2 * k] = __uint_as_float(ow[k] << 16);
            v[2 * k + 1] = __uint_as_float(ow[k] & 0xFFFF0000u);
          }
          if (added) {
            v[0] += __uint_as_float(r0[j].x); v[1] += __uint_as_float(r0[j].y);
            v[2] += __uint_as_float(r0[j].z); v[3] += __uint_as_float(r0[j].w);
            v[4] += __uint_as_float(r1[j].x); v[5] += __uint_as_float(r1[j].y);
            v[6] += __uint_as_float(r1[j].z); v[7] += __uint_as_float(r1[j].w);
          }
          if (a.msum != nullptr) {
            v[0] += m0[j].x; v[1] += m0[j].y; v[2] += m0[j].z; v[3] += m0[j].w;
            v[4] += m1[j].x; v[5] += m1[j].y; v[6] += m1[j].z; v[7] += m1[j].w;
          }
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const __nv_bfloat162 p = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
            ow[k] = *reinterpret_cast<const uint32_t*>(&p);
            const float lo = __uint_as_float(ow[k] << 16), hi = __uint_as_float(ow[k] & 0xFFFF0000u);
            ss += lo * lo + hi * hi;
          }
          reinterpret_cast<uint4*>(a.out)[i] = o[j];
        }
        if (a.rstd != nullptr && base + j < TURNS4) {
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
          const int i0 = (tid & ~31) + (base + j) * THREADS;
          if ((tid & 31) == 0 && i0 < n8) atomicAdd(&rsum[i0 / (H / 8)], ss);
        }
      }
    }
    __syncthreads();
    if (a.rstd != nullptr && tid < T) a.rstd[tid] = rsqrtf(rsum[tid] / (float)H + a.eps);
  } else if (status == ST_ADDED) {     // 8 values per thread and turn: two 16 B loads of float32, one of bfloat16
    // All of a thread's turns (<= MT x 448 / THREADS = 7) load first, then add and store: the result rows sit in host
    // memory (read over C2C), so one round trip instead of one per turn (8.8 us on the chain per call, 11-18 profile)
    constexpr int TURNS = (MT * (H / 8) + THREADS - 1) / THREADS;
    const int n8 = (int)tag_rows(tag) * (H / 8);
    uint4 r0[TURNS], r1[TURNS], o[TURNS];
#pragma unroll
    for (int j = 0; j < TURNS; j++) {
      const int i = tid + j * THREADS;
      if (i < n8) {
        r0[j] = ld_volatile_u4(res + RES_ROWS + 32 * (size_t)i);
        r1[j] = ld_volatile_u4(res + RES_ROWS + 32 * (size_t)i + 16);
        o[j] = reinterpret_cast<const uint4*>(a.out)[i];
      }
    }
#pragma unroll
    for (int j = 0; j < TURNS; j++) {
      const int i = tid + j * THREADS;
      if (i < n8) {
        const float rv[8] = {__uint_as_float(r0[j].x), __uint_as_float(r0[j].y), __uint_as_float(r0[j].z),
                             __uint_as_float(r0[j].w), __uint_as_float(r1[j].x), __uint_as_float(r1[j].y),
                             __uint_as_float(r1[j].z), __uint_as_float(r1[j].w)};
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o[j]);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          const float lo = __uint_as_float(ow[k] << 16) + rv[2 * k];
          const float hi = __uint_as_float(ow[k] & 0xFFFF0000u) + rv[2 * k + 1];
          const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
          ow[k] = *reinterpret_cast<const uint32_t*>(&p);
        }
        reinterpret_cast<uint4*>(a.out)[i] = o[j];
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    st->status = status;
    if (status == ST_ADDED) st->n_added += 1;
    else if (status == ST_ERROR) st->n_error += 1;
    else if (status == ST_EMPTY) st->n_empty += 1;
    else if (status == ST_DOWN) st->n_down += 1;
    else {
      st->n_timeout += 1;
      if (st->late_tag != 0) st->n_lost += 1;
      st->late_tag = tag;
      st->late_idx = st->idx;
    }
    st->idx += 1;
  }
}

int g_pdl = 1;

// One CTA of THREADS, with programmatic stream serialization unless rtxband_fused_pdl(0) turned it off.
int launch_pdl(void (*kernel)(const FusedArgs), const FusedArgs& args, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = g_pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, args);
  return e != cudaSuccess ? (int)e : (int)cudaGetLastError();
}

}  // namespace

extern "C" {

struct RtxBandFused {
  void* seg; void* st;
  const void* ids; const void* w; const void* row_map; const void* band_map;
  const void* x; void* out; void* stats;
  int T, layer, flags, sf_min;
  float alpha, beta;
  unsigned long long timeout_ns;
  void* served;
  const void* msum; void* rstd; float eps;
  float prior[MT]; int prior_on;
};
static_assert(sizeof(RtxBandFused) == sizeof(FusedArgs), "argument block");
static_assert(sizeof(FusedArgs) == 168, "argument block: k31_engine/band.py _Fused has the same 168 bytes");

void rtxband_fused_pdl(int on) { g_pdl = on; }

// B1 (K31_BAND_PRIOR): this build reads prior / prior_on (older builds ignore the fields past eps)
int rtxband_fused_prior(void) { return 1; }

int rtxband_fused_hint(const RtxBandFused* a, void* stream) {
  return launch_pdl(band_plan_hint, *reinterpret_cast<const FusedArgs*>(a), (cudaStream_t)stream);
}

int rtxband_fused_qsend(const RtxBandFused* a, void* stream) {
  return launch_pdl(band_qsend, *reinterpret_cast<const FusedArgs*>(a), (cudaStream_t)stream);
}

int rtxband_fused_finish(const RtxBandFused* a, void* stream) {
  return launch_pdl(band_finish_bf16, *reinterpret_cast<const FusedArgs*>(a), (cudaStream_t)stream);
}

}  // extern "C"
