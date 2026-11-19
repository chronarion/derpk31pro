// RTX band kernel of derpk31pro: Kimi K3.1 routed experts computed from 2-bit records on an RTX PRO 6000 Blackwell
// (sm_120a), for decode-sized batches (<= 8 rows, <= 16 distinct experts per call).
//
//   y[t] = sum over the call's experts e of w[e][t] * W2_e . Q8( SiTU(W13_e . x[t]) )
//
// x arrives as MXFP8 (E4M3 codes + one UE8M0 scale per 32 inputs), the weights are the hi2 records' depth-2 code map
// (e2m1 nibble = (pair << 2) | 3, scale byte as stored = native - 1), Q8 is the MXFP8 quantization of the
// intermediate (blocks of 32, DeepGEMM's / the runner's "ceil" scale rule).
//
// Design (numbers in ../../REPORT.md):
//   * Both GEMMs run on tensor cores with the block-scaled MMA
//       mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32  (A = e2m1 weights, B = e4m3 activations)
//     with the weights on the M = 16 side and the rows (tokens) on the N = 8 side: one MMA is 16 weight rows x 8 tokens
//     x 32 inputs, so up to 8 rows cost the same weight traffic as one.
//   * The MMA takes its operands from registers. A record is repacked once, when it enters the band, into the order
//     the fragments are consumed in: per 16-row tile and 128 inputs ("step") a uint4 per lane with the 2-bit codes of
//     its four MMAs (512 B), and the tile's 16 x 4 scale codes as 4-bit offsets from the record's lowest code (32 B;
//     the 1,136 K3.1 records looked at span at most 14 codes each). A lane expands 32 bits of codes into the four
//     8-bit-container registers of one MMA with two instructions per register and 16 bits of scale codes into the
//     four scale bytes of a step; the MMA's byte-id immediate picks the byte. Nothing is expanded in memory: the
//     kernel reads 8.77 MB per expert (the hi2 record's 9.29 MB minus half of its scale bytes) and nothing else.
//   * A warp owns one row tile and streams its steps with BAND_PF of them in flight (a DRAM miss takes ~430 ns here,
//     an MMA ~14 ns: without the lookahead a tile is a 12 us serial chain).
//   * In MegaMoE's kernel layout a 16-row tile of w13 is 8 gate rows over the 8 up rows of the same neurons, so a
//     lane's accumulators already hold (gate, up) pairs: SiTU, the routing weight, the block amax and the FP8 cast
//     run in registers, and the 32-neuron block of four warps writes one K-block of linear 2's input.
//   * Fixed grids, the work list is device data (CUDA-graph friendly). Either two launches (linear 1, linear 2) or
//     one fused launch whose linear-2 CTAs wait, per expert, for that expert's linear-1 CTAs. The CTAs of the last
//     expert sum the experts' partial outputs in expert order (deterministic; no floating-point atomics).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -shared -Xcompiler -fPIC -o librtxband.so rtxband.cu
#include "band_core.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {
using namespace band;

__global__ void __launch_bounds__(THREADS) band_l1(const BandCall* __restrict__ call, const Bufs b,
                                                   const uint8_t* __restrict__ xf) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int e = blockIdx.x / CTAS1, part = blockIdx.x % CTAS1;
  if (e >= call->n_exp) return;
  l1_body<false>(call, b, xf, e, part, smem, false);
}

__global__ void __launch_bounds__(THREADS) band_l2(const BandCall* __restrict__ call, const Bufs b) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int e = blockIdx.x / CTAS2, part = blockIdx.x % CTAS2;
  const int n_exp = call->n_exp;
  if (e >= n_exp) {
    if (n_exp == 0 && e == 0) zero_y(b.y, part);
    return;
  }
  l2_body<false>(call, b, e, part, n_exp, MAXT, smem, false);
}

// Both linears in one launch: CTAs [0, 16 CTAS1) are linear 1, the rest linear 2. CTAs are dispatched in index
// order, so every linear-1 CTA is running or done before a linear-2 CTA can occupy a slot and wait for it.
__global__ void __launch_bounds__(THREADS) band_fused(const BandCall* __restrict__ call, const Bufs b,
                                                      const uint8_t* __restrict__ xf) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int n_exp = call->n_exp;
  if (blockIdx.x < CTAS1 * MAXE) {
    const int e = blockIdx.x / CTAS1, part = blockIdx.x % CTAS1;
    if (e >= n_exp) return;
    l1_body<false>(call, b, xf, e, part, smem, true);
  } else {
    const int i = blockIdx.x - CTAS1 * MAXE;
    const int e = i / CTAS2, part = i % CTAS2;
    if (e >= n_exp) {
      if (n_exp == 0 && e == 0) zero_y(b.y, part);
      return;
    }
    l2_body<false>(call, b, e, part, n_exp, MAXT, smem, true);
  }
}

// Row-major MXFP8 rows -> the call's input in fragment order (what the sender should write directly).
// xq [T][3584] E4M3 codes, xs [T][112] UE8M0 codes; rows >= T become zero. One CTA per step record.
__global__ void band_pack_x(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, int T,
                            uint8_t* __restrict__ xf) {
  const int kq = blockIdx.x;
  uint8_t* rec = xf + kq * BREC;
  for (int f = threadIdx.x; f < 32 * 4; f += blockDim.x) {
    const int lane = f >> 2, kap = f & 3, gg = lane >> 2, tt = lane & 3;
    uint32_t v0 = 0, v1 = 0;
    if (gg < T) {
      const uint8_t* p = xq + gg * H + (kq * 4 + kap) * 32 + tt * 4;
      v0 = *reinterpret_cast<const uint32_t*>(p);
      v1 = *reinterpret_cast<const uint32_t*>(p + 16);
    }
    uint32_t* d = reinterpret_cast<uint32_t*>(rec + lane * 32 + kap * 8);
    d[0] = v0; d[1] = v1;
  }
  if (threadIdx.x < MAXT) {
    const int gg = threadIdx.x;
    *reinterpret_cast<uint32_t*>(rec + 1024 + gg * 4) =
        gg < T ? *reinterpret_cast<const uint32_t*>(xs + gg * (H / 32) + kq * 4) : 0u;
  }
}

// Bulk units (prompt chunks): unit u's rows (rows[u][0..7], -1 = none) of the chunk's row-major MXFP8 rows -> the
// unit's input xf[u] in fragment order (band_pack_x for gathered rows). Grid (KQ1, units), 128 threads.
__global__ void band_pack_units(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                const int* __restrict__ rows, uint8_t* __restrict__ xf) {
  const int kq = blockIdx.x, u = blockIdx.y;
  uint8_t* rec = xf + ((long long)u * KQ1 + kq) * BREC;
  const int* ru = rows + u * MAXT;
  for (int f = threadIdx.x; f < 32 * 4; f += blockDim.x) {
    const int lane = f >> 2, kap = f & 3, gg = lane >> 2, tt = lane & 3;
    const int r = ru[gg];
    uint32_t v0 = 0, v1 = 0;
    if (r >= 0) {
      const uint8_t* p = xq + (long long)r * H + (kq * 4 + kap) * 32 + tt * 4;
      v0 = *reinterpret_cast<const uint32_t*>(p);
      v1 = *reinterpret_cast<const uint32_t*>(p + 16);
    }
    uint32_t* d = reinterpret_cast<uint32_t*>(rec + lane * 32 + kap * 8);
    d[0] = v0; d[1] = v1;
  }
  if (threadIdx.x < MAXT) {
    const int r = ru[threadIdx.x];
    *reinterpret_cast<uint32_t*>(rec + 1024 + threadIdx.x * 4) =
        r >= 0 ? *reinterpret_cast<const uint32_t*>(xs + (long long)r * (H / 32) + kq * 4) : 0u;
  }
}

// Bulk units: out[t] = sum over k of yu[idx[t][k]] (float32), idx [T][kmax] in ascending unit order, -1 ends a row's
// list: every row's picks are added in the same order on every run (no atomics). One CTA per row.
__global__ void band_gather_units(const float* __restrict__ yu, const int* __restrict__ idx, int kmax,
                                  float* __restrict__ out) {
  const int t = blockIdx.x;
  const int* it = idx + (long long)t * kmax;
  for (int c = threadIdx.x * 4; c < H; c += blockDim.x * 4) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < kmax; k++) {
      const int j = it[k];
      if (j < 0) break;
      const float4 v = *reinterpret_cast<const float4*>(yu + (long long)j * H + c);
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    *reinterpret_cast<float4*>(out + (long long)t * H + c) = acc;
  }
}

// L2 prefetch of the records of a work list's experts (cp.async.bulk.prefetch.L2 in 64 KB pieces): a hint, issued
// while the band idles, for experts a coming call is likely to use. The L2 cache holds 128 MB = 14 records; a call
// whose records are there is compute-bound (about 3x faster at 8 experts). Results never depend on it.
#ifndef BAND_PREFETCH_READ
#define BAND_PREFETCH_READ 0   // 1: read the bytes instead of bulk-prefetching them (a probe of the hint's coverage)
#endif
__global__ void band_prefetch(const BandCall* __restrict__ call, const uint8_t* __restrict__ store, int* sink) {
  const int n_exp = min(call->n_exp, MAXE);
  const int total = n_exp * PIECES;
  uint32_t acc = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const int e = i / PIECES, j = i % PIECES;
#if BAND_PREFETCH_READ
    const uint8_t* a = store + (long long)call->slot[e] * REC_BYTES + (long long)j * PIECE;
    const uint32_t bytes = (uint32_t)min(PIECE, REC_BYTES - (long long)j * PIECE);
    for (uint32_t o = 0; o + 16 <= bytes; o += 16) acc += reinterpret_cast<const uint4*>(a + o)->x;
#else
    prefetch_piece(store, call->slot[e], j);
#endif
  }
  if (acc == 0x9E3779B9u) sink[0] = 1;
}

// hi2 record -> band record, on the device (what runs when a record enters the band). Two launches:
//   band_scale_range: range[0] = max(255 - code), range[1] = max(code) over the record's scale bytes
//                     (range must be zero before; the pack kernel leaves it zero again)
//   band_repack:      permutes the 2-bit codes into fragment order and writes the scale codes as 4-bit offsets from
//                     the lowest one; meta[0] = that base, meta[1] = 1 if the record fits (no zero code, span <= 16)
#if BAND_K2
// BAND_K2 (hi2-k2 records): range[0] = 1 if the record does not fit (a status word set, or a row with more than two
// exception entries), range[1] = 1 if it has exception entries; band_repack copies the rows' bases and 2-bit codes
// and the exception entries; meta[0] = range[1] (for sbase[slot]), meta[1] = 1 if the record fits.
__global__ void band_scale_range(const uint8_t* __restrict__ hi2, int* __restrict__ range) {
  uint32_t unfit = 0, hasx = 0;
  scan_share<false>(hi2, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x, unfit, hasx);
  if (unfit) atomicMax(&range[0], 1);
  if (hasx) atomicMax(&range[1], 1);
}

__global__ void band_repack(const uint8_t* __restrict__ hi2, uint8_t* __restrict__ rec, int* __restrict__ range,
                            uint8_t* __restrict__ meta) {
  const int unfit = range[0], hasx = range[1];
  repack_share<false>(hi2, rec, (blockIdx.x * blockDim.x + threadIdx.x) >> 5, (gridDim.x * blockDim.x) >> 5, 0u);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < XTILES * XENT; i += gridDim.x * blockDim.x)
    xent_part<false>(hi2, rec, i);
  if (blockIdx.x == 0) {
    __syncthreads();
    if (threadIdx.x == 0) {
      meta[0] = (uint8_t)hasx;
      meta[1] = (uint8_t)(unfit == 0);
      range[0] = 0; range[1] = 0;
    }
  }
}
#else
__global__ void band_scale_range(const uint8_t* __restrict__ hi2, int* __restrict__ range) {
  uint32_t lo = 0xFFFFFFFFu, hi = 0;                       // bytewise min / max over this thread's words
  scale_range_share<false>(hi2, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x, lo, hi);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
    hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
  }
  if ((threadIdx.x & 31) == 0 && lo <= 255u) {
    atomicMax(&range[0], 255 - (int)lo);
    atomicMax(&range[1], (int)hi);
  }
}

__global__ void band_repack(const uint8_t* __restrict__ hi2, uint8_t* __restrict__ rec, int* __restrict__ range,
                            uint8_t* __restrict__ meta) {
  const int lo = 255 - range[0], hi = range[1];
  const uint32_t base4 = (uint32_t)lo * 0x01010101u;
  repack_share<false>(hi2, rec, (blockIdx.x * blockDim.x + threadIdx.x) >> 5, (gridDim.x * blockDim.x) >> 5, base4);
  if (blockIdx.x == 0) {
    __syncthreads();
    if (threadIdx.x == 0) {
      meta[0] = (uint8_t)lo;
      meta[1] = (uint8_t)(lo > 0 && hi - lo <= 15);
      range[0] = 0; range[1] = 0;
    }
  }
}
#endif

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

extern "C" {

// A band's fixed device buffers (see Bufs). The caller keeps this struct alive and passes it to every call.
struct RtxBand {
  const void* store; const void* sbase; void* aq; void* ypart; void* y; void* sync;
};

// sizes a caller needs: [record bytes, call bytes, xf bytes, aq bytes, ypart bytes, y bytes, sync bytes, CTAs of
// linear 1, CTAs of linear 2, threads per CTA, lookahead, shared memory per CTA]
void rtxband_sizes(long long* out) {
  out[0] = REC_BYTES; out[1] = sizeof(BandCall); out[2] = XF_BYTES; out[3] = (long long)MAXE * AQ_BYTES;
  out[4] = (long long)(MAXE - 1) * MAXT * H * 4; out[5] = (long long)MAXT * H * 4; out[6] = SYNC_WORDS * 4;
  out[7] = CTAS1 * MAXE; out[8] = CTAS2 * MAXE; out[9] = THREADS; out[10] = BAND_PF; out[11] = SMEM_BYTES;
}

// the record formats this library was built for: [hi2 record bytes, 1 = hi2-k2 records and 2-bit band scale codes]
void rtxband_format(long long* out) { out[0] = HI2_BYTES; out[1] = BAND_K2; }

static Bufs bufs(const RtxBand* r) {
  return Bufs{(const uint8_t*)r->store, (const uint8_t*)r->sbase, (uint8_t*)r->aq, (float*)r->ypart, (float*)r->y,
              (int*)r->sync};
}

int rtxband_pack_x(const void* xq, const void* xs, int n_rows, void* xf, void* stream) {
  band_pack_x<<<KQ1, 128, 0, (cudaStream_t)stream>>>((const uint8_t*)xq, (const uint8_t*)xs, n_rows, (uint8_t*)xf);
  return (int)cudaGetLastError();
}

int rtxband_l1(const RtxBand* band, const void* call, const void* xf, void* stream) {
  band_l1<<<CTAS1 * MAXE, THREADS, SMEM_BYTES, (cudaStream_t)stream>>>((const BandCall*)call, bufs(band),
                                                                       (const uint8_t*)xf);
  return (int)cudaGetLastError();
}

int rtxband_l2(const RtxBand* band, const void* call, void* stream) {
  band_l2<<<CTAS2 * MAXE, THREADS, SMEM_BYTES, (cudaStream_t)stream>>>((const BandCall*)call, bufs(band));
  return (int)cudaGetLastError();
}

// Hint: pull the records of the work list's slots (n_exp, slot[]) into the L2 cache. Returns at once; the
// transfer takes about 5.5 us per record of DRAM time.
int rtxband_prefetch(const RtxBand* band, const void* call, void* stream) {
#ifndef BAND_PREFETCH_CTAS
#define BAND_PREFETCH_CTAS 8
#endif
#ifndef BAND_PREFETCH_THREADS
#define BAND_PREFETCH_THREADS 1
#endif
  // few issuers on purpose: 67 x 32 threads issuing at once lost about half of the pieces (bench/probe_hint.py)
  band_prefetch<<<BAND_PREFETCH_CTAS, BAND_PREFETCH_THREADS, 0, (cudaStream_t)stream>>>((const BandCall*)call, (const uint8_t*)band->store,
                                                                 (int*)band->sync);
  return (int)cudaGetLastError();
}

// hi2 record (device memory, 9,289,728 B) -> band record `rec` (8,773,632 B, e.g. its slot in the store).
// range: two zero int32 of scratch per repack in flight; meta[0] = the record's scale base (for sbase[slot]),
// meta[1] = 1 if the record fits the band's 4-bit scale code, else 0 (the output is then unusable). Asynchronous.
int rtxband_repack(const void* hi2, void* rec, void* range, void* meta, void* stream) {
  band_scale_range<<<188, 128, 0, (cudaStream_t)stream>>>((const uint8_t*)hi2, (int*)range);
  band_repack<<<376, 256, 0, (cudaStream_t)stream>>>((const uint8_t*)hi2, (uint8_t*)rec, (int*)range, (uint8_t*)meta);
  return (int)cudaGetLastError();
}

// One call on `stream` (0 = the default stream), asynchronous. fused = 0: two launches; 1: one launch.
int rtxband_run(const RtxBand* band, const void* call, const void* xf, int fused, void* stream) {
  if (fused) {
    band_fused<<<(CTAS1 + CTAS2) * MAXE, THREADS, SMEM_BYTES, (cudaStream_t)stream>>>((const BandCall*)call,
                                                                                      bufs(band), (const uint8_t*)xf);
    return (int)cudaGetLastError();
  }
  int rc = rtxband_l1(band, call, xf, stream);
  if (rc) return rc;
  return rtxband_l2(band, call, stream);
}

// Bulk units (prompt chunks). rtxband_pack_units: n units' inputs (xf: n x XF_BYTES) from the chunk's MXFP8 rows
// (xq [T][3584], xs [T][112]) and each unit's 8 row indices (rows [n][8] int32, -1 = none).
int rtxband_pack_units(const void* xq, const void* xs, const void* rows, int n_units, void* xf, void* stream) {
  if (n_units <= 0) return 0;
  band_pack_units<<<dim3(KQ1, n_units), 128, 0, (cudaStream_t)stream>>>((const uint8_t*)xq, (const uint8_t*)xs,
                                                                        (const int*)rows, (uint8_t*)xf);
  return (int)cudaGetLastError();
}

// n_calls fused launches: call i (calls + i, FLAG_PER_EXPERT set, up to 16 units) reads its units' inputs at
// xf + 16 i XF_BYTES and writes unit e's 8 weighted rows at y + (16 i + e) MAXT H (float32).
int rtxband_run_bulk(const RtxBand* band, const void* calls, int n_calls, const void* xf, void* y, void* stream) {
  Bufs b = bufs(band);
  for (int i = 0; i < n_calls; i++) {
    b.y = (float*)y + (long long)i * MAXE * MAXT * H;
    band_fused<<<(CTAS1 + CTAS2) * MAXE, THREADS, SMEM_BYTES, (cudaStream_t)stream>>>(
        (const BandCall*)calls + i, b, (const uint8_t*)xf + (long long)i * MAXE * XF_BYTES);
  }
  return (int)cudaGetLastError();
}

// out [T][3584] float32 = per row, the sum of its units' rows yu[idx[t][k]] in list order (idx [T][kmax] int32).
int rtxband_gather_units(const void* yu, const void* idx, int kmax, int T, void* out, void* stream) {
  if (T <= 0) return 0;
  band_gather_units<<<T, 224, 0, (cudaStream_t)stream>>>((const float*)yu, (const int*)idx, kmax, (float*)out);
  return (int)cudaGetLastError();
}

// Timing of the entry path and of the hint. what = 0: device repack of n hi2 records (hi2 = n x 9,289,728 B in device
// memory) into n band records at `rec`, queued back to back; out[0] = seconds per record.
// what = 1: L2 prefetch hints for the work lists `calls` (n of them) on `band`, each launched and synchronized;
// out[0] = median seconds per hint (the time until the pieces are issued and accepted).
int rtxband_bench_aux(int what, const RtxBand* band, const void* hi2_or_calls, int n, void* rec, void* range,
                      void* meta, int iters, double* out) {
  cudaStream_t st;
  cudaStreamCreate(&st);
  if (what == 0) {
    const uint8_t* src = (const uint8_t*)hi2_or_calls;
    for (int i = 0; i < 2; i++) rtxband_repack(src, rec, range, meta, st);
    cudaStreamSynchronize(st);
    double best = 1e30, sum = 0;
    for (int r = 0; r < iters; r++) {
      const double t0 = now();
      for (int i = 0; i < n; i++)
        rtxband_repack(src + (long long)i * HI2_BYTES, (uint8_t*)rec + (long long)i * REC_BYTES, range, meta, st);
      cudaStreamSynchronize(st);
      const double t = (now() - t0) / n;
      best = std::min(best, t); sum += t;
    }
    out[0] = sum / iters; out[1] = best;
  } else {
    const BandCall* cs = (const BandCall*)hi2_or_calls;
    std::vector<double> ts(iters);
    for (int i = 0; i < iters; i++) {
      const double t0 = now();
      rtxband_prefetch(band, cs + (i % n), st);
      cudaStreamSynchronize(st);
      ts[i] = now() - t0;
    }
    std::sort(ts.begin(), ts.end());
    out[0] = ts[iters / 2]; out[1] = ts[iters * 9 / 10];
  }
  cudaError_t e = cudaStreamSynchronize(st);
  cudaStreamDestroy(st);
  return (int)e;
}

// Timing. `calls` = n_calls BandCall structs in device memory, used round-robin (give enough different experts that
// a record is not re-read while it is still in the 128 MB L2 cache). mode: 0 = each call launched and synchronized
// (median and p90 of `iters`), 1 = `iters` calls queued back to back, one synchronize (mean per call),
// 2 = a CUDA graph of all n_calls calls replayed `iters` times (mean per call), 3 = a one-call graph per work list,
// launched and synchronized (median, p90), 4 = as 0 after an L2 prefetch of the call's experts and a pause of
// pause_us (the hinted case), 5 = as 0 with an idle pause of pause_us before each call (a duty cycle). which: 3 = both linears in two launches, 1 = linear 1 only, 2 = linear 2
// only, 4 = the fused launch. out[0] = seconds per call, out[1] = p90 (modes 0, 3, 4) or the best repetition.
int rtxband_bench(const RtxBand* band, const void* calls, int n_calls, const void* xf, int mode, int which, int iters,
                  int pause_us, double* out) {
  cudaStream_t st;
  cudaStreamCreate(&st);
  const BandCall* cs = (const BandCall*)calls;
  auto one = [&](int i, cudaStream_t s) {
    const void* c = cs + (i % n_calls);
    if (which == 4) { rtxband_run(band, c, xf, 1, s); return; }
    if (which & 1) rtxband_l1(band, c, xf, s);
    if (which & 2) rtxband_l2(band, c, s);
  };
  int rc = 0;
  for (int i = 0; i < std::min(n_calls, 8); i++) one(i, st);
  cudaStreamSynchronize(st);
  if (mode == 0 || mode == 4 || mode == 5) {
    std::vector<double> ts(iters);
    for (int i = 0; i < iters; i++) {
      if (mode == 4) {
        rtxband_prefetch(band, cs + (i % n_calls), st);
        cudaStreamSynchronize(st);
      }
      if (mode != 0) {
        const double until = now() + 1e-6 * pause_us;
        while (now() < until) {}
      }
      const double t0 = now();
      one(i, st);
      cudaStreamSynchronize(st);
      ts[i] = now() - t0;
    }
    std::sort(ts.begin(), ts.end());
    out[0] = ts[iters / 2]; out[1] = ts[iters * 9 / 10];
  } else if (mode == 1) {
    double best = 1e30, sum = 0;
    const int reps = 5;
    for (int r = 0; r < reps; r++) {
      const double t0 = now();
      for (int i = 0; i < iters; i++) one(r * iters + i, st);
      cudaStreamSynchronize(st);
      const double t = (now() - t0) / iters;
      best = std::min(best, t); sum += t;
    }
    out[0] = sum / reps; out[1] = best;
  } else if (mode == 2) {
    cudaGraph_t graph; cudaGraphExec_t exec;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < n_calls; i++) one(i, st);
    cudaStreamEndCapture(st, &graph);
    rc = (int)cudaGraphInstantiate(&exec, graph, 0);
    if (!rc) {
      cudaGraphLaunch(exec, st); cudaStreamSynchronize(st);
      double best = 1e30, sum = 0;
      for (int r = 0; r < iters; r++) {
        const double t0 = now();
        cudaGraphLaunch(exec, st);
        cudaStreamSynchronize(st);
        const double t = (now() - t0) / n_calls;
        best = std::min(best, t); sum += t;
      }
      out[0] = sum / iters; out[1] = best;
      cudaGraphExecDestroy(exec);
    }
    cudaGraphDestroy(graph);
  } else if (mode == 3) {
    std::vector<cudaGraphExec_t> execs(n_calls);
    std::vector<cudaGraph_t> graphs(n_calls);
    for (int i = 0; i < n_calls && !rc; i++) {
      cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
      one(i, st);
      cudaStreamEndCapture(st, &graphs[i]);
      rc = (int)cudaGraphInstantiate(&execs[i], graphs[i], 0);
    }
    if (!rc) {
      for (int i = 0; i < n_calls; i++) cudaGraphLaunch(execs[i], st);
      cudaStreamSynchronize(st);
      std::vector<double> ts(iters);
      for (int i = 0; i < iters; i++) {
        const double t0 = now();
        cudaGraphLaunch(execs[i % n_calls], st);
        cudaStreamSynchronize(st);
        ts[i] = now() - t0;
      }
      std::sort(ts.begin(), ts.end());
      out[0] = ts[iters / 2]; out[1] = ts[iters * 9 / 10];
    }
    for (int i = 0; i < n_calls; i++) { cudaGraphExecDestroy(execs[i]); cudaGraphDestroy(graphs[i]); }
  }
  cudaError_t e = cudaStreamSynchronize(st);
  cudaStreamDestroy(st);
  return rc ? rc : (int)e;
}

}  // extern "C"
