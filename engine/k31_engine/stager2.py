"""The stager's kernels for hi2-k2 records and for 6-bit scale rows (stager.py hands over to this module when the
record store is hi2-k2 or the layer's rows carry 6-bit scale codes; hi2-k1 records into plain rows stay in stager.py).

Two things vary, independently:
  the record     hi2-k1: the scale bytes as they lie in the kernel tensors; hi2-k2: per scale row a base byte and per
                 scale a 2-bit offset, in self-contained units of 512 rows with exact exception entries (recfmt.py)
  the row        int32 scale words as vLLM registers them, or 6-bit codes in the MegaMoE fork's layout 2 (sf6.py:
                 [N / 128 tiles][K groups][32 chunks of 12 bytes], code = scale byte - E0)
The code bytes are the same in every combination (the 256-entry table of stager.py). Every kernel here writes the bytes
the reference codecs give (tests/test_stager2.py: byte-exact on the GPU against recfmt.py and sf6.py):
  stage     records in pinned Grace memory -> staging rows (one launch per MoE call; a ring variant writes two rows)
  expand    records in an HBM scratch -> rows (the prompt path's copy-engine lane, celane.py)
  merge     a hi2 and a lo2 record -> the exact native row (promotions; the scales come from lo2)
  extract   a native row -> its hi2 record (an evicted expert's record goes back to the Grace bank); for hi2-k2 this
            is the canonical encoder of recfmt.py on the device
"""
import os

import torch

from . import recfmt
from . import sf6 as _sf6

CHUNK = 16384
STAGES = 6
SLOTS = 16
KQ_MAX = 32
ARCH = os.environ.get("K31_CUDA_ARCH", "103a")          # tests on another GPU: K31_CUDA_ARCH=120a
KQ = (28, 24)               # K groups (128 inputs) of w13 and w2: Kimi K3.1 Pro's routed experts (latent 3584, 3072)
_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <vector>

constexpr int CHUNK = %(CHUNK)d, STAGES = %(STAGES)d, SLOTS = %(SLOTS)d, THREADS = 256, UNROLL = 4;
constexpr int UROWS = %(UROWS)d, EXC = %(EXC)d, XT = %(XT)d, UHEAD = UROWS + 4 * EXC, KQ_MAX = %(KQ_MAX)d;   // XT: entries per 128-row tile
constexpr uint32_t E0 = %(E0)d, E0W = E0 * 0x01010101u, EMPTY = 0xFFFFFFFFu;
constexpr int KQ13 = %(KQ13)d, KQ2 = %(KQ2)d;      // K groups of w13 and w2 (the encoder keeps a row's in registers)
constexpr int TILE = %(TILE)d;                     // rows of a tile

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%%0], 1;" :: "r"(sa(b))); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], %%1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%%0], %%1;\n@!p bra W;\n}\n" :: "r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%%0], [%%1], %%2, [%%3];"
               :: "r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory"); }

struct Dst { uint8_t* base[4]; long long stride[4]; };
// k2: the records are hi2-k2 (else hi2-k1); sf6: the rows' scales are 6-bit codes (else int32 words). Per scale
// tensor t (0 = w13, 1 = w2): n scale rows, kq K groups, c code chunks of 16 KiB, ns scale units of sz bytes per record
// (hi2-k1: chunks of 16 KiB; hi2-k2: units of 512 rows).
struct Geo { int k2, sf6, n[2], kq[2], c[2], ns[2], sz[2]; long long c13b, codesb, s13b, s2b; };   // + byte sizes (host)

// ------------------------------------------------------------------------------------------------ scale codecs
// four 2-bit codes of a byte -> the low two bits of a word's four bytes
__device__ __forceinline__ uint32_t spread2(uint32_t b) { return (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u; }
// hi2-k2: four code bytes and the four base bytes of scale rows P .. P + 3 -> their 16 scale bytes (4 words)
__device__ __forceinline__ uint4 dec4(uint32_t cw, uint32_t bw) {
  return make_uint4(spread2(cw & 0xFF) + (bw & 0xFF) * 0x01010101u,
                    spread2((cw >> 8) & 0xFF) + ((bw >> 8) & 0xFF) * 0x01010101u,
                    spread2((cw >> 16) & 0xFF) + ((bw >> 16) & 0xFF) * 0x01010101u,
                    spread2(cw >> 24) + (bw >> 24) * 0x01010101u);
}
// 6-bit rows: a word of four codes (each < 64, one per byte) <-> 24 bits
__device__ __forceinline__ uint32_t pack4(uint32_t w) {
  return (w & 0x3Fu) | ((w >> 2) & 0xFC0u) | ((w >> 4) & 0x3F000u) | ((w >> 6) & 0xFC0000u);
}
__device__ __forceinline__ uint32_t unpack4(uint32_t p) {
  return (p & 0x3Fu) | ((p & 0xFC0u) << 2) | ((p & 0x3F000u) << 4) | ((p & 0xFC0000u) << 6);
}
// the 12 bytes of scale rows P .. P + 3 (P a multiple of 4) of K group kq in a 6-bit row
__device__ __forceinline__ uint8_t* chunk6(uint8_t* s, int KQ, int kq, int P) {
  return s + ((long long)(P >> 7) * KQ + kq) * 384 + ((P & 127) >> 2) * 12;
}
// 16 scale bytes (rows P .. P + 3 of K group kq; w = kq * N + P: their first word in the int32 layout) into a row
__device__ __forceinline__ void put16(uint8_t* s, int sf6, int KQ, int kq, int P, long long w, uint4 v) {
  if (!sf6) { *reinterpret_cast<uint4*>(s + 4 * w) = v; return; }
  const uint32_t p0 = pack4(v.x - E0W), p1 = pack4(v.y - E0W), p2 = pack4(v.z - E0W), p3 = pack4(v.w - E0W);
  uint32_t* d = reinterpret_cast<uint32_t*>(chunk6(s, KQ, kq, P));
  d[0] = p0 | (p1 << 24); d[1] = (p1 >> 8) | (p2 << 16); d[2] = (p2 >> 16) | (p3 << 8);
}
__device__ __forceinline__ uint4 get16(const uint8_t* s, int sf6, int KQ, int kq, int P, long long w) {
  if (!sf6) return *reinterpret_cast<const uint4*>(s + 4 * w);
  const uint32_t* d = reinterpret_cast<const uint32_t*>(chunk6(const_cast<uint8_t*>(s), KQ, kq, P));
  const uint32_t a = d[0], b = d[1], c = d[2];
  return make_uint4(unpack4(a & 0xFFFFFFu) + E0W, unpack4(((a >> 24) | (b << 8)) & 0xFFFFFFu) + E0W,
                    unpack4(((b >> 16) | (c << 16)) & 0xFFFFFFu) + E0W, unpack4(c >> 8) + E0W);
}
// the four scale bytes of row P, K group kq
__device__ __forceinline__ uint32_t get4(const uint8_t* s, int sf6, int N, int KQ, int kq, int P) {
  if (!sf6) return *reinterpret_cast<const uint32_t*>(s + ((long long)kq * N + P) * 4);
  const uint8_t* d = chunk6(const_cast<uint8_t*>(s), KQ, kq, P & ~3) + 3 * (P & 3);
  return unpack4((uint32_t)d[0] | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16)) + E0W;
}
// one scale byte (row P, K block k) of a row
__device__ __forceinline__ void put1(uint8_t* s, int sf6, int N, int KQ, int P, int k, uint32_t val) {
  const int kq = k >> 2, j = k & 3;
  if (!sf6) { s[((long long)kq * N + P) * 4 + j] = (uint8_t)val; return; }
  uint8_t* d = chunk6(s, KQ, kq, P & ~3) + 3 * (P & 3);
  uint32_t w = (uint32_t)d[0] | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16);
  w = (w & ~(63u << (6 * j))) | (((val - E0) & 63u) << (6 * j));
  d[0] = (uint8_t)w; d[1] = (uint8_t)(w >> 8); d[2] = (uint8_t)(w >> 16);
}
// a record's scale bytes are the native ones lowered by one; zero stays zero
__device__ __forceinline__ uint32_t lower4(uint32_t w) {
  const uint32_t nz = (((w | 0x80808080u) - 0x01010101u) | w) & 0x80808080u;       // bit 7 of every non-zero byte
  return w - (nz >> 7);
}
// the exception entries of a hi2-k2 scale unit (rows u * UROWS ..) into a row; one thread, after the unit's codes
__device__ __forceinline__ void patch(const uint32_t* ex, uint8_t* s, int sf6, int N, int KQ, int u) {
#pragma unroll 1
  for (int i = 0; i < EXC; i++) {
    const uint32_t e = ex[i];
    if (e != EMPTY) put1(s, sf6, N, KQ, u * UROWS + (int)(e & 511u), (int)((e >> 9) & 127u), (e >> 16) & 255u);
  }
}

// ------------------------------------------------------------------------------------------------------ stage
// Slot k's record stage[k] (>= 0) of the bank -> row k of dst (ring == nullptr, or late), and -> row ring[k] of dst2
// (ring != nullptr and ring[k] >= 0). The grid's CTAs share the active slots' units round robin; a unit is a 16 KiB
// chunk of code bytes (expanded through the table) or a scale unit (hi2-k1: 16 KiB of scale bytes; hi2-k2: 512 scale
// rows with their bases and exceptions), read with one bulk copy into a ring of shared-memory stages.
__global__ void __launch_bounds__(THREADS) stage_kernel(const uint8_t* __restrict__ bank, long long rec_bytes,
    const int* __restrict__ stage, const int* __restrict__ ring, Dst dst, Dst dst2, int late, int gdc, Geo g,
    const uint16_t* __restrict__ lut_g) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint64_t bars[STAGES];
  __shared__ uint16_t lut[256];
  __shared__ int active[SLOTS];
  __shared__ int n_active;
  if (gdc) {                 // the decode stager (PDL): wait for the stage list's writer, let the next kernel launch
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  const bool to1 = ring == nullptr || late;
  if (threadIdx.x == 0) {
    int n = 0;
    for (int k = 0; k < SLOTS; k++) if (stage[k] >= 0 && (to1 || ring[k] >= 0)) active[n++] = k;
    n_active = n;
  }
  __syncthreads();
  if (n_active == 0) return;
  for (int i = threadIdx.x; i < 256; i += THREADS) lut[i] = lut_g[i];
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++) mbar_init(&bars[s]);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  const int nc = g.c[0] + g.c[1], per = nc + g.ns[0] + g.ns[1], total = per * n_active;
  const long long codes = (long long)nc * CHUNK;
  const int n = (int)blockIdx.x < total ? (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto issue = [&](int i) {
    const int u = blockIdx.x + i * gridDim.x, k = active[u / per];
    int c = u %% per, bytes = CHUNK;
    long long off = (long long)c * CHUNK;
    if (c >= nc) {
      c -= nc;
      if (c < g.ns[0]) { off = codes + (long long)c * g.sz[0]; bytes = g.sz[0]; }
      else { off = codes + (long long)g.ns[0] * g.sz[0] + (long long)(c - g.ns[0]) * g.sz[1]; bytes = g.sz[1]; }
    }
    mbar_expect(&bars[i %% STAGES], bytes);
    bulk_g2s(smem + (i %% STAGES) * CHUNK, bank + (long long)stage[k] * rec_bytes + off, bytes, &bars[i %% STAGES]);
  };
  if (threadIdx.x == 0) for (int i = 0; i < STAGES && i < n; i++) issue(i);
  for (int i = 0; i < n; i++) {
    const int s = i %% STAGES;
    mbar_wait(&bars[s], (i / STAGES) & 1);
    const int u = blockIdx.x + i * gridDim.x, k = active[u / per];
    int c = u %% per;
    const uint8_t* sp = smem + s * CHUNK;
    const int rk = ring ? ring[k] : -1;
    int t = -1;                                             // scale tensor of this unit, or -1: code bytes
    uint8_t *o1 = nullptr, *o2 = nullptr;
    if (c < nc) {                                           // codes: 16 record bytes -> 32 native bytes per step
      const int seg = c >= g.c[0];
      if (seg) c -= g.c[0];
      if (to1) o1 = dst.base[seg] + (long long)k * dst.stride[seg] + 2LL * c * CHUNK;
      if (rk >= 0) o2 = dst2.base[seg] + (long long)rk * dst2.stride[seg] + 2LL * c * CHUNK;
      for (int j = threadIdx.x * 16; j < CHUNK; j += THREADS * 16) {
        uint4 v = *reinterpret_cast<const uint4*>(sp + j);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        uint4 x0, x1;
        uint16_t* p0 = reinterpret_cast<uint16_t*>(&x0);
        uint16_t* p1 = reinterpret_cast<uint16_t*>(&x1);
#pragma unroll
        for (int q = 0; q < 8; q++) { p0[q] = lut[b[q]]; p1[q] = lut[b[8 + q]]; }
        if (o1) { uint4* d = reinterpret_cast<uint4*>(o1 + 2 * j); d[0] = x0; d[1] = x1; }
        if (o2) { uint4* d = reinterpret_cast<uint4*>(o2 + 2 * j); d[0] = x0; d[1] = x1; }
      }
    } else {
      c -= nc;
      t = c >= g.ns[0];
      if (t) c -= g.ns[0];
      const int N = g.n[t], KQ = g.kq[t];
      if (to1) o1 = dst.base[2 + t] + (long long)k * dst.stride[2 + t];
      if (rk >= 0) o2 = dst2.base[2 + t] + (long long)rk * dst2.stride[2 + t];
      if (!g.k2) {                                          // 16 KiB of scale bytes: words c * 4096 ..
        for (int j = threadIdx.x; j < CHUNK / 16; j += THREADS) {
          const uint4 v = *reinterpret_cast<const uint4*>(sp + 16 * j);
          const long long w = (long long)c * (CHUNK / 4) + 4 * j;
          int kq = 0, P = 0;
          if (g.sf6) { kq = (int)(w / N); P = (int)(w - (long long)kq * N); }
          if (o1) put16(o1, g.sf6, KQ, kq, P, w, v);
          if (o2) put16(o2, g.sf6, KQ, kq, P, w, v);
        }
      } else {                                              // a unit: bases, exceptions, KQ x 512 code bytes
        const uint8_t* cb = sp + UHEAD;
        for (int q = threadIdx.x; q < KQ * (UROWS / 4); q += THREADS) {
          const int kq = q / (UROWS / 4), a = q %% (UROWS / 4), P = c * UROWS + 4 * a;
          const uint4 v = dec4(*reinterpret_cast<const uint32_t*>(cb + kq * UROWS + 4 * a),
                               *reinterpret_cast<const uint32_t*>(sp + 4 * a));
          const long long w = (long long)kq * N + P;
          if (o1) put16(o1, g.sf6, KQ, kq, P, w, v);
          if (o2) put16(o2, g.sf6, KQ, kq, P, w, v);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (t >= 0 && g.k2) {                                 // before the stage's memory is reused
        const uint32_t* ex = reinterpret_cast<const uint32_t*>(sp + UROWS);
        if (o1) patch(ex, o1, g.sf6, g.n[t], g.kq[t], c);
        if (o2) patch(ex, o2, g.sf6, g.n[t], g.kq[t], c);
      }
      if (i + STAGES < n) issue(i + STAGES);
    }
  }
}

// ----------------------------------------------------------------------------------------------------- expand
// The copy-engine lane (celane.py): raw records in an HBM scratch (record j at src + j * rec_bytes) -> rows row0 + j.
// blockIdx.y = the record; blocks along x stride over the 16-byte units of its code bytes (and of hi2-k1 scale bytes),
// UNROLL loads in flight per thread; hi2-k2 scale units are decoded one whole unit per block.
__global__ void __launch_bounds__(THREADS) expand_kernel(const uint8_t* __restrict__ src, long long rec_bytes, Dst dst,
    long long row0, Geo g, const uint16_t* __restrict__ lut_g) {
  __shared__ uint16_t lut[256];
  __shared__ uint32_t bs[UROWS / 4];
  for (int i = threadIdx.x; i < 256; i += THREADS) lut[i] = lut_g[i];
  __syncthreads();
  const long long row = row0 + blockIdx.y;
  const uint8_t* recb = src + (long long)blockIdx.y * rec_bytes;
  const uint4* rec = reinterpret_cast<const uint4*>(recb);
  const long long c13 = g.c13b, codes = g.codesb, s13 = g.s13b, s2 = g.s2b;
  const long long n16 = (codes + (g.k2 ? 0 : s13 + s2)) >> 4, step = (long long)gridDim.x * THREADS;
  uint8_t* const o13 = dst.base[0] + row * dst.stride[0];
  uint8_t* const o2 = dst.base[1] + row * dst.stride[1];
  uint8_t* const os[2] = {dst.base[2] + row * dst.stride[2], dst.base[3] + row * dst.stride[3]};
  for (long long u0 = (long long)blockIdx.x * THREADS + threadIdx.x; u0 < n16; u0 += UNROLL * step) {
    uint4 v[UNROLL];
#pragma unroll
    for (int q = 0; q < UNROLL; q++) {
      const long long u = u0 + q * step;
      v[q] = u < n16 ? __ldcs(rec + u) : make_uint4(0, 0, 0, 0);       // read once: evict first
    }
#pragma unroll
    for (int q = 0; q < UNROLL; q++) {
      const long long u = u0 + q * step;
      if (u >= n16) break;
      const long long off = u << 4;
      if (off < codes) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v[q]);
        uint4 w0, w1;
        uint16_t* p0 = reinterpret_cast<uint16_t*>(&w0);
        uint16_t* p1 = reinterpret_cast<uint16_t*>(&w1);
#pragma unroll
        for (int k = 0; k < 8; k++) { p0[k] = lut[b[k]]; p1[k] = lut[b[8 + k]]; }
        uint4* d = reinterpret_cast<uint4*>(off < c13 ? o13 + 2 * off : o2 + 2 * (off - c13));
        d[0] = w0; d[1] = w1;
      } else {                               // hi2-k1 scale bytes
        long long so = off - codes;
        const int t = so >= s13;
        if (t) so -= s13;
        const long long w = so >> 2;
        int kq = 0, P = 0;
        if (g.sf6) { kq = (int)(w / g.n[t]); P = (int)(w - (long long)kq * g.n[t]); }
        put16(os[t], g.sf6, g.kq[t], kq, P, w, v[q]);
      }
    }
  }
  if (!g.k2) return;
  __syncthreads();
  for (int U = blockIdx.x; U < g.ns[0] + g.ns[1]; U += gridDim.x) {
    const int t = U >= g.ns[0], c = t ? U - g.ns[0] : U, N = g.n[t], KQ = g.kq[t];
    const uint8_t* up = recb + codes + (t ? (long long)g.ns[0] * g.sz[0] : 0) + (long long)c * g.sz[t];
    for (int a = threadIdx.x; a < UROWS / 4; a += THREADS) bs[a] = *reinterpret_cast<const uint32_t*>(up + 4 * a);
    __syncthreads();
    const uint8_t* cb = up + UHEAD;
    for (int q = threadIdx.x; q < KQ * (UROWS / 4); q += THREADS) {
      const int kq = q / (UROWS / 4), a = q %% (UROWS / 4), P = c * UROWS + 4 * a;
      const uint4 v = dec4(*reinterpret_cast<const uint32_t*>(cb + kq * UROWS + 4 * a), bs[a]);
      put16(os[t], g.sf6, KQ, kq, P, (long long)kq * N + P, v);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      patch(reinterpret_cast<const uint32_t*>(up + UROWS), os[t], g.sf6, N, KQ, c);
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------------ merge
// Promotion: a hi2 record (either format: only its code bytes are read) and the expert's lo2 record (hi2-k1 layout:
// the low bit planes and the unlowered scale bytes) -> the exact native row. Plain 16-byte loads (host memory is fine).
__global__ void __launch_bounds__(THREADS) merge_kernel(const uint8_t* __restrict__ hi, const uint8_t* __restrict__ lo,
    Dst dst, Geo g) {
  const long long c13 = g.c13b, codes = g.codesb, s13 = g.s13b;
  const long long total16 = (codes + s13 + g.s2b) >> 4;
  uint8_t* const d13 = dst.base[0];
  uint8_t* const d2 = dst.base[1];
  uint8_t* const ds13 = dst.base[2];
  uint8_t* const ds2 = dst.base[3];
  const int six = g.sf6, n13 = g.n[0], n2 = g.n[1], kq13 = g.kq[0], kq2 = g.kq[1];
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total16; i += (long long)gridDim.x * blockDim.x) {
    const long long off = i * 16;
    const uint4 l = *reinterpret_cast<const uint4*>(lo + off);
    if (off < codes) {
      const uint4 h = *reinterpret_cast<const uint4*>(hi + off);
      const uint8_t* hb = reinterpret_cast<const uint8_t*>(&h);
      const uint8_t* lb = reinterpret_cast<const uint8_t*>(&l);
      uint4 o0, o1;
      uint8_t* ob0 = reinterpret_cast<uint8_t*>(&o0);
      uint8_t* ob1 = reinterpret_cast<uint8_t*>(&o1);
#pragma unroll
      for (int q = 0; q < 16; q++) {
        const uint32_t x = hb[q], y = lb[q];
        const uint32_t n0 = ((x & 3) << 2) | (y & 3), n1 = (((x >> 2) & 3) << 2) | ((y >> 2) & 3);
        const uint32_t n2_ = (((x >> 4) & 3) << 2) | ((y >> 4) & 3), n3 = (((x >> 6) & 3) << 2) | ((y >> 6) & 3);
        uint8_t* ob = q < 8 ? ob0 : ob1;
        ob[2 * (q & 7)] = (uint8_t)(n0 | (n1 << 4));
        ob[2 * (q & 7) + 1] = (uint8_t)(n2_ | (n3 << 4));
      }
      uint4* d = reinterpret_cast<uint4*>(off < c13 ? d13 + 2 * off : d2 + 2 * (off - c13));
      d[0] = o0; d[1] = o1;
    } else {
      const long long so = off - codes;
      if (!six) {
        *reinterpret_cast<uint4*>(so < s13 ? ds13 + so : ds2 + (so - s13)) = l;
      } else {
        const bool t = so >= s13;
        const long long w = (t ? so - s13 : so) >> 2;
        const int N = t ? n2 : n13, kq = (int)(w / N);
        put16(t ? ds2 : ds13, 1, t ? kq2 : kq13, kq, (int)(w - (long long)kq * N), w, l);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------- extract
// A native row -> its hi2 record: the top two bits of every nibble, and the scale bytes lowered by one (zero stays
// zero), as hi2-k1 bytes or hi2-k2 units (the canonical encoder of recfmt.py: a row's base is its lowest byte, or for
// a row of more than four values the lowest start of the 4-wide window that holds most; a tile's exceptions sorted
// by row and K block in the tile's own entries; its status word counts the exceptions it could not hold). The
// record's tail is zeroed.
// hi2-k2: a block encodes one tile of 128 scale rows at a time: every thread a few (K group, four rows) words, the
// rows' minima through shared memory, then the codes; rows that need more (a zero byte, a byte above 127, more than
// four values) are redone exactly by one thread each. With a large grid the first blocks only encode tiles and the
// others stride over the code bytes (a thread then runs a few hundred instructions either way).
__device__ __forceinline__ uint32_t minbyte(uint32_t w) {
  return min(min(w & 255u, (w >> 8) & 255u), min((w >> 16) & 255u, w >> 24));
}
__device__ __forceinline__ uint32_t fold2(uint32_t d) { return (d | (d >> 6) | (d >> 12) | (d >> 18)) & 255u; }

struct TileMem {                                             // one block's scratch for a tile
  uint8_t rmin[KQ_MAX * TILE];                               // [K group][row]: the lowest of the word's four bytes
  uint8_t base[TILE];                                        // a row's lowest (native) byte
  uint8_t slow[TILE];
  uint8_t xcnt[TILE];
  uint32_t xent[TILE * XT];
  int anyslow;
};

template <int KQ>
__device__ __forceinline__ void encode_tile(const uint8_t* __restrict__ srow, int six, int N, int tile,
                                            uint8_t* __restrict__ unit0, int unit_bytes, uint32_t* __restrict__ status,
                                            TileMem& m) {
  constexpr int NQ = KQ * (TILE / 4), PER = (NQ + THREADS - 1) / THREADS;    // (K group, 4 rows) words; per thread
  const int tid = threadIdx.x;
  const int P0 = TILE * tile;                                // the tile's first scale row
  uint8_t* const unit = unit0 + (long long)(P0 / UROWS) * unit_bytes;
  const int r0 = P0 %% UROWS;                                 // its row in the unit
  if (tid < TILE) m.slow[tid] = 0;
  if (tid == 0) m.anyslow = 0;
  __syncthreads();
  uint4 w[PER];
#pragma unroll
  for (int i = 0; i < PER; i++) {
    const int q = tid + i * THREADS;
    if (q < NQ) {
      const int kq = q / (TILE / 4), a = q %% (TILE / 4);
      const uint4 v = get16(srow, six, KQ, kq, P0 + 4 * a, (long long)kq * N + P0 + 4 * a);
      w[i] = v;
      *reinterpret_cast<uint32_t*>(m.rmin + kq * TILE + 4 * a) =
          minbyte(v.x) | (minbyte(v.y) << 8) | (minbyte(v.z) << 16) | (minbyte(v.w) << 24);
      if ((v.x | v.y | v.z | v.w) & 0x80808080u) {            // a byte above 127: the exact path
        if (v.x & 0x80808080u) m.slow[4 * a] = 1;
        if (v.y & 0x80808080u) m.slow[4 * a + 1] = 1;
        if (v.z & 0x80808080u) m.slow[4 * a + 2] = 1;
        if (v.w & 0x80808080u) m.slow[4 * a + 3] = 1;
        m.anyslow = 1;
      }
    }
  }
  __syncthreads();
  if (tid < TILE) {
    uint32_t lo = 255u;
#pragma unroll 4
    for (int kq = 0; kq < KQ; kq++) lo = min(lo, (uint32_t)m.rmin[kq * TILE + tid]);
    m.base[tid] = (uint8_t)lo;
    if (lo == 0u) { m.slow[tid] = 1; m.anyslow = 1; }          // a zero byte stays zero: the exact path
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < PER; i++) {
    const int q = tid + i * THREADS;
    if (q < NQ) {
      const int kq = q / (TILE / 4), a = q %% (TILE / 4);
      const uint32_t bw = *reinterpret_cast<const uint32_t*>(m.base + 4 * a);
      const uint32_t d0 = w[i].x - (bw & 255u) * 0x01010101u, d1 = w[i].y - ((bw >> 8) & 255u) * 0x01010101u;
      const uint32_t d2 = w[i].z - ((bw >> 16) & 255u) * 0x01010101u, d3 = w[i].w - (bw >> 24) * 0x01010101u;
      if ((d0 | d1 | d2 | d3) & 0xFCFCFCFCu) {                // a row of more than four values: the exact path
        if (d0 & 0xFCFCFCFCu) m.slow[4 * a] = 1;
        if (d1 & 0xFCFCFCFCu) m.slow[4 * a + 1] = 1;
        if (d2 & 0xFCFCFCFCu) m.slow[4 * a + 2] = 1;
        if (d3 & 0xFCFCFCFCu) m.slow[4 * a + 3] = 1;
        m.anyslow = 1;
      }
      *reinterpret_cast<uint32_t*>(unit + UHEAD + kq * UROWS + r0 + 4 * a) =
          fold2(d0) | (fold2(d1) << 8) | (fold2(d2) << 16) | (fold2(d3) << 24);
    }
  }
  if (tid < TILE / 4)                                        // the stored base: the lowest byte, lowered by one
    *reinterpret_cast<uint32_t*>(unit + r0 + 4 * tid) = lower4(*reinterpret_cast<const uint32_t*>(m.base + 4 * tid));
  __syncthreads();
  uint32_t* const ex = reinterpret_cast<uint32_t*>(unit + UROWS) + XT * (r0 / TILE);     // the tile's entries
  if (!m.anyslow) {
    if (tid < XT) ex[tid] = EMPTY;
    if (tid == 0) *status = 0u;
    return;
  }
  if (tid < TILE) {                                          // the general encoder: thread r redoes row r if marked
    int nx = 0;                                              // (rare: loops kept rolled, the row re-read from memory)
    if (m.slow[tid]) {
      const int P = P0 + tid;
      uint32_t lo = 255u, hi = 0u;
#pragma unroll 1
      for (int k = 0; k < 4 * KQ; k++) {
        const uint32_t v = (lower4(get4(srow, six, N, KQ, k >> 2, P)) >> (8 * (k & 3))) & 255u;
        lo = min(lo, v); hi = max(hi, v);
      }
      uint32_t base = lo;
      if (hi - lo > 3u) {                                    // the best 4-wide window, lowest first
        int best = -1;
#pragma unroll 1
        for (uint32_t st = lo; st + 3u <= hi; st++) {
          int c = 0;
#pragma unroll 1
          for (int k = 0; k < 4 * KQ; k++)
            c += (((lower4(get4(srow, six, N, KQ, k >> 2, P)) >> (8 * (k & 3))) & 255u) - st) <= 3u;
          if (c > best) { best = c; base = st; }
        }
      }
      unit[r0 + tid] = (uint8_t)base;
      uint32_t cbyte = 0;
#pragma unroll 1
      for (int k = 0; k < 4 * KQ; k++) {
        const uint32_t v = (lower4(get4(srow, six, N, KQ, k >> 2, P)) >> (8 * (k & 3))) & 255u;
        uint32_t d = v - base;
        if (d > 3u) {
          if (nx < XT) m.xent[tid * XT + nx] = (uint32_t)(r0 + tid) | ((uint32_t)k << 9) | (v << 16);
          nx++;
          d = 0;
        }
        cbyte |= d << (2 * (k & 3));
        if ((k & 3) == 3) { unit[UHEAD + (k >> 2) * UROWS + r0 + tid] = (uint8_t)cbyte; cbyte = 0; }
      }
    }
    m.xcnt[tid] = (uint8_t)min(nx, 255);
  }
  __syncthreads();
  if (tid == 0) {                                            // the tile's entries in row order
    int n = 0;
    for (int r = 0; r < TILE; r++) {
      const int c = m.xcnt[r];
      for (int j = 0; j < min(c, XT); j++) if (n + j < XT) ex[n + j] = m.xent[r * XT + j];
      n += c;
    }
    for (int i = min(n, XT); i < XT; i++) ex[i] = EMPTY;
    *status = n > XT ? (uint32_t)(n - XT) : 0u;
  }
  __syncthreads();
}

// (six blocks per SM, as today's extract kernel fits: the tile encoder must not cost the code blocks their slots)
__global__ void __launch_bounds__(THREADS, 6) extract_kernel(Dst src, uint8_t* __restrict__ out, Geo g, long long rec_bytes) {
  __shared__ TileMem tm;
  const long long c13 = g.c13b, codes = g.codesb, s13 = g.s13b, s2 = g.s2b;
  const int nt13 = g.n[0] / TILE, nt = g.k2 ? nt13 + g.n[1] / TILE : 0;     // scale tiles of the record
  // hi2-k2 on a large grid: blocks 0 .. nt - 1 encode one tile each and take no code bytes
  const bool split = nt && (int)gridDim.x >= 8 * nt;
  if (!split || (int)blockIdx.x >= nt) {
    const long long total16 = (codes + (g.k2 ? 0 : s13 + s2)) >> 4;
    const long long first = split ? (int)blockIdx.x - nt : (int)blockIdx.x, nblk = split ? (int)gridDim.x - nt : (int)gridDim.x;
    const uint8_t* const p13 = src.base[0];
    const uint8_t* const p2 = src.base[1];
    for (long long i = first * blockDim.x + threadIdx.x; i < total16; i += nblk * blockDim.x) {
      const long long off = i * 16;
      uint4 o;
      uint8_t* ob = reinterpret_cast<uint8_t*>(&o);
      if (off < codes) {
        const uint8_t* p = off < c13 ? p13 + 2 * off : p2 + 2 * (off - c13);
        const uint4 a = *reinterpret_cast<const uint4*>(p);
        const uint4 b = *reinterpret_cast<const uint4*>(p + 16);
        const uint8_t* na = reinterpret_cast<const uint8_t*>(&a);
        const uint8_t* nb = reinterpret_cast<const uint8_t*>(&b);
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const uint32_t n0 = q < 8 ? na[2 * q] : nb[2 * (q - 8)];
          const uint32_t n1 = q < 8 ? na[2 * q + 1] : nb[2 * (q - 8) + 1];
          ob[q] = (uint8_t)(((n0 & 0xF) >> 2) | ((n0 >> 6) << 2) | (((n1 & 0xF) >> 2) << 4) | ((n1 >> 6) << 6));
        }
      } else {                               // hi2-k1 scale bytes
        const long long so = off - codes;
        const bool t = so >= s13;
        const long long w = (t ? so - s13 : so) >> 2;
        int kq = 0, P = 0;
        if (g.sf6) { const int N = t ? g.n[1] : g.n[0]; kq = (int)(w / N); P = (int)(w - (long long)kq * N); }
        const uint4 v = get16(t ? src.base[3] : src.base[2], g.sf6, t ? g.kq[1] : g.kq[0], kq, P, w);
        o = make_uint4(lower4(v.x), lower4(v.y), lower4(v.z), lower4(v.w));
      }
      *reinterpret_cast<uint4*>(out + off) = o;
    }
  }
  const long long p13b = (long long)g.ns[0] * g.sz[0];
  const long long status = codes + (g.k2 ? p13b + (long long)g.ns[1] * g.sz[1] : s13 + s2);
  if (blockIdx.x == gridDim.x - 1)           // zero the record's tail (after the status words)
    for (long long j = status + 4 * nt + threadIdx.x; j < rec_bytes; j += THREADS) out[j] = 0;
  if (nt == 0 || (split && (int)blockIdx.x >= nt)) return;
  uint32_t* const st = reinterpret_cast<uint32_t*>(out + status);
  for (int T = blockIdx.x; T < nt; T += gridDim.x) {
    if (T < nt13) encode_tile<KQ13>(src.base[2], g.sf6, g.n[0], T, out + codes, g.sz[0], st + T, tm);
    else encode_tile<KQ2>(src.base[3], g.sf6, g.n[1], T - nt13, out + codes + p13b, g.sz[1], st + T, tm);
  }
}

// --------------------------------------------------------------------------------------------------- bindings
static Geo geo_of(const std::vector<int64_t>& v) {
  TORCH_CHECK(v.size() == 12, "geometry: 12 values");
  Geo g;
  g.k2 = (int)v[0]; g.sf6 = (int)v[1];
  for (int t = 0; t < 2; t++) {
    g.n[t] = (int)v[2 + t]; g.kq[t] = (int)v[4 + t]; g.c[t] = (int)v[6 + t]; g.ns[t] = (int)v[8 + t]; g.sz[t] = (int)v[10 + t];
    TORCH_CHECK(g.kq[t] <= KQ_MAX && g.sz[t] <= CHUNK && g.sz[t] %% 16 == 0 && g.n[t] %% UROWS == 0, "scale geometry");
    TORCH_CHECK(!g.k2 || (g.ns[t] * UROWS == g.n[t] && g.sz[t] == UHEAD + g.kq[t] * UROWS), "hi2-k2 scale units");
    TORCH_CHECK(g.k2 || (long long)g.ns[t] * g.sz[t] == (long long)g.n[t] * g.kq[t] * 4, "hi2-k1 scale chunks");
  }
  g.c13b = (long long)g.c[0] * CHUNK; g.codesb = g.c13b + (long long)g.c[1] * CHUNK;
  g.s13b = (long long)g.n[0] * g.kq[0] * 4; g.s2b = (long long)g.n[1] * g.kq[1] * 4;
  return g;
}

static Dst dst_of(const std::vector<int64_t>& bases, const std::vector<int64_t>& strides) {
  TORCH_CHECK(bases.size() == 4 && (strides.empty() || strides.size() == 4));
  Dst d;
  for (int i = 0; i < 4; i++) {
    d.base[i] = reinterpret_cast<uint8_t*>(bases[i]);
    d.stride[i] = strides.empty() ? 0 : strides[i];
    TORCH_CHECK(bases[i] %% 16 == 0 && d.stride[i] %% 4 == 0, "rows must be 16-byte aligned");
  }
  return d;
}

// mode 0: the decode stager (one destination; launched with programmatic stream serialization when K31_PDL=1, as
// stager.py's); mode 1: the ring stager (src slots, ring rows, two destinations)
void stage(int64_t bank, int64_t rec_bytes, torch::Tensor stage_idx, torch::Tensor ring, std::vector<int64_t> bases,
           std::vector<int64_t> bases2, std::vector<int64_t> strides, int64_t late, int64_t mode,
           std::vector<int64_t> geo, torch::Tensor lut, int64_t sms) {
  TORCH_CHECK(stage_idx.is_cuda() && stage_idx.scalar_type() == torch::kInt32 && stage_idx.numel() == SLOTS);
  TORCH_CHECK(mode == 0 || (ring.is_cuda() && ring.scalar_type() == torch::kInt32 && ring.numel() == SLOTS));
  TORCH_CHECK(bank %% 16 == 0 && rec_bytes %% 16 == 0, "records must be 16-byte aligned");
  const Geo g = geo_of(geo);
  const Dst d = dst_of(bases, strides), d2 = dst_of(mode == 0 ? bases : bases2, strides);
  static bool attr = false;
  const size_t smem = (size_t)STAGES * CHUNK;
  if (!attr) { cudaFuncSetAttribute(stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)sms); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute lattr[1];
  lattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  lattr[0].val.programmaticStreamSerializationAllowed = 1;
  static const int pdl = getenv("K31_PDL") != nullptr && getenv("K31_PDL")[0] == '1';
  cfg.attrs = lattr; cfg.numAttrs = (mode == 0 && pdl) ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, stage_kernel, reinterpret_cast<const uint8_t*>(bank), (long long)rec_bytes,
      (const int*)stage_idx.data_ptr<int>(), mode == 0 ? (const int*)nullptr : (const int*)ring.data_ptr<int>(), d, d2,
      (int)late, (int)(mode == 0), g, reinterpret_cast<const uint16_t*>(lut.data_ptr())));
}

void expand(int64_t src, int64_t rec_bytes, int64_t n, std::vector<int64_t> bases, std::vector<int64_t> strides,
            int64_t row0, std::vector<int64_t> geo, torch::Tensor lut, int64_t blocks) {
  if (n <= 0) return;
  TORCH_CHECK(lut.is_cuda() && lut.numel() == 512, "the stager's table: 256 x uint16 as 512 bytes");
  TORCH_CHECK(src %% 16 == 0 && rec_bytes %% 16 == 0, "records must be 16-byte aligned");
  TORCH_CHECK(n <= 65535, "at most 65535 records per launch");
  const Geo g = geo_of(geo);
  const Dst d = dst_of(bases, strides);
  const int64_t bx = std::max<int64_t>(1, (blocks + n - 1) / n);
  expand_kernel<<<dim3((unsigned)bx, (unsigned)n), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(src), (long long)rec_bytes, d, (long long)row0, g,
      reinterpret_cast<const uint16_t*>(lut.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge(int64_t hi, int64_t lo, std::vector<int64_t> bases, std::vector<int64_t> geo, int64_t blocks) {
  merge_kernel<<<(int)blocks, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(hi), reinterpret_cast<const uint8_t*>(lo), dst_of(bases, {}), geo_of(geo));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void extract(int64_t out, std::vector<int64_t> bases, std::vector<int64_t> geo, int64_t rec_bytes, int64_t blocks) {
  TORCH_CHECK(geo.size() == 12 && geo[4] == KQ13 && geo[5] == KQ2, "this build encodes scale units of other K groups");
  extract_kernel<<<(int)blocks, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      dst_of(bases, {}), reinterpret_cast<uint8_t*>(out), geo_of(geo), (long long)rec_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stage", &stage); m.def("expand", &expand); m.def("merge", &merge); m.def("extract", &extract);
}
"""
_MOD = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{ARCH[:-2]}.{ARCH[-2:]}")
        src = _SRC % dict(CHUNK=CHUNK, STAGES=STAGES, SLOTS=SLOTS, UROWS=recfmt.UNIT_ROWS, EXC=recfmt.EXC,
                          KQ_MAX=KQ_MAX, E0=_sf6.E0, KQ13=KQ[0], KQ2=KQ[1], XT=recfmt.EXC_TILE, TILE=recfmt.TILE_ROWS)
        _MOD.append(load_inline(name=f"k31_stager2_{ARCH}", cpp_sources="", cuda_sources=src,
                                extra_cuda_cflags=["-O3", "-std=c++20", f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}"],
                                verbose=False))
    return _MOD[0]


# ------------------------------------------------------------------------------------------------ rows and geometry
def coded(mod):
    """The module's 6-bit scale banks (uint8 [rows, row bytes] for w13 and w2; set by the hook with K31_SF6=1), or None."""
    return getattr(mod, "_k31_sf6", None)


def active(layout: dict, mod=None) -> bool:
    """True if this module's kernels serve the combination (hi2-k2 records, or 6-bit scale rows)."""
    return recfmt.fmt(layout) == recfmt.K2 or (mod is not None and coded(mod) is not None)


def rows(mod):
    """(row-0 addresses, row pitches in bytes, 6-bit rows?) of the module's four kernel tensors: w13 and w2 codes, w13
    and w2 scales (cached on the module while its tensors are the same objects)."""
    c = coded(mod)
    ts = [mod.w13_weight, mod.w2_weight]
    ts += [mod.w13_weight_scale, mod.w2_weight_scale] if c is None else list(c)
    key = tuple(id(t) for t in ts)
    hit = getattr(mod, "_k31_rows", None)
    if hit is not None and hit[0] == key:
        return hit[1]
    out = _rows(ts, c is not None)
    try:
        object.__setattr__(mod, "_k31_rows", (key, out, ts))     # ts: the keyed objects stay alive, so ids are unique
    except Exception:                            # noqa: BLE001  (a module that refuses attributes: no cache)
        pass
    return out


def _rows(ts, six):
    bases, pitch = [], []
    for t in ts:
        row = t[0].numel() * t.element_size()
        assert t.stride(0) * t.element_size() == row, "rows must be dense blocks"
        bases.append(t.data_ptr())               # data_ptr of a transposed view = its storage start
        pitch.append(row)
    return bases, pitch, six


_GEO: dict = {}


def geo(layout: dict, k2: bool, six: bool) -> list:
    """The kernels' geometry argument for records of `layout` read or written as hi2-k2 (k2) or hi2-k1, and rows with
    6-bit scales (six) or int32 scale words (cached per layout object)."""
    key = (id(layout), k2, six)
    hit = _GEO.get(key)
    if hit is not None and hit[0] is layout:
        return hit[1]
    out = _geo(layout, k2, six)
    _GEO[key] = (layout, out)
    return out


def _geo(layout: dict, k2: bool, six: bool) -> list:
    g = recfmt.geom(layout)
    assert g["c13"] % CHUNK == 0 and g["c2"] % CHUNK == 0, "code segments must be whole chunks"
    if k2:
        ns, sz = [g["units13"], g["units2"]], [g["unit13"], g["unit2"]]
    else:
        assert g["s13"] % CHUNK == 0 and g["s2"] % CHUNK == 0, "scale segments must be whole chunks"
        ns, sz = [g["s13"] // CHUNK, g["s2"] // CHUNK], [CHUNK, CHUNK]
    if six:
        assert _sf6.LAYOUT == 2 and g["n13"] % 128 == 0 and g["n2"] % 128 == 0, "6-bit rows: layout 2, whole tiles"
    return [int(k2), int(six), g["n13"], g["n2"], g["kq13"], g["kq2"], g["c13"] // CHUNK, g["c2"] // CHUNK, *ns, *sz]


def _lut(dev):
    from .stager import _lut16_cached
    return _lut16_cached(dev)


# ------------------------------------------------------------------------------------------------------- stagers
class Stager2:
    """stager.Stager for hi2-k2 records and / or 6-bit scale rows (same call)."""

    def __init__(self, mod, first_slot: int, layout: dict, sms: int = 24):
        self.m = _module()
        self.sms = sms
        self.lut = _lut(mod.w13_weight.device)
        b, self.strides, six = rows(mod)
        self.bases = [x + first_slot * p for x, p in zip(b, self.strides)]
        self.geo = geo(layout, recfmt.fmt(layout) == recfmt.K2, six)
        self.rec_bytes = layout["record_bytes"]
        self._none = torch.empty(0, dtype=torch.int32, device=mod.w13_weight.device)

    def __call__(self, bank_ptr: int, stage_idx: torch.Tensor) -> None:
        self.m.stage(bank_ptr, self.rec_bytes, stage_idx, self._none, self.bases, self.bases, self.strides, 0, 0,
                     self.geo, self.lut, self.sms)


class RingStager2(Stager2):
    """stager.RingStager for hi2-k2 records and / or 6-bit scale rows."""

    def __init__(self, mod, first_slot: int, ring_base: int, layout: dict, sms: int = 24):
        super().__init__(mod, first_slot, layout, sms)
        b, p, _ = rows(mod)
        self.bases2 = [x + ring_base * y for x, y in zip(b, p)]

    def __call__(self, bank_ptr: int, src: torch.Tensor, ring: torch.Tensor, late: bool) -> None:
        self.m.stage(bank_ptr, self.rec_bytes, src, ring, self.bases, self.bases2, self.strides, int(late), 1,
                     self.geo, self.lut, self.sms)


def _row(mod, row: int):
    b, p, six = rows(mod)
    return [x + row * y for x, y in zip(b, p)], six


def merge(hi_ptr: int, lo_ptr: int, mod, row: int, layout: dict, blocks: int = 32) -> None:
    """stager.merge: expert row `row` from its hi2 record (either format) and its lo2 record."""
    bases, six = _row(mod, row)
    _module().merge(hi_ptr, lo_ptr, bases, geo(layout, False, six), blocks)


def extract_hi2(mod, row: int, layout: dict, out: torch.Tensor, blocks: int = 152) -> None:
    """stager.extract_hi2: native row `row` -> its hi2 record in the store's format, out[:record_bytes]."""
    bases, six = _row(mod, row)
    assert out.numel() >= layout["record_bytes"] and out.data_ptr() % 16 == 0
    _module().extract(out.data_ptr(), bases, geo(layout, recfmt.fmt(layout) == recfmt.K2, six), layout["record_bytes"],
                      blocks)


def expand(src_ptr: int, rec_bytes: int, n: int, mod, row0: int, layout: dict, blocks: int) -> None:
    """celane's expand: n records at src_ptr (HBM) -> rows row0 .. row0 + n - 1."""
    b, p, six = rows(mod)
    _module().expand(src_ptr, rec_bytes, n, b, p, row0, geo(layout, recfmt.fmt(layout) == recfmt.K2, six),
                     _lut(mod.w13_weight.device), blocks)


# ---------------------------------------------------------------------------------- references (torch, any device)
def row_scales(mod, row: int) -> torch.Tensor:
    """Row `row`'s scale bytes in hi2-k1 order (w13 then w2, native values), whatever the row's scale format."""
    c = coded(mod)
    if c is None:
        from .stager import _row_bytes
        return torch.cat((_row_bytes(mod.w13_weight_scale, row), _row_bytes(mod.w2_weight_scale, row)))
    out = []
    for bank, raw in zip(c, (mod.w13_weight_scale, mod.w2_weight_scale)):
        n, kb = raw.shape[1], raw.shape[2]
        t = _sf6.decode(bank[row:row + 1], n, kb)                    # int32 [1, N, KB] over [KB][N] memory
        out.append(t.permute(0, 2, 1).contiguous().view(torch.uint8).reshape(-1))
    return torch.cat(out)


def extract_record(mod, row: int, layout: dict, out: torch.Tensor, low: bool = False) -> None:
    """stager.extract_record by the reference codecs: row -> its hi2 record in the store's format (or lo2 with `low`,
    always the hi2-k1 layout)."""
    from .stager import _row_bytes
    codes = torch.cat((_row_bytes(mod.w13_weight, row), _row_bytes(mod.w2_weight, row)))
    if low:
        lo, hi = codes & 0x03, (codes >> 4) & 0x03
    else:
        lo, hi = (codes & 0x0F) >> 2, codes >> 6
    b = (lo | (hi << 2)).view(-1, 2)
    g = recfmt.geom(layout)
    k1 = torch.zeros(g["k1_bytes"], dtype=torch.uint8, device=out.device)
    k1[:g["codes"]] = b[:, 0] | (b[:, 1] << 4)
    sf = row_scales(mod, row)
    k1[g["codes"]:g["codes"] + sf.numel()] = sf if low else torch.where(sf > 0, sf - 1, sf)
    if low or recfmt.fmt(layout) != recfmt.K2:
        out[:k1.numel()] = k1
    else:
        out[:g["k2_bytes"]] = recfmt.k1_to_k2(k1.unsqueeze(0), layout)[0][0]
