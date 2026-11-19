"""The stager: 2-bit expert records in pinned Grace memory -> native MXFP4 rows in a layer's HBM staging slots.

One launch per MoE call (CUDA-graph safe: fixed grid, the work list is device data). For each staging slot k with
`stage[k] >= 0`, record `stage[k]` of the layer's Grace bank is read with 1-D TMA bulk copies (cp.async.bulk) into a
ring of shared-memory stages, its code bytes expanded through a 256-entry table (record byte -> two native bytes, the
depth-2 code map: nibble = (pair << 2) | 3) and its scale bytes copied, into slot k's rows of the four kernel tensors
(w13, w2 codes; w13, w2 scales). Byte layout of a record: store/build_store.py (format hi2-k1). Measured as a
standalone kernel (engine/tests/stager_e3b.cu, 11-16): 24 SMs 306 GB/s of record bytes alone, 304 GB/s beside an
HBM-streaming kernel on the other 128 SMs (which slowed 12%); byte-exact against a CPU expansion.
"""
import os

import torch

CHUNK = 16384
STAGES = 6
SLOTS = 16
_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

constexpr int CHUNK = %(CHUNK)d, STAGES = %(STAGES)d, SLOTS = %(SLOTS)d, THREADS = 256;

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

// chunks per record segment: w13 codes, w2 codes (expanded x2), w13 scales, w2 scales (copied)
__global__ void __launch_bounds__(THREADS) stage_kernel(const uint8_t* __restrict__ bank, long long rec_bytes,
    const int* __restrict__ stage, Dst dst, int c13, int c2, int s13, int s2, const uint16_t* __restrict__ lut_g) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint64_t bars[STAGES];
  __shared__ uint16_t lut[256];
  __shared__ int active[SLOTS];
  __shared__ int n_active;
  // PDL (launched with programmatic stream serialization): wait for the stage list's writer, let the next kernel
  // launch; an empty stage list (the usual case once the RTX band serves strong misses) returns at once
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
    int n = 0;
    for (int k = 0; k < SLOTS; k++) if (stage[k] >= 0) active[n++] = k;
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
  const int per = c13 + c2 + s13 + s2, total = per * n_active;
  const int n = (int)blockIdx.x < total ? (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto issue = [&](int i) {
    int u = blockIdx.x + i * gridDim.x, k = active[u / per], c = u %% per;
    mbar_expect(&bars[i %% STAGES], CHUNK);
    bulk_g2s(smem + (i %% STAGES) * CHUNK, bank + (long long)stage[k] * rec_bytes + (long long)c * CHUNK, CHUNK,
             &bars[i %% STAGES]);
  };
  if (threadIdx.x == 0) for (int i = 0; i < STAGES && i < n; i++) issue(i);
  for (int i = 0; i < n; i++) {
    const int s = i %% STAGES;
    mbar_wait(&bars[s], (i / STAGES) & 1);
    const int u = blockIdx.x + i * gridDim.x, k = active[u / per];
    int c = u %% per;
    const uint8_t* src = smem + s * CHUNK;
    int seg = 0;
    if (c >= c13) { c -= c13; seg = 1;
      if (c >= c2) { c -= c2; seg = 2;
        if (c >= s13) { c -= s13; seg = 3; } } }
    uint8_t* out = dst.base[seg] + (long long)k * dst.stride[seg];
    if (seg < 2) {                        // codes: 16 record bytes -> 32 native bytes per thread step
      uint8_t* o = out + 2LL * c * CHUNK;
      for (int j = threadIdx.x * 16; j < CHUNK; j += THREADS * 16) {
        uint4 v = *reinterpret_cast<const uint4*>(src + j);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        uint4 o0, o1;
        uint16_t* p0 = reinterpret_cast<uint16_t*>(&o0);
        uint16_t* p1 = reinterpret_cast<uint16_t*>(&o1);
#pragma unroll
        for (int q = 0; q < 8; q++) { p0[q] = lut[b[q]]; p1[q] = lut[b[8 + q]]; }
        uint4* d = reinterpret_cast<uint4*>(o + 2 * j);
        d[0] = o0; d[1] = o1;
      }
    } else {                              // scale bytes: copy
      uint8_t* o = out + (long long)c * CHUNK;
      for (int j = threadIdx.x * 16; j < CHUNK; j += THREADS * 16)
        *reinterpret_cast<uint4*>(o + j) = *reinterpret_cast<const uint4*>(src + j);
    }
    __syncthreads();
    if (threadIdx.x == 0 && i + STAGES < n) issue(i + STAGES);
  }
}

// stage 1 ring: slot k's record (src[k] >= 0) is read once and written to slot k of dst (the shared deferral
// rows, if late) and to row ring[k] of dst2 (the layer's ring rows, if ring[k] >= 0)
__global__ void __launch_bounds__(THREADS) stage2_kernel(const uint8_t* __restrict__ bank, long long rec_bytes,
    const int* __restrict__ src, const int* __restrict__ ring, Dst dst, Dst dst2, int late, int c13, int c2, int s13,
    int s2, const uint16_t* __restrict__ lut_g) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint64_t bars[STAGES];
  __shared__ uint16_t lut[256];
  __shared__ int active[SLOTS];
  __shared__ int n_active;
  for (int i = threadIdx.x; i < 256; i += THREADS) lut[i] = lut_g[i];
  if (threadIdx.x == 0) {
    int n = 0;
    for (int k = 0; k < SLOTS; k++) if (src[k] >= 0 && (late || ring[k] >= 0)) active[n++] = k;
    n_active = n;
    for (int s = 0; s < STAGES; s++) mbar_init(&bars[s]);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  const int per = c13 + c2 + s13 + s2, total = per * n_active;
  const int n = (int)blockIdx.x < total ? (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto issue = [&](int i) {
    int u = blockIdx.x + i * gridDim.x, k = active[u / per], c = u %% per;
    mbar_expect(&bars[i %% STAGES], CHUNK);
    bulk_g2s(smem + (i %% STAGES) * CHUNK, bank + (long long)src[k] * rec_bytes + (long long)c * CHUNK, CHUNK,
             &bars[i %% STAGES]);
  };
  if (threadIdx.x == 0) for (int i = 0; i < STAGES && i < n; i++) issue(i);
  for (int i = 0; i < n; i++) {
    const int s = i %% STAGES;
    mbar_wait(&bars[s], (i / STAGES) & 1);
    const int u = blockIdx.x + i * gridDim.x, k = active[u / per];
    int c = u %% per;
    const uint8_t* sp = smem + s * CHUNK;
    int seg = 0;
    if (c >= c13) { c -= c13; seg = 1;
      if (c >= c2) { c -= c2; seg = 2;
        if (c >= s13) { c -= s13; seg = 3; } } }
    const int rk = ring[k];
    uint8_t* out = late ? dst.base[seg] + (long long)k * dst.stride[seg] : nullptr;
    uint8_t* out2 = rk >= 0 ? dst2.base[seg] + (long long)rk * dst2.stride[seg] : nullptr;
    if (seg < 2) {
      for (int j = threadIdx.x * 16; j < CHUNK; j += THREADS * 16) {
        uint4 v = *reinterpret_cast<const uint4*>(sp + j);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        uint4 o0, o1;
        uint16_t* p0 = reinterpret_cast<uint16_t*>(&o0);
        uint16_t* p1 = reinterpret_cast<uint16_t*>(&o1);
#pragma unroll
        for (int q = 0; q < 8; q++) { p0[q] = lut[b[q]]; p1[q] = lut[b[8 + q]]; }
        if (out) { uint4* d = reinterpret_cast<uint4*>(out + 2LL * c * CHUNK + 2 * j); d[0] = o0; d[1] = o1; }
        if (out2) { uint4* d = reinterpret_cast<uint4*>(out2 + 2LL * c * CHUNK + 2 * j); d[0] = o0; d[1] = o1; }
      }
    } else {
      for (int j = threadIdx.x * 16; j < CHUNK; j += THREADS * 16) {
        const uint4 v = *reinterpret_cast<const uint4*>(sp + j);
        if (out) *reinterpret_cast<uint4*>(out + (long long)c * CHUNK + j) = v;
        if (out2) *reinterpret_cast<uint4*>(out2 + (long long)c * CHUNK + j) = v;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && i + STAGES < n) issue(i + STAGES);
  }
}

// promotion (residency.py): a hi2 record and a lo2 record of one expert -> its native row (nibble = hi << 2 | lo; the
// scale bytes from lo2, which keeps them unlowered). Plain 16-byte loads (the sources may be host memory: C2C).
__global__ void __launch_bounds__(THREADS) merge_kernel(const uint8_t* __restrict__ hi, const uint8_t* __restrict__ lo,
    Dst dst, long long c13, long long c2, long long s13, long long s2) {
  const long long total16 = (c13 + c2 + s13 + s2) / 16;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total16; i += (long long)gridDim.x * blockDim.x) {
    const long long off = i * 16;
    const uint4 l = *reinterpret_cast<const uint4*>(lo + off);
    if (off < c13 + c2) {
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
        const uint32_t n2 = (((x >> 4) & 3) << 2) | ((y >> 4) & 3), n3 = (((x >> 6) & 3) << 2) | ((y >> 6) & 3);
        uint8_t* ob = q < 8 ? ob0 : ob1;
        ob[2 * (q & 7)] = (uint8_t)(n0 | (n1 << 4));
        ob[2 * (q & 7) + 1] = (uint8_t)(n2 | (n3 << 4));
      }
      uint4* d = reinterpret_cast<uint4*>(off < c13 ? dst.base[0] + 2 * off : dst.base[1] + 2 * (off - c13));
      d[0] = o0; d[1] = o1;
    } else {
      const long long so = off - c13 - c2;
      *reinterpret_cast<uint4*>(so < s13 ? dst.base[2] + so : dst.base[3] + (so - s13)) = l;
    }
  }
}

// the inverse of merge_kernel for the hi2 half: a native row -> its hi2 record (top two bits of every nibble: record
// byte q = lo(n0) | hi(n0) << 2 | lo(n1) << 4 | hi(n1) << 6 for native bytes n0 = 2q, n1 = 2q + 1, lo(b) = bits 3-2,
// hi(b) = bits 7-6; scale bytes lowered by one, zero stays zero). Used to write an evicted expert back to the bank.
__global__ void __launch_bounds__(THREADS) extract_kernel(Dst src, uint8_t* __restrict__ out, long long c13,
    long long c2, long long s13, long long s2) {
  const long long total16 = (c13 + c2 + s13 + s2) / 16;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total16; i += (long long)gridDim.x * blockDim.x) {
    const long long off = i * 16;
    uint4 o;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&o);
    if (off < c13 + c2) {
      const uint8_t* p = off < c13 ? src.base[0] + 2 * off : src.base[1] + 2 * (off - c13);
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
    } else {
      const long long so = off - c13 - c2;
      o = *reinterpret_cast<const uint4*>(so < s13 ? src.base[2] + so : src.base[3] + (so - s13));
#pragma unroll
      for (int q = 0; q < 16; q++) ob[q] = ob[q] ? (uint8_t)(ob[q] - 1) : (uint8_t)0;
    }
    *reinterpret_cast<uint4*>(out + off) = o;
  }
}

void extract(int64_t out, std::vector<int64_t> bases, std::vector<int64_t> sizes, int64_t blocks) {
  Dst d;
  for (int i = 0; i < 4; i++) { d.base[i] = reinterpret_cast<uint8_t*>(bases[i]); d.stride[i] = 0; }
  extract_kernel<<<(int)blocks, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      d, reinterpret_cast<uint8_t*>(out), sizes[0], sizes[1], sizes[2], sizes[3]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge(int64_t hi, int64_t lo, std::vector<int64_t> bases, std::vector<int64_t> sizes, int64_t blocks) {
  Dst d;
  for (int i = 0; i < 4; i++) { d.base[i] = reinterpret_cast<uint8_t*>(bases[i]); d.stride[i] = 0; }
  merge_kernel<<<(int)blocks, THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(hi), reinterpret_cast<const uint8_t*>(lo), d, sizes[0], sizes[1], sizes[2], sizes[3]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void stage(int64_t bank, int64_t rec_bytes, torch::Tensor stage_idx, std::vector<int64_t> bases,
           std::vector<int64_t> strides, std::vector<int64_t> chunks, torch::Tensor lut, int64_t sms) {
  TORCH_CHECK(stage_idx.is_cuda() && stage_idx.scalar_type() == torch::kInt32 && stage_idx.numel() == SLOTS);
  Dst d;
  for (int i = 0; i < 4; i++) { d.base[i] = reinterpret_cast<uint8_t*>(bases[i]); d.stride[i] = strides[i]; }
  static bool attr = false;
  const size_t smem = (size_t)STAGES * CHUNK;
  if (!attr) { cudaFuncSetAttribute(stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)sms); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute lattr[1];
  lattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  lattr[0].val.programmaticStreamSerializationAllowed = 1;
  static const int pdl = getenv("K31_PDL") != nullptr && getenv("K31_PDL")[0] == '1';   // opt-in, as w8a16 / unitk
  cfg.attrs = lattr; cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, stage_kernel, reinterpret_cast<const uint8_t*>(bank), (long long)rec_bytes,
      (const int*)stage_idx.data_ptr<int>(), d, (int)chunks[0], (int)chunks[1], (int)chunks[2], (int)chunks[3],
      reinterpret_cast<const uint16_t*>(lut.data_ptr())));
}

void stage2(int64_t bank, int64_t rec_bytes, torch::Tensor src, torch::Tensor ring, std::vector<int64_t> bases,
            std::vector<int64_t> bases2, std::vector<int64_t> strides, int64_t late, std::vector<int64_t> chunks,
            torch::Tensor lut, int64_t sms) {
  TORCH_CHECK(src.is_cuda() && src.scalar_type() == torch::kInt32 && src.numel() == SLOTS);
  TORCH_CHECK(ring.is_cuda() && ring.scalar_type() == torch::kInt32 && ring.numel() == SLOTS);
  Dst d, d2;
  for (int i = 0; i < 4; i++) {
    d.base[i] = reinterpret_cast<uint8_t*>(bases[i]); d.stride[i] = strides[i];
    d2.base[i] = reinterpret_cast<uint8_t*>(bases2[i]); d2.stride[i] = strides[i];
  }
  static bool attr = false;
  const size_t smem = (size_t)STAGES * CHUNK;
  if (!attr) { cudaFuncSetAttribute(stage2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  stage2_kernel<<<(int)sms, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(bank), rec_bytes, src.data_ptr<int>(), ring.data_ptr<int>(), d, d2, (int)late,
      (int)chunks[0], (int)chunks[1], (int)chunks[2], (int)chunks[3], reinterpret_cast<const uint16_t*>(lut.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stage", &stage); m.def("stage2", &stage2); m.def("merge", &merge); m.def("extract", &extract);
}
"""
_MOD = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
        _MOD.append(load_inline(name="k31_stager", cpp_sources="", cuda_sources=_SRC % dict(CHUNK=CHUNK, STAGES=STAGES, SLOTS=SLOTS),
                                extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_103a,code=sm_103a"],
                                verbose=False))
    return _MOD[0]


def _lut16(t: torch.Tensor, device) -> torch.Tensor:
    lo = (t & 0xFF).to(torch.uint8)
    hi = ((t >> 8) & 0xFF).to(torch.uint8)
    return torch.stack((lo, hi), dim=-1).reshape(-1).contiguous().to(device)    # 512 bytes, uint16 little-endian


class Stager:
    """Per layer: the destination slot rows (rows first..first+SLOTS-1 of the module's four kernel tensors)."""

    def __init__(self, mod, first_slot: int, layout: dict, sms: int = 24):
        self.m = _module()
        self.sms = sms
        dev = mod.w13_weight.device
        self.lut = _lut16_cached(dev)
        ts = [mod.w13_weight, mod.w2_weight, mod.w13_weight_scale, mod.w2_weight_scale]
        self.bases, self.strides = [], []
        for t in ts:
            row = t[0].numel() * t.element_size()
            assert t.stride(0) * t.element_size() == row, "rows must be dense blocks"
            self.bases.append(t.data_ptr() + first_slot * row)        # data_ptr of a transposed view = its storage start
            self.strides.append(row)
        nb = layout["native_bytes"]
        self.chunks = [nb["w13"] // 2 // CHUNK, nb["w2"] // 2 // CHUNK, nb["w13s"] // CHUNK, nb["w2s"] // CHUNK]
        assert [c * CHUNK for c in self.chunks] == [nb["w13"] // 2, nb["w2"] // 2, nb["w13s"], nb["w2s"]], \
            "record segments must be whole chunks"
        self.rec_bytes = layout["record_bytes"]

    def __call__(self, bank_ptr: int, stage_idx: torch.Tensor) -> None:
        self.m.stage(bank_ptr, self.rec_bytes, stage_idx, self.bases, self.strides, self.chunks, self.lut, self.sms)


class RingStager(Stager):
    """Stage 1 ring: per layer, the shared deferral rows (first_slot..) and the layer's ring rows (ring_base..) of the
    same module tensors; one read per record, written to both (or only to the ring row when the layer does not
    compute late)."""

    def __init__(self, mod, first_slot: int, ring_base: int, layout: dict, sms: int = 24):
        super().__init__(mod, first_slot, layout, sms)
        ts = [mod.w13_weight, mod.w2_weight, mod.w13_weight_scale, mod.w2_weight_scale]
        self.bases2 = [t.data_ptr() + ring_base * (t[0].numel() * t.element_size()) for t in ts]

    def __call__(self, bank_ptr: int, src: torch.Tensor, ring: torch.Tensor, late: bool) -> None:
        self.m.stage2(bank_ptr, self.rec_bytes, src, ring, self.bases, self.bases2, self.strides, int(late),
                      self.chunks, self.lut, self.sms)


def merge(hi_ptr: int, lo_ptr: int, mod, row: int, layout: dict, blocks: int = 32) -> None:
    """Rebuild expert row `row` of the module's four kernel tensors from its hi2 and lo2 records (device-visible
    addresses; host memory is fine), on the current stream."""
    ts = [mod.w13_weight, mod.w2_weight, mod.w13_weight_scale, mod.w2_weight_scale]
    bases = []
    for t in ts:
        rb = t[0].numel() * t.element_size()
        bases.append(t.data_ptr() + row * rb)
    nb = layout["native_bytes"]
    _module().merge(hi_ptr, lo_ptr, bases, [nb["w13"] // 2, nb["w2"] // 2, nb["w13s"], nb["w2s"]], blocks)


def _row_bytes(t: torch.Tensor, row: int) -> torch.Tensor:
    """Row `row` of a kernel tensor as its bytes in memory order (rows are dense blocks; inner layouts as stored)."""
    per = t[0].numel()
    return torch.as_strided(t, (t.shape[0], per), (per, 1))[row].view(torch.uint8)


def extract_hi2(mod, row: int, layout: dict, out: torch.Tensor, blocks: int = 152) -> None:
    """extract_record(.., low=False) as one kernel (the inverse of merge for the hi2 half): ~27 MB of HBM traffic."""
    ts = [mod.w13_weight, mod.w2_weight, mod.w13_weight_scale, mod.w2_weight_scale]
    bases = [t.data_ptr() + row * (t[0].numel() * t.element_size()) for t in ts]
    nb = layout["native_bytes"]
    _module().extract(out.data_ptr(), bases, [nb["w13"] // 2, nb["w2"] // 2, nb["w13s"], nb["w2s"]], blocks)


def extract_record(mod, row: int, layout: dict, out: torch.Tensor, low: bool = False) -> None:
    """The inverse of the stager / merge (store/build_store.py `hi2-k1`): native row `row` of the module's four kernel
    tensors -> its hi2 record (the top two bits of every nibble, scale bytes lowered by one) or with `low` its lo2
    record (the bottom two bits, scale bytes as they are), written into out[:record_bytes] on the current stream.
    Used when a promotion evicts a native row: the evicted expert's 2-bit record goes back to the Grace bank."""
    codes = torch.cat((_row_bytes(mod.w13_weight, row), _row_bytes(mod.w2_weight, row)))
    if low:
        lo, hi = codes & 0x03, (codes >> 4) & 0x03
    else:
        lo, hi = (codes & 0x0F) >> 2, codes >> 6
    b = (lo | (hi << 2)).view(-1, 2)
    cb = layout["codes_bytes"]
    out[:cb] = b[:, 0] | (b[:, 1] << 4)
    sf = torch.cat((_row_bytes(mod.w13_weight_scale, row), _row_bytes(mod.w2_weight_scale, row)))
    out[cb:cb + sf.numel()] = sf if low else torch.where(sf > 0, sf - 1, sf)


_LUTS = {}


def _lut16_cached(dev):
    if dev not in _LUTS:
        t = torch.zeros(256, dtype=torch.int64)
        for b in range(256):
            v = 0
            for j in range(4):
                v |= ((((b >> (2 * j)) & 3) << 2) | 3) << (4 * j)
            t[b] = v
        _LUTS[dev] = _lut16(t, dev)
    return _LUTS[dev]
