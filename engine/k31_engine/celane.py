"""The copy-engine lane of the prompt path: K31_PROMPT_CELANE=<share> (0 or unset = off, the default).

A prompt chunk streams its layer's bank records (2-bit, pinned Grace memory) into the 128 shared staging rows in
rounds (prompt_path.py); the stager (stager.py) reads them with TMA bulk copies on 24 SMs at ~287-306 GB/s of record
bytes. The lane moves a share of every round with the GPU's copy engines instead: cudaMemcpyAsync of the raw records
into an HBM scratch on its own copy stream(s), then an expand kernel turns them into native rows with the stager's
arithmetic (the same 256-entry table, the same segment order: byte-exact). The copies use no SMs, so they also run
while MegaMoE computes the previous round, and beside the stager's own reads (whether the link then carries more than
the stager alone is experiment E1: tests/bench_celane.py).

A staging unit (a round of 128 rows; a half-round of 64 rows with K31_PROMPT_AHEAD=1) holds m records, places
p0 .. p0 + m - 1 in rows q0 .. q0 + m - 1:
  - the lane takes the unit's last l = floor(share x m + 1/2) records, the stager the first h = m - l, as before;
  - copies of unit u: on the copy stream(s), into scratch buffer u % BUFS once the expand of unit u - BUFS has read it;
  - expand of unit u: on the stream that stages (main, or the stage-ahead stream), after the stager's launches of the
    unit, waiting for the unit's copies;
  - the unit's MegaMoE call then reads the same rows as without the lane: its output is bit-identical.
The host needs the kept records' bank slots to address the copies: the layer's one host sync (n_keep) becomes one copy
of [n_keep, kept] (host_kept). The copy streams first wait for the main stream (the plan, and the residency's bank
writes joined into it), and every copy of a layer completes before that layer's last expand on the stream that
stages, so the lane reads the bank inside the window in which the stager reads it.
Knobs, read at every prompt chunk through flags.py (the environment, or the control file a running server re-reads
between steps):
  K31_PROMPT_CELANE_LANES   copy streams (default 1); a unit's lane records are split between them in two halves
  K31_PROMPT_CELANE_BUFS    scratch buffers (default 1); 2 lets the copies run a whole unit ahead (twice the HBM)
  K31_PROMPT_CELANE_XSIDE=1 (rounds without K31_PROMPT_AHEAD only) the expand on its own stream, beside the stager's
                            launches of the same round instead of after them
  K31_PROMPT_CELANE_XBLOCKS expand kernel blocks (default 608, four per SM)
  K31_PROMPT_CELANE_CHECK=n the first n lane units of each layer since the start (a runtime flag: raise it to check
                            more units later, e.g. under mixed traffic): every lane row compared with the same
                            record staged by the stager into the same row, byte for byte (logged; a difference
                            raises)
  K31_PROMPT_SPEC=1         (rounds without K31_PROMPT_AHEAD only; off by default) the scratch as a pool with copies
                            ahead of the plan: while a layer's dense phase runs, the next layer's first lane records
                            cross the link (class Pool). Same rows, same MegaMoE calls: bit-identical output
HBM: BUFS x floor(share x 128 + 1/2) records of 9.29 MB (share 0.25: 297 MB per buffer), allocated at the first chunk
that uses the lane. A change of the share, LANES or BUFS rebuilds the lane (after a device sync); switching the lane
off (share 0) frees it (prompt_path.forward_prompt calls release()).
"""
import os
import sys

import torch

from . import flags
from .stager import _lut16_cached

UNIT = 128                                   # rows of the largest staging unit (a round): prompt_path.STAGE_ROWS
_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <vector>

constexpr int THREADS = 256, UNROLL = 4;

struct Dst { uint8_t* base[4]; long long stride[4]; };

// Raw records in HBM (the lane's scratch, record j at src + j * rec_bytes) -> native rows row0 + j of the four kernel
// tensors, the stager's arithmetic: record bytes [0, c13) and [c13, c13 + c2) are the w13 and w2 codes, each byte
// expanded through the 256-entry table into two native bytes (little-endian: the table entry's low byte first); the
// next s13 and s2 bytes are the w13 and w2 scale bytes, copied. blockIdx.y = the record; blocks along x stride over
// its 16-byte units, UNROLL loads in flight per thread. Segment ends are multiples of 16 KiB (stager.py asserts it),
// so the 512 bytes of a warp never straddle two segments.
__global__ void __launch_bounds__(THREADS) expand_kernel(const uint8_t* __restrict__ src, long long rec_bytes, Dst dst,
    long long row0, long long c13, long long c2, long long s13, long long s2, const uint16_t* __restrict__ lut_g) {
  __shared__ uint16_t lut[256];
  for (int i = threadIdx.x; i < 256; i += THREADS) lut[i] = lut_g[i];
  __syncthreads();
  const long long row = row0 + blockIdx.y;
  const uint4* rec = reinterpret_cast<const uint4*>(src + (long long)blockIdx.y * rec_bytes);
  const long long codes = c13 + c2, n16 = (codes + s13 + s2) >> 4, step = (long long)gridDim.x * THREADS;
  uint8_t* const o13 = dst.base[0] + row * dst.stride[0];
  uint8_t* const o2 = dst.base[1] + row * dst.stride[1];
  uint8_t* const os13 = dst.base[2] + row * dst.stride[2];
  uint8_t* const os2 = dst.base[3] + row * dst.stride[3];
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
      if (off < codes) {                     // codes: 16 record bytes -> 32 native bytes, as stage_kernel
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v[q]);
        uint4 w0, w1;
        uint16_t* p0 = reinterpret_cast<uint16_t*>(&w0);
        uint16_t* p1 = reinterpret_cast<uint16_t*>(&w1);
#pragma unroll
        for (int k = 0; k < 8; k++) { p0[k] = lut[b[k]]; p1[k] = lut[b[8 + k]]; }
        uint4* d = reinterpret_cast<uint4*>(off < c13 ? o13 + 2 * off : o2 + 2 * (off - c13));
        d[0] = w0; d[1] = w1;
      } else {                               // scale bytes: copy
        const long long so = off - codes;
        *reinterpret_cast<uint4*>(so < s13 ? os13 + so : os2 + (so - s13)) = v[q];
      }
    }
  }
}

void expand(int64_t src, int64_t rec_bytes, int64_t n, std::vector<int64_t> bases, std::vector<int64_t> strides,
            int64_t row0, std::vector<int64_t> segs, torch::Tensor lut, int64_t blocks) {
  if (n <= 0) return;
  TORCH_CHECK(bases.size() == 4 && strides.size() == 4 && segs.size() == 4);
  TORCH_CHECK(lut.is_cuda() && lut.numel() == 512, "the stager's table: 256 x uint16 as 512 bytes");
  TORCH_CHECK(src % 16 == 0 && rec_bytes % 16 == 0, "records must be 16-byte aligned");
  TORCH_CHECK(n <= 65535, "at most 65535 records per launch");
  for (int i = 0; i < 4; i++) {
    TORCH_CHECK(segs[i] % 16 == 0 && strides[i] % 16 == 0 && bases[i] % 16 == 0, "16-byte units");
  }
  TORCH_CHECK(segs[0] + segs[1] + segs[2] + segs[3] <= rec_bytes, "segments exceed the record");
  Dst d;
  for (int i = 0; i < 4; i++) { d.base[i] = reinterpret_cast<uint8_t*>(bases[i]); d.stride[i] = strides[i]; }
  const int64_t bx = std::max<int64_t>(1, (blocks + n - 1) / n);
  expand_kernel<<<dim3((unsigned)bx, (unsigned)n), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(src), (long long)rec_bytes, d, (long long)row0, (long long)segs[0],
      (long long)segs[1], (long long)segs[2], (long long)segs[3], reinterpret_cast<const uint16_t*>(lut.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// expand_kernel with a table: record j of the launch is read at scratch slot map.src[j] (src + map.src[j] * rec_bytes)
// and written to row map.dst[j]. The pool (K31_PROMPT_SPEC) expands a unit whose records sit anywhere in the scratch:
// copied ahead of the layer's plan, or into whatever slots were free. Same arithmetic, same bytes per row.
struct Map { uint16_t src[128]; uint16_t dst[128]; };

__global__ void __launch_bounds__(THREADS) expand_map_kernel(const uint8_t* __restrict__ src, long long rec_bytes,
    Dst dst, Map map, long long c13, long long c2, long long s13, long long s2, const uint16_t* __restrict__ lut_g) {
  __shared__ uint16_t lut[256];
  for (int i = threadIdx.x; i < 256; i += THREADS) lut[i] = lut_g[i];
  __syncthreads();
  const long long row = map.dst[blockIdx.y];
  const uint4* rec = reinterpret_cast<const uint4*>(src + (long long)map.src[blockIdx.y] * rec_bytes);
  const long long codes = c13 + c2, n16 = (codes + s13 + s2) >> 4, step = (long long)gridDim.x * THREADS;
  uint8_t* const o13 = dst.base[0] + row * dst.stride[0];
  uint8_t* const o2 = dst.base[1] + row * dst.stride[1];
  uint8_t* const os13 = dst.base[2] + row * dst.stride[2];
  uint8_t* const os2 = dst.base[3] + row * dst.stride[3];
  for (long long u0 = (long long)blockIdx.x * THREADS + threadIdx.x; u0 < n16; u0 += UNROLL * step) {
    uint4 v[UNROLL];
#pragma unroll
    for (int q = 0; q < UNROLL; q++) {
      const long long u = u0 + q * step;
      v[q] = u < n16 ? __ldcs(rec + u) : make_uint4(0, 0, 0, 0);
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
      } else {
        const long long so = off - codes;
        *reinterpret_cast<uint4*>(so < s13 ? os13 + so : os2 + (so - s13)) = v[q];
      }
    }
  }
}

void expand_map(int64_t src, int64_t rec_bytes, std::vector<int64_t> src_idx, std::vector<int64_t> dst_row,
                std::vector<int64_t> bases, std::vector<int64_t> strides, std::vector<int64_t> segs,
                torch::Tensor lut, int64_t blocks) {
  const int64_t n = (int64_t)src_idx.size();
  if (n <= 0) return;
  TORCH_CHECK(n <= 128 && (int64_t)dst_row.size() == n, "at most 128 records per launch, one row each");
  TORCH_CHECK(bases.size() == 4 && strides.size() == 4 && segs.size() == 4);
  TORCH_CHECK(lut.is_cuda() && lut.numel() == 512, "the stager's table: 256 x uint16 as 512 bytes");
  TORCH_CHECK(src % 16 == 0 && rec_bytes % 16 == 0, "records must be 16-byte aligned");
  for (int i = 0; i < 4; i++) {
    TORCH_CHECK(segs[i] % 16 == 0 && strides[i] % 16 == 0 && bases[i] % 16 == 0, "16-byte units");
  }
  TORCH_CHECK(segs[0] + segs[1] + segs[2] + segs[3] <= rec_bytes, "segments exceed the record");
  Map m;
  for (int64_t i = 0; i < n; i++) {
    TORCH_CHECK(src_idx[i] >= 0 && src_idx[i] < 65536 && dst_row[i] >= 0 && dst_row[i] < 65536, "table entry");
    m.src[i] = (uint16_t)src_idx[i]; m.dst[i] = (uint16_t)dst_row[i];
  }
  for (int64_t i = n; i < 128; i++) { m.src[i] = 0; m.dst[i] = 0; }
  Dst d;
  for (int i = 0; i < 4; i++) { d.base[i] = reinterpret_cast<uint8_t*>(bases[i]); d.stride[i] = strides[i]; }
  const int64_t bx = std::max<int64_t>(1, (blocks + n - 1) / n);
  expand_map_kernel<<<dim3((unsigned)bx, (unsigned)n), THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(src), (long long)rec_bytes, d, m, (long long)segs[0], (long long)segs[1],
      (long long)segs[2], (long long)segs[3], reinterpret_cast<const uint16_t*>(lut.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One cudaMemcpyAsync on the current stream per (dst offset, src offset, bytes) triple: the copy engines. src is
// pinned host memory (registered, UVA), dst device memory. Returns the number of copies.
int64_t copy(int64_t dst, int64_t src, std::vector<int64_t> triples) {
  TORCH_CHECK(triples.size() % 3 == 0);
  const cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  for (size_t i = 0; i < triples.size(); i += 3)
    C10_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dst + triples[i]),
                                   reinterpret_cast<const void*>(src + triples[i + 1]), (size_t)triples[i + 2],
                                   cudaMemcpyDefault, s));
  return (int64_t)(triples.size() / 3);
}

// cudaDeviceGetAttribute on the current device (the probe reports the copy engine count, attribute 40)
int64_t attr(int64_t a) {
  int dev = 0, v = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&v, (cudaDeviceAttr)a, dev));
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expand", &expand); m.def("expand_map", &expand_map); m.def("copy", &copy); m.def("attr", &attr);
}
"""
_MOD = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline
        from .stager2 import ARCH                 # K31_CUDA_ARCH (default 103a, the GB300)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{ARCH[:-2]}.{ARCH[-2:]}")
        _MOD.append(load_inline(name="k31_celane" + ("" if ARCH == "103a" else "_" + ARCH), cpp_sources="", cuda_sources=_SRC,
                                extra_cuda_cflags=["-O3", "-std=c++20", f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}"],
                                verbose=False))
    return _MOD[0]


# ------------------------------------------------------------------------------------------- plan (pure Python)
def take(share: float, m: int) -> int:
    """Records of a unit of m that the lane stages (the unit's last ones): floor(share x m + 1/2), within [0, m]."""
    return max(0, min(m, int(share * m + 0.5)))


def parts(n: int, lanes: int) -> list:
    """[a, b) ranges of n records, one per copy stream: contiguous, sizes differing by at most one."""
    return [(n * k // lanes, n * (k + 1) // lanes) for k in range(lanes)]


def runs(slots, rec_bytes: int, dst0: int = 0) -> list:
    """Bank slots in place order -> flat (dst offset, src offset, bytes) triples, one per run of consecutive slots:
    record i of the list lands at dst0 + i x rec_bytes of the scratch, slot s is read at s x rec_bytes of the bank."""
    out = []
    i, n = 0, len(slots)
    while i < n:
        j = i
        while j + 1 < n and slots[j + 1] == slots[j] + 1:
            j += 1
        out += [dst0 + i * rec_bytes, slots[i] * rec_bytes, (j - i + 1) * rec_bytes]
        i = j + 1
    return out


def segs_of(layout: dict) -> list:
    """Bytes of a record's four segments in order: w13 codes, w2 codes, w13 scales, w2 scales."""
    nb = layout["native_bytes"]
    return [nb["w13"] // 2, nb["w2"] // 2, nb["w13s"], nb["w2s"]]


def host_kept(n_keep: torch.Tensor, kept: torch.Tensor) -> list:
    """The layer's one host sync with the lane on: n_keep and the kept bank slots in one device -> host copy."""
    t = torch.cat((n_keep.reshape(1).to(kept.dtype), kept)).cpu()
    return t[1:1 + int(t[0])].tolist()


def bank_base(st) -> int:
    """The address the copies read the layer's bank from: its pinned host tensor (the stager reads the same memory
    through its device view, st.bank_ptr; under UVA both are one address)."""
    t = getattr(st, "bank_cpu", None)
    return t.data_ptr() if t is not None else int(st.bank_ptr)


_DST: dict = {}


def expand_map(lane, base: int, slots: list, rows: list, mod) -> None:
    """Records at base + slot x R (device memory) -> the module's rows `rows`, in the record format of lane.layout.
    hi2-k1 records into plain rows: celane's map kernel, one launch. hi2-k2 records or 6-bit scale rows: stager2's
    expand, one launch per run of consecutive slots and rows (the pool hands out free slots in order, so runs are
    long). The pool's copies ahead land here (Pool.land)."""
    from . import stager2
    if stager2.active(lane.layout, mod):
        i, n = 0, len(slots)
        while i < n:
            j = i + 1
            while j < n and slots[j] == slots[i] + (j - i) and rows[j] == rows[i] + (j - i):
                j += 1
            stager2.expand(base + slots[i] * lane.R, lane.R, j - i, mod, rows[i], lane.layout, lane.xblocks)
            i = j
        return
    bases, strides = dst_of(mod)
    lane.m.expand_map(base, lane.R, list(slots), list(rows), bases, strides, lane.segs, lane.lut, lane.xblocks)


def dst_of(mod):
    """Row 0 of the module's four kernel tensors and their row strides in bytes (the shared staging rows are rows
    0 .. 127), computed as stager.Stager does; cached by the four tensors' addresses and row strides."""
    ts = (mod.w13_weight, mod.w2_weight, mod.w13_weight_scale, mod.w2_weight_scale)
    key = tuple((t.data_ptr(), t.stride(0), t.element_size()) for t in ts)
    hit = _DST.get(key)
    if hit is None:
        bases, strides = [], []
        for t in ts:
            row = t[0].numel() * t.element_size()
            assert t.stride(0) * t.element_size() == row, "rows must be dense blocks"
            bases.append(t.data_ptr())               # data_ptr of a transposed view = its storage start
            strides.append(row)
        hit = _DST[key] = (bases, strides)
    return hit


# ------------------------------------------------------------------------------------------------ the lane
class Lane:
    """Copy streams, the HBM scratch (bufs buffers of cap records) and the per-unit protocol: begin() once per layer on
    the main stream, issue(u, ..) for unit u's copies, land(u, ..) on the stream that stages for its expand."""

    def __init__(self, share: float, layout: dict, dev, lanes: int = None, bufs: int = None, xblocks: int = None):
        lanes, bufs, xblocks = _knobs(lanes, bufs, xblocks)
        assert 0.0 < share <= 1.0, f"K31_PROMPT_CELANE share {share} must be in (0, 1]"
        self.share, self.lanes, self.bufs, self.xblocks = share, lanes, bufs, xblocks
        self.R = int(layout["record_bytes"])
        self.layout = layout
        self.segs = segs_of(layout)
        self.cap = take(share, UNIT)                  # most records any unit (<= 128 rows) gives the lane
        self.dev = torch.device(dev)
        self.m = _module()
        self.lut = _lut16_cached(self.dev)
        self.scratch = torch.empty((bufs, max(self.cap, 1), self.R), dtype=torch.uint8, device=self.dev)
        self.streams = [torch.cuda.Stream(self.dev) for _ in range(lanes)]
        for s in self.streams if self.scratch.is_cuda else ():   # freed scratch waits for the copies queued on them
            self.scratch.record_stream(s)
        self.free = [None] * bufs                     # per buffer: the event after the expand that last read it
        self.copied: dict = {}                        # unit -> (copy events, records)
        self.n_records = self.n_copies = self.n_units = 0

    def scratch_bytes(self) -> int:
        return self.scratch.numel()

    def begin(self) -> None:
        """A layer's first unit: the copy streams wait for everything enqueued on the current stream so far (the plan,
        and the residency's bank writes, which the main stream has joined)."""
        cur = torch.cuda.current_stream(self.dev)
        for s in self.streams:
            s.wait_stream(cur)

    def issue(self, u: int, src: int, slots: list) -> None:
        """Copies of unit u's lane records (bank slots in place order, read at src + slot x R) into buffer u % bufs,
        after the expand that last read that buffer."""
        evs = []
        if slots:
            b = u % self.bufs
            base = self.scratch[b].data_ptr()
            for s, (a, e) in zip(self.streams, parts(len(slots), self.lanes)):
                if a == e:
                    continue
                if self.free[b] is not None:
                    s.wait_event(self.free[b])
                with torch.cuda.stream(s):
                    self.n_copies += self.m.copy(base, src, runs(slots[a:e], self.R, a * self.R))
                    ev = torch.cuda.Event()
                    ev.record(s)
                evs.append(ev)
            self.n_records += len(slots)
        self.copied[u] = (evs, len(slots))

    def land(self, u: int, mod, row0: int) -> int:
        """On the current stream: wait for unit u's copies and expand its records into rows row0 .. row0 + n - 1 of
        the module's four kernel tensors. Returns n."""
        evs, n = self.copied.pop(u)
        if n == 0:
            return 0
        cur = torch.cuda.current_stream(self.dev)
        for ev in evs:
            cur.wait_event(ev)
        b = u % self.bufs
        from . import stager2
        if stager2.active(self.layout, mod):          # hi2-k2 records or 6-bit scale rows: stager2's expand
            stager2.expand(self.scratch[b].data_ptr(), self.R, n, mod, row0, self.layout, self.xblocks)
        else:
            bases, strides = dst_of(mod)
            self.m.expand(self.scratch[b].data_ptr(), self.R, n, bases, strides, row0, self.segs, self.lut, self.xblocks)
        ev = torch.cuda.Event()
        ev.record(cur)
        self.free[b] = ev
        self.n_units += 1
        return n


def _knobs(lanes=None, bufs=None, xblocks=None):
    """(copy streams, scratch buffers, expand blocks): the arguments, else the runtime flags; each at least 1."""
    lanes = flags.num("K31_PROMPT_CELANE_LANES", 1) if lanes is None else lanes
    bufs = flags.num("K31_PROMPT_CELANE_BUFS", 1) if bufs is None else bufs
    xblocks = flags.num("K31_PROMPT_CELANE_XBLOCKS", 608) if xblocks is None else xblocks
    return max(1, int(lanes)), max(1, int(bufs)), max(1, int(xblocks))


_LANE: list = []
_FAILED: list = [None]                       # the (share, lanes, bufs) whose scratch did not fit, until they change


def get(st, dev, share: float):
    """The process's lane (one for all layers: their prompt chunks stage one layer after another), made at the first
    prompt chunk that uses it and rebuilt when the share, the copy streams or the buffers change at runtime (after a
    device sync: the old lane's copies and expands are done before its scratch is freed). None when its scratch does
    not fit in HBM (logged once per setting): the stager then stages every record."""
    lanes, bufs, xblocks = _knobs()
    cfg = (share, lanes, bufs)
    old = _LANE[0] if _LANE else None
    if old is None or (old.share, old.lanes, old.bufs) != cfg:
        if old is not None:
            torch.cuda.synchronize()
            _LANE.clear()
            del old
        if _FAILED[0] == cfg:
            return None
        try:
            lane = Lane(share, st.layout, dev, lanes=lanes, bufs=bufs, xblocks=xblocks)
        except torch.cuda.OutOfMemoryError:
            _FAILED[0] = cfg
            mib = bufs * take(share, UNIT) * int(st.layout["record_bytes"]) / 2**20
            _LOG(f"prompt path copy-engine lane: no room for its scratch ({bufs} x {take(share, UNIT)} records, "
                 f"{mib:.0f} MiB of HBM): the stager stages every record")
            return None
        _FAILED[0] = None
        _LANE[:] = [lane]
        _LOG(f"prompt path copy-engine lane: share {share} of every staging unit, {lane.lanes} copy stream(s), "
             f"{lane.bufs} scratch buffer(s) of {lane.cap} records ({lane.scratch_bytes() / 2**20:.0f} MiB of HBM)"
             + (", expand beside the stager" if flags.on("K31_PROMPT_CELANE_XSIDE") else "")
             + (", byte check on" if flags.num("K31_PROMPT_CELANE_CHECK", 0) > 0 else ""))
    _LANE[0].xblocks = xblocks
    return _LANE[0]


def release() -> None:
    """The lane switched off at runtime: wait for its last copies and expands, then drop it (its scratch goes back to
    torch's cache)."""
    if _LANE:
        torch.cuda.synchronize()
        _LANE.clear()
        _LOG("prompt path copy-engine lane: off")


# ------------------------------------------------------------------------------------------------ the rounds
def stage_head(st, kept: torch.Tensor, p0: int, h: int, g0: int) -> None:
    """The stager's part of a unit: places p0 .. p0 + h - 1 into rows 16 g0 .. 16 g0 + h - 1, one launch per slot
    group that holds any of them (later groups hold only the lane's rows and are not launched)."""
    if h <= 0:
        return
    from .prompt_path import SLOTS
    ng = -(-h // SLOTS)
    lst = kept[p0:p0 + h]
    if lst.numel() < ng * SLOTS:
        lst = torch.cat([lst, torch.full((ng * SLOTS - lst.numel(),), -1, dtype=lst.dtype, device=lst.device)])
    lst = lst.to(torch.int32)
    for g in range(ng):
        st.stagers[g0 + g](st.bank_ptr, lst[SLOTS * g:SLOTS * (g + 1)].contiguous())


_CHECKED: dict = {}


def _row_bytes(mod, row: int) -> torch.Tensor:
    from .stager import _row_bytes as rb
    from .stager2 import coded
    c = coded(mod)                                   # 6-bit scale rows: the coded banks hold the scale bytes
    sc = [rb(mod.w13_weight_scale, row), rb(mod.w2_weight_scale, row)] if c is None else [c[0][row], c[1][row]]
    return torch.cat([rb(mod.w13_weight, row), rb(mod.w2_weight, row)] + sc)


def check_rows(st, kept: torch.Tensor, p0: int, q0: int, a: int, b: int) -> int:
    """Lane rows q0 + a .. q0 + b - 1 (places p0 + a ..) against the same records staged again by the stager into the
    same rows, one row at a time, byte for byte (a correct lane leaves the rows as they were). Returns the number of
    rows that differed. Host syncs: a check mode only."""
    from .prompt_path import SLOTS
    bad = 0
    for k in range(a, b):
        row = q0 + k
        before = _row_bytes(st.mod, row)                 # a copy (torch.cat)
        lst = torch.full((SLOTS,), -1, dtype=torch.int32, device=kept.device)
        lst[row % SLOTS] = kept[p0 + k].to(torch.int32)
        st.stagers[row // SLOTS](st.bank_ptr, lst)
        bad += 0 if torch.equal(before, _row_bytes(st.mod, row)) else 1
    return bad


def _check(st, kept, p0, q0, h, m) -> None:
    budget = flags.num("K31_PROMPT_CELANE_CHECK", 0)     # lane units checked per layer since the start (1: the first)
    if m <= h or budget <= 0 or _CHECKED.get(st.idx, 0) >= budget:
        return
    _CHECKED[st.idx] = _CHECKED.get(st.idx, 0) + 1
    bad = check_rows(st, kept, p0, q0, h, m)
    _LOG(f"celane check layer {st.idx}: {m - h - bad} of {m - h} lane rows equal to the stager's")
    if bad:
        raise RuntimeError(f"K31 celane: {bad} lane rows of layer {st.idx} differ from the stager's")


def rounds(orig, mod, st, x, w, shared_experts, kept, kept_h, place, nat, n_kept, dtype, share, ahead=False,
           lane=None):
    """forward_prompt's staging rounds with the lane: the same rounds (or half-rounds with K31_PROMPT_AHEAD), the same
    MegaMoE calls on the same rows; the last share of every unit's records arrives through the copy engines."""
    lane = lane if lane is not None else get(st, x.device, share)
    assert lane is not None, "K31 celane: no lane (its scratch did not fit); forward_prompt then stages by the stager"
    if ahead:
        return _rounds_ahead(orig, mod, st, x, w, shared_experts, kept, kept_h, place, nat, n_kept, dtype, lane)
    from . import prompt_path as pp
    R = pp.STAGE_ROWS
    n_rounds = max(1, -(-n_kept // R))
    rnd = torch.div(place, R, rounding_mode="floor")
    pos = place - rnd * R
    src = bank_base(st)

    def unit(r):
        m = max(0, min(R, n_kept - r * R))
        return m, m - take(lane.share, m)

    def issue(r):
        if r < n_rounds:
            m, h = unit(r)
            lane.issue(r, src, kept_h[r * R + h:r * R + m])

    lane.begin()
    for r in range(min(lane.bufs, n_rounds)):
        issue(r)
    xside = flags.on("K31_PROMPT_CELANE_XSIDE")
    main = torch.cuda.current_stream() if xside else None
    xs = _xside(main) if xside else None
    y = None
    for r in range(n_rounds):
        m, h = unit(r)
        if xs is not None:                            # the expand beside the stager's launches, both after call r - 1
            xs.wait_stream(main)
            with torch.cuda.stream(xs):
                lane.land(r, st.mod, h)
                ev = torch.cuda.Event()
                ev.record(xs)
            stage_head(st, kept, r * R, h, 0)
            main.wait_event(ev)
        else:
            stage_head(st, kept, r * R, h, 0)
            lane.land(r, st.mod, h)
        _check(st, kept, r * R, 0, h, m)
        issue(r + lane.bufs)                          # into the buffer this round's expand has just read
        ids = torch.where((place >= 0) & (rnd == r), pos, nat if r == 0 else torch.full_like(nat, -1))
        out = pp._megamoe(orig, mod, x, w, ids, shared_experts if r == 0 else None, dtype)
        y = out if y is None else y + out             # out of place: a round's output may be MegaMoE's own buffer
        if pp.pprof.on():
            pp.pprof.round_mark(st.idx, r)
    return y


# ------------------------------------------------------------------------------------------------ the pool
class Pool:
    """K31_PROMPT_SPEC=1: the lane's scratch as one pool of N = bufs x cap record slots, so that copies can run ahead
    of a layer's plan.

    Without it the link idles through every layer's dense phase (attention, norms, dense GEMMs, the residency's
    tick): the lane's copies of a layer are issued after that layer's plan. With the pool, once a layer's last units
    are queued, the next layer's first lane records are copied into the slots those units free, on the guess that the
    next plan stages every record that is mapped now, in slot order. When that plan arrives, a record copied ahead is
    used only if (1) it is a lane record of one of the layer's first `bufs` units and (2) the residency has neither
    unmapped nor remapped its bank slot since the copy was issued (owner and generation as they were). Every other
    lane record is copied as before, into slots whose content is not needed. The staging order, the rows and the
    MegaMoE calls are those of the lane without the pool: only where a record's bytes wait changes, so a layer's
    output is bit-identical.

    A slot's life: a copy fills it (its `ready` event), an expand reads it (its `reader` event), then the next copy
    into it waits for both. A unit's expand names each record's slot and row in a table (expand_map). A slot named by
    a unit that has not been expanded yet is `live` and never given to a copy."""

    def __init__(self, lane):
        self.lane = lane
        self.N = lane.bufs * max(lane.cap, 1)
        self.R = lane.R
        self.base = lane.scratch.data_ptr()          # slot p at base + p x R: the [bufs, cap, R] scratch is one block
        self.ready = [None] * self.N
        self.reader = [None] * self.N
        self.live = set()
        self.spec = None
        self.last = {}                               # layer idx -> the bank slots its last plan staged, in order
        self.n = {"layers": 0, "lane": 0, "ahead": 0, "used": 0, "stale": 0}
        self.logged = 0

    # -- copies
    def _copy(self, src: int, pairs: list) -> None:
        """Copies of (pool slot, bank slot) pairs, in order, split between the copy streams as the lane splits a
        unit; one cudaMemcpyAsync per run of consecutive pool slots and bank slots. Each stream first waits for the
        events of the slots it writes: the expand that last read them, a copy still filling them."""
        lane = self.lane
        for s, (a, e) in zip(lane.streams, parts(len(pairs), lane.lanes)):
            if a == e:
                continue
            part = pairs[a:e]
            waits = {}
            for p, _ in part:
                for ev in (self.reader[p], self.ready[p]):
                    if ev is not None:
                        waits[id(ev)] = ev
            for ev in waits.values():
                s.wait_event(ev)
            with torch.cuda.stream(s):
                lane.n_copies += lane.m.copy(self.base, src, pair_runs(part, self.R))
                ev = torch.cuda.Event()
                ev.record(s)
            for p, _ in part:
                self.ready[p] = ev
        lane.n_records += len(pairs)

    def fill(self, src: int, recs: list) -> list:
        """Fresh copies of the records [(bank slot, row)] into free slots -> [(pool slot, row)]."""
        if not recs:
            return []
        free = [p for p in range(self.N) if p not in self.live][:len(recs)]
        assert len(free) == len(recs), f"K31 celane pool: {len(recs)} records for {len(free)} free slots"
        self._copy(src, [(p, b) for p, (b, _) in zip(free, recs)])
        self.live.update(free)
        return [(p, row) for p, (_, row) in zip(free, recs)]

    # -- the expand of a unit
    def land(self, mod, mapping: list) -> int:
        """On the current stream: wait for the copies of the unit's records [(pool slot, row)] and expand them into
        their rows of the module's four kernel tensors. Returns the number of records."""
        n = len(mapping)
        if n == 0:
            return 0
        lane = self.lane
        cur = torch.cuda.current_stream(lane.dev)
        waits = {}
        for p, _ in mapping:
            ev = self.ready[p]
            if ev is not None:
                waits[id(ev)] = ev
        for ev in waits.values():
            cur.wait_event(ev)
        expand_map(lane, self.base, [p for p, _ in mapping], [row for _, row in mapping], mod)
        ev = torch.cuda.Event()
        ev.record(cur)
        for p, _ in mapping:
            self.reader[p] = ev
            self.live.discard(p)
        lane.n_units += 1
        return n

    # -- ahead of the plan
    def speculate(self, st, resv) -> None:
        """Copy layer st's first lane records ahead of its plan: the lane records of its first `bufs` units if every
        record mapped now is staged, in slot order. Called once the current layer's last unit is queued: every slot is
        free after its reader."""
        lane = self.lane
        G = int(getattr(st, "bank_n", 0) or len(st.bank))
        if G <= 0 or self.live:
            return
        r = resv[0] if resv else None
        j = None
        if r is not None:
            j = _layer_j(r, st.idx)
            if j is None or getattr(r, "bank_gen", None) is None:
                return
            cand = _np().nonzero(r.bank_owner[j, :G] >= 0)[0]
        else:
            cand = _np().arange(G)
        # The guess of the plan's order. Without history: every mapped record, in slot order (a chunk of 4,096 rows
        # hits 78-97% of them). With the layer's last plan (K31_PROMPT_SPEC_LAST=1, a runtime flag, off by default):
        # the records that plan staged and that are still mapped, in its order, then the other mapped ones;
        # consecutive chunks of one prompt pick nearly the same records, so fewer copies ahead are wasted.
        last = self.last.get(st.idx) if flags.on("K31_PROMPT_SPEC_LAST") else None
        if last is not None and len(last):
            np_ = _np()
            keep = last[(last < G)]
            if r is not None:
                keep = keep[r.bank_owner[j, keep] >= 0]
            rest = cand[~np_.isin(cand, keep)]
            cand = np_.concatenate([keep, rest])
        n = int(cand.shape[0])
        slots = []
        for u in range(lane.bufs):
            m = max(0, min(UNIT, n - u * UNIT))
            slots.extend(cand[u * UNIT + m - take(lane.share, m):u * UNIT + m].tolist())
        slots = slots[:self.N]
        if not slots:
            return
        src = bank_base(st)
        self._copy(src, list(enumerate(slots)))
        arr = _np().asarray(slots, dtype=_np().int64)
        self.spec = {"idx": st.idx, "src": src, "slots": slots, "arr": arr, "res": r, "j": j,
                     "owner": None if r is None else r.bank_owner[j, arr].copy(),
                     "gen": None if r is None else r.bank_gen[j, arr].copy()}
        self.n["ahead"] += len(slots)

    def claim(self, st, resv) -> dict:
        """{bank slot: pool slot} of the records copied ahead for layer st that are still what the bank holds."""
        sp, self.spec = self.spec, None
        if sp is None or sp["idx"] != st.idx or sp["src"] != bank_base(st):
            return {}
        r = resv[0] if resv else None
        if sp["res"] is not r:                       # issued under another residency (or none): cannot be checked
            return {}
        if r is None:
            return {b: p for p, b in enumerate(sp["slots"])}
        ok = (r.bank_owner[sp["j"], sp["arr"]] == sp["owner"]) & (r.bank_gen[sp["j"], sp["arr"]] == sp["gen"])
        self.n["stale"] += int(ok.size - ok.sum())
        return {b: p for p, (b, o) in enumerate(zip(sp["slots"], ok.tolist())) if o}

    def log(self, every: int = 920) -> None:
        self.n["layers"] += 1
        if self.n["layers"] - self.logged >= every:
            self.logged = self.n["layers"]
            n = self.n
            _LOG(f"prompt path pool (K31_PROMPT_SPEC): {n['layers']} layer calls; of {n['lane']} lane records "
                 f"{n['used']} ({100.0 * n['used'] / max(1, n['lane']):.1f}%) were copied ahead of their plan; "
                 f"{n['ahead']} records copied ahead in all, {n['stale']} of them dropped because the bank slot "
                 f"changed")


def _np():
    import numpy
    return numpy


_J: dict = {}


def _layer_j(r, idx: int):
    """The residency's row index of layer idx (its tables are [layers, ...] in the order of r.idx)."""
    m = _J.get(id(r))
    if m is None or m[0] is not r:
        m = _J[id(r)] = (r, {v: k for k, v in enumerate(r.idx)})
    return m[1].get(idx)


def pair_runs(pairs: list, rec_bytes: int) -> list:
    """(pool slot, bank slot) pairs -> flat (dst offset, src offset, bytes) triples, one per run in which both the
    pool slots and the bank slots are consecutive."""
    out = []
    i, n = 0, len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[j][0] + 1 and pairs[j + 1][1] == pairs[j][1] + 1:
            j += 1
        out += [pairs[i][0] * rec_bytes, pairs[i][1] * rec_bytes, (j - i + 1) * rec_bytes]
        i = j + 1
    return out


def set_mode(lane, pool: bool) -> None:
    """The lane's scratch is driven either by its buffers (rounds) or as a pool (rounds_pool). A change of mode
    waits for everything queued and starts from an empty scratch."""
    if getattr(lane, "pool_mode", False) == pool:
        return
    torch.cuda.synchronize()
    lane.free = [None] * lane.bufs
    lane.copied = {}
    lane.pool = Pool(lane) if pool else None
    lane.pool_mode = pool
    _LOG(f"prompt path copy-engine lane: {'pool with copies ahead of the plan (K31_PROMPT_SPEC)' if pool else 'buffers per unit'}")


def speculate(lane, next_st, resv=None) -> None:
    """After a layer's rounds are queued (rounds_pool): copy the next layer's first lane records ahead of its plan
    (next_st: its layer state, or None)."""
    pool = getattr(lane, "pool", None)
    if pool is not None and next_st is not None:
        pool.speculate(next_st, resv)


def rounds_pool(orig, mod, st, x, w, shared_experts, kept, kept_h, place, nat, n_kept, dtype, lane, resv=None):
    """rounds() with the scratch as a pool: the same rounds, rows and MegaMoE calls; lane records copied ahead of this
    layer's plan are expanded from where they are, the others copied as in rounds(). The caller then calls
    speculate() for the next layer."""
    from . import prompt_path as pp
    R = pp.STAGE_ROWS
    pool = lane.pool
    n_rounds = max(1, -(-n_kept // R))
    rnd = torch.div(place, R, rounding_mode="floor")
    pos = place - rnd * R
    src = bank_base(st)

    def unit(r):
        m = max(0, min(R, n_kept - r * R))
        return m, m - take(lane.share, m)

    def recs(r):                                      # the unit's lane records: (bank slot, row)
        m, h = unit(r)
        return [(kept_h[r * R + k], k) for k in range(h, m)]

    held = pool.claim(st, resv)
    pool.last[st.idx] = _np().asarray(kept_h[:n_kept], dtype=_np().int64)[:lane.bufs * R]    # for the next guess
    first = min(lane.bufs, n_rounds)
    maps, miss = {}, {}
    for r in range(first):
        mp, ms = [], []
        for b, row in recs(r):
            p = held.get(b)
            if p is not None:
                mp.append((p, row))
            else:
                ms.append((b, row))
        maps[r], miss[r] = mp, ms
        pool.n["used"] += len(mp)
    pool.live = {p for r in range(first) for p, _ in maps[r]}     # every other slot is free (after its reader)
    lane.begin()
    for r in range(first):
        maps[r] += pool.fill(src, miss[r])
    xside = flags.on("K31_PROMPT_CELANE_XSIDE")
    main = torch.cuda.current_stream() if xside else None
    xs = _xside(main) if xside else None
    y = None
    for r in range(n_rounds):
        m, h = unit(r)
        pool.n["lane"] += m - h
        if xs is not None:                            # the expand beside the stager's launches, both after call r - 1
            xs.wait_stream(main)
            with torch.cuda.stream(xs):
                pool.land(st.mod, maps.pop(r))
                ev = torch.cuda.Event()
                ev.record(xs)
            stage_head(st, kept, r * R, h, 0)
            main.wait_event(ev)
        else:
            stage_head(st, kept, r * R, h, 0)
            pool.land(st.mod, maps.pop(r))
        _check(st, kept, r * R, 0, h, m)
        if r + lane.bufs < n_rounds:                  # into the slots this round's expand has just read
            maps[r + lane.bufs] = pool.fill(src, recs(r + lane.bufs))
        ids = torch.where((place >= 0) & (rnd == r), pos, nat if r == 0 else torch.full_like(nat, -1))
        out = pp._megamoe(orig, mod, x, w, ids, shared_experts if r == 0 else None, dtype)
        y = out if y is None else y + out             # out of place: a round's output may be MegaMoE's own buffer
        if pp.pprof.on():
            pp.pprof.round_mark(st.idx, r)
    pool.log()
    return y


_XS: dict = {"stream": None}


def _xside(main):
    if _XS["stream"] is None:
        _XS["stream"] = torch.cuda.Stream(device=main.device)
    return _XS["stream"]


def _rounds_ahead(orig, mod, st, x, w, shared_experts, kept, kept_h, place, nat, n_kept, dtype, lane):
    """prompt_path._rounds_ahead with the lane: half-round h (places 64 h ..) is staged on the stage-ahead stream into
    half h % 2 of the shared rows, its first records by the stager, its last ones landed by the lane's expand on the
    same stream; MegaMoE h waits for both, as it waited for the stager alone."""
    from . import prompt_path as pp
    HALF, R, G = pp.HALF, pp.STAGE_ROWS, pp.GROUPS
    main = torch.cuda.current_stream()
    if pp._AH["stream"] is None:
        pp._AH["stream"] = torch.cuda.Stream(device=main.device)
    side = pp._AH["stream"]
    halves = max(1, -(-n_kept // HALF))
    hr = torch.div(place, HALF, rounding_mode="floor")
    row = place - torch.div(place, R, rounding_mode="floor") * R
    moe_done = [None] * (halves + 1)
    staged = [None] * (halves + 1)
    src = bank_base(st)

    def unit(h):
        m = max(0, min(HALF, n_kept - h * HALF))
        return m, m - take(lane.share, m)

    def issue(h):
        if h < halves:
            m, hh = unit(h)
            lane.issue(h, src, kept_h[h * HALF + hh:h * HALF + m])

    def stage(h):
        if h >= halves:
            return
        side.wait_stream(main) if h == 0 else None
        if h >= 2 and moe_done[h - 2] is not None:
            side.wait_event(moe_done[h - 2])                 # the call that last read this half
        m, hh = unit(h)
        q0 = (h % 2) * HALF
        with torch.cuda.stream(side):
            stage_head(st, kept, h * HALF, hh, (h % 2) * (G // 2))
            lane.land(h, st.mod, q0 + hh)
            _check(st, kept, h * HALF, q0, hh, m)
            ev = torch.cuda.Event()
            ev.record(side)
        staged[h] = ev
        issue(h + lane.bufs)

    lane.begin()
    for h in range(min(lane.bufs, halves)):
        issue(h)
    stage(0)
    stage(1)
    y = None
    for h in range(halves):
        main.wait_event(staged[h])
        ids = torch.where((place >= 0) & (hr == h), row, nat if h == 0 else torch.full_like(nat, -1))
        out = pp._megamoe(orig, mod, x, w, ids, shared_experts if h == 0 else None, dtype)
        ev = torch.cuda.Event()
        ev.record(main)
        moe_done[h] = ev
        stage(h + 2)                                         # refill the half this call has just read
        y = out if y is None else y + out
    return y
