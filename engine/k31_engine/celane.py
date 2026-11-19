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
  K31_PROMPT_CELANE_CHECK=1 the first lane unit of each layer: every lane row compared with the same record staged by
                            the stager into the same row, byte for byte (logged; a difference raises)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("expand", &expand); m.def("copy", &copy); m.def("attr", &attr); }
"""
_MOD = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
        _MOD.append(load_inline(name="k31_celane", cpp_sources="", cuda_sources=_SRC,
                                extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_103a,code=sm_103a"],
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
        bases, strides = dst_of(mod)
        b = u % self.bufs
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
             + (", byte check on" if flags.on("K31_PROMPT_CELANE_CHECK") else ""))
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
    return torch.cat([rb(t, row) for t in (mod.w13_weight, mod.w2_weight, mod.w13_weight_scale, mod.w2_weight_scale)])


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
    if m <= h or _CHECKED.get(st.idx, 0) >= 1 or not flags.on("K31_PROMPT_CELANE_CHECK"):
        return
    _CHECKED[st.idx] = 1
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
