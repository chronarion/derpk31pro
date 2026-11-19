"""RTX band, engine side (K31_BAND): picks without a resident row whose expert has a record in the RTX band are
computed on the RTX PRO 6000 by the band sidecar's persistent service kernel (rtxband/code/band/rtxband_svc.cu,
started by rtxband/code/band/sidecar.py) from 2-bit records, on time.

Per MoE call of T <= 8 rows: `plan` builds the call's work list (band::BandCall, 152 int32: the distinct band experts
by their largest weight, at most 16, and each row's routing weight for them), `quant` turns the call's latent rows into
MXFP8 (MegaMoE's input rule), `Gb.send` writes both into the mailbox (before the call's staging and MegaMoE) and
`Gb.finish` waits for the result rows (a bounded spin) and adds them, float32, into `dst`, which the caller adds into the
routed latent sum. Every launch goes on the current stream, so a CUDA graph captures it. The GB300 kernels are the ones
of rtxband_gb300.cu (librtxband_gb300.so, sm_103a), through ctypes.
"""
import ctypes
import mmap
import os
import struct
import time

import numpy as np
import torch
import triton
import triton.language as tl

H = 3584
MAXE, MAXT = 16, 8
CALL_INTS = 8 + MAXE + MAXE * MAXT                       # 152 (band::BandCall)
SEG_BYTES = 659456                                       # band_mailbox.h: SEG_BYTES
OFF_DOOR, DOOR_REQ, OFF_STATE, STATE_RUNNING, STATE_SERVED, STATE_CALLS = 128, 8, 256, 0, 24, 32
NSLOT = 4
FLAG_OUT_BF16 = 16
FLAGS_MEGAMOE, SF_MIN_MEGAMOE = FLAG_OUT_BF16, 105       # DeepGEMM MegaMoE's arithmetic (rtxband.py)
# K31_BAND_MAXE=n (1-15): the fused send serves at most n distinct band experts per call (the strongest by weight; the
# rest are cut), bounding the RTX's time per call at 5+ rows; 16 = the call's full size. Fused path only (flags bits
# 16-23, stripped before the call header reaches the RTX).
BAND_MAXE = int(os.environ.get("K31_BAND_MAXE", "16"))
SITU_ALPHA, SITU_BETA = 4.0, 25.0


def parse_prior(spec: str) -> tuple:
    """K31_BAND_PRIOR "1,0.64,0.36,0.19,0.09" -> (1.0, 0.64, ...): the prior chance that row i of a request's verify
    block is accepted (row 0 = the last sampled token, always; row i = draft i, kept when drafts 1..i are accepted).
    Empty -> () (off)."""
    vals = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (v >= 0.0 and v < float("inf")):
            raise ValueError(f"K31_BAND_PRIOR: {part!r} is not a finite value >= 0")
        vals.append(v)
    return tuple(vals)


def prior_rows(table: tuple, period: int, n: int = 8) -> list:
    """The prior of rows 0..n-1 of a band call whose requests' verify blocks of `period` rows lie back to back: row t
    takes table[t % period] (the table's last value past its end)."""
    period = max(1, int(period))
    return [float(table[min(t % period, len(table) - 1)]) for t in range(n)]


# K31_BAND_PRIOR=p0,p1,... (B1, fused send only): a call with more distinct band experts than its cap keeps those with
# the largest weight x their row's prior chance of being accepted, instead of the largest weight. The kept experts keep
# today's order in the work list, so calls under the cap send the same bytes. Calls recorded into CUDA graphs only
# (decode verify blocks); eager calls keep today's plan. Unset: today's plan.
BAND_PRIOR = parse_prior(os.environ.get("K31_BAND_PRIOR", ""))
STATE_WORDS = ("seq", "tag", "idx", "hint_seq", "status", "late_tag", "late_idx", "late_status", "n_added",
               "n_timeout", "n_error", "n_empty", "n_down", "n_late_added", "n_lost", "t_pub", "t_hint")


def _fbits(x: float) -> int:
    return struct.unpack("<i", struct.pack("<f", x))[0]


@triton.jit
def _bplan(ids_ptr, w_ptr, row_ptr, band_ptr, call_ptr, stats_ptr, T, base, flags, sf_min, alpha_bits, beta_bits,
           TB: tl.constexpr, K: tl.constexpr, NE: tl.constexpr, MT: tl.constexpr):
    N: tl.constexpr = TB * K
    n = tl.arange(0, N)
    t = n // K
    m = t < T
    e = tl.load(ids_ptr + n, mask=m, other=-1).to(tl.int32)
    w = tl.load(w_ptr + n, mask=m, other=0.0).to(tl.float32)
    valid = m & (e >= 0)
    ee = tl.where(valid, e, 0)
    nat = tl.load(row_ptr + ee, mask=valid, other=-1)
    bs = tl.load(band_ptr + ee, mask=valid, other=-1)
    bp = valid & (nat < 0) & (bs >= 0)                                   # picks the band serves
    same = bp[:, None] & bp[None, :] & (e[:, None] == e[None, :])         # [N, N]
    first = bp & (tl.sum((same & (n[None, :] < n[:, None])).to(tl.int32), axis=1) == 0)
    score = tl.max(tl.where(same, w[None, :], float("-inf")), axis=1)    # the expert's largest weight
    better = first[None, :] & ((score[None, :] > score[:, None]) |
                               ((score[None, :] == score[:, None]) & (e[None, :] < e[:, None])))
    pos = tl.sum(better.to(tl.int32), axis=1) - base                     # place of the pick's expert in this call
    ok = bp & (pos >= 0) & (pos < NE)
    n_exp = tl.minimum(tl.maximum(tl.sum(first.to(tl.int32), axis=0) - base, 0), NE)
    h = tl.arange(0, 8)
    hv = tl.where(h == 0, n_exp, tl.where(h == 1, T, tl.where(h == 2, flags, tl.where(h == 3, sf_min, tl.where(
        h == 4, alpha_bits, tl.where(h == 5, beta_bits, 0))))))
    tl.store(call_ptr + h, hv)
    s = tl.arange(0, NE)
    slot = tl.sum(tl.where(first[None, :] & (pos[None, :] == s[:, None]), bs[None, :] + 1, 0), axis=1) - 1
    tl.store(call_ptr + 8 + s, tl.maximum(slot, 0))      # unused entries 0: the service checks all 16 are in the store
    q = tl.arange(0, NE * MT)
    jj = q // MT
    tt = q % MT
    wv = tl.sum(tl.where(ok[None, :] & (pos[None, :] == jj[:, None]) & (t[None, :] == tt[:, None]), w[None, :], 0.0),
                axis=1)
    tl.store(call_ptr + 8 + NE + q, wv.to(tl.int32, bitcast=True))
    tl.atomic_add(stats_ptr + 0, tl.sum(ok.to(tl.int32), axis=0))                 # picks served
    tl.atomic_add(stats_ptr + 1, tl.sum((bp & (pos >= NE)).to(tl.int32), axis=0))  # picks past this call's 16 experts
    tl.atomic_add(stats_ptr + 2, tl.sum(valid.to(tl.int32), axis=0))              # all picks


def plan(ids: torch.Tensor, w: torch.Tensor, row_map: torch.Tensor, band_map: torch.Tensor, call: torch.Tensor,
         stats: torch.Tensor, base: int = 0, flags: int = FLAGS_MEGAMOE, sf_min: int = SF_MIN_MEGAMOE) -> None:
    """The work list of one call into `call` (int32 [152]): picks (ids, w [T, 16], T <= 8) with no resident row
    (row_map < 0) and a band slot (band_map >= 0); their distinct experts ordered by largest weight (ties: lower id),
    those at places base .. base + 15 in this call. stats (int32 [3]) += served, past the 16, all picks."""
    T, K = ids.shape
    assert K == 16 and 1 <= T <= MAXT
    TB = max(1, triton.next_power_of_2(T))
    _bplan[(1,)](ids if ids.dtype == torch.int32 else ids.int(), w.contiguous(), row_map, band_map, call, stats, T,
                 base, flags, sf_min, _fbits(SITU_ALPHA), _fbits(SITU_BETA), TB=TB, K=K, NE=MAXE, MT=MAXT,
                 num_warps=4)


@triton.jit
def _q8(x_ptr, q_ptr, s_ptr, NB: tl.constexpr, NBP: tl.constexpr):
    r = tl.program_id(0)
    b = tl.arange(0, NBP)
    c = tl.arange(0, 32)
    bm = b < NB
    off = r * (NB * 32) + b[:, None] * 32 + c[None, :]
    x = tl.load(x_ptr + off, mask=bm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    bits = amax.to(tl.int32, bitcast=True)
    code = tl.maximum(((bits + 0x1FFFFF) >> 23) - 8, 0)                  # DeepGEMM's get_ue8m0_sf_exp, no floor
    inv = ((254 - code) << 23).to(tl.float32, bitcast=True)              # 2^(127 - code), exact
    y = tl.minimum(tl.maximum(x * inv[:, None], -448.0), 448.0)
    tl.store(q_ptr + off, y.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=bm[:, None])
    tl.store(s_ptr + r * NB + b, code.to(tl.uint8), mask=bm)


def quant(x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> None:
    """MXFP8 of the rows x [T, 3584] (bf16): E4M3 codes into xq [>= T, 3584] uint8, UE8M0 codes into xs [>= T, 112]
    uint8 (the smallest power-of-two scale with amax / scale <= 448; round to nearest even)."""
    T = x.shape[0]
    assert x.shape[1] == H and x.is_contiguous()
    _q8[(T,)](x, xq, xs, NB=H // 32, NBP=128, num_warps=4)


class _Gb(ctypes.Structure):
    _fields_ = [("seg", ctypes.c_void_p), ("st", ctypes.c_void_p),
                ("xq", ctypes.c_void_p), ("xq_stride", ctypes.c_longlong),
                ("xs", ctypes.c_void_p), ("xs_stride", ctypes.c_longlong),
                ("calls", ctypes.c_void_p), ("call_stride", ctypes.c_longlong),
                ("hints", ctypes.c_void_p), ("hint_stride", ctypes.c_longlong),
                ("dst", ctypes.c_void_p), ("dst_stride", ctypes.c_longlong),
                ("late", ctypes.c_void_p), ("late_stride", ctypes.c_longlong),
                ("log", ctypes.c_void_p), ("log_stride", ctypes.c_longlong),
                ("layers", ctypes.c_int), ("pad", ctypes.c_int),
                ("timeout_ns", ctypes.c_ulonglong), ("work_ns", ctypes.c_ulonglong),
                ("lead_ns", ctypes.c_ulonglong)]


class Gb:
    """The mailbox as this GB300 process uses it: the segment mapped through ATS, the state block that carries a call's
    seq from send to finish, and the call buffers (input rows, result rows) every layer shares."""

    def __init__(self, lib: str, seg_path: str, dev, timeout_us: float = 60.0):
        self.lib = ctypes.CDLL(lib)
        sizes = (ctypes.c_longlong * 2)()
        self.lib.rtxband_gb_sizes(sizes)
        assert sizes[0] == 8 * 20, f"state block {sizes[0]} B"
        fd = os.open(seg_path, os.O_RDWR)
        try:
            self.buf = mmap.mmap(fd, SEG_BYTES)
        finally:
            os.close(fd)
        self.u8 = np.frombuffer(self.buf, dtype=np.uint8)
        self.u64 = np.frombuffer(self.buf, dtype=np.uint64)
        if bytes(self.u8[:8]) != b"RTXBAND1":
            raise RuntimeError(f"K31 band: {seg_path} is not a band mailbox")
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        V = ctypes.c_void_p
        self.lib.rtxband_gb_map.argtypes = [V, ctypes.c_longlong, ctypes.POINTER(V)]
        for f in ("rtxband_gb_send", "rtxband_gb_finish", "rtxband_gb_late"):
            getattr(self.lib, f).argtypes = [ctypes.POINTER(_Gb), V]
        dptr = V()
        rc = self.lib.rtxband_gb_map(self.addr, SEG_BYTES, ctypes.byref(dptr))
        if rc != 0:
            raise RuntimeError(f"K31 band: registering the mailbox failed (CUDA error {rc})")
        self.seg = dptr.value
        self.dev = dev
        self.st = torch.zeros(20, dtype=torch.int64, device=dev)
        self.st[0] = self.next_seq() - 1                 # the first call's seq is above anything the mailbox has seen
        self.xq = torch.zeros((MAXT, H), dtype=torch.uint8, device=dev)
        self.xs = torch.zeros((MAXT, H // 32), dtype=torch.uint8, device=dev)
        self.dst = torch.zeros((MAXT, H), dtype=torch.float32, device=dev)
        self.timeout_ns = int(timeout_us * 1000)

    def word(self, off: int) -> int:
        return int(self.u64[off // 8])

    def next_seq(self) -> int:
        top = self.word(OFF_STATE + STATE_SERVED)
        for s in range(NSLOT):
            top = max(top, self.word(OFF_DOOR + DOOR_REQ + 8 * s) >> 16)
        return top + 1

    def running(self) -> bool:
        return self.word(OFF_STATE + STATE_RUNNING) != 0

    def args(self, call: torch.Tensor, timeout_us: float = None) -> _Gb:
        t = self.timeout_ns if timeout_us is None else int(timeout_us * 1000)
        return _Gb(self.seg, self.st.data_ptr(), self.xq.data_ptr(), 0, self.xs.data_ptr(), 0, call.data_ptr(), 0,
                   None, 0, self.dst.data_ptr(), 0, None, 0, None, 0, 1, 0, t, 0, 0)

    def send(self, a: _Gb) -> None:
        rc = self.lib.rtxband_gb_send(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: send launch failed ({rc})")

    def finish(self, a: _Gb) -> None:
        rc = self.lib.rtxband_gb_finish(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: finish launch failed ({rc})")

    def counters(self) -> dict:
        v = self.st.tolist()
        return {k: v[i] for i, k in enumerate(STATE_WORDS)}


# The sidecar's command ring (<segment>.cmd): record entries into band slots while the service runs.
#   0: magic "K31BCMD1", 8: u64 entries (1024), 64: u64 acked (the last seq done; entries complete in order),
#   72: u64 entries that fit, 80: u64 unfit, 88: u64 errors, 96: u64 entry microseconds in total
#   128 + 32 * (seq % 1024): u64 seq (stored last, release), i32 layer, i32 expert, i32 slot, i32 status (1 entered,
#   2 the record does not fit the band's scale code, 3 error), written by the sidecar before it moves `acked` on.
CMD_MAGIC, CMD_N = b"K31BCMD1", 1024
CMD_BYTES = 128 + 32 * CMD_N
CMD_ACKED, CMD_OK, CMD_UNFIT, CMD_ERR, CMD_US = 64, 72, 80, 88, 96


class BandCmd:
    """One end of the command ring (the engine posts, the sidecar serves); acquire / release stores through
    libbandhost.so."""

    def __init__(self, path: str, host_lib: str, create: bool = False):
        self.host = ctypes.CDLL(host_lib)
        V, U = ctypes.c_void_p, ctypes.c_uint64
        self.host.bandhost_get.argtypes = [V, U]
        self.host.bandhost_get.restype = U
        self.host.bandhost_set.argtypes = [V, U, U]
        fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o666)
        try:
            if create:
                os.ftruncate(fd, CMD_BYTES)
            self.buf = mmap.mmap(fd, CMD_BYTES)
        finally:
            os.close(fd)
        self.u8 = np.frombuffer(self.buf, dtype=np.uint8)
        self.i32 = np.frombuffer(self.buf, dtype=np.int32)
        if create:
            self.u8[:] = 0
            self.u8[:8] = np.frombuffer(CMD_MAGIC, dtype=np.uint8)
            self.u8[8] = CMD_N & 0xFF
            self.u8[9] = CMD_N >> 8
        if bytes(self.u8[:8]) != CMD_MAGIC:
            raise RuntimeError(f"K31 band: {path} is not a band command ring")
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self.posted = self.done = self.get(CMD_ACKED)

    def get(self, off: int) -> int:
        return int(self.host.bandhost_get(self.addr, off))

    def set(self, off: int, v: int) -> None:
        self.host.bandhost_set(self.addr, off, v)

    @staticmethod
    def slot_off(seq: int) -> int:
        return 128 + 32 * (seq % CMD_N)

    # the engine's end
    def post(self, layer: int, expert: int, slot: int, rec: int = 0) -> int:
        """rec > 0: the record is in row rec - 1 of the sidecar's /dev/shm ring (<segment>.rec, K31_BAND_RING), copied
        there from the Grace bank; 0: the sidecar reads it from the model drive."""
        seq = self.posted + 1
        o = self.slot_off(seq)
        self.i32[(o + 8) // 4:(o + 28) // 4] = (layer, expert, slot, 0, rec)
        self.set(o, seq)                                # release: the fields are visible before the seq
        self.posted = seq
        return seq

    def poll(self) -> list:
        """[(seq, status)] of the entries completed since the last poll, in order."""
        a = self.get(CMD_ACKED)
        out = []
        while self.done < a:
            self.done += 1
            out.append((self.done, int(self.i32[(self.slot_off(self.done) + 20) // 4])))
        return out

    def in_flight(self) -> int:
        return self.posted - self.done

    def rec(self, seq: int) -> int:
        """The ring row + 1 of posted entry `seq` (0: from the drive)."""
        return int(self.i32[(self.slot_off(seq) + 24) // 4])

    # the sidecar's end
    def next_job(self):
        """(seq, layer, expert, slot) of the next posted entry, or None."""
        seq = self.get(CMD_ACKED) + 1
        o = self.slot_off(seq)
        if self.get(o) != seq:
            return None
        layer, expert, slot = (int(v) for v in self.i32[(o + 8) // 4:(o + 20) // 4])
        return seq, layer, expert, slot

    def peek(self, seq: int):
        """(seq, layer, expert, slot) of the posted entry `seq` (ahead of the acknowledged one), or None: lets the
        sidecar read records ahead while it still completes the entries in order."""
        if seq > self.get(CMD_ACKED) + CMD_N - 1:
            return None
        o = self.slot_off(seq)
        if self.get(o) != seq:
            return None
        layer, expert, slot = (int(v) for v in self.i32[(o + 8) // 4:(o + 20) // 4])
        return seq, layer, expert, slot

    def complete(self, seq: int, status: int, us: int) -> None:
        o = self.slot_off(seq)
        self.i32[(o + 20) // 4] = status
        k = {1: CMD_OK, 2: CMD_UNFIT}.get(status, CMD_ERR)
        self.set(k, self.get(k) + 1)
        self.set(CMD_US, self.get(CMD_US) + us)
        self.set(CMD_ACKED, seq)                        # release: status and counters before the ack


class _Fused(ctypes.Structure):
    _fields_ = [("seg", ctypes.c_void_p), ("st", ctypes.c_void_p), ("ids", ctypes.c_void_p), ("w", ctypes.c_void_p),
                ("row_map", ctypes.c_void_p), ("band_map", ctypes.c_void_p), ("x", ctypes.c_void_p),
                ("out", ctypes.c_void_p), ("stats", ctypes.c_void_p), ("T", ctypes.c_int), ("layer", ctypes.c_int),
                ("flags", ctypes.c_int), ("sf_min", ctypes.c_int), ("alpha", ctypes.c_float), ("beta", ctypes.c_float),
                ("timeout_ns", ctypes.c_ulonglong), ("served", ctypes.c_void_p),   # served: v3+, msum / rstd / eps:
                ("msum", ctypes.c_void_p), ("rstd", ctypes.c_void_p), ("eps", ctypes.c_float),   # v4 (older builds
                ("prior", ctypes.c_float * MAXT), ("prior_on", ctypes.c_int)]    # read the fields before them)
    # prior / prior_on (B1): builds that export rtxband_fused_prior read them; older builds stop at eps


assert ctypes.sizeof(_Fused) == 168, "rtxband_fused.cu FusedArgs is 168 bytes"


class Fused:
    """The two fused GB300 kernels of a band call (rtxband_fused.cu): qsend = work list + MXFP8 rows + tag in one
    launch, finish = bounded wait + the rows added into the bfloat16 routed sum in place. They share `gb`'s state
    block, so fused and unfused calls can mix."""

    def __init__(self, lib: str, gb: Gb):
        self.lib = ctypes.CDLL(lib)
        for f in ("rtxband_fused_qsend", "rtxband_fused_finish"):
            getattr(self.lib, f).argtypes = [ctypes.POINTER(_Fused), ctypes.c_void_p]
        self.has_hint = hasattr(self.lib, "rtxband_fused_hint")       # librtxband_fused_v5 and later
        if self.has_hint:
            self.lib.rtxband_fused_hint.argtypes = [ctypes.POINTER(_Fused), ctypes.c_void_p]
        self.lib.rtxband_fused_pdl(1 if os.environ.get("K31_PDL", "0") == "1" else 0)   # opt-in, as w8a16 / unitk
        self.gb = gb
        self.has_prior = hasattr(self.lib, "rtxband_fused_prior")     # B1: builds that read prior / prior_on
        self.lib_path = lib
        self.prior = None                                             # (c_float * 8) per row of a call, or None
        self.prior_graphs_only = True

    def set_prior(self, rows, graphs_only: bool = True) -> None:
        """B1 (K31_BAND_PRIOR): later qsend / hint launches order the cap by weight x rows[t] (row t of the call; at
        most 8 values, the last repeated); None = today's order. graphs_only: only launches recorded into a CUDA
        graph (decode verify calls) take the prior; eager calls, such as a prompt's last chunk of <= 8 rows whose rows
        are all real tokens, keep today's order."""
        self.prior_graphs_only = bool(graphs_only)
        if rows is None:
            self.prior = None
            return
        if not self.has_prior:
            raise RuntimeError(f"K31_BAND_PRIOR: {self.lib_path} does not read a prior (rebuild it: ./build.sh)")
        vals = [float(v) for v in rows][:MAXT]
        if not vals or not all(0.0 <= v < float("inf") for v in vals):
            raise ValueError(f"K31_BAND_PRIOR: row priors {rows} must be finite values >= 0")
        vals += [vals[-1]] * (MAXT - len(vals))
        self.prior = (ctypes.c_float * MAXT)(*vals)

    def _prior(self, a: _Fused) -> _Fused:
        if self.prior is not None and (not self.prior_graphs_only or torch.cuda.is_current_stream_capturing()):
            a.prior = self.prior
            a.prior_on = 1
        return a

    def qsend(self, ids: torch.Tensor, w: torch.Tensor, row_map: torch.Tensor, band_map: torch.Tensor, x: torch.Tensor,
              stats: torch.Tensor, layer: int = 0, flags: int = FLAGS_MEGAMOE, sf_min: int = SF_MIN_MEGAMOE,
              cap: int = None, served: torch.Tensor = None) -> None:
        T = ids.shape[0]
        cap = BAND_MAXE if cap is None else int(cap)     # distinct band experts per call (K31_BAND_MAXE / the policy)
        assert 1 <= T <= MAXT and ids.dtype == torch.int32 and w.dtype == torch.float32 and x.dtype == torch.bfloat16
        assert ids.is_contiguous() and w.is_contiguous() and x.is_contiguous() and x.shape == (T, H)
        a = self._prior(_Fused(self.gb.seg, self.gb.st.data_ptr(), ids.data_ptr(), w.data_ptr(), row_map.data_ptr(),
                               band_map.data_ptr(), x.data_ptr(), None,
                               stats.data_ptr() if stats is not None else None, T, layer & 0xFF,
                               flags | (((cap & 0xFF) << 16) if 0 < cap < MAXE else 0), sf_min, SITU_ALPHA, SITU_BETA,
                               0, served.data_ptr() if served is not None else None))
        rc = self.lib.rtxband_fused_qsend(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: fused send launch failed ({rc})")

    def hint(self, ids: torch.Tensor, w: torch.Tensor, row_map: torch.Tensor, band_map: torch.Tensor,
             stats: torch.Tensor, layer: int = 0, cap: int = None, served: torch.Tensor = None) -> None:
        """K31_BAND_HINT: the call's band plan right after top-k: served flags and stats written here,
        the call's band slots published as a hint (the RTX pulls their records into its L2 before the rows arrive).
        The following qsend then gets served = stats = None."""
        T = ids.shape[0]
        cap = BAND_MAXE if cap is None else int(cap)
        assert 1 <= T <= MAXT and ids.dtype == torch.int32 and w.dtype == torch.float32
        a = self._prior(_Fused(self.gb.seg, self.gb.st.data_ptr(), ids.data_ptr(), w.data_ptr(), row_map.data_ptr(),
                               band_map.data_ptr(), None, None, stats.data_ptr() if stats is not None else None, T,
                               layer & 0xFF, FLAGS_MEGAMOE | (((cap & 0xFF) << 16) if 0 < cap < MAXE else 0),
                               SF_MIN_MEGAMOE, SITU_ALPHA, SITU_BETA, 0,
                               served.data_ptr() if served is not None else None))
        rc = self.lib.rtxband_fused_hint(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: hint launch failed ({rc})")

    def finish(self, out: torch.Tensor, timeout_us: float = None, msum: torch.Tensor = None, rstd: torch.Tensor = None,
               eps: float = 0.0) -> None:
        """v4 (librtxband_fused_v4.so) with msum / rstd: out = bf16(out + the band rows + msum) in one rounding and
        rstd[t] = rsqrt(mean(out[t]^2) + eps) of the result (the routed up-projection's RMSNorm statistic)."""
        assert out.dtype == torch.bfloat16 and out.is_contiguous() and out.shape[1] == H
        assert msum is None or (msum.dtype == torch.float32 and msum.is_contiguous() and msum.shape[1] == H)
        assert rstd is None or rstd.dtype == torch.float32
        t = self.gb.timeout_ns if timeout_us is None else int(timeout_us * 1000)
        a = _Fused(self.gb.seg, self.gb.st.data_ptr(), None, None, None, None, None, out.data_ptr(), None,
                   out.shape[0], 0, 0, 0, 0.0, 0.0, t, None, msum.data_ptr() if msum is not None else None,
                   rstd.data_ptr() if rstd is not None else None, float(eps))
        rc = self.lib.rtxband_fused_finish(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: fused finish launch failed ({rc})")


class Bulk:
    """The bulk band request, engine side (K31_BAND_BULK=1; the sidecar's band/bulk.py serves it): a prompt chunk's
    band picks go to the RTX in one request per MoE layer instead of decode calls of 8 rows, and their summed rows come
    back in one result that `finish` adds after the layer's MegaMoE rounds. While a prompt is read the sidecar's
    service kernel is parked: `set_mode(True)` before a prompt chunk, `set_mode(False)` before a decode step (both wait
    for the sidecar's acknowledgement)."""
    MAGIC = b"RTXBULK1"
    OFF_DOOR, OFF_MODE, OFF_ACK, OFF_DONE, OFF_STATUS, OFF_TIMES = 128, 256, 384, 512, 520, 528

    def __init__(self, lib: str, host_lib: str, seg_path: str, dev, timeout_us: float = 200000.0):
        fd = os.open(seg_path, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            self.buf = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        if bytes(self.buf[0:8]) != self.MAGIC:
            raise RuntimeError(f"K31 band: {seg_path} is not a bulk segment")
        xq, xs, slot, w, res, nbytes, max_rows = struct.unpack_from("<7Q", self.buf, 16)
        assert nbytes == size, f"bulk segment: {size} B against its layout's {nbytes}"
        self.offs = (ctypes.c_longlong * 4)(xq, xs, slot, w)
        self.off_res, self.max_rows = res, max_rows
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self.lib = ctypes.CDLL(lib)
        V = ctypes.c_void_p
        self.lib.bandbulk_map.argtypes = [V, ctypes.c_longlong, ctypes.POINTER(V)]
        self.lib.bandbulk_post.argtypes = [V, ctypes.POINTER(ctypes.c_longlong), V, V, V, V, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
                                           ctypes.c_ulonglong, V, V]
        self.lib.bandbulk_finish.argtypes = [V, ctypes.c_longlong, V, ctypes.c_longlong, ctypes.c_int,
                                             ctypes.c_ulonglong, ctypes.c_ulonglong, V, V]
        dptr = V()
        rc = self.lib.bandbulk_map(self.addr, size, ctypes.byref(dptr))
        if rc != 0:
            raise RuntimeError(f"K31 band: registering the bulk segment failed (CUDA error {rc})")
        self.seg = dptr.value
        self.host = ctypes.CDLL(host_lib)
        self.host.bandhost_get.restype = ctypes.c_uint64
        self.host.bandhost_get.argtypes = [V, ctypes.c_uint64]
        self.host.bandhost_set.argtypes = [V, ctypes.c_uint64, ctypes.c_uint64]
        mx = max_rows
        self.dev = dev
        self.xq = torch.empty((mx, H), dtype=torch.uint8, device=dev)
        self.xs = torch.empty((mx, H // 32), dtype=torch.uint8, device=dev)
        self.slot = torch.empty((mx, 16), dtype=torch.int16, device=dev)
        self.w = torch.empty((mx, 16), dtype=torch.float32, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.err = torch.zeros(2, dtype=torch.int32, device=dev)       # timeouts, error results
        self.timeout_ns = int(timeout_us * 1000)
        self.seq = (max(self.get(self.OFF_DOOR), self.get(self.OFF_DONE)) >> 24) + 1
        self.mode = self.get(self.OFF_ACK) == 1
        self.n = 0
        self.T = 0
        self.tag = 0

    def get(self, off: int) -> int:
        return int(self.host.bandhost_get(self.addr, off))

    def set_mode(self, on: bool, timeout_s: float = 10.0) -> None:
        """Ask the sidecar to park (on) or restart (off) its service kernel and wait until it has."""
        if self.mode == on:
            return
        want = 1 if on else 0
        self.host.bandhost_set(self.addr, self.OFF_MODE, want)
        t0 = time.time()
        while self.get(self.OFF_ACK) != want:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"K31 band: the sidecar did not switch to bulk mode {want} in {timeout_s} s")
            time.sleep(0.0002)
        self.mode = on

    def send(self, st, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Post layer st.idx's request for the rows x [T, 3584] (bf16): picks without a resident row whose expert has
        a band record (the band slot in st.band_map)."""
        T = x.shape[0]
        assert self.mode and 1 <= T <= self.max_rows and topk_ids.shape[1] == 16
        quant(x.contiguous(), self.xq, self.xs)
        valid = topk_ids >= 0
        ids = topk_ids.long().clamp_min(0)
        band = st.band_map[ids]
        is_band = valid & (st.row_map[ids] < 0) & (band >= 0)
        self.slot[:T].copy_(torch.where(is_band, band, -1))
        self.w[:T].copy_(topk_weights)
        self.tag = (self.seq << 24) | ((st.idx & 0xFF) << 16) | ((T - 1) & 0xFFFF)
        self.seq += 1
        self.T = T
        rc = self.lib.bandbulk_post(self.seg, self.offs, self.xq.data_ptr(), self.xs.data_ptr(), self.slot.data_ptr(),
                                    self.w.data_ptr(), T, st.idx, FLAGS_MEGAMOE, SF_MIN_MEGAMOE, SITU_ALPHA, SITU_BETA,
                                    self.tag, self.counter.data_ptr(),
                                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: bulk post launch failed ({rc})")

    def finish(self, st, y: torch.Tensor) -> None:
        """y [T, 3584] (bf16, rows contiguous) += the request's result, once the sidecar has served it."""
        assert y.shape[0] == self.T and y.stride(1) == 1 and y.dtype == torch.bfloat16
        rc = self.lib.bandbulk_finish(self.seg, self.off_res, y.data_ptr(), y.stride(0), self.T, self.tag,
                                      self.timeout_ns, self.err.data_ptr(),
                                      ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"K31 band: bulk finish launch failed ({rc})")
        self.n += 1
