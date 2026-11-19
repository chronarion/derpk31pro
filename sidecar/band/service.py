"""RTX band of derpk31pro: host side of the persistent service kernel (band/rtxband_svc.cu), and the trace tools the
tests and the round-trip benchmark share.

- `Mailbox`: the pinned segment (band/band_mailbox.h), a /dev/shm file mapped into this process.
- `Service`: a band store on the RTX (a `Band`) plus the service kernel's buffers; `start()` launches the kernel,
  `stop()` asks it to leave. While the kernel runs this process must not launch other kernels on the RTX or
  synchronize the device: the service occupies the SMs for good (only copy-engine transfers go through).
- `make_trace()`: realistic work lists (several distinct experts per call, fresh records for every call), their
  inputs, hint lists of a given recall, and the reference outputs from the launched kernels.
"""
import ctypes
import mmap
import os
import time

import numpy as np
import torch

from .rtxband import H, HI2_REC, MAXE, MAXT, REC, XF_BYTES, Band, mxfp8_encode, pack_x

ROW_IN, ROW_OUT, CALL_BYTES, HINT_MAX = 3696, 3584 * 4, 608, 16


def row_perm() -> np.ndarray:
    """perm[k] = byte offset of input k inside a row as sent (the band's fragment order)."""
    k = np.arange(H)
    return (k // 128) * 128 + ((k % 16) // 4) * 32 + ((k % 128) // 32) * 8 + ((k % 32) // 16) * 4 + k % 4


class Mailbox:
    """This process's view of the mailbox segment."""

    def __init__(self, host_lib: str, path: str, create: bool):
        self.host = ctypes.CDLL(host_lib)
        lay = (ctypes.c_longlong * 12)()
        self.host.bandhost_layout(lay)
        (self.bytes, self.OFF_DOOR, self.OFF_STATE, self.OFF_HINT, self.OFF_REQ, self.REQ_BYTES, self.OFF_RES,
         self.RES_BYTES, self.RES_ROWS, self.NSLOT, _, _) = list(lay)
        self.path = path
        fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o666)
        try:
            if create:
                os.ftruncate(fd, self.bytes)
            self.buf = mmap.mmap(fd, self.bytes)
        finally:
            os.close(fd)
        self.np = np.frombuffer(self.buf, dtype=np.uint8)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        if create:
            self.np[:] = 0
            self.np[:8] = np.frombuffer(b"RTXBAND1", dtype=np.uint8)
            self.np[8] = 1
        V, U = ctypes.c_void_p, ctypes.c_uint64
        self.host.bandhost_get.argtypes = [V, U]; self.host.bandhost_get.restype = U
        self.host.bandhost_set.argtypes = [V, U, U]
        self.host.bandhost_tag.argtypes = [U, ctypes.c_int, ctypes.c_int]; self.host.bandhost_tag.restype = U
        self.host.bandhost_send.argtypes = [V, U, V, V, ctypes.c_int]
        self.host.bandhost_wait.argtypes = [V, U, ctypes.c_longlong]; self.host.bandhost_wait.restype = ctypes.c_longlong
        self.host.bandhost_hint.argtypes = [V, U, V, ctypes.c_int]
        self.host.bandhost_repack.argtypes = [V, U, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        self.host.bandhost_repack.restype = ctypes.c_longlong
        self.host.bandhost_trace.argtypes = [V, V, V, V, ctypes.c_int, ctypes.c_int, U, ctypes.c_int, ctypes.c_longlong,
                                             ctypes.c_longlong, ctypes.c_longlong, V, V, U]

    def get(self, off: int) -> int:
        return int(self.host.bandhost_get(self.addr, off))

    def set(self, off: int, v: int):
        self.host.bandhost_set(self.addr, off, v)

    def state(self) -> dict:
        s = self.OFF_STATE
        return dict(running=self.get(s), error=self.get(s + 8), repack_ack=self.get(s + 16), served=self.get(s + 24),
                    calls=self.get(s + 32), hints=self.get(s + 40), poll_turns=self.get(s + 48),
                    poll_ns=self.get(s + 56))

    def next_seq(self) -> int:
        """A seq above everything this mailbox has seen: where a new sender starts."""
        top = self.get(self.OFF_STATE + 24)
        for s in range(self.NSLOT):
            top = max(top, self.get(self.OFF_DOOR + 8 + 8 * s) >> 16)
        return top + 1

    def result_rows(self, seq: int, rows: int) -> np.ndarray:
        o = self.OFF_RES + (seq % self.NSLOT) * self.RES_BYTES + self.RES_ROWS
        return self.np[o:o + rows * ROW_OUT].view(np.float32).reshape(rows, H).copy()

    def trace(self, calls: np.ndarray, rows: np.ndarray, hints, T: int, seq0: int, period_us: float = 0.0,
              hint_lead_us: float = 140.0, timeout_us: float = 5000.0, layers: int = 92, hint_seq0: int = 1,
              keep_rows: bool = True):
        """Replay calls (uint8 [n, 608]) with rows (uint8 [n, T, 3696]) from this thread (the GB300's stand-in).
        Returns (ns int64 [n, 9]: round trip, lateness of the send, status, then RTX ns from the tag seen to: done
        published, call released, first worker at work, last unit's sum stored, and fenced, and the RTX ns the
        store of the previous call's done tag took; rows float32 [n, T, 3584] or None; number of timeouts)."""
        n = calls.shape[0]
        calls = np.ascontiguousarray(calls, dtype=np.uint8); rows = np.ascontiguousarray(rows, dtype=np.uint8)
        out = np.zeros((n, 9), dtype=np.int64)
        res = np.zeros((n, T, H), dtype=np.float32) if keep_rows else None
        hp = None
        if hints is not None:
            hints = np.ascontiguousarray(hints, dtype=np.int32)
            hp = hints.ctypes.data
        late = self.host.bandhost_trace(self.addr, calls.ctypes.data, rows.ctypes.data, hp, n, T, seq0, layers,
                                        int(period_us * 1000), int(hint_lead_us * 1000), int(timeout_us * 1000),
                                        out.ctypes.data, res.ctypes.data if keep_rows else None, hint_seq0)
        return out, res, late


class _RtxSvc(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("seg", "store", "sbase", "xf", "aq", "ypart", "ctl", "landing")] + \
               [("idle_limit_ns", ctypes.c_ulonglong), ("wait_limit_ns", ctypes.c_ulonglong),
                ("n_slots", ctypes.c_longlong), ("repack_after_ns", ctypes.c_ulonglong),
                ("repack_idle_ns", ctypes.c_ulonglong)]


class Service:
    """The service kernel on the device of `band`, sharing its store."""

    def __init__(self, band: Band, svc_lib: str, mailbox: Mailbox, landings: int = 2):
        self.band, self.mb = band, mailbox
        self.lib = ctypes.CDLL(svc_lib)
        sizes = (ctypes.c_longlong * 9)()
        self.lib.rtxband_svc_sizes(sizes)
        self.sizes = list(sizes)
        assert self.sizes[2] == mailbox.bytes and self.sizes[3] == HI2_REC and self.sizes[4] == REC
        info = (ctypes.c_int * 3)()
        rc = self.lib.rtxband_svc_preload(info)
        assert rc == 0, f"preload failed: {rc}"
        self.regs, self.per_sm, self.sms = list(info)
        dev = band.dev
        self.xf = torch.zeros(XF_BYTES, dtype=torch.uint8, device=dev)
        self.ctl = torch.zeros(self.sizes[0], dtype=torch.uint8, device=dev)
        self.landing = torch.zeros((landings, HI2_REC), dtype=torch.uint8, device=dev)
        dptr = ctypes.c_void_p()
        self.lib.rtxband_svc_map.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]
        rc = self.lib.rtxband_svc_map(mailbox.addr, mailbox.bytes, ctypes.byref(dptr))
        assert rc == 0, f"mailbox registration failed: {rc}"
        self.seg_dev = dptr.value
        self.params = _RtxSvc(self.seg_dev, band.store.data_ptr(), band.sbase.data_ptr(), self.xf.data_ptr(),
                              band.aq.data_ptr(), band.ypart.data_ptr(), self.ctl.data_ptr(),
                              self.landing.data_ptr(), 0, 0, band.store.shape[0], 15000, 1000000)
        self.lib.rtxband_svc_launch.argtypes = [ctypes.POINTER(_RtxSvc), ctypes.c_int]
        self.lib.rtxband_svc_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        self.lib.rtxband_svc_pin.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        self.lib.rtxband_svc_reset.argtypes = [ctypes.c_void_p]
        self.grid = 0
        self.rseq = 1
        torch.cuda.synchronize(dev)

    def start(self, grid: int = 0, idle_limit_s: float = 0.0, wait_limit_s: float = 0.5, repack_after_us: float = 15.0,
              repack_idle_us: float = 1000.0):
        """Launch the kernel. grid 0 = as many CTAs as fit at once. A repack starts within repack_after_us of a
        call's end or once no call has ended for repack_idle_us (both 0: whenever the poller sees it)."""
        assert not self.running()
        self.params.repack_after_ns = int(repack_after_us * 1000)
        self.params.repack_idle_ns = int(repack_idle_us * 1000)
        self.grid = grid or self.per_sm * self.sms
        assert 2 <= self.grid <= self.per_sm * self.sms, "the grid must be co-resident"
        rc = self.lib.rtxband_svc_reset(self.ctl.data_ptr())
        assert rc == 0
        torch.cuda.synchronize(self.band.dev)
        self.params.idle_limit_ns = int(idle_limit_s * 1e9)
        self.params.wait_limit_ns = int(wait_limit_s * 1e9)
        self.mb.set(self.mb.OFF_DOOR, 0)                 # control word
        self.mb.set(self.mb.OFF_STATE + 8, 0)            # error word
        self.mb.set(self.mb.OFF_STATE, 1)                # running
        rc = self.lib.rtxband_svc_launch(ctypes.byref(self.params), self.grid)
        assert rc == 0, f"launch failed: {rc}"

    def running(self) -> bool:
        r = self.lib.rtxband_svc_running() if self.grid else 0
        if r < 0:                                        # the kernel died without clearing its running word:
            self.mb.set(self.mb.OFF_STATE, 0)            # the GB300's finish must not wait for it
        assert r >= 0, f"the service kernel died: CUDA error {-r}"
        return bool(r)

    def stop(self, timeout_s: float = 5.0):
        """Ask the kernel to leave and wait for it."""
        if not self.grid:
            return
        self.mb.set(self.mb.OFF_DOOR, 1)
        t0 = time.time()
        while self.running():
            assert time.time() - t0 < timeout_s, "the service kernel did not leave"
            time.sleep(0.0005)
        rc = self.lib.rtxband_svc_wait_exit()
        assert rc == 0, f"service kernel exit: {rc}"
        self.mb.set(self.mb.OFF_DOOR, 0)

    def enter(self, hi2_host: np.ndarray, slot: int, landing: int = 0, timeout_s: float = 1.0):
        """A record enters the band while the kernel runs: copy-engine transfer of the hi2 record (pinned host
        memory) into a landing buffer, then a repack item. Returns (fits, scale base, seconds for the copy,
        seconds for the repack)."""
        assert hi2_host.dtype == np.uint8 and hi2_host.size == HI2_REC
        t0 = time.perf_counter()
        rc = self.lib.rtxband_svc_copy(self.landing[landing].data_ptr(), hi2_host.ctypes.data, HI2_REC)
        assert rc == 0
        while True:
            d = self.lib.rtxband_svc_copy_done()
            assert d >= 0
            if d:
                break
        t1 = time.perf_counter()
        ack = self.mb.host.bandhost_repack(self.mb.addr, self.rseq, landing, slot, int(timeout_s * 1e9))
        t2 = time.perf_counter()
        self.rseq += 1
        assert ack >= 0, "no acknowledgement of the repack"
        return bool(ack >> 8), ack & 0xFF, t1 - t0, t2 - t1

    def feed_start(self, records: np.ndarray, recs: np.ndarray, slots: np.ndarray, per_second: float = 0.0,
                   chunk_bytes: int = 0, gate: int = 0, mode: int = 0):
        """Records enter the band from a host thread while the kernel serves calls: entry i brings
        records[recs[i % m]] (uint8 [.., hi2 bytes], pinned here) into store slot slots[i % m]. per_second 0 = as
        fast as the rest allows; chunk_bytes = one copy-engine transfer (0 = a whole record); gate 1: a transfer
        starts only right after a call has ended (or after a quiet millisecond); gate 2: transfers start as long as
        no call is in flight."""
        assert records.dtype == np.uint8 and records.shape[1] == HI2_REC and records.flags.c_contiguous
        assert self.landing.shape[0] >= 2
        self._feed = (records, np.ascontiguousarray(recs, dtype=np.int32), np.ascontiguousarray(slots, dtype=np.int32))
        if getattr(self, "_pinned", None) is not records:
            rc = self.lib.rtxband_svc_pin(records.ctypes.data, records.nbytes)
            assert rc == 0, f"pinning the records failed: {rc}"
            self._pinned = records
        self.lib.rtxband_svc_feed_start.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_longlong,
                                                                            ctypes.c_longlong, ctypes.c_int,
                                                                            ctypes.c_int, ctypes.c_ulonglong]
        rc = self.lib.rtxband_svc_feed_start(self.mb.addr, self.landing.data_ptr(), records.ctypes.data,
                                             self._feed[1].ctypes.data, self._feed[2].ctypes.data, len(self._feed[1]),
                                             int(1e9 / per_second) if per_second else 0, int(chunk_bytes), int(gate),
                                             int(mode), self.rseq)
        assert rc == 0

    def feed_stop(self) -> dict:
        out = (ctypes.c_longlong * 11)()
        rc = self.lib.rtxband_svc_feed_stop(out)
        assert rc == 0
        self.rseq = int(out[7])
        n = max(int(out[0]), 1)
        return dict(entries=int(out[0]), copy_us=out[1] / n / 1000, repack_us=out[2] / n / 1000,
                    repack_max_us=out[3] / 1000, unfit=int(out[4]), errors=int(out[5]), behind=int(out[6]),
                    transfers=int(out[8]), transfer_us=out[9] / max(int(out[8]), 1) / 1000,
                    transfer_max_us=out[10] / 1000)


def expert_counts(rng, n_calls: int, e_lo: int, e_hi: int, dist: str = "uniform", cap: int = 12) -> np.ndarray:
    """Distinct band experts per call. "uniform": e_lo..e_hi. "rows3": the simulated band at three rows per step
    (mean 9.6, p90 14, p99 19: a gamma of shape 7.1), "rows2": at two tokens per step (mean 7.0, p90 11, p99 14:
    shape 6.1); both capped at `cap` (what goes beyond the cap takes the engine's late path, not this call)."""
    if dist == "uniform":
        return rng.integers(e_lo, e_hi + 1, n_calls)
    shape, mean = {"rows3": (7.1, 9.6), "rows2": (6.1, 7.0)}[dist]
    return np.clip(np.rint(rng.gamma(shape, mean / shape, n_calls)), 1, cap).astype(np.int64)


def make_trace(band: Band, n_records: int, n_calls: int, T: int = 3, e_lo: int = 7, e_hi: int = 10, seed: int = 0,
               recall: float = 0.8, flags=None, reference: bool = True, dist: str = "uniform", cap: int = 12,
               layers: int = 0, repeat: float = 0.25):
    """Work lists for n_calls layer calls: e_lo..e_hi distinct experts each (or `dist`, see expert_counts), T rows,
    each expert used by a random non-empty subset of the rows.
    layers = 0: slots walk through a shuffled list of all records, so a record comes back only after the whole
    store has been read (nothing is served from L2 unless a hint put it there).
    layers = L: the store is L layers of n_records // L slots; call i belongs to layer i % L and takes its experts
    from that layer's slots, a share `repeat` of them from the same layer's call one step earlier.
    Hints: the call's own experts, each replaced with probability 1 - recall by a record the call does not use (far
    away in the walk, or another slot of the layer). Returns a dict of numpy arrays: calls [n, 608] uint8, rows
    [n, T, 3696] uint8 (as sent), xq [n, T, 3584], xs [n, T, 112] (row-major, for the GB300's send), hints [n, 17]
    int32, n_exp [n], and ref [n, T, 3584] float32 (the launched kernels' output; needs the service kernel not to be
    running)."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = row_perm()
    order = rng.permutation(n_records)
    ptr = 0
    counts = expert_counts(rng, n_calls, e_lo, e_hi, dist, cap)
    per_layer = n_records // layers if layers else 0
    last = {}
    out = dict(calls=np.zeros((n_calls, CALL_BYTES), np.uint8), rows=np.zeros((n_calls, T, ROW_IN), np.uint8),
               xq=np.zeros((n_calls, T, H), np.uint8), xs=np.zeros((n_calls, T, H // 32), np.uint8),
               hints=np.zeros((n_calls, 1 + HINT_MAX), np.int32), n_exp=np.zeros(n_calls, np.int32),
               ref=np.zeros((n_calls, T, H), np.float32) if reference else None)
    kw = {} if flags is None else dict(flags=flags)
    for i in range(n_calls):
        n = int(counts[i])
        if layers:
            L = i % layers
            prev = last.get(L, [])
            keep = [s for s in prev if rng.random() < repeat][:n]
            pool = np.setdiff1d(np.arange(L * per_layer, (L + 1) * per_layer), keep)
            slots = keep + [int(s) for s in rng.choice(pool, n - len(keep), replace=False)]
            last[L] = slots
            others = np.setdiff1d(pool, slots)
        else:
            if ptr + n > n_records:
                order, ptr = rng.permutation(n_records), 0
            slots = [int(s) for s in order[ptr:ptr + n]]
            ptr += n
        w = torch.zeros(n, MAXT)
        for e in range(n):
            use = torch.rand(T, generator=gen) < 0.5
            use[int(torch.randint(0, T, (1,), generator=gen))] = True
            w[e, :T] = torch.where(use, torch.rand(T, generator=gen) * 0.2 + 0.01, torch.zeros(T))
        x = torch.randn(T, H, generator=gen).to(band.dev)
        xq, xs, _ = mxfp8_encode(x, 0)
        call = band.make_call(slots, w, T, **kw)
        out["calls"][i] = call.cpu().numpy().view(np.uint8)
        xqn, xsn = xq.cpu().numpy(), xs.cpu().numpy()
        out["xq"][i], out["xs"][i] = xqn, xsn
        out["rows"][i, :, perm] = xqn.T
        out["rows"][i, :, H:] = xsn
        out["n_exp"][i] = n
        # the hint for this call: its experts, some of them wrong (records far away in the walk)
        hs = []
        for s in slots:
            if rng.random() < recall:
                hs.append(s)
            elif layers:
                hs.append(int(others[(len(hs) * 7 + i) % len(others)]))
            else:
                hs.append(int(order[(ptr + n_records // 2 + len(hs)) % n_records]))
        out["hints"][i, 0] = n
        out["hints"][i, 1:1 + n] = hs
        if reference:
            out["ref"][i] = band.run(call, pack_x(xq, xs), fused=False)[:T].cpu().numpy()
    return out
