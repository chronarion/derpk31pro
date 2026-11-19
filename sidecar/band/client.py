"""RTX band of derpk31pro: the GB300 side as a Python object (band/rtxband_gb300.cu): the send, finish, late and
hint kernels over device buffers, and the replay of a trace of calls through them.

An engine launches the same kernels inside its own graph (or inlines band/band_gb300.cuh into its kernels):
`rtxband_gb_send` once `unitk` has written the layer's work list, `rtxband_gb_finish` before the routed norm,
`rtxband_gb_late` where the late completion of the previous layer is added; the status words of the state block say
what happened.
"""
import ctypes

import numpy as np
import torch

from .service import Mailbox

H = 3584
ST_ADDED, ST_TIMEOUT, ST_ERROR, ST_EMPTY, ST_DOWN = 0, 1, 2, 3, 4
LATE_NONE, LATE_ADDED, LATE_LOST, LATE_ERROR = 0, 1, 2, 3
STATE_WORDS = ("seq", "tag", "idx", "hint_seq", "status", "late_tag", "late_idx", "late_status", "n_added",
               "n_timeout", "n_error", "n_empty", "n_down", "n_late_added", "n_lost", "t_pub", "t_hint", "pad0",
               "pad1", "pad2")


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


class Gb300:
    """The mailbox as the GB300 uses it. One per process and device."""

    def __init__(self, gb_lib: str, mailbox: Mailbox, device="cuda"):
        self.lib = ctypes.CDLL(gb_lib)
        self.mb = mailbox
        self.dev = torch.device(device)
        torch.zeros(1, device=self.dev)                  # the context
        sizes = (ctypes.c_longlong * 2)()
        self.lib.rtxband_gb_sizes(sizes)
        assert sizes[0] == 8 * len(STATE_WORDS)
        self.log_words = int(sizes[1])
        self.lib.rtxband_gb_map.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_void_p)]
        self.seg_dev = self._map(mailbox.addr, mailbox.bytes)
        self.state = torch.zeros(len(STATE_WORDS), dtype=torch.int64, device=self.dev)       # GbState
        self.lib.rtxband_gb_trace.argtypes = [ctypes.POINTER(_Gb), ctypes.c_int, ctypes.c_longlong, ctypes.c_int,
                                              ctypes.c_void_p]
        for f in ("rtxband_gb_send", "rtxband_gb_finish", "rtxband_gb_late", "rtxband_gb_hint"):
            getattr(self.lib, f).argtypes = [ctypes.POINTER(_Gb), ctypes.c_void_p]
        self.lib.rtxband_gb_load_start.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_longlong]
        self.lib.rtxband_gb_load_stop.argtypes = [ctypes.c_void_p]
        self.t = None
        self.load_buf = None
        torch.cuda.synchronize(self.dev)

    def _map(self, addr: int, nbytes: int) -> int:
        dptr = ctypes.c_void_p()
        rc = self.lib.rtxband_gb_map(addr, nbytes, ctypes.byref(dptr))
        assert rc == 0, f"host registration failed: {rc}"
        return dptr.value

    def load_trace(self, calls: np.ndarray, xq: np.ndarray, xs: np.ndarray, hints: np.ndarray):
        """calls uint8 [n, 608], xq uint8 [n, T, 3584], xs uint8 [n, T, 112], hints int32 [n, 17]."""
        n, T = xq.shape[0], xq.shape[1]
        d = self.dev
        self.t = dict(n=n, T=T, calls=torch.from_numpy(np.ascontiguousarray(calls)).to(d),
                      xq=torch.from_numpy(np.ascontiguousarray(xq)).to(d),
                      xs=torch.from_numpy(np.ascontiguousarray(xs)).to(d),
                      hints=torch.from_numpy(np.ascontiguousarray(hints, dtype=np.int32)).to(d),
                      dst=torch.zeros((n, T, H), dtype=torch.float32, device=d),
                      late=torch.zeros((n, T, H), dtype=torch.float32, device=d),
                      log=torch.zeros((n, self.log_words), dtype=torch.int64, device=d))
        torch.cuda.synchronize(d)

    def args(self, hint_lead_us, timeout_us: float, work_us: float = 0.0, layers: int = 92) -> _Gb:
        t = self.t
        hinted = hint_lead_us is not None
        return _Gb(self.seg_dev, self.state.data_ptr(), t["xq"].data_ptr(), t["T"] * H, t["xs"].data_ptr(),
                   t["T"] * (H // 32), t["calls"].data_ptr(), 608, t["hints"].data_ptr() if hinted else None, 17,
                   t["dst"].data_ptr(), t["T"] * H, t["late"].data_ptr(), t["T"] * H, t["log"].data_ptr(),
                   self.log_words, layers, 0, int(timeout_us * 1000), int(work_us * 1000),
                   int(hint_lead_us * 1000) if hinted else 0)

    def run(self, period_us: float = 234.0, hint_lead_us=None, timeout_us: float = 2000.0, work_us: float = 0.0,
            with_late: int = 0, n: int = None):
        """Replay the loaded trace (its first n calls): per call [late,] [hint,] send, finish. hint_lead_us: the
        call's hint list goes out first and the send follows that long after it. work_us holds the finish back
        until that long after the send (the layer's own kernels). with_late 1: band_late before every send (the
        engine's order), 2: right behind the finish (a late call is then always lost). Returns a dict: log int64
        [n, 16], dst and late float32 [n, T, 3584] (rows added on time / collected late), host int64 [n, 2] ns,
        state (dict)."""
        t = self.t
        n = n or t["n"]
        t["dst"].zero_(); t["late"].zero_(); t["log"].zero_()
        self.state.zero_()
        self.state[0] = self.mb.next_seq() - 1           # continue above the mailbox's sequence
        torch.cuda.synchronize(self.dev)
        a = self.args(hint_lead_us, timeout_us, work_us)
        host = np.zeros((n, 2), dtype=np.int64)
        rc = self.lib.rtxband_gb_trace(ctypes.byref(a), n, int(period_us * 1000), int(with_late), host.ctypes.data)
        assert rc == 0, f"trace failed: CUDA error {rc}"
        st = self.state.cpu().numpy()
        return dict(log=t["log"][:n].cpu().numpy(), dst=t["dst"][:n].cpu().numpy(), late=t["late"][:n].cpu().numpy(),
                    host=host, state={k: int(v) for k, v in zip(STATE_WORDS, st)})

    # --- a stand-in for the engine's stager: the GB300 streams host memory while the trace runs
    def load_start(self, gib: float = 2.0, grid: int = 128, seconds: float = 2.0):
        if self.load_buf is None or self.load_buf[0].nbytes != int(gib * 2**30):
            buf = np.empty(int(gib * 2**30), dtype=np.uint8)
            buf[::65536] = 1                             # touch every page
            self.load_buf = (buf, self._map(buf.ctypes.data, buf.nbytes))
        rc = self.lib.rtxband_gb_load_start(self.load_buf[1], self.load_buf[0].nbytes, grid, int(seconds * 1e9))
        assert rc == 0, f"load launch failed: {rc}"

    def load_stop(self) -> float:
        """Stop the stream; returns its rate in GB/s."""
        out = (ctypes.c_longlong * 2)()
        rc = self.lib.rtxband_gb_load_stop(out)
        assert rc == 0, f"load stop failed: {rc}"
        return out[0] / max(out[1], 1)
