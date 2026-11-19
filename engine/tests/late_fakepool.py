"""A stand-in for the Grace pool (engine/pool, libcpustage.so) in the late tier's tests: one Python thread that serves
the jobs of a cpustage_seg2.h segment the way the slot-ring worker does (cpustage_fifo.cpp.txt), with the pool's
arithmetic in float64 (k31_engine.late.pool_reference on late.decode_hi2's weights) instead of the NEON kernel.

  - jobs run in sequence order: `next` follows the door's newest sequence number; a job already NSLOT - 1 behind the
    door is skipped as stale, as is a slot whose header tag is not the one expected (overwritten by a newer job);
  - a header the pool cannot run (counts, tag fields, unknown layer, a record past the bank) is refused: status
    CPUSEG_ST_ERROR and the done tag, no rows;
  - rows, then the status word, then the done tag, each control word a release store (late_host_set), so the GB300's
    acquire of the done tag sees the rows; the door is read with an acquire load (late_host_get);
  - test knobs: `delay_s` before publishing, `drop` (sequence numbers never answered), `refuse` (answered with an error).
The control words go through liblate.so's host helpers (latetier.Segment.get / set)."""
import atexit
import struct
import threading
import time

import numpy as np
import torch

from k31_engine import late

_LIVE = []                                     # pools still running at exit are stopped there


@atexit.register
def _stop_all():
    for p in list(_LIVE):
        p.stop()


HDR = struct.Struct("<QIIIIffiI8i64f")        # CpuJob2: tag, n_pick, n_rows, layer, flags, alpha, beta, e_min, pad,
assert HDR.size == 328                          # slot[8], w[8][8]
ERROR = 1


def dequant_rows(xq: np.ndarray, xe: np.ndarray) -> torch.Tensor:
    """The segment's int8 rows (row-planar: byte 8 p + i of a block = input 4 i + p) and exponent bytes (e + 128)
    -> float64 [T, 3584]."""
    T = xq.shape[0]
    q = xq.view(np.int8).reshape(T, late.H // 32, 32)
    k = np.arange(32)
    vals = q[:, :, 8 * (k & 3) + (k >> 2)].astype(np.float64)
    e = xe.reshape(T, late.H // 32).astype(np.float64) - 128.0
    return torch.from_numpy((vals * np.exp2(e)[..., None]).reshape(T, late.H))


def pack_header(tag, n_pick, n_rows, layer, flags, alpha, beta, e_min, slots, w) -> bytes:
    """CpuJob2's bytes (little-endian, padded to the job slot's 512 B)."""
    s = list(slots) + [-1] * (8 - len(slots))
    wf = np.zeros((8, 8), dtype=np.float32)
    wa = np.asarray(w, dtype=np.float32)
    wf[:wa.shape[0], :wa.shape[1]] = wa
    return HDR.pack(tag, n_pick, n_rows, layer, flags, alpha, beta, e_min, 0, *s[:8], *wf.reshape(-1).tolist()) + \
        bytes(512 - HDR.size)


class FakePool:
    def __init__(self, seg, banks: dict, device="cpu", delay_s: float = 0.0):
        """seg: latetier.Segment; banks: {layer: uint8 tensor [records, 9,289,728] (host)}."""
        self.seg, self.lo, self.banks = seg, seg.lo, banks
        self.device = device
        self.delay_s = delay_s
        self.drop, self.refuse = set(), set()
        self.cache = {}
        self.served = []                     # (seq, status) in the order published
        self.stale = 0
        self.errors = []
        self.next = (seg.get(self.lo["door"]) >> 16) + 1
        self.stop_flag = False
        self.th = None
        # its GPU math on a stream of its own: torch's pooled streams are non-blocking, so nothing here waits behind
        # (or holds up) the legacy default stream, where a landing may be spinning for this very job
        self.cuda_stream = torch.cuda.Stream() if str(device).startswith("cuda") else None

    def start(self) -> None:
        self.seg.set(0, self.lo["magic"])
        self.seg.set(self.lo["state"], 1)                 # running
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        _LIVE.append(self)

    def stop(self) -> None:
        self.stop_flag = True
        if self.th is not None:
            self.th.join()
            self.th = None
        self.seg.set(self.lo["state"], 0)
        if self in _LIVE:
            _LIVE.remove(self)

    def wait_served(self, n: int, timeout_s: float = 120.0) -> bool:
        t0 = time.time()
        while len(self.served) < n:
            if time.time() - t0 > timeout_s or self.errors:
                return False
            time.sleep(0.001)
        return True

    def _experts(self, layer: int, slot: int):
        k = (layer, slot)
        if k not in self.cache:
            self.cache[k] = late.decode_hi2(self.banks[layer][slot].to(self.device))
        return self.cache[k]

    def _job(self, s: int, hdr):
        tag, n_pick, n_rows, layer, flags, alpha, beta, e_min = hdr[:8]
        slots = hdr[9:17]
        w = np.asarray(hdr[17:], dtype=np.float32).reshape(8, 8)
        bank = self.banks.get(layer)
        if not (1 <= n_pick <= 8 and 1 <= n_rows <= 8 and n_rows == (tag & 0xFF) and
                (layer & 0xFF) == ((tag >> 8) & 0xFF) and bank is not None and alpha > 0 and beta >= 0 and
                all(0 <= slots[p] < bank.shape[0] for p in range(n_pick))):
            return ERROR, None
        T = n_rows
        xo = self.seg.x_off(s)
        xq = self.seg.u8[xo:xo + T * late.H].copy().reshape(T, late.H)
        xe = self.seg.u8[xo + late.MAXT * late.H:xo + late.MAXT * late.H + T * (late.H // 32)].copy()
        xd = dequant_rows(xq, xe).to(self.device)
        experts = [self._experts(layer, slots[p]) for p in range(n_pick)]
        wt = torch.from_numpy(w[:n_pick, :T].copy()).to(self.device)
        rows = late.pool_reference(experts, xd, wt, flags, alpha, beta)
        return 0, rows.float().cpu().numpy()

    def _run(self) -> None:
        NS = self.lo["nslot"]
        try:
            while not self.stop_flag:
                newest = self.seg.get(self.lo["door"]) >> 16
                if newest < self.next:
                    if newest + NS < self.next:            # a restarted sender: follow the door down
                        self.next = newest + 1
                    time.sleep(0.0002)
                    continue
                if newest - self.next >= NS - 1:           # too far behind: the oldest jobs are past their landing
                    self.stale += newest - self.next - (NS - 2)
                    self.next = newest - (NS - 2)
                seq = self.next
                self.next += 1
                s = seq % NS
                hdr = HDR.unpack_from(self.seg.buf, self.seg.job_off(s))
                if hdr[0] >> 16 != seq:                    # overwritten by a newer job
                    self.stale += 1
                    continue
                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        status, rows = self._job(s, hdr)
                else:
                    status, rows = self._job(s, hdr)
                if seq in self.drop:
                    continue
                if self.delay_s:
                    time.sleep(self.delay_s)
                if seq in self.refuse:
                    status, rows = ERROR, None
                if rows is not None:
                    self.seg.rows(s, rows.shape[0])[:] = rows
                done = self.seg.done_off(s)
                self.seg.set(done + 8, status)            # release: the rows first
                self.seg.set(done, hdr[0])                # release: rows and status before the tag
                self.served.append((seq, status))
        except Exception as e:                             # noqa: BLE001 (reported by the test)
            self.errors.append(repr(e))
            raise
