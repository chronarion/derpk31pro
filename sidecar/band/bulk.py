"""The bulk band request (prompt chunks), sidecar side.

A prompt chunk's band picks cost one 2-bit record pass on the RTX per (8-row call, expert) when they go through the
decode calls: 8 consecutive rows hold ~19 band picks of ~17 distinct experts, so a call reads ~17 records for ~19
picks, and every call is a round trip with a host sync on the GB300. In bulk, a chunk's band picks travel once per MoE
layer: the GB300 posts the chunk's MXFP8 rows and each pick's band store slot and weight into a second shared segment
(<mailbox>.bulk), this process groups the picks expert by expert into units of at most 8 picks of one expert, runs the
band's launched kernels on 16 units per launch (FLAG_PER_EXPERT: a unit's own 8 rows in, its own 8 weighted rows
out), sums every row's unit outputs in a fixed order and returns one [T, 3584] float32 result. A record pass then
serves 8 picks of one expert: (picks / 8 + experts) passes instead of (rows / 8 x ~17).

The persistent service kernel holds every SM of the RTX, so it is parked while a prompt is read: the engine sets
OFF_MODE = 1 before a prompt chunk and 0 before a decode step; this process stops or restarts the kernel and answers in
OFF_ACK. While it is parked, records that enter the band go through the launched repack.

Segment layout (written into the segment for the engine: u64 offsets of xq, xs, slot, w, result, the size and the row
cap at bytes 16-72). Control words own whole 128 B lines:
  OFF_MAGIC "RTXBULK1"; OFF_DOOR u64 tag of the newest request (the GB300's post kernel, release); OFF_MODE u64 (the
  engine); OFF_ACK u64 (this process: the mode it has switched to); OFF_DONE u64 tag of the newest served request
  (written last, release); OFF_STATUS u64 (0 ok, 1 error, 2 no band pick); OFF_TIMES u64[4] ns (copy in, plan + pack,
  launches, gather + copy out) of the newest request; OFF_HDR the request's header (T, layer, flags, sf_min, alpha,
  beta, n_units). Then xq [MAX_ROWS][3584] E4M3, xs [MAX_ROWS][112] UE8M0, slot [MAX_ROWS][16] int16 (-1 = not a band
  pick), w [MAX_ROWS][16] float32, result [MAX_ROWS][3584] float32.
A tag is seq << 24 | layer << 16 | (T - 1): never zero, never reused within a server's life.
"""
import ctypes
import mmap
import os
import struct
import time

import numpy as np
import torch

H, MAXT, MAXE, K = 3584, 8, 16, 16
XF_BYTES = 29568
CALL_INTS = 8 + MAXE + MAXE * MAXT
FLAG_PER_EXPERT = 1024
MAX_ROWS = int(os.environ.get("K31_BAND_BULK_ROWS", "4096"))
MAGIC = b"RTXBULK1"
OFF_MAGIC, OFF_DOOR, OFF_MODE, OFF_ACK, OFF_DONE = 0, 128, 256, 384, 512
OFF_STATUS, OFF_TIMES, OFF_HDR = 520, 528, 640


def _align(n: int) -> int:
    return (n + 4095) // 4096 * 4096


def layout(max_rows: int = MAX_ROWS) -> dict:
    o = {"xq": 4096}
    o["xs"] = o["xq"] + max_rows * H
    o["slot"] = o["xs"] + max_rows * (H // 32)
    o["w"] = o["slot"] + max_rows * K * 2
    o["res"] = _align(o["w"] + max_rows * K * 4)
    o["bytes"] = _align(o["res"] + max_rows * H * 4)
    return o


HDR = struct.Struct("<IIIiffI")             # T, layer, flags, sf_min, alpha, beta, n_units (written back)


class BulkServer:
    """This process's side of the bulk segment: created here, served from the sidecar's main loop."""

    def __init__(self, band, host_lib: str, path: str, max_rows: int = MAX_ROWS):
        self.band, self.dev, self.max_rows = band, band.dev, max_rows
        self.L = layout(max_rows)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            os.ftruncate(fd, self.L["bytes"])
            self.buf = mmap.mmap(fd, self.L["bytes"])
        finally:
            os.close(fd)
        os.chmod(path, 0o666)
        self.path = path
        u8 = np.frombuffer(self.buf, dtype=np.uint8)
        u8[:4096] = 0
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        rc = int(torch.cuda.cudart().cudaHostRegister(self.addr, self.L["bytes"], 0))
        if rc != 0:
            raise RuntimeError(f"bulk segment: cudaHostRegister failed ({rc})")
        self.host = ctypes.CDLL(host_lib)
        self.host.bandhost_get.restype = ctypes.c_uint64
        self.host.bandhost_get.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.host.bandhost_set.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        mx = max_rows
        t = torch.frombuffer(self.buf, dtype=torch.uint8)
        self.xq_h = t[self.L["xq"]:self.L["xq"] + mx * H].view(mx, H)
        self.xs_h = t[self.L["xs"]:self.L["xs"] + mx * (H // 32)].view(mx, H // 32)
        self.slot_h = t[self.L["slot"]:self.L["slot"] + mx * K * 2].view(torch.int16).view(mx, K)
        self.w_h = t[self.L["w"]:self.L["w"] + mx * K * 4].view(torch.float32).view(mx, K)
        self.res_h = t[self.L["res"]:self.L["res"] + mx * H * 4].view(torch.float32).view(mx, H)
        d = self.dev
        self.xq = torch.empty((mx, H), dtype=torch.uint8, device=d)
        self.xs = torch.empty((mx, H // 32), dtype=torch.uint8, device=d)
        self.slot = torch.empty((mx, K), dtype=torch.int16, device=d)
        self.w = torch.empty((mx, K), dtype=torch.float32, device=d)
        self.res = torch.empty((mx, H), dtype=torch.float32, device=d)
        self.xf = self.yu = None                 # unit buffers, grown on demand
        V = ctypes.c_void_p
        lib = band.lib
        lib.rtxband_pack_units.argtypes = [V, V, V, ctypes.c_int, V, V]
        lib.rtxband_run_bulk.argtypes = [ctypes.c_void_p, V, ctypes.c_int, V, V, V]
        lib.rtxband_gather_units.argtypes = [V, V, ctypes.c_int, ctypes.c_int, V, V]
        self.last = 0
        self.mode = 0
        self.n = 0
        self.sum_s = [0.0, 0.0, 0.0, 0.0]
        struct.pack_into("<7Q", self.buf, 16, self.L["xq"], self.L["xs"], self.L["slot"], self.L["w"], self.L["res"],
                         self.L["bytes"], max_rows)  # the layout, for the engine (k31_engine/band.py Bulk)
        self.buf[0:8] = MAGIC                    # last: the engine checks it before it maps the segment

    def get(self, off: int) -> int:
        return int(self.host.bandhost_get(self.addr, off))

    def set(self, off: int, v: int) -> None:
        self.host.bandhost_set(self.addr, off, v)

    def wanted(self) -> int:
        return 1 if self.get(OFF_MODE) == 1 else 0

    def ack(self, mode: int) -> None:
        self.mode = mode
        self.set(OFF_ACK, mode)

    def pending(self) -> bool:
        door = self.get(OFF_DOOR)
        return door != 0 and door != self.last

    def _units(self, T: int):
        """Picks -> units of <= 8 picks of one expert (ascending store slot, then ascending row), and for every row
        the output rows of its units in ascending unit order (the gather's fixed summation order)."""
        d = self.dev
        slot = self.slot[:T].to(torch.int32).reshape(-1)
        w = self.w[:T].reshape(-1)
        rows = torch.arange(T, dtype=torch.int64, device=d).repeat_interleave(K)
        m = slot >= 0
        s, r, ww = slot[m].to(torch.int64), rows[m], w[m]
        P = s.numel()
        if P == 0:
            return 0, None, None, None, None
        order = torch.argsort(s * T + r)                    # (slot, row) keys are unique: top-k experts are distinct
        s, r, ww = s[order], r[order], ww[order]
        new = torch.ones(P, dtype=torch.bool, device=d)
        new[1:] = s[1:] != s[:-1]
        run = torch.cumsum(new.to(torch.int64), 0) - 1
        start = torch.nonzero(new).squeeze(1)
        rank = torch.arange(P, device=d) - start[run]
        cnt = torch.diff(torch.cat([start, torch.tensor([P], device=d)]))
        upr = (cnt + MAXT - 1) // MAXT
        ubase = torch.cumsum(upr, 0) - upr
        U = int(upr.sum())
        unit = ubase[run] + rank // MAXT
        pos = rank % MAXT
        NC = (U + MAXE - 1) // MAXE
        U16 = NC * MAXE
        u_slot = torch.zeros(U16, dtype=torch.int32, device=d)
        u_slot[unit] = s.to(torch.int32)
        u_rows = torch.full((U16, MAXT), -1, dtype=torch.int32, device=d)
        u_rows[unit, pos] = r.to(torch.int32)
        u_w = torch.zeros((U16, MAXT), dtype=torch.float32, device=d)
        u_w[unit, pos] = ww
        j = unit * MAXT + pos                               # the pick's output row among the units' rows
        o2 = torch.argsort(r * U + unit)                    # per row, ascending unit
        r2, j2 = r[o2], j[o2]
        newr = torch.ones(P, dtype=torch.bool, device=d)
        newr[1:] = r2[1:] != r2[:-1]
        rs = torch.nonzero(newr).squeeze(1)
        rid = torch.cumsum(newr.to(torch.int64), 0) - 1
        k = torch.arange(P, device=d) - rs[rid]
        idx = torch.full((T, K), -1, dtype=torch.int32, device=d)
        idx[r2, k] = j2.to(torch.int32)
        return U, u_slot, u_rows, u_w, idx

    def serve(self) -> None:
        """One request: rows in, units, launches, gather, result out, then the done tag."""
        tag = self.get(OFF_DOOR)
        T, layer, flags, sf_min, alpha, beta, _ = HDR.unpack_from(self.buf, OFF_HDR)
        status = 0
        t0 = time.perf_counter()
        try:
            if not 1 <= T <= self.max_rows:
                raise ValueError(f"request of {T} rows (at most {self.max_rows})")
            self.xq[:T].copy_(self.xq_h[:T], non_blocking=True)
            self.xs[:T].copy_(self.xs_h[:T], non_blocking=True)
            self.slot[:T].copy_(self.slot_h[:T], non_blocking=True)
            self.w[:T].copy_(self.w_h[:T], non_blocking=True)
            t1 = time.perf_counter()
            U, u_slot, u_rows, u_w, idx = self._units(T)
            if U == 0:
                self.res_h[:T].zero_()
                status = 2
                t2 = t3 = t1
            else:
                NC = (U + MAXE - 1) // MAXE
                if self.xf is None or self.xf.shape[0] < NC * MAXE:
                    self.xf = torch.empty((NC * MAXE, XF_BYTES), dtype=torch.uint8, device=self.dev)
                    self.yu = torch.empty((NC * MAXE * MAXT, H), dtype=torch.float32, device=self.dev)
                calls = torch.zeros((NC, CALL_INTS), dtype=torch.int32, device=self.dev)
                calls[:, 0] = torch.clamp(U - MAXE * torch.arange(NC, device=self.dev), max=MAXE).to(torch.int32)
                calls[:, 1] = MAXT
                calls[:, 2] = flags | FLAG_PER_EXPERT
                calls[:, 3] = sf_min
                calls[:, 4] = struct.unpack("<i", struct.pack("<f", alpha))[0]
                calls[:, 5] = struct.unpack("<i", struct.pack("<f", beta))[0]
                calls[:, 8:8 + MAXE] = u_slot.view(NC, MAXE)
                calls[:, 8 + MAXE:] = u_w.view(NC, MAXE * MAXT).view(torch.int32)
                st = torch.cuda.current_stream(self.dev).cuda_stream
                lib = self.band.lib
                rc = lib.rtxband_pack_units(self.xq.data_ptr(), self.xs.data_ptr(), u_rows.data_ptr(), U,
                                            self.xf.data_ptr(), st)
                assert rc == 0, f"pack launch failed ({rc})"
                t2 = time.perf_counter()
                rc = lib.rtxband_run_bulk(ctypes.byref(self.band.c), calls.data_ptr(), NC, self.xf.data_ptr(),
                                          self.yu.data_ptr(), st)
                assert rc == 0, f"bulk launches failed ({rc})"
                t3 = time.perf_counter()
                rc = lib.rtxband_gather_units(self.yu.data_ptr(), idx.data_ptr(), K, T, self.res.data_ptr(), st)
                assert rc == 0, f"gather launch failed ({rc})"
                self.res_h[:T].copy_(self.res[:T], non_blocking=True)
                HDR.pack_into(self.buf, OFF_HDR, T, layer, flags, sf_min, alpha, beta, U)
            torch.cuda.synchronize(self.dev)
        except Exception as ex:                  # noqa: BLE001  (reported through the status word)
            print(f"[band bulk] request {tag:#x} failed: {ex}", flush=True)
            status = 1
            t2 = t3 = t1 = t0
        t4 = time.perf_counter()
        for i, v in enumerate((t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            self.set(OFF_TIMES + 8 * i, int(v * 1e9))
            self.sum_s[i] += v
        self.set(OFF_STATUS, status)
        self.set(OFF_DONE, tag)                  # last (release): the GB300's finish waits for it
        self.last = tag
        self.n += 1

    def stats(self) -> str:
        n = max(self.n, 1)
        return (f"bulk: {self.n} requests, ms each: copy in {1e3 * self.sum_s[0] / n:.2f}, plan + pack "
                f"{1e3 * self.sum_s[1] / n:.2f}, launches {1e3 * self.sum_s[2] / n:.2f}, gather + out "
                f"{1e3 * self.sum_s[3] / n:.2f}")
