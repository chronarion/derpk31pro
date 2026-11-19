"""Adaptive residency for mode staged: swaps experts into and out of each layer's resident HBM rows at a bounded
rate, the engine's version of the runner's budgeted promotion (k31/policy.py `promote`).

Inputs (device, written by unitk inside every MoE call): per layer the decayed scores of all 896 experts (decay per
token, + squared routing weight per pick) and pick counts (resident, staged, cut, all).
Each tick (after a step's sample_tokens, on the engine thread, never waiting for the GPU):
  1. promotions whose copy has landed are mapped in: their row-map word is written on the main stream (later steps
     route the expert's picks to its row);
  2. the scores of an earlier step (a pinned D2H ring, read only once its copy has landed) give, per layer with
     credit (K31_PROMOTE per layer per step), swaps: the strongest-scored expert of the ideal top-S that is not
     resident against the weakest-scored resident row not in flight, if it scores more;
  3. a swap first unmaps the victim (its row-map word -> -1 on the main stream, so no later step reads the row; it
     can still be staged from the bank), records an event, and hands the new expert to a reader thread: it preads the
     expert's six tensors from the official shards (NVMe), waits for the event on its own stream, writes them into the
     row with the module's kernel-layout loader, and records a done event (1);
  4. starts this step's D2H copy of the scores and counts.
Weights are read-only: eviction never writes back. Logs every 30 s: picks resident / staged / cut, promotions, NVMe
read rate.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

RING = 4
MAX_FLIGHT = int(os.environ.get("K31_PROMOTE_FLIGHT", "16"))      # promotions in flight over all layers
MERGE_BLOCKS = int(os.environ.get("K31_MERGE_BLOCKS", "152"))     # the merge runs between steps: all SMs
# the write-back extract's grid (11-18 sweep, bench_merge_blocks.py: merge 11.4 -> 8.2 us and extract 8.5 -> 6.4 us at
# 1216 blocks, byte-identical output; both grid-stride loops hold one 16 B load per thread in flight)
EXTRACT_BLOCKS = int(os.environ.get("K31_EXTRACT_BLOCKS", "152"))
BANK_FLIGHT = int(os.environ.get("K31_BANK_FLIGHT", "32"))        # bank refills in flight over all layers
LOG_S = float(os.environ.get("K31_RES_LOG_S", "30"))              # seconds between the residency's log lines
# RTX band swaps (K31_BAND_SWAP=1): a band expert promoted to a native row hands its band slot to the evicted native
# expert; the sidecar enters that record (~1 ms each, one at a time) and the band map changes when it acks
BAND_SWAP = os.environ.get("K31_BAND_SWAP", "0") == "1"
# K31_BAND_FEED=P: per layer and accepted token P credits of band entries; with a
# credit, the strongest-scored expert that has a bank record, no native row and no band slot replaces the weakest-scored
# band member if it scores 10% more (at most 2 per layer per tick). The victim is unmapped at once, the entry is posted
# once the GPU is past this tick's map write, the entrant is mapped when the sidecar acks, and the victim gets the
# entrant's freed bank slot (a refill from NVMe), so the band holds the experts just under the native set and the bank
# the tier below. Needs the sidecar's command ring (created for K31_BAND_SWAP or K31_BAND_FEED).
BAND_FEED = float(os.environ.get("K31_BAND_FEED", "0") or 0)
BAND_FEED_MARGIN = float(os.environ.get("K31_BAND_FEED_MARGIN", "1.1"))
BAND_SWAP_FLIGHT = int(os.environ.get("K31_BAND_SWAP_FLIGHT", "64"))
# K31_PROMOTE_BAND_MARGIN: a band member (served on time from the RTX) is promoted to HBM only if it
# outscores the weakest resident expert by this factor: the hand-off costs two NVMe records and a band entry for a
# tenth of what a bank -> band entry buys (78% of promotions and 52% of band entries were hand-offs, fw 11-18)
PROMOTE_BAND_MARGIN = float(os.environ.get("K31_PROMOTE_BAND_MARGIN", "1.0"))
# K31_FADV_RANDOM=1: the residency's record reads (promotions, refills) without the kernel's read-ahead: a 9.3 / 17.6 MB
# record read pulled ~1.2 MB more past its end (~1.6 GB/s of the one model drive)
FADV_RANDOM = os.environ.get("K31_FADV_RANDOM", "0") == "1"
# K31_HAND_FEED=1: a band member promoted to HBM frees its band slot (per-layer free list) instead of
# taking the evicted expert in (a hand-off: an NVMe read for the band and a refill, ~2.1 GB/s); the evicted expert
# goes to the bank by the write-back path; _band_feed fills free slots first with the layer's best bank candidate
# (no victim, no margin), and on the ack that entrant's bank slot is freed for the refills
HAND_FEED = os.environ.get("K31_HAND_FEED", "0") == "1"
# K31_CREDIT_WEIGHTS=FILE ({"layers": {"<layer>": w}}, mean 1): promotion, bank and feed credit accrue per layer x w,
# and the "largest gains first" orders use w x gain (the pipeline is capacity-bound, layers' gains differ)
CREDIT_WEIGHTS = os.environ.get("K31_CREDIT_WEIGHTS", "")
# K31_BAND_RING=R (> 0): a band entrant that holds a Grace bank record enters from it: a
# worker copies the record into a row of the sidecar's /dev/shm ring (<segment>.rec, R rows, created and pinned by the
# sidecar) and the entry is posted with rec = row + 1, so the sidecar reads nothing from the model drive for it; the
# bank slot is held (bank_ref) during the copy and the row is free again at the acknowledgement. Entrants without a
# record (hand-offs) still come from the drive. Must match the sidecar's K31_BAND_RING (k31.sh passes it).
BAND_RING = int(os.environ.get("K31_BAND_RING", "0") or 0)
# K31_RES_STREAM=1: the tick's GPU work (write-back extracts, promotion merges, the three map writes) on its own stream,
# started after the forward in flight and joined by the next CUDA graph replay (or, without one, by the first MoE
# layer's call): the target head, the sampler and the drafter, which read neither expert rows nor the maps, run beside
# it instead of behind it (~0.1-0.2 ms per step). Every write still follows the forward that might read the old state,
# and the next forward still follows every write.
RES_STREAM = os.environ.get("K31_RES_STREAM", "0") == "1"


def _install_join(res) -> None:
    """Wrap CUDAGraph.replay (outermost, so the replay timer does not count the wait): before a replay, the current
    stream waits for the last tick's GPU work."""
    orig = torch.cuda.CUDAGraph.replay
    if getattr(orig, "_k31_join", False):
        return

    def replay(self, _orig=orig):
        ev = res.tick_ev
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            res.tick_ev = None
        return _orig(self)
    replay._k31_join = True
    torch.cuda.CUDAGraph.replay = replay
_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")                  # noqa: E731


class Residency:
    def __init__(self, layers: dict, native_dir: str, promote: float, workers: int, dev, write_expert,
                 store: str = "", layout: dict = None, bank_promote: float = 0.0):
        self.idx = sorted(layers)
        self.st = [layers[i] for i in self.idx]
        self.P = promote
        self.dev = dev
        self.native = native_dir
        self.write_expert = write_expert
        self.store, self.layout = store, layout          # the hi2 / lo2 record store (promotions by merge)
        self.merge_ok = layout is not None and all(
            os.path.exists(os.path.join(store, f"L{st.idx:02d}.lo2")) for st in self.st)
        R = layout["record_bytes"] if layout else 0
        # a lo2 record keeps the hi2-k1 layout whatever the hi2 format (recfmt.py): its own size
        self.RL = int(layout.get("lo2_record_bytes", R)) if layout else 0
        self.k2 = bool(layout) and layout.get("format") == "hi2-k2"
        self.n_wb_bad = 0
        self.bufs = [(torch.empty(R, dtype=torch.uint8, device=dev), torch.empty(self.RL, dtype=torch.uint8, device=dev))
                     for _ in range(MAX_FLIGHT)] if self.merge_ok else []
        self.buf_ev = [None] * MAX_FLIGHT
        self.free = list(range(MAX_FLIGHT))
        self.upd_pin = [torch.zeros((4 * MAX_FLIGHT + 4 * BANK_FLIGHT + 8, 3), dtype=torch.int64, pin_memory=True)
                        for _ in range(8)]
        self.upd_ev = [None] * 8
        self.upd_head = 0
        n = len(self.idx)
        # the adaptive Grace bank (K31_BANK_PROMOTE refills per layer per token): each layer's bank slots follow the
        # strongest non-resident experts by score; slot -> expert in bank_owner, expert -> slot in bank_slot
        self.PB = bank_promote
        self.Gl = np.array([len(st.bank) for st in self.st], dtype=np.int64)      # K31_BANK_MAP: per layer
        self.G = int(self.Gl.max()) if n else 0
        self.bank_slot = np.stack([np.asarray(st.bank_idx, dtype=np.int64) for st in self.st])        # [L, 896]
        self.bank_owner = np.full((n, max(self.G, 1)), -1, dtype=np.int64)
        for j in range(n):
            for e in np.nonzero(self.bank_slot[j] >= 0)[0]:
                self.bank_owner[j, self.bank_slot[j, e]] = e
        self.bank_ref = np.zeros((n, max(self.G, 1)), dtype=np.int64)       # promotions reading a slot
        # counts the times a slot was mapped to a record (after a refill or a write-back landed): a reader that
        # copied a slot's record ahead of its use (celane's pool, K31_PROMPT_SPEC) checks owner and count again
        self.bank_gen = np.zeros((n, max(self.G, 1)), dtype=np.int64)
        self.bank_busy = np.zeros((n, max(self.G, 1)), dtype=bool)          # slots being refilled
        for j in range(n):                       # a smaller layer's padding: busy for good, never a victim or target
            self.bank_busy[j, self.Gl[j]:] = True
        self.bank_coming = np.zeros((n, 896), dtype=bool)
        # experts with a record in the RTX band (static): the bank never refills them and an evicted one is not
        # written back (the band serves it)
        self.banded = np.zeros((n, 896), dtype=bool)
        for j, st in enumerate(self.st):
            if getattr(st, "band", None):
                self.banded[j, np.asarray(st.band, dtype=np.int64)] = True
        self.band_slot = np.full((n, 896), -1, dtype=np.int64)
        self.band_map = self.band_cmd = None
        self.band_pending = {}                                              # seq -> (layer j, expert, slot)
        self.band_wait = []          # [event, j, expert, slot]: posted once the GPU is past the tick that freed the slot
        self.n_bswap = self.n_bswap_ok = self.n_bswap_skip = 0
        self.band_coming = np.zeros((n, 896), dtype=bool)   # score-fed entries posted or waiting, not yet acked
        self.band_ok = np.array([getattr(st, "band_ok", True) for st in self.st], dtype=bool)   # K31_POLICY band off
        self.band_free = [[] for _ in range(n)]          # K31_HAND_FEED: band slots freed by promotions, per layer
        self.rec_ring, self.ring_free, self.ring_copy = None, [], []   # K31_BAND_RING (bound with the command ring)
        self.n_ring = self.n_ring_drive = self.n_ring_err = 0
        self.n_handfree = self.n_handfeed = self.n_handfeed_ok = 0
        self.cw = np.ones(n)
        if CREDIT_WEIGHTS:
            import json
            raw = json.load(open(CREDIT_WEIGHTS))
            wmap = {int(k): float(v) for k, v in raw.get("layers", raw).items()}
            self.cw = np.array([wmap.get(st.idx, 1.0) for st in self.st])
        self.fcredit = np.zeros(n)
        self.feed_bjobs = []                                 # (j, victim, bank slot): refills queued by acked entries
        self.n_feed = self.n_feed_ok = self.n_feed_fail = self.n_feed_refill = 0
        if all(getattr(st, "band_map", None) is not None for st in self.st):
            for j, st in enumerate(self.st):
                self.band_slot[j] = st.band_map.cpu().numpy()
            self.banded = self.band_slot >= 0
            self.band_map = torch.stack([st.band_map for st in self.st])        # pooled (before any graph capture)
            for j, st in enumerate(self.st):
                st.band_map = self.band_map[j]
            from .hook import BAND, BAND_LIB
            if BAND_SWAP or BAND_FEED > 0:
                from .band import BandCmd
                self.band_cmd = BandCmd(BAND + ".cmd", os.path.join(os.path.dirname(BAND_LIB), "libbandhost.so"))
                if BAND_RING > 0:
                    R = self.layout["record_bytes"]
                    rows = os.path.getsize(BAND + ".rec") // R
                    if rows != BAND_RING:
                        raise RuntimeError(f"K31_BAND_RING={BAND_RING} but the sidecar's ring {BAND}.rec has {rows} rows")
                    self.rec_ring = np.memmap(BAND + ".rec", dtype=np.uint8, mode="r+", shape=(rows, R))
                    self.ring_free = list(range(rows))
        self.bcredit = np.zeros(n)
        self.bank_flight = {}                                                # (layer j, slot) -> (expert, future)
        self.n_bank = 0
        if self.G and all(st.bank_map is not None for st in self.st):
            self.bank_map = torch.stack([st.bank_map for st in self.st])        # pooled, like the row map
            for j, st in enumerate(self.st):
                st.bank_map = self.bank_map[j]
        self.tick_s, self.n_tick, self.tick_s_log, self.n_tick_log = 0.0, 0, 0.0, 0
        n = len(self.idx)
        self.score = torch.zeros((n, 896), dtype=torch.float32, device=dev)
        self.stats = torch.zeros((n, 4), dtype=torch.int32, device=dev)
        # K31_ACCEPTED: scores and promotion credit from accepted rows only (rejected draft rows route on tokens that
        # never happened); unitk parks a call's picks in pend_*, commit() adds the accepted rows after sampling
        self.accepted = os.environ.get("K31_ACCEPTED", "0") == "1"
        # (16 rows: a call; with K31_DECODE_SPLIT a step's rows come as several calls per layer: hook.PEND_ROWS)
        pr = 16
        if os.environ.get("K31_DECODE_SPLIT", "0") == "1":
            from .hook import PEND_ROWS as pr
        self.pend_ids = torch.full((n, pr, 16), -1, dtype=torch.int32, device=dev)
        self.pend_w2 = torch.zeros((n, pr, 16), dtype=torch.float32, device=dev)
        self.pend_t = torch.zeros(n, dtype=torch.int32, device=dev)
        self.acc = torch.zeros(1, dtype=torch.int64, device=dev)
        # K31_BANK_EXCL: promotions swap with the bank (write-back of the evicted row's hi2 record into the promoted
        # expert's slot); a small pool of device staging buffers for the extracted records
        self.swap_wb = os.environ.get("K31_BANK_EXCL", "0") == "1" and layout is not None
        nwb = int(os.environ.get("K31_WB_BUFS", "64")) if self.swap_wb else 0   # ~15 promotions land per tick; a
                                            # buffer returns ~3 ticks later (8 held write-backs at ~17% of promotions)
        self.wb_bufs = [torch.empty(layout["record_bytes"], dtype=torch.uint8, device=dev) for _ in range(nwb)]
        self.wb_free = list(range(nwb))
        self.wb_flight = []
        self.wb_stream = torch.cuda.Stream(dev) if nwb else None
        self.n_wb = 0
        self.n_wb_evict = 0                 # write-backs that took a weaker record's slot (no slot freed by a swap)
        self.last_scores = None
        self.acc_seen = 0
        self.n_acc = 0
        self.eager_tokens = 0              # rows of eager rounds (prompt chunks) since the last credit
        self.row_map = torch.stack([st.row_map for st in self.st])          # one pooled row map: one write per tick
        for j, st in enumerate(self.st):
            st.score, st.stats, st.row_map = self.score[j], self.stats[j], self.row_map[j]
            st.pend = (self.pend_ids[j], self.pend_w2[j], self.pend_t[j:j + 1]) if self.accepted else None
        self.ring = [(torch.empty((n, 896), dtype=torch.float32, pin_memory=True),
                      torch.empty((n, 4), dtype=torch.int32, pin_memory=True),
                      torch.zeros(1, dtype=torch.int64, pin_memory=True)) for _ in range(RING)]
        self.ring_ev = [None] * RING
        self.ring_head = 0                 # next ring slot to write
        self.ring_tail = 0                 # oldest copy not yet read
        self.side = torch.cuda.Stream(dev)
        self.rstream, self.tick_ev = None, None          # K31_RES_STREAM
        if RES_STREAM:
            _install_join(self)
        self.credit = np.zeros(n)
        self.row_of = [np.full(896, -1, dtype=np.int64) for _ in self.st]    # expert -> resident row (or -1)
        self.res = []                                                        # row -> expert, per layer
        # stage 1 ring (K31_RING): the last RING rows of a layer belong to the device's ring (defer.plan_ring maps and
        # evicts them); the host promotes into and evicts from the native rows only, and a ring member counts as
        # not resident here (it can be promoted to a native row; its bank record is kept)
        self.ring_rows = int(os.environ.get("K31_RING", "0")) if os.environ.get("K31_DEFER", "0") == "1" else 0
        for j, st in enumerate(self.st):
            r = {}
            lim = st.slots + st.S - (self.ring_rows if st.ring_ok else 0)
            for e, row in enumerate(st.row):
                if 0 <= row < lim:
                    self.row_of[j][e] = row
                    r[row] = e
            self.res.append(r)
        self.flight = {}                   # (layer j, row) -> (expert, future, buffer)
        self.resident = np.stack([r >= 0 for r in self.row_of])             # [layers, 896]
        self.coming_m = np.zeros_like(self.resident)
        nat = [st.S - (self.ring_rows if st.ring_ok else 0) for st in self.st]
        # the ideal set's size for promotions, per layer (K31_HBM_MAP gives layers different resident row counts)
        self.Hl = np.array(nat, dtype=np.int64)
        self.H = int(self.Hl.max())
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="k31-promote")
        self.local = threading.local()
        self.n_done = 0
        self.n_bytes = 0
        self.last_stats = None
        self.t_log = time.time()
        self.n_done_log = 0
        self.bytes_log = 0

    # ------------------------------------------------------------------------------------------- reader threads
    def _job(self, j, e, buf, slot=-1):
        """Reader thread: the expert's hi2 record (the layer's Grace bank, or NVMe) and lo2 record (NVMe) into device
        buffer pair `buf` by copy-engine DMA (no SMs: a kernel beside the persistent MegaMoE calls would stall
        them). The merge into the row runs on the main stream between steps (tick). Without lo2 files: the official
        shards, written by the module's loader on this thread's stream (old path)."""
        st = self.st[j]
        loc = self.local
        if not hasattr(loc, "stream"):
            torch.cuda.set_device(self.dev)
            loc.stream = torch.cuda.Stream(self.dev)
            loc.fds = {}
        s = loc.stream
        if self.merge_ok:
            R, RL = self.layout["record_bytes"], self.RL
            if not hasattr(loc, "lo"):
                loc.lo = torch.empty(RL, dtype=torch.uint8, pin_memory=True)
                loc.hi = torch.empty(R, dtype=torch.uint8, pin_memory=True)
            ev_free = self.buf_ev[buf]
            if ev_free is not None:
                ev_free.synchronize()                    # the previous merge from this buffer pair has run
            d_hi, d_lo = self.bufs[buf]
            nbytes = self._pread(loc, os.path.join(self.store, f"L{st.idx:02d}.lo2"), loc.lo, e * RL)
            with torch.cuda.stream(s):
                d_lo.copy_(loc.lo, non_blocking=True)
                if slot >= 0:                            # the slot is held (bank_ref) until this promotion lands
                    d_hi.copy_(st.bank_cpu[slot], non_blocking=True)
                else:
                    nbytes += self._pread(loc, os.path.join(self.store, f"L{st.idx:02d}.hi2"), loc.hi, e * R)
                    d_hi.copy_(loc.hi, non_blocking=True)
                done = torch.cuda.Event()
                done.record(s)
            done.synchronize()                           # the thread's pinned buffers are reused by its next job
            return done, nbytes
        from .ckpt import Experts
        if not hasattr(loc, "ck"):
            loc.ck = Experts(self.native, workers=2, buf_bytes=64 << 20)
            loc.ck.fadv_random = FADV_RANDOM
        t = loc.ck.read_layer(st.idx, [e])[e]
        return t, loc.ck.last_bytes

    def _bank_job(self, j, e, slot, unmapped):
        """Reader thread: refill bank slot `slot` of layer j with expert e's hi2 record, once no queued step can
        still read the slot (the event recorded after its unmap)."""
        loc = self.local
        if not hasattr(loc, "stream"):
            torch.cuda.set_device(self.dev)
            loc.stream = torch.cuda.Stream(self.dev)
            loc.fds = {}
        unmapped.synchronize()
        st = self.st[j]
        R = self.layout["record_bytes"]
        return self._pread(loc, os.path.join(self.store, f"L{st.idx:02d}.hi2"), st.bank_cpu[slot], e * R)

    @staticmethod
    def _pread(loc, path, buf, pos) -> int:
        fd = loc.fds.get(path)
        if fd is None:
            fd = loc.fds[path] = os.open(path, os.O_RDONLY)
            if FADV_RANDOM:                      # no read-ahead past a record (~1.6 GB/s of the model drive)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        view, n = memoryview(buf.numpy()).cast("B"), 0
        while view:
            k = os.preadv(fd, [view], pos)
            if k <= 0:
                raise IOError(f"short read in {path} at {pos}")
            view, pos, n = view[k:], pos + k, n + k
        return n

    # ------------------------------------------------------------------------------------------------- tick
    def tick(self, tokens: int = 1):
        t0 = time.perf_counter()
        self.tokens = tokens
        self._tick()
        self.tick_s += time.perf_counter() - t0
        self.n_tick += 1

    def _lap(self, name, t):
        now = time.perf_counter()
        self.sect[name] = self.sect.get(name, 0.0) + now - t
        return now

    def _tick(self):
        if not hasattr(self, "sect"):
            self.sect, self.sect_log = {}, {}
        t = time.perf_counter()
        main = torch.cuda.current_stream(self.dev)
        gs = main                                                    # the stream of this tick's GPU work
        if RES_STREAM:
            if self.rstream is None:
                self.rstream = torch.cuda.Stream(self.dev)
            gs = self.rstream
            gs.wait_stream(main)                                     # after the forward in flight
        upd = []                                                     # (layer j, expert, row or -1), applied at once
        merged = []
        bupd = []                                                    # bank map updates (layer, expert, slot or -1)
        band_upd = []                                                # band map updates (layer, expert, slot or -1)
        if self.band_wait or self.ring_copy:                         # slots no step queued before can still read
            self._band_post_ready()
        if self.band_cmd is not None and self.band_pending:          # band entries the sidecar has acknowledged
            for seq, status in self.band_cmd.poll():
                j, v, s_b, victim, r = self.band_pending.pop(seq)
                if r >= 0:
                    self.ring_free.append(r)
                self.band_coming[j, v] = False
                if status == 1:
                    self.band_slot[j, v] = s_b
                    self.banded[j, v] = True
                    band_upd.append((j, v, s_b))
                    self.n_bswap_ok += 1
                    if victim == -2:                     # hand-feed: the entrant's bank record is free for refills
                        self.n_handfeed_ok += 1
                        b = int(self.bank_slot[j, v])
                        if b >= 0 and not self.bank_busy[j, b] and self.bank_ref[j, b] == 0:
                            self.bank_slot[j, v] = -1
                            self.bank_owner[j, b] = -1
                            bupd.append((j, v, -1))
                    if victim >= 0:
                        self.n_feed_ok += 1
                        b = int(self.bank_slot[j, v])        # the entrant's bank record is redundant now: the victim's
                        if b >= 0 and not self.bank_busy[j, b] and self.bank_ref[j, b] == 0 and \
                                not self.resident[j, victim] and not self.banded[j, victim] and \
                                self.bank_slot[j, victim] < 0 and not self.bank_coming[j, victim]:
                            self.feed_bjobs.append((j, victim, b))
                elif victim >= 0:
                    self.n_feed_fail += 1
                elif victim == -2:                       # the entry failed: its band slot is free again
                    self.band_free[j].append(s_b)
        wb_issue = []                                                # write-backs to start after bupd is applied
        for key in [k for k, (_, f) in self.bank_flight.items() if f.done()]:  # 0. landed bank refills
            e, fut = self.bank_flight.pop(key)
            nb = fut.result()
            j, slot = key
            bupd.append((j, e, slot))
            self.bank_slot[j, e] = slot
            self.bank_owner[j, slot] = e
            self.bank_gen[j, slot] += 1
            self.bank_busy[j, slot] = False
            self.bank_coming[j, e] = False
            self.n_bank += 1
            self.n_bytes += nb
        for key in [k for k, (_, f, _, _, _) in self.flight.items() if f.done()]:  # 1. landed promotions
            e, fut, buf, bslot, v_out = self.flight[key]
            if bslot >= 0:
                self.bank_ref[key[0], bslot] -= 1
            res, nb = fut.result()
            j, row = key
            st = self.st[j]
            band_swap = False
            if HAND_FEED and self.band_cmd is not None and self.band_slot[j, e] >= 0:
                s_b = int(self.band_slot[j, e])                  # the promoted expert's band slot goes free; the evicted
                self.band_free[j].append(s_b)                    # expert takes the write-back branch below
                self.band_slot[j, e] = -1
                self.banded[j, e] = False
                band_upd.append((j, e, -1))
                self.n_handfree += 1
            elif self.band_cmd is not None and v_out >= 0 and self.band_slot[j, e] >= 0 and not self.banded[j, v_out]:
                if self._band_busy() < BAND_SWAP_FLIGHT:
                    s_b = int(self.band_slot[j, e])              # the promoted expert's band slot -> the evicted one,
                    self.band_wait.append([None, j, int(v_out), s_b])   # entered once no queued step can read it
                    self.band_slot[j, e] = -1
                    self.banded[j, e] = False
                    band_upd.append((j, e, -1))
                    band_swap = True
                    self.n_bswap += 1
                else:
                    self.n_bswap_skip += 1
            wslot = -1
            if not band_swap and self.swap_wb and v_out >= 0 and self.merge_ok and self.wb_free and \
                    self.bank_slot[j, v_out] < 0 and not self.banded[j, v_out]:
                if bslot >= 0 and self.bank_ref[j, bslot] == 0:
                    wslot = bslot                       # the swap: the promoted expert's slot
                else:
                    wslot = self._wb_slot(j, v_out)     # else the weakest slot v outscores
                    if wslot >= 0:
                        old = int(self.bank_owner[j, wslot])
                        if old >= 0:
                            self.bank_slot[j, old] = -1
                            bupd.append((j, old, -1))
                        self.bank_owner[j, wslot] = -1
                        self.n_wb_evict += 1
            if wslot >= 0:
                # the swap (K31_BANK_EXCL): the promoted expert leaves the bank, its slot takes the evicted expert's
                # 2-bit record, cut out of the row before the merge overwrites it and copied back by DMA
                from .stager import extract_hi2
                wb = self.wb_free.pop()
                with torch.cuda.stream(gs):
                    extract_hi2(st.mod, row, self.layout, self.wb_bufs[wb], blocks=EXTRACT_BLOCKS)   # before the merge
                if wslot == bslot:                       # the promoted expert leaves the bank
                    self.bank_slot[j, e] = -1
                    self.bank_owner[j, bslot] = -1
                    bupd.append((j, e, -1))
                self.bank_busy[j, wslot] = True
                wb_issue.append((j, v_out, wslot, wb, st))   # the DMA goes after this tick's bank-map write
            if self.merge_ok:                     # main stream, between steps: after the victim's unmap (stream order)
                from .stager import merge
                gs.wait_event(res)
                d_hi, d_lo = self.bufs[buf]
                with torch.cuda.stream(gs):
                    merge(d_hi.data_ptr(), d_lo.data_ptr(), st.mod, row, self.layout, blocks=MERGE_BLOCKS)
                merged.append(buf)
            else:
                with torch.cuda.stream(gs):
                    self.write_expert(st.mod, row, res, self.dev)
            upd.append((j, e, row))
            self.row_of[j][e] = row
            self.res[j][row] = e
            self.resident[j, e] = True
            self.coming_m[j, e] = False
            del self.flight[key]
            self.n_done += 1
            self.n_bytes += nb
        if merged:
            ev = torch.cuda.Event()
            ev.record(gs)
            for buf in merged:
                self.buf_ev[buf] = ev
                self.free.append(buf)
        keep = []
        for j, v, slot, ev_w, wb in self.wb_flight:                  # 1b. landed write-backs
            if not ev_w.query():
                keep.append((j, v, slot, ev_w, wb))
                continue
            self.bank_busy[j, slot] = False
            if self.k2 and self._wb_unfit(j, slot):                # a row the record format cannot hold: no record
                self.bank_owner[j, slot] = -1
            elif self.resident[j, v] or self.bank_slot[j, v] >= 0:   # promoted again meanwhile, or already banked
                self.bank_owner[j, slot] = -1                      # (the slot stays free for a refill)
            else:
                self.bank_slot[j, v] = slot
                self.bank_owner[j, slot] = v
                self.bank_gen[j, slot] += 1
                bupd.append((j, v, slot))
                self.n_wb += 1
            self.wb_free.append(wb)
        self.wb_flight = keep
        t = self._lap("land", t)
        scores = None
        ntok = self.tokens
        if self.ring_tail != self.ring_head and self.ring_ev[self.ring_tail % RING].query():   # 2. landed scores
            sc, cnt, acc = self.ring[self.ring_tail % RING]
            scores = sc.numpy()
            self.last_scores = scores.copy()
            self.last_stats = cnt.numpy().copy()
            self.ring_tail += 1
            if self.accepted:                     # credit = the accepted tokens committed since the last snapshot
                a = int(acc[0])               # + the prompt rows of eager rounds
                ntok, self.acc_seen = a - self.acc_seen + self.eager_tokens, a
                self.eager_tokens = 0
                self.n_acc += ntok
        t = self._lap("scores", t)
        jobs = []
        if scores is not None and self.P > 0:                                   # 3. swaps, all layers at once
            self.credit = np.minimum(self.credit + self.P * ntok * self.cw, np.maximum(1.0, self.P * ntok * self.cw) + 1.0)
            for _ in range(min(4, int(self.credit.max()))):             # up to 4 swaps per layer per tick
                n0 = len(jobs)
                self._swaps_all(scores, upd, jobs)
                if len(jobs) == n0:
                    break
        t = self._lap("swaps", t)
        bjobs = []
        if scores is not None and self.PB > 0 and self.G:                       # 3b. bank refills, all layers
            self.bcredit = np.minimum(self.bcredit + self.PB * ntok * self.cw,
                                      np.maximum(1.0, self.PB * ntok * self.cw) + 1.0)
            for _ in range(min(4, int(self.bcredit.max()))):
                n0 = len(bjobs)
                self._bank_swaps(scores, bupd, bjobs)
                if len(bjobs) == n0:
                    break
        if self.feed_bjobs:                            # victims of acked band entries take the entrants' bank slots
            for j, e, slot in self.feed_bjobs:
                if self.bank_busy[j, slot] or self.bank_ref[j, slot] > 0 or self.resident[j, e] or self.banded[j, e]:
                    continue
                old = int(self.bank_owner[j, slot])
                if old >= 0:
                    bupd.append((j, old, -1))
                    self.bank_slot[j, old] = -1
                self.bank_owner[j, slot] = -1
                self.bank_busy[j, slot] = True
                self.bank_coming[j, e] = True
                bjobs.append((j, e, slot))
                self.n_feed_refill += 1
            self.feed_bjobs = []
        if scores is not None and BAND_FEED > 0 and self.band_cmd is not None:   # 3c. score-fed band entries
            self.fcredit = np.minimum(self.fcredit + BAND_FEED * ntok * self.cw,
                                      np.maximum(1.0, BAND_FEED * ntok * self.cw) + 1.0)
            for _ in range(min(2, int(self.fcredit.max()))):
                n0 = len(self.band_wait)
                self._band_feed(scores, band_upd)
                if len(self.band_wait) == n0:
                    break
        t = self._lap("bank", t)
        if bupd:
            with torch.cuda.stream(gs):
                self._write(self.bank_map, bupd, gs)
        if wb_issue:              # after the unmaps: no step queued from here on reads those slots; the copies wait
            ev_u = torch.cuda.Event()                              # for every earlier step that still might
            ev_u.record(gs)
            self.wb_stream.wait_event(ev_u)
            for j, v_out, wslot, wb, st in wb_issue:
                with torch.cuda.stream(self.wb_stream):
                    st.bank_cpu[wslot].copy_(self.wb_bufs[wb], non_blocking=True)
                    ev_w = torch.cuda.Event()
                    ev_w.record(self.wb_stream)
                self.wb_flight.append((j, v_out, wslot, ev_w, wb))
        if bjobs:
            ev_b = torch.cuda.Event()
            ev_b.record(gs)                         # after the victims' unmaps: no later step reads those slots
            for j, e, slot in bjobs:
                self.bank_flight[(j, slot)] = (e, self.pool.submit(self._bank_job, j, e, slot, ev_b))
        if upd:                       # one indexed write, from a pinned ring (no host wait)
            with torch.cuda.stream(gs):
                self._write(self.row_map, upd, gs)
        if band_upd:
            with torch.cuda.stream(gs):
                self._write(self.band_map, band_upd, gs)
        if any(w[0] is None for w in self.band_wait):     # after this tick's map writes: once the GPU is past them, no
            ev_s = torch.cuda.Event()                     # step still sends a call that names a handed-over slot
            ev_s.record(gs)
            for w in self.band_wait:
                if w[0] is None:
                    w[0] = ev_s
        if RES_STREAM:                                    # the next replay (or the first MoE layer) waits for this
            self.tick_ev = torch.cuda.Event()
            self.tick_ev.record(gs)
        t = self._lap("writes", t)
        for j, e_in, row, v_out in jobs:          # the victims are unmapped (above) before any merge into their rows
            buf = self.free.pop()
            bslot = int(self.bank_slot[j, e_in]) if self.G else -1
            if bslot >= 0 and self.bank_busy[j, bslot]:
                bslot = -1
            if bslot >= 0:
                self.bank_ref[j, bslot] += 1        # the slot cannot be refilled while this promotion reads it
            self.flight[(j, row)] = (e_in, self.pool.submit(self._job, j, e_in, buf, bslot), buf, bslot, v_out)
        t = self._lap("submit", t)
        if self.ring_head - self.ring_tail < RING:                              # 4. this step's scores
            slot = self.ring_head % RING
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                self.ring[slot][0].copy_(self.score, non_blocking=True)
                self.ring[slot][1].copy_(self.stats, non_blocking=True)
                self.ring[slot][2].copy_(self.acc, non_blocking=True)
                ev2 = torch.cuda.Event()
                ev2.record(self.side)
            self.ring_ev[slot] = ev2
            self.ring_head += 1
        self._lap("d2h", t)
        if time.time() - self.t_log > LOG_S:
            self._log()

    def join_eager(self) -> None:
        """K31_RES_STREAM, paths without a graph replay since the tick (eager prefill chunks): the first MoE layer's
        call waits for the last tick's GPU work. Not while capturing (the event is outside the capture)."""
        ev = self.tick_ev
        if ev is not None and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream().wait_event(ev)
            self.tick_ev = None

    def rewarm(self) -> int:
        """Re-warm: before a request's first decode step, fill each layer's ring with the
        strongest-scored experts that have a bank record and no row (the prompt's routing is in the scores), each
        replacing the weakest ring member it outscores (stale rows first). One fetch round of the ring stager per
        layer on the current stream (~46 ms for 92 x 16 records at the link's ~300 GB/s)."""
        n = 0
        for j, st in enumerate(self.st):
            if not st.ring_ok:
                continue
            R = st.ring_owner.numel()
            sc, rm, bm, own = self.score[j], self.row_map[j], self.bank_map[j], st.ring_owner
            cand = torch.where((rm < 0) & (bm >= 0), sc, torch.full_like(sc, -1.0))
            cv, ce = cand.topk(R)                                    # strongest candidates first
            o = own.long().clamp_min(0)
            ar = torch.arange(R, device=own.device)
            live = (own >= 0) & (rm[o] == st.ring_base + ar)
            sc_o = sc[o]
            kv, kr = torch.where(live, sc_o, torch.full_like(sc_o, -1.0)).sort()   # weakest rows first
            do = (cv > kv) & (cv > 0)
            rows, exps = kr[do], ce[do]
            k = int(do.sum())
            if k == 0:
                continue
            oldo, oldlive = own[rows].long(), live[rows]
            rm[oldo[oldlive]] = -1
            rm[exps] = (st.ring_base + rows).to(rm.dtype)
            own[rows] = exps.to(own.dtype)
            src = torch.full((16,), -1, dtype=torch.int32, device=own.device)
            ring = torch.full((16,), -1, dtype=torch.int32, device=own.device)
            src[:k] = bm[exps].to(torch.int32)
            ring[:k] = rows.to(torch.int32)
            st.ring_stager(st.bank_ptr, src, ring, False)
            n += k
        self.n_rewarm = getattr(self, "n_rewarm", 0) + 1
        return n

    def _wb_unfit(self, j: int, slot: int) -> bool:
        """hi2-k2: True if the record just written back into bank slot `slot` has a status word set (a tile with more
        exceptions than it holds: recfmt.py; none of the checkpoint's experts is such a row). The slot then stays empty."""
        if not hasattr(self, "_wb_np"):
            from . import recfmt
            g = recfmt.geom(self.layout)
            self._wb_np = ([st.bank_cpu.numpy() for st in self.st], g["status"], g["status"] + 4 * g["status_words"])
        banks, a, b = self._wb_np
        if not banks[j][slot, a:b].any():
            return False
        self.n_wb_bad += 1
        if self.n_wb_bad <= 3:
            _LOG(f"residency: layer {self.st[j].idx}: an evicted row does not fit a hi2-k2 record; no record written back")
        return True

    def _wb_slot(self, j: int, v: int) -> int:
        """A bank slot for an evicted expert's record when its promoted replacement freed none: an empty slot, else the
        weakest-scored owner's slot (by the last landed scores) if v outscores it; never a slot being refilled, written
        back or read by a promotion. -1 if none."""
        own = self.bank_owner[j]
        ok = ~self.bank_busy[j] & (self.bank_ref[j] == 0)
        if not ok.any() or self.last_scores is None:
            return -1
        sc = self.last_scores[j]
        val = np.where(own >= 0, sc[np.maximum(own, 0)], -np.inf)
        val = np.where(ok, val, np.inf)
        k = int(val.argmin())
        if val[k] == np.inf:                     # every slot is busy or referenced
            return -1
        return k if val[k] < sc[v] else -1       # an empty slot (-inf) always, else only a weaker owner

    def commit(self, qsl, num_sampled, nreq: int) -> None:
        """After a step's sampling (hook wraps the runner's sample): fold the accepted rows into the scores."""
        if self.accepted and nreq > 0:
            from .unitk import commit
            from .hook import LAM
            commit(self.score, self.pend_ids, self.pend_w2, self.pend_t, qsl, num_sampled, nreq, LAM, self.acc)

    def _swaps_all(self, scores, upd, jobs) -> None:
        """At most one swap per layer with credit: the strongest-scored expert of the layer's ideal top-S that is
        neither resident nor coming, against its weakest-scored resident expert, if it scores more (numpy over all
        layers at once; a row being filled holds no resident expert, so it is never a victim)."""
        room = min(MAX_FLIGHT - len(self.flight), len(self.free)) - len(jobs)
        cl = np.nonzero(self.credit >= 1)[0]
        if room <= 0 or cl.size == 0:
            return
        S = scores[cl]
        if PROMOTE_BAND_MARGIN != 1.0 and self.band_slot is not None:
            # band members count at score / margin in the ideal set and against the weakest resident: one under the
            # margin yields its place to the next outsider (residents never have a band slot)
            S = np.where(self.band_slot[cl] >= 0, S / PROMOTE_BAND_MARGIN, S)
        held = self.resident[cl] | self.coming_m[cl]
        # the best expert neither resident nor coming: when it outscores the weakest resident it is inside the layer's
        # own top-H_l (only held experts can score above it), so no ideal-set sort is needed for any H_l
        cs = np.where(held, -np.inf, S)
        e_in = cs.argmax(1)
        best = cs[np.arange(cl.size), e_in]
        mem = np.where(self.resident[cl], S, np.inf)
        vi = mem.argmin(1)
        worst = mem[np.arange(cl.size), vi]
        bar = worst
        do = np.nonzero((best > bar) & np.isfinite(best))[0]
        for k in do[np.argsort(-(self.cw[cl[do]] * (best[do] - bar[do])))][:room]:   # largest weighted gains first
            j, e, v = int(cl[k]), int(e_in[k]), int(vi[k])
            row = int(self.row_of[j][v])
            self.row_of[j][v] = -1
            self.resident[j, v] = False
            self.res[j].pop(row, None)
            self.coming_m[j, e] = True
            upd.append((j, v, -1))
            jobs.append((j, e, row, v))
            self.credit[j] -= 1
        self.credit[cl] = np.minimum(self.credit[cl], np.where(np.isin(np.arange(cl.size), do), self.credit[cl], 1.0))

    def _band_busy(self) -> int:
        """Band entries posted and not acknowledged, waiting for their slot, or being copied into the ring."""
        return self.band_cmd.in_flight() + len(self.band_wait) + len(self.ring_copy)

    def _band_post_ready(self) -> None:
        """Post the waiting band entries whose slot no queued step can read any more; with the ring (K31_BAND_RING) an
        entrant holding a bank record is copied into a ring row by a worker first and posted when the copy is done."""
        keep = []
        for w in self.band_wait:
            if w[0] is not None and w[0].query():
                j, e = w[1], w[2]
                b = int(self.bank_slot[j, e]) if self.rec_ring is not None else -1
                if b >= 0 and self.ring_free and not self.bank_busy[j, b]:
                    r = self.ring_free.pop()
                    self.bank_ref[j, b] += 1                 # no refill or write-back into the slot while it is copied
                    self.ring_copy.append([self.pool.submit(self._ring_copy, j, b, r), w, r, b])
                else:
                    self._band_post(w, -1)
            else:
                keep.append(w)
        self.band_wait = keep
        if self.ring_copy:
            keep = []
            for c in self.ring_copy:
                fut, w, r, b = c
                if not fut.done():
                    keep.append(c)
                    continue
                self.bank_ref[w[1], b] -= 1
                try:
                    fut.result()
                    self._band_post(w, r)
                except Exception as ex:                      # noqa: BLE001  (from the drive instead)
                    self.n_ring_err += 1
                    if self.n_ring_err <= 3:
                        print(f"K31 record ring: copy of layer {self.st[w[1]].idx} expert {w[2]} failed: {ex}",
                              flush=True)
                    self.ring_free.append(r)
                    self._band_post(w, -1)
            self.ring_copy = keep

    def _band_post(self, w, r: int) -> None:
        seq = self.band_cmd.post(self.st[w[1]].idx, w[2], w[3], rec=r + 1 if r >= 0 else 0)
        self.band_pending[seq] = (w[1], w[2], w[3], w[4] if len(w) > 4 else -1, r)
        if r >= 0:
            self.n_ring += 1
        elif self.rec_ring is not None:
            self.n_ring_drive += 1

    def _ring_copy(self, j: int, b: int, r: int) -> None:
        np.copyto(self.rec_ring[r], self.st[j].bank_cpu[b].numpy())

    def _band_feed(self, scores, band_upd) -> None:
        """At most one band entry per layer with feed credit: the strongest-scored expert with a bank record, no native
        row, no band slot and no entry in flight, against the weakest-scored band member, if it scores BAND_FEED_MARGIN
        times more. The victim is unmapped now; the entry waits in band_wait for this tick's map-write event."""
        room = BAND_SWAP_FLIGHT - self._band_busy()
        if HAND_FEED and room > 0:                  # free slots first: the layer's best bank candidate, no victim
            fl = np.array([j for j in range(len(self.band_free)) if self.band_free[j] and self.band_ok[j]], dtype=np.int64)
            if fl.size:
                S = scores[fl]
                ok = (self.bank_slot[fl] >= 0) & ~self.resident[fl] & ~self.coming_m[fl] & ~self.banded[fl] & \
                    ~self.band_coming[fl]
                cs = np.where(ok, S, -np.inf)
                ci = cs.argmax(1)
                cbest = cs[np.arange(fl.size), ci]
                for k in np.argsort(-(self.cw[fl] * np.where(np.isfinite(cbest), cbest, -1e30)))[:room]:
                    if not np.isfinite(cbest[k]):
                        continue
                    j, e = int(fl[k]), int(ci[k])
                    slot = self.band_free[j].pop()
                    self.band_coming[j, e] = True
                    self.band_wait.append([None, j, e, slot, -2])
                    self.n_handfeed += 1
                    room -= 1
        cl = np.nonzero((self.fcredit >= 1) & self.band_ok)[0]
        if room <= 0 or cl.size == 0:
            return
        ar = np.arange(cl.size)
        S = scores[cl]
        ok = (self.bank_slot[cl] >= 0) & ~self.resident[cl] & ~self.coming_m[cl] & ~self.banded[cl] & \
            ~self.band_coming[cl]
        cs = np.where(ok, S, -np.inf)
        ci = cs.argmax(1)
        cbest = cs[ar, ci]
        mem = np.where(self.banded[cl] & (self.band_slot[cl] >= 0), S, np.inf)
        vi = mem.argmin(1)
        vworst = mem[ar, vi]
        do = np.nonzero(np.isfinite(cbest) & np.isfinite(vworst) & (cbest > 0) &
                        (cbest > BAND_FEED_MARGIN * np.maximum(vworst, 0.0)))[0]
        for k in do[np.argsort(-(self.cw[cl[do]] * (cbest[do] - vworst[do])))][:room]:
            j, e, v = int(cl[k]), int(ci[k]), int(vi[k])
            slot = int(self.band_slot[j, v])
            self.band_slot[j, v] = -1
            self.banded[j, v] = False
            band_upd.append((j, v, -1))
            self.band_coming[j, e] = True
            self.band_wait.append([None, j, e, slot, v])
            self.fcredit[j] -= 1
            self.n_feed += 1
        self.fcredit[cl] = np.minimum(self.fcredit[cl], np.where(np.isin(ar, do), self.fcredit[cl], 1.0))

    def _write(self, table, upd, main):
        """One indexed write of (layer, expert, value) triples into a pooled device table, from a pinned ring."""
        k = self.upd_head % len(self.upd_pin)
        pin, ev_used = self.upd_pin[k], self.upd_ev[k]
        if ev_used is not None and not ev_used.query():
            ev_used.synchronize()
        n = min(len(upd), pin.shape[0])
        pin.numpy()[:n] = np.asarray(upd[:n], dtype=np.int64)
        u = pin[:n].to(self.dev, non_blocking=True)
        table[u[:, 0], u[:, 1]] = u[:, 2].to(torch.int32)
        ev = torch.cuda.Event()
        ev.record(main)
        self.upd_ev[k] = ev
        self.upd_head += 1

    def _bank_swaps(self, scores, bupd, bjobs) -> None:
        """At most one refill per layer with bank credit: the strongest-scored expert that is neither resident nor
        in the bank (nor coming) against the weakest bank slot that is free to change (no promotion reading it, not
        being refilled; slots holding resident experts count as weakest), if it scores more."""
        room = BANK_FLIGHT - len(self.bank_flight) - len(bjobs)
        cl = np.nonzero(self.bcredit >= 1)[0]
        if room <= 0 or cl.size == 0:
            return
        ar = np.arange(cl.size)
        S = scores[cl]
        ok = ~self.resident[cl] & (self.bank_slot[cl] < 0) & ~self.bank_coming[cl] & ~self.coming_m[cl] & \
            ~self.banded[cl]
        cs = np.where(ok, S, -np.inf)
        ci = cs.argmax(1)
        cbest = cs[ar, ci]
        own = self.bank_owner[cl]
        hs = np.take_along_axis(S, np.maximum(own, 0), 1)
        res_hold = np.take_along_axis(self.resident[cl], np.maximum(own, 0), 1)
        hs = np.where((own < 0) | res_hold, -np.inf, hs)
        hs = np.where(self.bank_busy[cl] | (self.bank_ref[cl] > 0), np.inf, hs)
        vi = hs.argmin(1)
        vworst = hs[ar, vi]
        do = np.nonzero((cbest > vworst) & np.isfinite(cbest))[0]
        gain = cbest[do] - np.where(np.isfinite(vworst[do]), vworst[do], -1e30)
        for k in do[np.argsort(-(self.cw[cl[do]] * gain))][:room]:            # largest weighted gains first
            j, e, slot = int(cl[k]), int(ci[k]), int(vi[k])
            old = int(self.bank_owner[j, slot])
            if old >= 0:
                bupd.append((j, old, -1))
                self.bank_slot[j, old] = -1
            self.bank_owner[j, slot] = -1
            self.bank_busy[j, slot] = True
            self.bank_coming[j, e] = True
            bjobs.append((j, e, slot))
            self.bcredit[j] -= 1
        self.bcredit[cl] = np.minimum(self.bcredit[cl], np.where(np.isin(ar, do), self.bcredit[cl], 1.0))

    def _log(self):
        dt = time.time() - self.t_log
        nt = self.n_tick - self.n_tick_log
        from .hook import _STEPS
        steps = _STEPS[0] - getattr(self, "steps_log", 0)
        self.steps_log = _STEPS[0]
        msg = f"residency: {self.n_done - self.n_done_log} promotions in {dt:.0f} s ({len(self.flight)} in flight), " \
              f"{self.n_bank - getattr(self, 'n_bank_log', 0)} bank refills ({len(self.bank_flight)} in flight), " \
              f"NVMe {(self.n_bytes - self.bytes_log) / dt / 1e9:.2f} GB/s, {nt} ticks of " \
              f"{1e6 * (self.tick_s - self.tick_s_log) / max(nt, 1):.0f} us on the host, {steps} execute_model calls"
        from .hook import _DEF
        if _DEF.get("count") is not None:
            nd = int(_DEF["count"].item())
            msg += f", {nd - getattr(self, 'ndef_log', 0)} picks deferred"
            self.ndef_log = nd
        sect = getattr(self, "sect", {})
        if nt and sect:
            msg += " (" + ", ".join(f"{k} {1e6 * (v - self.sect_log.get(k, 0.0)) / nt:.0f}" for k, v in sect.items()) + " us)"
            self.sect_log = dict(sect)
        self.tick_s_log, self.n_tick_log = self.tick_s, self.n_tick
        if self.swap_wb:
            msg += (f", {self.n_wb - getattr(self, 'n_wb_log', 0)} evicted records written back to the bank "
                    f"({self.n_wb_evict - getattr(self, 'n_wbe_log', 0)} into a weaker record's slot)")
            self.n_wb_log, self.n_wbe_log = self.n_wb, self.n_wb_evict
        if self.accepted:
            msg += f", {self.n_acc - getattr(self, 'n_acc_log', 0)} accepted tokens credited"
            self.n_acc_log = self.n_acc
        if self.last_stats is not None:
            tot = self.last_stats.sum(0).astype(np.float64)
            if tot[3] > 0:
                msg += (f"; picks since start: resident {100 * tot[0] / tot[3]:.1f}%, staged {100 * tot[1] / tot[3]:.1f}%, "
                        f"cut {100 * tot[2] / tot[3]:.1f}%")
        if self.band_cmd is not None:
            msg += (f"; band swaps {self.n_bswap} posted, {self.n_bswap_ok} entered, {self.n_bswap_skip} skipped "
                    f"(ring full), {len(self.band_wait) + len(self.band_pending)} waiting")
            if BAND_FEED > 0:
                msg += (f"; band feed {self.n_feed} posted, {self.n_feed_ok} entered, {self.n_feed_fail} failed, "
                        f"{self.n_feed_refill} victims refilled into the bank")
            if HAND_FEED:
                msg += (f"; hand-feed {self.n_handfree} band slots freed by promotions, {self.n_handfeed} posted, "
                        f"{self.n_handfeed_ok} entered, {sum(len(f) for f in self.band_free)} free")
            if self.rec_ring is not None:
                msg += (f"; record ring: {self.n_ring} entries from the bank, {self.n_ring_drive} from the drive, "
                        f"{len(self.ring_copy)} copying" + (f", {self.n_ring_err} copy errors" if self.n_ring_err else ""))
        from .hook import _BAND
        if _BAND.get("gb") is not None:          # the RTX band: picks it served (counted among "cut" above), calls
            b = _BAND["stats"].sum(0).tolist()
            c = _BAND["gb"].counters()
            msg += (f"; RTX band: {100 * b[0] / max(b[2], 1):.1f}% of picks served, {100 * b[1] / max(b[2], 1):.2f}% "
                    f"past a call's 16 experts; calls added {c['n_added']}, timed out {c['n_timeout']}, empty "
                    f"{c['n_empty']}, service down {c['n_down']}, errors {c['n_error']}")
        _LOG(msg)
        self.t_log, self.n_done_log, self.bytes_log = time.time(), self.n_done, self.n_bytes
        self.n_bank_log = self.n_bank
