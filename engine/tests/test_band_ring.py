"""K31_BAND_RING in residency._band_post_ready (CPU only, numpy + a thread pool, on a namespace): a ready band entry
whose entrant holds a bank record is copied into a ring row and posted with rec = row + 1 (bank slot held during the
copy, released after); an entrant without a record, or a full ring, is posted from the drive (rec 0); an entry whose
slot event has not passed keeps waiting."""
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import residency  # noqa: E402
from k31_engine.residency import Residency  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


class Ev:
    def __init__(self, ready):
        self.ready = ready

    def query(self):
        return self.ready


REC = 64
ns = types.SimpleNamespace()
n = 2
ns.st = [types.SimpleNamespace(idx=10 + j, bank_cpu=torch.arange(8 * REC, dtype=torch.int64).remainder(251).to(
    torch.uint8).view(8, REC) + j) for j in range(n)]
ns.bank_slot = np.full((n, 896), -1, dtype=np.int64)
ns.bank_slot[:, 100:108] = np.arange(8)                      # experts 100..107 hold bank slots 0..7
ns.bank_busy = np.zeros((n, 8), dtype=bool)
ns.bank_ref = np.zeros((n, 8), dtype=np.int64)
ns.rec_ring = np.zeros((2, REC), dtype=np.uint8)
ns.ring_free = [0, 1]
ns.ring_copy = []
ns.band_pending = {}
ns.n_ring = ns.n_ring_drive = ns.n_ring_err = 0
ns.pool = ThreadPoolExecutor(2)
posts = []


class Cmd:
    seq = 0

    def post(self, layer, expert, slot, rec=0):
        Cmd.seq += 1
        posts.append((Cmd.seq, layer, expert, slot, rec))
        return Cmd.seq


ns.band_cmd = Cmd()
ns._band_post = lambda w, r: Residency._band_post(ns, w, r)
ns._ring_copy = lambda j, b, r: (time.sleep(0.3), Residency._ring_copy(ns, j, b, r))   # slow: the hold is visible
# entries: [event, layer j, expert, band slot, victim]
ns.band_wait = [[Ev(True), 0, 103, 7, -2],                    # bank record (slot 3): the ring
                [Ev(True), 1, 300, 8, -2],                    # no bank record: the drive
                [Ev(False), 0, 104, 9, -2],                   # slot still readable: waits
                [Ev(True), 1, 105, 10, 55],                   # bank record (slot 5): the ring
                [Ev(True), 0, 106, 11, -2]]                   # bank record but the ring is full: the drive
Residency._band_post_ready(ns)
held = ns.bank_ref.copy()
t0 = time.time()
while ns.ring_copy and time.time() - t0 < 5:
    time.sleep(0.01)
    Residency._band_post_ready(ns)
check("ready entries: bank-record entrants go to ring rows, the rest to the drive, unready ones wait",
      sorted((p[2], p[4] > 0) for p in posts) == [(103, True), (105, True), (106, False), (300, False)]
      and [w[2] for w in ns.band_wait] == [104], f"(posts {posts}, waiting {[w[2] for w in ns.band_wait]})")
check("bank slots held during the copy, released after", held[0, 3] == 1 and held[1, 5] == 1 and not ns.bank_ref.any(),
      f"(held {held[0, 3]}, {held[1, 5]}; after {int(ns.bank_ref.sum())})")
ring_of = {p[2]: p[4] - 1 for p in posts if p[4] > 0}
ok = all(np.array_equal(ns.rec_ring[ring_of[e]], ns.st[j].bank_cpu[b].numpy()) for e, j, b in ((103, 0, 3), (105, 1, 5)))
check("ring rows hold the entrants' bank records", ok)
pend = {v[1]: v[4] for v in ns.band_pending.values()}
check("pending entries remember their ring row (-1 from the drive)",
      pend == {103: ring_of[103], 105: ring_of[105], 106: -1, 300: -1} and ns.ring_free == [], f"({pend})")
check("counters", ns.n_ring == 2 and ns.n_ring_drive == 2 and ns.n_ring_err == 0)
print("BAND RING:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
