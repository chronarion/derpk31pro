"""K31_HAND_FEED + K31_CREDIT_WEIGHTS in residency._band_feed (CPU only, numpy, on a namespace like
test_band_feed.py): a layer with a free band slot posts its best bank candidate into it with no victim (-2) and no
margin test, before any victim feed; layers with the band off are skipped; when room is short the larger
weight x score goes first; the victim feed's order is weighted too."""
import os
import sys
import types

import numpy as np

os.environ["K31_BAND_FEED"] = "0.25"
os.environ["K31_HAND_FEED"] = "1"
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import residency  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def make(n, room):
    ns = types.SimpleNamespace()
    ns.bank_slot = np.full((n, 896), -1, dtype=np.int64)
    ns.bank_slot[:, 100:200] = np.arange(100)               # experts 100..199 hold bank records
    ns.resident = np.zeros((n, 896), dtype=bool)
    ns.resident[:, :50] = True
    ns.coming_m = np.zeros((n, 896), dtype=bool)
    ns.band_slot = np.full((n, 896), -1, dtype=np.int64)
    ns.band_slot[:, 50:60] = np.arange(10)
    ns.banded = ns.band_slot >= 0
    ns.band_coming = np.zeros((n, 896), dtype=bool)
    ns.band_ok = np.ones(n, dtype=bool)
    ns.band_wait = []
    ns.fcredit = np.zeros(n)                                # no victim feed unless a test gives credit
    ns.n_feed = ns.n_handfeed = 0
    ns.cw = np.ones(n)
    ns.band_free = [[] for _ in range(n)]
    ns.band_cmd = types.SimpleNamespace(in_flight=lambda: residency.BAND_SWAP_FLIGHT - room)
    ns.ring_copy = []
    ns._band_busy = types.MethodType(residency.Residency._band_busy, ns)
    return ns


assert residency.HAND_FEED
# 1. free slots: layer 0 has slot 7 free (its member at 57 was promoted), layer 1 has none, layer 2 has the band off
n = 3
ns = make(n, room=8)
ns.band_slot[0, 57] = -1
ns.banded[0, 57] = False
ns.band_free[0].append(7)
ns.band_free[2].append(3)
ns.band_ok[2] = False
scores = np.zeros((n, 896), dtype=np.float32)
scores[:, 100:200] = np.linspace(0.1, 0.3, 100)
scores[0, 150] = 0.9                                      # layer 0's best bank candidate
scores[0, 199] = 0.95
ns.band_coming[0, 199] = True                             # in flight: never chosen
scores[0, 20] = 2.0                                       # resident: never chosen
band_upd = []
residency.Residency._band_feed(ns, scores, band_upd)
check("free slot takes the layer's best bank candidate, no victim", [w[1:] for w in ns.band_wait] == [[0, 150, 7, -2]],
      f"(band_wait {[w[1:] for w in ns.band_wait]})")
check("state: slot taken from the free list, entrant coming, nothing unmapped",
      ns.band_free[0] == [] and ns.band_coming[0, 150] and band_upd == [] and ns.n_handfeed == 1)
check("band-off layer keeps its free slot", ns.band_free[2] == [3] and not ns.band_coming[2].any())
# 2. room for one entry: the larger weight x score wins (layer 1: 0.5 x 1.5 = 0.75 > layer 0: 0.9 x 0.5 = 0.45)
ns = make(2, room=1)
for j in (0, 1):
    ns.band_free[j].append(2)
scores = np.zeros((2, 896), dtype=np.float32)
scores[0, 120] = 0.9
scores[1, 130] = 0.5
ns.cw = np.array([0.5, 1.5])
residency.Residency._band_feed(ns, scores, [])
check("short room: weight x score orders the free-slot fills", [w[1:] for w in ns.band_wait] == [[1, 130, 2, -2]],
      f"(band_wait {[w[1:] for w in ns.band_wait]})")
# 3. no free slots, victim feed with credit in both layers and room for one: weighted gain decides
ns = make(2, room=1)
ns.fcredit = np.array([1.0, 1.0])
scores = np.zeros((2, 896), dtype=np.float32)
scores[:, 50:60] = 0.2                                    # band members
scores[0, 110] = 0.9                                      # gain 0.7 x w 0.5 = 0.35
scores[1, 111] = 0.6                                      # gain 0.4 x w 1.5 = 0.60
ns.cw = np.array([0.5, 1.5])
residency.Residency._band_feed(ns, scores, [])
got = [(w[1], w[2], w[4]) for w in ns.band_wait]
check("victim feed: weighted gain orders the layers", got == [(1, 111, 50)] or (len(got) == 1 and got[0][:2] == (1, 111)),
      f"(band_wait {got})")
print("HAND FEED:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
