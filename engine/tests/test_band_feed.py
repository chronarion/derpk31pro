"""residency._band_feed (K31_BAND_FEED) on synthetic state: per layer with credit, the strongest-scored expert with a
bank record, no native row, no band slot and no entry in flight replaces the weakest-scored band member when it scores
BAND_FEED_MARGIN x more; the victim is unmapped, the entry waits for the tick's map-write event; nothing else moves."""
import os
import sys
import types

os.environ["K31_BAND_FEED"] = "0.25"
sys.path.insert(0, "/engine")
import numpy as np  # noqa: E402

from k31_engine import residency  # noqa: E402

rng = np.random.default_rng(0)
n = 3
ns = types.SimpleNamespace()
ns.bank_slot = np.full((n, 896), -1, dtype=np.int64)
ns.bank_slot[:, 100:200] = np.arange(100)
ns.resident = np.zeros((n, 896), dtype=bool)
ns.resident[:, :50] = True
ns.coming_m = np.zeros((n, 896), dtype=bool)
ns.band_slot = np.full((n, 896), -1, dtype=np.int64)
ns.band_slot[:, 50:60] = np.arange(10)
ns.banded = ns.band_slot >= 0
ns.band_coming = np.zeros((n, 896), dtype=bool)
ns.band_ok = np.ones(n, dtype=bool)
ns.cw = np.ones(n)                                       # K31_CREDIT_WEIGHTS: flat
ns.band_free = [[] for _ in range(n)]                    # K31_HAND_FEED: no free slots
ns.n_handfeed = 0
ns.ring_copy = []                                        # K31_BAND_RING: nothing being copied
ns._band_busy = types.MethodType(residency.Residency._band_busy, ns)
ns.band_wait = []
ns.fcredit = np.array([1.0, 1.0, 0.0])                  # layer 2 has no credit
ns.n_feed = 0
ns.band_cmd = types.SimpleNamespace(in_flight=lambda: 0)
scores = rng.random((n, 896)).astype(np.float32) * 0.1
scores[0, 150] = 5.0                                     # layer 0: a strong bank expert -> enters
scores[0, 55] = 0.001                                    # ... replacing band member 55 (weakest)
scores[1, 120] = 0.02                                    # layer 1: the best bank expert is weak -> no entry
scores[1, 50:60] = 0.5
scores[2, 160] = 9.0                                     # layer 2: strong, but no credit
ns.band_coming[0, 199] = True                            # in flight: never chosen again
scores[0, 199] = 50.0
scores[0, 10] = 99.0                                     # resident: never a candidate
band_upd = []
residency.Residency._band_feed(ns, scores, band_upd)
ok = True
ok &= band_upd == [(0, 55, -1)]
ok &= len(ns.band_wait) == 1 and ns.band_wait[0][1:] == [0, 150, 5, 55]
ok &= ns.band_slot[0, 55] == -1 and not ns.banded[0, 55] and ns.band_coming[0, 150]
ok &= ns.fcredit[0] == 0.0 and ns.n_feed == 1
ok &= not ns.band_coming[1].any() and not ns.band_coming[2, 160]
print("band_upd", band_upd, "| band_wait", [w[1:] for w in ns.band_wait], "| credit", ns.fcredit.tolist())
# room: a full ring posts nothing
ns2 = types.SimpleNamespace(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(ns).items()})
ns2.band_wait, ns2.fcredit = [], np.array([1.0, 1.0, 1.0])
ns2.band_cmd = types.SimpleNamespace(in_flight=lambda: residency.BAND_SWAP_FLIGHT)
ns2._band_busy = types.MethodType(residency.Residency._band_busy, ns2)
residency.Residency._band_feed(ns2, scores, [])
ok &= ns2.band_wait == []
print("BAND FEED:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
