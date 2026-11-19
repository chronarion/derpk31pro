"""K31_HBM_MAP (resident rows per layer): residency._swaps_all promotes toward each layer's own ideal top-H_l set, and
the promotion margin for band members applies. CPU only (numpy), on a namespace like test_band_feed.py."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
os.environ.setdefault("K31_PROMOTE_BAND_MARGIN", "1.5")
from k31_engine import residency  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def make(n, rows):
    ns = types.SimpleNamespace()
    ns.Hl = np.array(rows, dtype=np.int64)
    ns.H = int(ns.Hl.max())
    ns.resident = np.zeros((n, 896), dtype=bool)
    ns.row_of = []
    ns.res = []
    for j, r in enumerate(rows):
        ns.resident[j, :r] = True                          # experts 0..r-1 resident
        ns.row_of.append(list(range(r)))
        ns.res.append({k: k for k in range(r)})
    ns.coming_m = np.zeros((n, 896), dtype=bool)
    ns.band_slot = np.full((n, 896), -1, dtype=np.int64)
    ns.credit = np.ones(n)
    ns.cw = np.ones(n)                                     # K31_CREDIT_WEIGHTS: flat
    ns.flight = []
    ns.free = list(range(64))
    return ns


# 1. per-layer H: layer 0 keeps 4 rows, layer 1 keeps 8; scores make experts 100.. the strongest outsiders
n = 2
ns = make(n, [4, 8])
scores = np.zeros((n, 896), dtype=np.float32)
scores[:, :8] = np.linspace(1.0, 0.5, 8)                  # residents (layer 0 holds only the first 4)
scores[0, 100] = 0.9                                      # layer 0: an outsider stronger than its weakest resident
scores[1, 100] = 0.45                                     # layer 1: weaker than its 8 residents
upd, jobs = [], []
residency.Residency._swaps_all(ns, scores, upd, jobs)
got = sorted((j, e) for j, e, _, _ in jobs)
check("per-layer ideal sets: layer 0 promotes 100, layer 1 nothing", got == [(0, 100)], f"(jobs {got})")
# 2. a layer whose own top-H_l excludes the candidate: layer 0 with H 4, the outsider ranks 5th by score
ns = make(n, [4, 8])
scores = np.zeros((n, 896), dtype=np.float32)
scores[:, :8] = np.linspace(1.0, 0.5, 8)
scores[0, 4:8] = 0.0                                      # layer 0's non-resident 4..7 score nothing
scores[0, 200] = 0.6                                      # beats resident 3 (0.786)? no: weaker than all 4 residents
upd, jobs = [], []
residency.Residency._swaps_all(ns, scores, upd, jobs)
check("an outsider weaker than the layer's residents stays out", jobs == [], f"(jobs {jobs})")
# 3. the band margin: a band member needs 1.5 x the weakest resident
ns = make(1, [4])
scores = np.zeros((1, 896), dtype=np.float32)
scores[0, :4] = [1.0, 0.9, 0.8, 0.5]
scores[0, 300] = 0.6                                      # band member: 0.6 < 1.5 x 0.5
scores[0, 301] = 0.55                                     # not in the band: 0.55 > 0.5 but weaker than 300
ns.band_slot[0, 300] = 7
upd, jobs = [], []
residency.Residency._swaps_all(ns, scores, upd, jobs)
check("band member below the margin yields to the next outsider (0.55 > 0.5 promoted)",
      [(j, e) for j, e, _, _ in jobs] == [(0, 301)], f"(jobs {jobs})")
scores[0, 300] = 0.8                                      # 0.8 / 1.5 = 0.53 > 0.5
scores[0, 301] = 0.0                                      # (with 301 at 0.55 it would win: 0.55 > 0.53)
ns = make(1, [4])
ns.band_slot[0, 300] = 7
upd, jobs = [], []
residency.Residency._swaps_all(ns, scores, upd, jobs)
check("band member above the margin is promoted", [(j, e) for j, e, _, _ in jobs] == [(0, 300)], f"(jobs {jobs})")
print("HBM MAP:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
