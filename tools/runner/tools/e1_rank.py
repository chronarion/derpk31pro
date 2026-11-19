"""The rank structure of K3.1's routing, from a runner pass's route.pt (universe 0's expert ids
and routing weights per layer and token, sorted by weight).

Reports, over all MoE layers and tokens:
  - routing energy by rank: each rank's share of the squared routing weights (gate: ranks 1-5 >= 65%);
  - reuse by rank: the share of rank-j picks whose expert the previous token also picked, and that the next token
    picks again;
  - an adaptive hot set (the runner's own rule, k31/policy.py: decaying sum of squared routing weights, half-life
    128 tokens, ties to lower ids) of 14% and 29% of a layer (125 and 260 experts): hit rate by rank and over all
    picks (gates: ranks 1-5 >= 75% at 14% and >= 85% at 29%; pick average within 5 points of 0.65 / 0.79),
    counted after the first `--warm` tokens of each window;
  - distinct experts per step of R consecutive rows (R = 1..8), per layer.
usage: e1_rank.py ROUTE_PT OUT_JSON [--warm 256]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31.policy import hot_scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("route")
ap.add_argument("out")
ap.add_argument("--warm", type=int, default=256)
ap.add_argument("--experts", type=int, default=896)
ap.add_argument("--halflife", type=float, default=128.0)
a = ap.parse_args()
d = torch.load(a.route)
ids, w = d["ids"], d["w"]
N, T = d["shape"]
L, _, K = ids.shape
E = a.experts
moe = [i for i in range(L) if bool(w[i].float().abs().sum() > 0)]
energy = torch.zeros(K, dtype=torch.float64)
prev_hit = torch.zeros(K, dtype=torch.float64)
next_hit = torch.zeros(K, dtype=torch.float64)
pairs = 0
sets = {"14%": 125, "29%": 260}
hits = {k: torch.zeros(K, dtype=torch.float64) for k in sets}
counted = 0
distinct = torch.zeros(8, dtype=torch.float64)
dsteps = torch.zeros(8, dtype=torch.float64)
for i in moe:
    I = ids[i].view(N, T, K).long()
    Wt = w[i].view(N, T, K).double()
    energy += (Wt ** 2).sum((0, 1))
    onehot = torch.zeros(N, T, E, dtype=torch.bool)
    onehot.scatter_(2, I, True)
    prev_hit += onehot[:, :-1].gather(2, I[:, 1:]).double().sum((0, 1))     # picked by the previous token
    next_hit += onehot[:, 1:].gather(2, I[:, :-1]).double().sum((0, 1))     # picked again by the next token
    pairs += N * (T - 1)
    score = hot_scores(I, Wt.float(), E, a.halflife)
    order = score.argsort(dim=2, descending=True, stable=True)
    place = torch.empty_like(order)
    place.scatter_(2, order, torch.arange(E).expand_as(order))
    pick = place.gather(2, I)[:, a.warm:]
    for k, size in sets.items():
        hits[k] += (pick < size).double().sum((0, 1))
    counted += N * (T - a.warm)
    for R in range(1, 9):                                    # distinct experts in steps of R consecutive rows
        steps = T // R
        blk = onehot[:, : steps * R].view(N, steps, R, E).any(2)
        distinct[R - 1] += blk.sum().double()
        dsteps[R - 1] += N * steps
    del onehot, score, order, place, pick

share = energy / energy.sum()
res = {
    "layers": len(moe), "tokens": N * T,
    "energy_share_by_rank": share.tolist(),
    "energy_ranks_1_5": float(share[:5].sum()), "energy_ranks_6_11": float(share[5:11].sum()),
    "energy_ranks_12_16": float(share[11:].sum()),
    "reuse_prev_by_rank": (prev_hit / pairs / len(moe)).tolist(),
    "reuse_next_by_rank": (next_hit / pairs / len(moe)).tolist(),
    "hot_hit_by_rank": {k: (v / counted / len(moe)).tolist() for k, v in hits.items()},
    "hot_hit_ranks_1_5": {k: float((v[:5] / counted / len(moe)).mean()) for k, v in hits.items()},
    "hot_hit_all": {k: float((v / counted / len(moe)).mean()) for k, v in hits.items()},
    "distinct_per_step": {R: float(distinct[R - 1] / dsteps[R - 1] / len(moe)) for R in range(1, 9)},
}
g = res
gates = {
    "ranks 1-5 >= 65% of energy": g["energy_ranks_1_5"] >= 0.65,
    "14% set holds >= 75% of rank-1-5 picks": g["hot_hit_ranks_1_5"]["14%"] >= 0.75,
    "29% set holds >= 85% of rank-1-5 picks": g["hot_hit_ranks_1_5"]["29%"] >= 0.85,
    "14% pick average within 5 points of 0.65": abs(g["hot_hit_all"]["14%"] - 0.65) <= 0.05,
    "29% pick average within 5 points of 0.79": abs(g["hot_hit_all"]["29%"] - 0.79) <= 0.05,
}
res["gates"] = gates
json.dump(res, open(a.out, "w"), indent=1)
f = lambda v: " ".join(f"{x:.3f}" for x in v)  # noqa: E731
print(f"{len(moe)} MoE layers, {N} windows x {T} tokens")
print(f"energy by rank: {f(share.tolist())}")
print(f"  ranks 1-5 {g['energy_ranks_1_5']:.3f}, 6-11 {g['energy_ranks_6_11']:.3f}, 12-16 {g['energy_ranks_12_16']:.3f}")
print(f"previous token also picked, by rank: {f(res['reuse_prev_by_rank'])}")
print(f"next token picks again, by rank:     {f(res['reuse_next_by_rank'])}")
for k in sets:
    print(f"hot set {k}: hit by rank {f(res['hot_hit_by_rank'][k])} | ranks 1-5 {g['hot_hit_ranks_1_5'][k]:.3f}, "
          f"all {g['hot_hit_all'][k]:.3f}")
print("distinct experts per step (R rows):", " ".join(f"{R}:{v:.1f}" for R, v in res["distinct_per_step"].items()))
for name, ok in gates.items():
    print(("PASS " if ok else "FAIL ") + name)
