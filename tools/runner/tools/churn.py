"""How much of K3.1's routing an adaptive HBM set serves when promotions are rate-limited (the engine cannot load
experts instantly: ~17.5 MB per native row over NVMe or NVLink-C2C), from a runner pass's route.pt.

Per MoE layer and window, over the window's tokens in order (= decode steps of one sequence), a set of H experts:
  - starts as the H most used experts of the layer (a static usage ranking: the route's own counts, or --rank JSON);
  - each step, scores decay (half-life `hl` steps) and add the step's squared routing weights; the set is updated
    BEFORE the step's picks are seen (they use the set as it was): up to P swaps per step (P fractional: a credit),
    each swapping the strongest-scored non-member of the ideal top-H for the weakest-scored member, if it scores more;
    a promotion becomes usable `delay` steps after it is decided;
  - a pick is a hit if its expert is in the usable set.
Reports hit rates (share of picks, by rank 1-3 / 4-8 / 9-16, and share of routing weight) after `--warm` tokens, for
P = 0 (static), fractional and integer budgets, and unlimited (the runner's hot-cut rule: the ideal top-H every step).
Runs on the GPU: layers and windows in parallel, one loop over steps.
usage: churn.py ROUTE_PT OUT_JSON [--sizes 115,125] [--budgets 0,0.25,0.5,1,2,4,8,inf] [--hl 32] [--delay 0]
"""
import argparse
import json
import math

import torch

ap = argparse.ArgumentParser()
ap.add_argument("route")
ap.add_argument("out")
ap.add_argument("--sizes", default="115,125")
ap.add_argument("--budgets", default="0,0.25,0.5,1,2,4,8,inf")
ap.add_argument("--hl", type=float, default=32.0)
ap.add_argument("--delay", type=int, default=0)
ap.add_argument("--warm", type=int, default=256)
ap.add_argument("--experts", type=int, default=896)
ap.add_argument("--rank", default="", help="static starting set from this usage ranking instead of the route's counts")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.load(a.route)
N, T = d["shape"]
ids_all, w_all = d["ids"], d["w"]
L, _, K = ids_all.shape
E = a.experts
moe = [i for i in range(L) if bool(w_all[i].float().abs().sum() > 0)]
ids = ids_all[moe].long().view(len(moe), N, T, K).to(dev)          # [L, N, T, K]
w = w_all[moe].float().view(len(moe), N, T, K).to(dev)
Lm = len(moe)
decay = 0.5 ** (1.0 / a.hl)
if a.rank:
    rk = json.load(open(a.rank))["layers"]
    order = torch.stack([torch.tensor(rk[str(i)]) for i in moe]).to(dev)           # [L, E] most used first
else:
    cnt = torch.zeros((Lm, E), device=dev)
    cnt.scatter_add_(1, ids.view(Lm, -1), torch.ones_like(ids.view(Lm, -1), dtype=torch.float))
    order = cnt.argsort(dim=1, descending=True, stable=True)
rank_bucket = torch.tensor([0] * 3 + [1] * 5 + [2] * 8, device=dev)                  # ranks 1-3, 4-8, 9-16


def run(H, P):
    in_set = torch.zeros((Lm, N, E), dtype=torch.bool, device=dev)
    in_set.scatter_(2, order[:, None, :H].expand(Lm, N, H), True)
    usable = in_set.clone()
    pending = []                                                    # (step usable, add idx, add mask, rm idx, rm mask)
    score = torch.zeros((Lm, N, E), device=dev)
    credit = torch.zeros((Lm, N), device=dev)
    hits = torch.zeros(3, device=dev); tot = torch.zeros(3, device=dev)
    whit = torch.zeros((), device=dev); wtot = torch.zeros((), device=dev)
    swaps = torch.zeros((), device=dev)
    pmax = 0 if P == 0 else (H if math.isinf(P) else max(1, math.ceil(P)))
    for t in range(T):
        if pmax:
            if math.isinf(P):
                ideal = torch.zeros_like(in_set)
                ideal.scatter_(2, score.topk(H, dim=2).indices, True)
                if t == 0:
                    ideal = in_set
                swaps += (ideal & ~in_set).sum()
                in_set = ideal
            else:
                credit += P
                n = credit.floor().clamp(max=pmax)
                credit -= n
                ideal = torch.zeros_like(in_set)
                ideal.scatter_(2, score.topk(H, dim=2).indices, True)
                cv, ci = torch.where(ideal & ~in_set, score, -1.0).topk(pmax, dim=2)        # strongest outsiders
                mv, mi = torch.where(in_set, score, float("inf")).topk(pmax, dim=2, largest=False)  # weakest members
                j = torch.arange(pmax, device=dev)
                do = (cv > mv) & (cv >= 0) & (j[None, None, :] < n[..., None])
                in_set.scatter_(2, ci, in_set.gather(2, ci) | do)
                in_set.scatter_(2, mi, in_set.gather(2, mi) & ~do)
                swaps += do.sum()
            if a.delay:
                pending.append((t + a.delay, in_set.clone()))
                while pending and pending[0][0] <= t:
                    usable = pending.pop(0)[1]
            else:
                usable = in_set
        pk, pw = ids[:, :, t], w[:, :, t]                               # [L, N, K]
        if t >= a.warm:
            h = usable.gather(2, pk)
            for b in range(3):
                m = rank_bucket == b
                hits[b] += h[..., m].sum(); tot[b] += m.sum() * Lm * N
            whit += (pw * h).sum(); wtot += pw.sum()
        score *= decay
        score.scatter_add_(2, pk, pw * pw)
    steps = (T - a.warm) * Lm * N
    allhit = hits.sum() / tot.sum()
    return {"H": H, "P": P, "hit": allhit.item(), "hit_r1_3": (hits[0] / tot[0]).item(),
            "hit_r4_8": (hits[1] / tot[1]).item(), "hit_r9_16": (hits[2] / tot[2]).item(),
            "weight_hit": (whit / wtot).item(), "swaps_per_layer_step": (swaps / (T * Lm * N)).item()}


res = []
for H in [int(x) for x in a.sizes.split(",")]:
    for P in [float(x) for x in a.budgets.split(",")]:
        r = run(H, P)
        res.append(r)
        print(f"H {H:4d} P {P:>5g}: picks hit {100 * r['hit']:5.1f}% (ranks 1-3 {100 * r['hit_r1_3']:5.1f}, 4-8 "
              f"{100 * r['hit_r4_8']:5.1f}, 9-16 {100 * r['hit_r9_16']:5.1f}), weight {100 * r['weight_hit']:5.1f}%, "
              f"swaps per layer-step {r['swaps_per_layer_step']:.2f}", flush=True)
json.dump({"halflife": a.hl, "delay": a.delay, "warm": a.warm, "layers": len(moe), "windows": N, "tokens": T,
           "start": a.rank or "route counts", "results": res}, open(a.out, "w"), indent=1)
