"""Where K3.1's expert picks land in a tiered design, and how many bytes per verify step each tier must move, from a
runner pass's route.pt (the reference's routing over forced-decode order).

Per MoE layer and window, over the window's tokens in order:
  - HBM: an adaptive set of H native experts, as in churn.py (static start, decayed squared-weight scores, P swaps per
    token as a credit, `delay` tokens before a promotion is usable);
  - RTX: a static set of R native experts, the next R in the usage ranking after the first H;
  - Grace bank: a static set (2-bit records): everything in the ranking's first H + R + B not in the other tiers;
  - tail: the rest (NVMe only: cut, or loaded too late for the token).
Steps are `step` consecutive tokens (a verify step with drafts): per step and layer, the distinct experts each tier
serves. Bytes: native rows 17.5 MB, 2-bit records 9.29 MB (hi2-k1).
usage: tiers.py ROUTE_PT OUT_JSON --designs "H:R:B:P,..." [--step 3] [--hl 32] [--warm 256] [--rank RANK_JSON]
"""
import argparse
import json
import math

import torch

ap = argparse.ArgumentParser()
ap.add_argument("route")
ap.add_argument("out")
ap.add_argument("--designs", default="105:0:400:0.1,105:55:400:0.1,105:55:500:0.25")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--hl", type=float, default=32.0)
ap.add_argument("--delay", type=int, default=2)
ap.add_argument("--warm", type=int, default=256)
ap.add_argument("--experts", type=int, default=896)
ap.add_argument("--rank", default="")
ap.add_argument("--windows", type=int, default=0, help="use only the first n windows (speed)")
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.load(a.route)
N, T = d["shape"]
ids_all, w_all = d["ids"], d["w"]
L, _, K = ids_all.shape
E = a.experts
moe = [i for i in range(L) if bool(w_all[i].float().abs().sum() > 0)]
ids = ids_all[moe].long().view(len(moe), N, T, K)
w = w_all[moe].float().view(len(moe), N, T, K)
if a.windows:
    N = a.windows
    ids, w = ids[:, :N], w[:, :N]
ids, w = ids.to(dev), w.to(dev)
Lm = len(moe)
decay = 0.5 ** (1.0 / a.hl)
if a.rank:
    rk = json.load(open(a.rank))["layers"]
    order = torch.stack([torch.tensor(rk[str(i)]) for i in moe]).to(dev)
else:
    cnt = torch.zeros((Lm, E), device=dev)
    cnt.scatter_add_(1, ids.reshape(Lm, -1), torch.ones_like(ids.reshape(Lm, -1), dtype=torch.float))
    order = cnt.argsort(dim=1, descending=True, stable=True)
pos = torch.empty_like(order)
pos.scatter_(1, order, torch.arange(E, device=dev).expand(Lm, E))       # usage position of each expert per layer


def run(H, R, B, P):
    in_set = torch.zeros((Lm, N, E), dtype=torch.bool, device=dev)
    in_set.scatter_(2, order[:, None, :H].expand(Lm, N, H), True)
    usable, pending = in_set.clone(), []
    score = torch.zeros((Lm, N, E), device=dev)
    credit = torch.zeros((Lm, N), device=dev)
    pmax = max(1, math.ceil(P))
    rtx = (pos >= H) & (pos < H + R)                                     # [Lm, E]
    bank = pos < H + R + B
    picks = torch.zeros(4, device=dev, dtype=torch.float64); wsum = torch.zeros(4, device=dev, dtype=torch.float64)
    uni = torch.zeros(4, device=dev, dtype=torch.float64); nsteps = 0
    stepmask = torch.zeros((4, Lm, N, E), dtype=torch.bool, device=dev)
    for t in range(T):
        credit += P
        n = credit.floor().clamp(max=pmax)
        credit -= n
        ideal = torch.zeros_like(in_set)
        ideal.scatter_(2, score.topk(H, dim=2).indices, True)
        cv, ci = torch.where(ideal & ~in_set, score, -1.0).topk(pmax, dim=2)
        mv, mi = torch.where(in_set, score, float("inf")).topk(pmax, dim=2, largest=False)
        j = torch.arange(pmax, device=dev)
        do = (cv > mv) & (cv >= 0) & (j[None, None, :] < n[..., None])
        in_set.scatter_(2, ci, in_set.gather(2, ci) | do)
        in_set.scatter_(2, mi, in_set.gather(2, mi) & ~do)
        pending.append((t + a.delay, in_set.clone()))
        while pending and pending[0][0] <= t:
            usable = pending.pop(0)[1]
        pk, pw = ids[:, :, t], w[:, :, t]                               # [Lm, N, K]
        h = usable.gather(2, pk)
        r = ~h & rtx.gather(1, pk.reshape(Lm, -1)).view_as(pk)
        bk = ~h & ~r & bank.gather(1, pk.reshape(Lm, -1)).view_as(pk)
        tl = ~h & ~r & ~bk
        if t >= a.warm:
            for i, m in enumerate((h, r, bk, tl)):
                picks[i] += m.sum(); wsum[i] += (pw * m).sum()
                stepmask[i].scatter_(2, pk, stepmask[i].gather(2, pk) | m)
            if (t - a.warm) % a.step == a.step - 1:
                uni += stepmask.sum(dim=(1, 2, 3)).double(); nsteps += 1
                stepmask.zero_()
        score *= decay
        score.scatter_add_(2, pk, pw * pw)
    ps, ws = picks / picks.sum(), wsum / wsum.sum()
    per = uni / max(nsteps, 1) / N                                       # distinct experts per step, all layers
    gb = {"hbm_native": per[0] * 17.5e-3, "rtx_native": per[1] * 17.5e-3, "grace_2bit": per[2] * 9.29e-3,
          "tail_2bit": per[3] * 9.29e-3}
    return {"H": H, "R": R, "B": B, "P": P, "picks": dict(zip(("hbm", "rtx", "grace", "tail"), ps.tolist())),
            "weight": dict(zip(("hbm", "rtx", "grace", "tail"), ws.tolist())),
            "distinct_per_step_all_layers": dict(zip(("hbm", "rtx", "grace", "tail"), per.tolist())),
            "GB_per_step": {k: float(v) for k, v in gb.items()}}


res = []
for dsg in a.designs.split(","):
    H, R, B, P = dsg.split(":")
    r = run(int(H), int(R), int(B), float(P))
    res.append(r)
    pk, wt, g = r["picks"], r["weight"], r["GB_per_step"]
    print(f"H {H} R {R} B {B} P {P}: picks HBM {100*pk['hbm']:.1f}% RTX {100*pk['rtx']:.1f}% Grace {100*pk['grace']:.1f}% "
          f"tail {100*pk['tail']:.1f}% | weight tail {100*wt['tail']:.2f}% Grace {100*wt['grace']:.1f}% | per {a.step}-token "
          f"step: Grace {g['grace_2bit']:.1f} GB, RTX {g['rtx_native']:.1f} GB, tail {g['tail_2bit']:.2f} GB, "
          f"HBM {g['hbm_native']:.1f} GB", flush=True)
json.dump({"step": a.step, "halflife": a.hl, "delay": a.delay, "warm": a.warm, "windows": N, "tokens": T,
           "start": a.rank or "route counts", "results": res}, open(a.out, "w"), indent=1)
