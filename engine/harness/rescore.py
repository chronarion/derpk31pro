"""Re-score an engine run (kld.py's .tokens.pt: the engine's top-64 per position) against any reference (a runner
kld.pt, or comma-separated halves in order), with kld.py's metrics: KL(ref || engine) coarse-grained over the shared
top-k tokens + one rest bucket, top-1 agreement, window-block bootstrap, percentiles, by domain.
    rescore.py ENGINE.tokens.pt REF[,REF2] OUT.json"""
import json
import sys

import torch

eng = torch.load(sys.argv[1], weights_only=False)
refs = [torch.load(f, weights_only=False) for f in sys.argv[2].split(",")]
ref_lp = torch.cat([r["ref_lp"] for r in refs]).double()
ref_id = torch.cat([r["ref_id"] for r in refs]).long()
sel = torch.tensor(eng["sel"])
R_lp, R_id = ref_lp[sel], ref_id[sel]                    # [n, T, K]
E_lp, E_id = eng["eng_lp"].double(), eng["eng_ids"].long()
n, T, K = E_lp.shape
kl = torch.zeros((n, T - 1), dtype=torch.float64)
top1 = torch.zeros((n, T - 1), dtype=torch.bool)
cover = torch.zeros((n, T - 1), dtype=torch.float64)
for i in range(n):
    rl, ri, el, ei = R_lp[i, :T - 1], R_id[i, :T - 1], E_lp[i, :T - 1], E_id[i, :T - 1]
    match = ri.unsqueeze(2) == ei.unsqueeze(1)           # [T-1, K_ref, K_eng]
    has = match.any(2)
    e_at = torch.where(match, el.unsqueeze(1), torch.tensor(float("-inf"), dtype=torch.float64)).amax(2)
    pr = torch.where(has, rl.exp(), 0.0)
    pe = torch.where(has, e_at.exp(), 0.0)
    rest_r = (1 - pr.sum(1)).clamp_min(1e-300)
    rest_e = (1 - pe.sum(1)).clamp_min(1e-300)
    v = (pr * (torch.where(has, rl, 0.0) - torch.where(has, e_at, 0.0))).sum(1) + rest_r * (rest_r.log() - rest_e.log())
    kl[i] = v.clamp_min(0)
    cover[i] = pr.sum(1)
    top1[i] = ei[:, 0] == ri[:, 0]
seq = kl.mean(1)
g = torch.Generator().manual_seed(0)
boots = torch.tensor([float(seq[torch.randint(0, n, (n,), generator=g)].mean()) for _ in range(2000)])
flat = kl.flatten().float()
res = {"kld": float(kl.mean()), "kld_ci95": [float(boots.quantile(0.025)), float(boots.quantile(0.975))],
       "median": float(flat.median()), "p90": float(flat.quantile(0.9)), "p99": float(flat.quantile(0.99)),
       "top1": float(top1.float().mean()), "cover": float(cover.mean()), "sequences": n, "reference": sys.argv[2]}
kinds = eng.get("kinds")
if kinds:
    dom = {}
    for i in range(n):
        dom.setdefault(kinds[i], []).append(float(seq[i]))
    res["by_domain"] = {k: sum(v) / len(v) for k, v in sorted(dom.items())}
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(f"KLD {res['kld']:.4f} [{res['kld_ci95'][0]:.4f}, {res['kld_ci95'][1]:.4f}] median {res['median']:.4f} "
      f"p90 {res['p90']:.3f} p99 {res['p99']:.2f} top-1 {100 * res['top1']:.2f}% cover {res['cover']:.3f} | "
      + " ".join(f"{k} {v:.3f}" for k, v in res.get("by_domain", {}).items()))
