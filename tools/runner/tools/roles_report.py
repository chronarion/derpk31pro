"""Read a runner pass by the role of the predicted token (the goal's KLD counts generated rows: assistant output or
plain text). Prints, per universe: all rows, generated rows with a cut-bootstrap interval,
assistant / plain text separately, the other roles, and generated rows by domain; optionally the paired difference of
every universe against one of them.

usage: roles_report.py PASS_DIR CORPUS.pt [--roles corpus/k31-bm-eval-roles.pt] [--against NAME] [--json OUT]
CORPUS.pt must carry `corpus_index` (the cuts' indices into the 512 of k31-bm-eval.pt) or be a rot32/rot128 file.
"""
import argparse
import json
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("pass_dir")
ap.add_argument("corpus")
ap.add_argument("--roles", default="/s/corpus/k31-bm-eval-roles.pt")
ap.add_argument("--against", default="")
ap.add_argument("--json", default="")
a = ap.parse_args()
d = torch.load(os.path.join(a.pass_dir, "kld.pt"), weights_only=False)
names = d.get("names") or [u["name"] for u in json.load(open(os.path.join(a.pass_dir, "universes.json")))]
kl = d["kl"].double()                                            # [S, N, T]
S, N, T = kl.shape
c = torch.load(a.corpus, weights_only=False)
if "corpus_index" in c:
    idx = list(c["corpus_index"])
elif "rot32_index" in c:
    idx = [16 * r for r in c["rot32_index"]]
elif N == 32:
    idx = [4 * w + w % 4 for w in range(0, 128, 4)]
elif N == 128:
    idx = [4 * w + w % 4 for w in range(128)]
else:
    raise SystemExit("cannot map the corpus to the 512 cuts")
kinds = c.get("kinds")
lab = torch.load(a.roles, weights_only=False)
lnames = lab["names"]
L = lab["labels"].view(-1, 2048)[torch.tensor(idx)][:, 1:T].long()   # role of the token each row predicts
x_all = kl[:, :, :T - 1]
gen = L <= 1
g = torch.Generator().manual_seed(0)
boot = [torch.randint(0, N, (N,), generator=g) for _ in range(2000)]
out = {}
ref_u = names.index(a.against) if a.against else None
print(f"{a.pass_dir}: {N} cuts, generated rows {int(gen.sum())} ({gen.double().mean():.1%}); reference = {names[0]}")
hdr = f"{'universe':28s} {'all':>7s} | {'GEN':>7s} [95% cut bootstrap] {'assist':>7s} {'text':>7s} | " + \
      " ".join(f"{n:>7s}" for n in lnames[2:6])
if ref_u is not None:
    hdr += f" | gen - {names[ref_u]} [95%]"
print(hdr)
for u in range(1, S):
    x = x_all[u]
    num, den = (x * gen).sum(1), gen.double().sum(1)
    bs = torch.tensor([float(num[b].sum() / den[b].sum().clamp_min(1)) for b in boot])
    r = {"all": float(x.mean()), "gen": float(x[gen].mean()),
         "gen_ci95": [float(torch.quantile(bs, 0.025)), float(torch.quantile(bs, 0.975))]}
    for i, n in enumerate(lnames):
        if (L == i).any():
            r[n] = float(x[L == i].mean())
    line = (f"{names[u][:28]:28s} {r['all']:7.4f} | {r['gen']:7.4f} [{r['gen_ci95'][0]:.4f}, {r['gen_ci95'][1]:.4f}] "
            f"{r.get('assistant', float('nan')):7.4f} {r.get('text', float('nan')):7.4f} | "
            + " ".join(f"{r.get(n, float('nan')):7.4f}" for n in lnames[2:6]))
    if ref_u is not None and u != ref_u:
        y = x_all[ref_u]
        dn = ((x - y) * gen).sum(1)
        bd = torch.tensor([float(dn[b].sum() / den[b].sum().clamp_min(1)) for b in boot])
        r["gen_minus_against"] = float((x - y)[gen].mean())
        r["gen_minus_against_ci95"] = [float(torch.quantile(bd, 0.025)), float(torch.quantile(bd, 0.975))]
        line += f" | {r['gen_minus_against']:+.4f} [{r['gen_minus_against_ci95'][0]:+.4f}, {r['gen_minus_against_ci95'][1]:+.4f}]"
    if kinds is not None:
        dom = {}
        for i in range(N):
            v = dom.setdefault(kinds[i], [0.0, 0.0])
            v[0] += float(num[i]); v[1] += float(den[i])
        r["gen_by_domain"] = {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(dom.items())}
    out[names[u]] = r
    print(line)
if kinds is not None:
    doms = sorted({k for k in kinds})
    print("generated rows by domain:")
    print(f"{'':28s} " + " ".join(f"{k[:8]:>8s}" for k in doms))
    for u in range(1, S):
        gd = out[names[u]]["gen_by_domain"]
        print(f"{names[u][:28]:28s} " + " ".join(f"{gd[k]:8.4f}" if gd.get(k) is not None else "       -" for k in doms))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
