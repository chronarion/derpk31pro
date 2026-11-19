"""Combine runner passes into one results table (markdown): per universe, KLD against the published model (mean,
by position in the window, by kind of text), top-1 agreement, perplexity, and the share of picks per treatment.
All passes must use the same token windows (same corpus file), so their universes are directly comparable.

usage: report.py CORPUS_PT OUT_MD PASS_DIR [PASS_DIR ...]
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31.policy import NTREAT  # noqa: E402

TREAT = ["native", "planes", "cut", "defer", "wdefer"][:NTREAT]
FAMILY = {"wiki": "Wikipedia", "web": "web", "code": "code", "math": "math", "chat": "chat", "sci": "science",
          "struct": "structured"}

corpus, out, passes = sys.argv[1], sys.argv[2], sys.argv[3:]
kinds = torch.load(corpus)["kinds"]
fam = [FAMILY.get(k.split("_")[0], k) for k in kinds]
families = sorted(set(fam))
rows, seen = [], set()
for p in passes:
    res = torch.load(os.path.join(p, "kld.pt"))
    stats = torch.load(os.path.join(p, "stats.pt"))["stats"]                 # [S, L, NTREAT, K]
    us = json.load(open(os.path.join(p, "universes.json")))
    for s, name in enumerate(res["names"]):
        if name in seen:
            continue
        seen.add(name)
        kl, top1, nll = res["kl"][s], res["top1"][s], res["nll"][s]
        T = kl.shape[1]
        st = stats[s].sum(dim=(0, 2)).double()
        share = (st / st.sum()).tolist() if st.sum() > 0 else [0.0] * NTREAT
        rows.append({
            "name": name, "pass": os.path.basename(p.rstrip("/")),
            "kld": kl.mean().item(), "kld_0_256": kl[:, :256].mean().item(),
            "kld_256_1024": kl[:, 256:1024].mean().item(), "kld_1024_end": kl[:, 1024:].mean().item(),
            "top1": top1.float().mean().item(), "ppl": float(torch.exp(nll.mean())),
            "by_family": {f: kl[[i for i, x in enumerate(fam) if x == f]].mean().item() for f in families},
            "share": dict(zip(TREAT, share)), "spec": us[s],
        })

rows.sort(key=lambda r: r["kld"])
lines = ["# derpk31pro: KLD against the published model", "",
         f"Corpus: `{corpus}` ({len(kinds)} windows of {T} tokens). Steady state = tokens 1024-{T}. "
         "Picks: share of routed picks per treatment.", "",
         "| universe | pass | KLD | 0-256 | 256-1024 | steady | top-1 | ppl | native / planes / cut / defer / wdefer |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
for r in rows:
    sh = " / ".join(f"{100 * r['share'][t]:.0f}" for t in TREAT)
    lines.append(f"| {r['name']} | {r['pass']} | {r['kld']:.4f} | {r['kld_0_256']:.4f} | {r['kld_256_1024']:.4f} | "
                 f"{r['kld_1024_end']:.4f} | {r['top1']:.4f} | {r['ppl']:.3f} | {sh} |")
lines += ["", "## By kind of text (mean KLD over all tokens of the window)", "",
          "| universe | " + " | ".join(families) + " |", "|---|" + "---:|" * len(families)]
for r in rows:
    lines.append(f"| {r['name']} | " + " | ".join(f"{r['by_family'][f]:.4f}" for f in families) + " |")
open(out, "w").write("\n".join(lines) + "\n")
json.dump(rows, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
print("\n".join(lines[:len(rows) + 6]))
