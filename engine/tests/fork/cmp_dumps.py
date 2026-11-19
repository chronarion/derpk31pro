"""Bitwise comparison of the MegaMoE outputs dumped by mb_fork.py: every arm in a dump directory against the stock arm
(same seeded experts, routing and activations in every run; the dumps carry the inputs, so a pair with different
inputs is reported, not compared). CPU only.
usage: python3 cmp_dumps.py DUMP_DIR [ARM ...]      (default arms: every prefix found besides stock)"""
import glob
import os
import sys

import torch

d = sys.argv[1]
stock = sorted(glob.glob(os.path.join(d, "stock-E*.pt")))
arms = sys.argv[2:] or sorted({os.path.basename(f).split("-E")[0] for f in glob.glob(os.path.join(d, "*-E*.pt"))} - {"stock"})
total = bad = 0
for f in stock:
    tag = os.path.basename(f)[len("stock-"):-3]
    a = torch.load(f)
    for arm in arms:
        g = os.path.join(d, f"{arm}-{tag}.pt")
        if not os.path.exists(g):
            print(f"{tag:30s} {arm:12s} no dump")
            continue
        b = torch.load(g)
        total += 1
        if not all(torch.equal(a[k], b[k]) for k in ("ids", "x", "w")):
            bad += 1
            print(f"{tag:30s} {arm:12s} DIFFERENT INPUTS (not compared)")
            continue
        ya, yb = a["y"], b["y"]
        eq = torch.equal(ya.view(torch.int16), yb.view(torch.int16)) and torch.equal(a["stats"], b["stats"])
        bad += not eq
        diff = (ya.float() - yb.float()).abs()
        print(f"{tag:30s} {arm:12s} {'BIT-EQUAL' if eq else 'DIFFER'}  nonzero {float((ya.float() != 0).float().mean()):.3f}  "
              f"rms {float(ya.float().pow(2).mean().sqrt()):.4f}"
              + ("" if eq else f"  differing elements {float((ya.view(torch.int16) != yb.view(torch.int16)).float().mean()):.5f}, "
                               f"max abs diff {float(diff.max()):.4g}, rel L2 "
                               f"{float(diff.pow(2).sum().sqrt() / ya.float().pow(2).sum().sqrt()):.3g}"))
print(f"{total} pairs compared, {bad} not equal")
sys.exit(1 if bad or not total else 0)
