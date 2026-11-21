"""Exact official floor for the on-policy tail score, from a reference pass's final hidden states.

    tailfloor.py MODEL_DIR RUNNER_DIR SET_DIR REF_DIR [--gen-cap 1024] [--top-p 0.95] [--out DIR] [--noise REF_DIR_B]

tailscore.py can bracket the official floor from the 64 tokens a reference pass stores per position; on flat
distributions (chat, refusals) those 64 do not cover the top-p nucleus and the bracket is wide. This rebuilds the
official model's full distribution at every generated position (REF_DIR/<bucket>/hidden.npy, the runner's
--dump-hidden, through the checkpoint's LM head: fp32 logits, float64 log-softmax, as the runner scores) and writes
REF_DIR/<bucket>/floor.pt (or under --out) = per sequence, per generated token (the first --gen-cap):

  floor [g, 3]  probability that the official model, sampling from its top-p nucleus here, writes a token it gives
                < 1% / < 0.1% / < 0.01%   (sum of p(v) / Z over nucleus tokens under the threshold)
  enll  [g]     its expected -log p under the same sampling (for the excess surprise)
  z     [g]     the nucleus mass Z

tailscore.py uses floor.pt when it is there. The hidden states can be deleted afterwards.

--noise REF_DIR_B: a second pass of the same set in another runner layout (batch / block sizes). Prints the
measurement's own floor: the rates a token sampled from pass A's nucleus would score under pass B (same model, other
rounding), i.e. what a bit-different but faithful engine would read.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

TAUS = (1e-2, 1e-3, 1e-4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model"); p.add_argument("runner"); p.add_argument("set_dir"); p.add_argument("ref_dir")
    p.add_argument("--gen-cap", type=int, default=1024)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--noise")
    p.add_argument("--out", help="directory for <bucket>/floor.pt (default REF_DIR)")
    p.add_argument("--device", default="cuda"); p.add_argument("--rows", type=int, default=256)
    a = p.parse_args()
    sys.path.insert(0, a.runner)
    from k31.stream import Shards
    name = "language_model.lm_head.weight"
    head = Shards(a.model).read([name], 1)[name].to(a.device).float()
    info = json.load(open(f"{a.set_dir}/sets.json"))
    tot, nz, nrows = np.zeros(5), np.zeros(9), 0

    def dist(x):
        h = torch.from_numpy(np.array(x).view(np.int16)).to(a.device).view(torch.bfloat16).float()
        lp = torch.log_softmax((h @ head.T).double(), -1)
        ps, order = lp.exp().sort(-1, descending=True)
        nuc = (ps.cumsum(-1) - ps) < a.top_p
        return lp, ps, order, nuc

    for b in info["main"]["buckets"]:
        hp = f"{a.ref_dir}/{b['name']}/hidden.npy"
        if not os.path.exists(hp):
            print(f"{b['name']}: no hidden.npy, skipped")
            continue
        c = torch.load(f"{a.set_dir}/{b['name']}.pt", weights_only=False)
        H = np.load(hp, mmap_mode="r")
        HB = np.load(f"{a.noise}/{b['name']}/hidden.npy", mmap_mode="r") if a.noise and os.path.exists(
            f"{a.noise}/{b['name']}/hidden.npy") else None
        out = {"ids": c["ids"], "gen_cap": a.gen_cap, "top_p": a.top_p, "taus": TAUS, "floor": [], "enll": [], "z": []}
        for n in range(H.shape[0]):
            L, G = int(c["lens"][n]), int(c["prompt"][n])
            r0, r1 = G - 1, min(L - 1, G - 1 + a.gen_cap)
            fl, en, zz = [], [], []
            for s in range(r0, r1, a.rows):
                e = min(r1, s + a.rows)
                lp, ps, order, nuc = dist(H[n, s:e])
                Z = (ps * nuc).sum(-1)
                fl.append(torch.stack([(ps * (nuc & (ps < t))).sum(-1) / Z for t in TAUS], -1).cpu())
                en.append(((ps * nuc * -ps.clamp_min(1e-300).log()).sum(-1) / Z).cpu())
                zz.append(Z.cpu())
                if HB is not None:
                    lb, pb, ob, nb = dist(HB[n, s:e])
                    w = ps * nuc / Z[:, None]                                  # A's sampling weights, sorted order
                    pbv = lb.exp().gather(-1, order)                           # B's probability of A's tokens
                    inb = torch.zeros_like(nb).scatter_(-1, ob, nb).gather(-1, order)   # A's tokens inside B's nucleus
                    lr = ps.clamp_min(1e-300).log() - pbv.clamp_min(1e-300).log()
                    v = [(w * (pbv < t)).sum() for t in TAUS] + [(w * ~inb).sum(), (w * lr).sum(),
                         (w * ((ps >= 0.5) & (pbv < 1e-2))).sum(), (w * (lr > math.log(100.0))).sum(),
                         (w * (-pbv.clamp_min(1e-300).log())).sum() - (w * -ps.clamp_min(1e-300).log()).sum(),
                         torch.tensor(float(e - s))]
                    nz += np.array([float(x) for x in v])
            if fl:
                out["floor"].append(torch.cat(fl).float()); out["enll"].append(torch.cat(en).float())
                out["z"].append(torch.cat(zz).float())
                tot[:3] += out["floor"][-1].double().sum(0).numpy(); tot[3] += float(out["enll"][-1].double().sum())
                tot[4] += float(out["z"][-1].double().sum()); nrows += r1 - r0
            else:
                out["floor"].append(torch.zeros((0, 3))); out["enll"].append(torch.zeros(0)); out["z"].append(torch.zeros(0))
        os.makedirs(f"{a.out or a.ref_dir}/{b['name']}", exist_ok=True)
        torch.save(out, f"{a.out or a.ref_dir}/{b['name']}/floor.pt")
        print(f"{b['name']}: floor.pt written ({H.shape[0]} sequences)", flush=True)
    if nrows:
        print(f"{nrows} generated rows: exact official floor per 1,000: p<1% {1e3 * tot[0] / nrows:.4f}, p<0.1% "
              f"{1e3 * tot[1] / nrows:.4f}, p<0.01% {1e3 * tot[2] / nrows:.4f}; expected NLL {tot[3] / nrows:.4f}; "
              f"nucleus mass {tot[4] / nrows:.4f}")
    if nz[8]:
        m = nz[8]
        print(f"measurement floor (pass A samples, pass B scores; {int(m)} rows) per 1,000: p<1% {1e3 * nz[0] / m:.4f}, "
              f"p<0.1% {1e3 * nz[1] / m:.4f}, p<0.01% {1e3 * nz[2] / m:.4f}, out of nucleus {1e3 * nz[3] / m:.4f}, "
              f"confident p<1% {1e3 * nz[5] / m:.4f}, ratio>100 {1e3 * nz[6] / m:.4f}; gap {nz[4] / m:.5f}; "
              f"excess surprise {nz[7] / m:.5f}")


if __name__ == "__main__":
    main()
