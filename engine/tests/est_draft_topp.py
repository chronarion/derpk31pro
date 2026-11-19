"""Estimate what the target's top-p does to draft acceptance, and what K31_DRAFT_TOPP / K31_DRAFT_ARGMAX would do, on
the target's real next-token distributions (CPU only, no model run).

Target distributions p: the BF16 reference's top-64 log-probabilities at every position of the KLD corpus (kld.pt,
ref_lp [sequences, positions, 64], temperature 1), completed to 4,096 tokens by a Zipf tail continuing from the 64th
probability that carries the remaining mass (flat when that mass exceeds 4,032 x p_64). Drafter model: log q =
(log p + sigma * eps) / tau, eps iid N(0, 1) per token, normalised: tau < 1 is a drafter sharper (more confident) than
the target at the same temperature, sigma its misalignment. For each tau, sigma is fitted so the stock acceptance at
top_p 1 (sum min(p, q), the first draft's chance) matches a measured value; then, per position:
  A1    = sum min(p, q)                     top_p 1, drafts sampled from q (today at lil-bench's literal default)
  A95   = sum min(p', q)                    target top_p 0.95 (p' = p's nucleus renormalised), drafts from full q
  A95t  = sum min(p', q')                   K31_DRAFT_TOPP: q' = q's own nucleus at 0.95, renormalised (exact sort)
  A95a  = p'(argmax q)                      K31_DRAFT_ARGMAX / DRAFT_SAMPLE=greedy at top_p 0.95
  A1a   = p(argmax q)                       argmax drafts at top_p 1
  G     = 1[argmax q = argmax p]            greedy requests (temperature 0): the greedy bench's acceptance
The ratios A95 / A1 are compared with the measured ones (lil-bench decode cell, block verification, position 1:
preview 0.62 / 0.639 = 0.97, final drafter 0.711 / 0.644 = 1.10), and the per-position measured acceptances are
scaled by the predicted ratios (position k by the product of the per-position ratios) for an accepted length.

Run (from engine/; needs the reference file, read-only):
    docker run --rm --network none --cpuset-cpus 50-51 --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/engine:ro \
        -v $STATE/kld/bmref0:/ref:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /engine/tests/est_draft_topp.py /ref/kld.pt
"""
import sys

import torch

torch.manual_seed(0)
NTOP, VEFF, NPOS, CHUNK, TOPP = 64, 4096, int(sys.argv[2]) if len(sys.argv) > 2 else 24000, 3000, 0.95
# measured (lil-bench decode cell, 4 drafts, block verification, temperature 1): cumulative acceptance at positions 1-4
PREVIEW_TP1 = [0.639, 0.362, 0.192, 0.093]
PREVIEW_TP95 = [0.62, 0.34, 0.17, 0.08]
FINAL_TP1 = [0.644, 0.370, 0.196, 0.104]
FINAL_TP95 = [0.711, 0.439, 0.242, 0.138]


def target_rows(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    lp = d["ref_lp"].reshape(-1, NTOP).float()
    lp = lp[torch.randperm(lp.shape[0])[:NPOS]]
    p64 = lp.exp()
    r = (1.0 - p64.sum(1)).clamp(min=0.0)
    ranks = torch.arange(NTOP + 1, VEFF + 1, dtype=torch.float64)
    lr = torch.log(ranks / NTOP)
    last = p64[:, -1].double()

    def tail_mass(s):
        return last[:, None] * torch.exp(-s[:, None] * lr[None, :])

    lo = torch.zeros(NPOS, dtype=torch.float64)
    hi = torch.full((NPOS,), 20.0, dtype=torch.float64)
    for _ in range(50):                                   # the Zipf exponent whose tail carries r (bisection)
        mid = (lo + hi) / 2
        big = tail_mass(mid).sum(1) > r.double()
        lo = torch.where(big, mid, lo)
        hi = torch.where(big, hi, mid)
    tail = tail_mass(lo)
    flat = r.double() > tail_mass(torch.zeros(NPOS, dtype=torch.float64)).sum(1)
    tail = torch.where(flat[:, None], (r.double() / (VEFF - NTOP))[:, None].expand_as(tail), tail)
    tail = tail * (r.double() / tail.sum(1).clamp(min=1e-30))[:, None]
    p = torch.cat([p64.double(), tail], 1)
    return (p / p.sum(1, keepdim=True)).float()


def nucleus(x, top_p):
    """x [N, V] probabilities -> renormalised top_p nucleus (exact sort, ties broken by order)."""
    v, i = torch.sort(x, 1, descending=True)
    keep = (torch.cumsum(v, 1) - v) < top_p
    out = torch.zeros_like(x).scatter_(1, i, torch.where(keep, v, torch.zeros_like(v)))
    return out / out.sum(1, keepdim=True)


def metrics(P, tau, sigma, seed=1):
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(6, dtype=torch.float64)
    for c in range(0, P.shape[0], CHUNK):
        p = P[c:c + CHUNK]
        lq = (torch.log(p.clamp(min=1e-30)) + sigma * torch.randn(p.shape, generator=g)) / tau
        q = torch.softmax(lq, 1)
        pn = nucleus(p, TOPP)
        qn = nucleus(q, TOPP)
        am = q.argmax(1, keepdim=True)
        acc += torch.stack([torch.minimum(p, q).sum(1).sum(), torch.minimum(pn, q).sum(1).sum(),
                            torch.minimum(pn, qn).sum(1).sum(), pn.gather(1, am).sum(), p.gather(1, am).sum(),
                            (am[:, 0] == p.argmax(1)).float().sum()]).double()
    return (acc / P.shape[0]).tolist()          # A1, A95, A95t, A95a, A1a, G


def fit_sigma(P, tau, a1):
    lo, hi = 0.0, 12.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if metrics(P[:6000], tau, mid)[0] > a1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def length(cum):
    return 1.0 + sum(cum)


def main():
    P = target_rows(sys.argv[1] if len(sys.argv) > 1 else "/ref/kld.pt")
    pm = P.max(1).values
    print(f"{P.shape[0]} positions: median p_max {float(pm.median()):.3f}, p_max >= 0.95 in "
          f"{float((pm >= 0.95).float().mean()):.1%}")
    # model-free bounds on the drafter-specific effect of the target's top-p at a fixed context: for drafts sampled
    # from any q, sum_N min(p/Z, q) - sum min(p, q) lies in [-(1 - Z), 1 - Z] (Z = the target nucleus's mass)
    v, _ = torch.sort(P, 1, descending=True)
    Z = torch.where((torch.cumsum(v, 1) - v) < TOPP, v, torch.zeros_like(v)).sum(1)
    print(f"target nucleus mass Z at top_p {TOPP}: mean {float(Z.mean()):.4f}; E[1 - Z] = {float((1 - Z).mean()):.4f} "
          f"(the most any sampled drafter can gain or lose per position, absolute); a perfect argmax drafter gains "
          f"E[p_max (1/Z - 1)] = {float((pm * (1 / Z - 1)).mean()):.4f}, a perfectly calibrated sampler (q = p) "
          f"loses E[1 - Z]")
    cond_tp1 = [PREVIEW_TP1[0]] + [PREVIEW_TP1[k] / PREVIEW_TP1[k - 1] for k in range(1, 4)]
    print("tau  sigma | A1     A95    A95t   A95a   A1a    G     | A95/A1 A95t/A1 A95a/A1")
    rows = {}
    for tau in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        s = fit_sigma(P, tau, PREVIEW_TP1[0])
        m = metrics(P, tau, s)
        rows[tau] = (s, m)
        print(f"{tau:.1f}  {s:5.2f} | " + " ".join(f"{v:.4f}" for v in m) +
              f" | {m[1] / m[0]:.3f}  {m[2] / m[0]:.3f}   {m[3] / m[0]:.3f}")
    print(f"measured A95/A1 at position 1: preview {PREVIEW_TP95[0] / PREVIEW_TP1[0]:.3f}, "
          f"final {FINAL_TP95[0] / FINAL_TP1[0]:.3f}")
    for name, tp1, tp95 in (("preview", PREVIEW_TP1, PREVIEW_TP95), ("final", FINAL_TP1, FINAL_TP95)):
        want = tp95[0] / tp1[0]
        tau = min(rows, key=lambda t: abs(rows[t][1][1] / rows[t][1][0] - want))
        print(f"{name}: best-matching tau {tau} (A95/A1 {rows[tau][1][1] / rows[tau][1][0]:.3f} vs {want:.3f})")
        # per position: refit sigma to each conditional acceptance at top_p 1, scale by that position's ratios
        cond = [tp1[0]] + [tp1[k] / tp1[k - 1] for k in range(1, 4)]
        pred = {"stock tp95": [], "K31_DRAFT_TOPP": [], "K31_DRAFT_ARGMAX": [], "argmax tp1": []}
        for k, c in enumerate(cond):
            s = fit_sigma(P, tau, c)
            m = metrics(P, tau, s, seed=7 + k)
            for key, v in (("stock tp95", m[1]), ("K31_DRAFT_TOPP", m[2]), ("K31_DRAFT_ARGMAX", m[3]),
                           ("argmax tp1", m[4])):
                pred[key].append(v / m[0] * c)
        cum = {key: [float(torch.tensor(v[:k + 1]).prod()) for k in range(4)] for key, v in pred.items()}
        print(f"  measured tp1 {tp1} (length {length(tp1):.2f}), tp95 {tp95} (length {length(tp95):.2f})")
        for key, v in cum.items():
            print(f"  predicted {key:17s} " + " ".join(f"{x:.3f}" for x in v) + f"  length {length(v):.2f}")
    del cond_tp1


if __name__ == "__main__":
    main()
