"""On-policy tail score: how often the served engine writes a token the official model would almost never write.

    tailscore.py score SET_DIR REF_DIR CAPTURE.jsonl [...] --out NAME.json [--gen-cap 1024] [--top-p 0.95]
                       [--ids MANIFEST.json] [--label TEXT] [--boot 2000]
    tailscore.py compare A.json B.json [--boot 2000] [--force]
    tailscore.py selftest

`score` reads a capture (capture.py: the engine's own sampled answers with its log-probability of every token), the
set built from it (build_sets.py, SET_DIR/sets.json + buckets) and the reference pass of the official model over that
set (REF_DIR/<bucket>/kld.pt: `nll` = the official model's exact -log p of each next token, `ref_lp` / `ref_id` = its
64 most likely tokens per position). No engine and no GPU are needed. For every generated token x at a position with
official distribution p and served distribution q it takes

  gap              mean of log q(x) - log p(x): a one-sample estimate of the reverse KL of the served engine against
                   the official model (q is the engine's own report, raw, before top-p)
  excess surprise  mean of -log p(x) minus what the official model would average itself when sampling from its top-p
                   nucleus at the same position. Needs no engine log-probabilities. ~0 for a faithful engine,
                   negative for one that is sharper than the original
  out of nucleus   x is outside the official model's top-p nucleus (the official sampler could not have produced it)
  tail curve       p(x) < 1%, < 0.1% ("near-impossible"), < 0.01%, per 1,000 tokens, each beside its
  official floor   the rate at which the official model itself, sampling from its nucleus at the same positions,
                   writes a token under the same threshold: sum of p(v) / Z over nucleus tokens with p(v) < tau.
                   Exact where the 64 stored tokens cover the nucleus; where they do not, the unseen nucleus mass is
                   counted as under the threshold when the 64th token is (exact up to one boundary token) and
                   otherwise bounded (none .. all of it): `floor_lo` / `floor_hi`, their middle is used for
                   excess = observed - floor and ratio = observed / floor. With REF_DIR/<bucket>/floor.pt
                   (tailfloor.py, from the pass's hidden states) the floor and the excess surprise are exact
  confident errors the engine gave x >= 50% and the official model gave it < 1% (and < 0.1%); floor ~0 by construction
  self tail        tokens the engine itself gave < tau (its own floor, for comparison with the official floor)

Each answer counts its first --gen-cap generated tokens; tokens from a detected loop's start on are left out (loops
are counted separately: inside one, both models agree). Pooled values are token-weighted over the prompts scored.
Intervals: 95% percentile bootstrap over prompts, resampled within each prompt kind (answers cluster their errors, so
tokens are not independent). `precision` in the output gives, per count, the events, the Poisson and the bootstrap
half-widths, the design effect and the tokens a +-10% interval would need.

`compare` pairs two scores by prompt id (same prompts and seeds): the difference B - A of every metric with a paired
bootstrap interval over the common prompts. It refuses captures whose recorded sampling settings differ.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

TAUS = (1e-2, 1e-3, 1e-4)
TAU_NAMES = ("p<1%", "p<0.1%", "p<0.01%")
BINS = ((0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 1 << 30))
F = ["n", "gap", "nll", "enll", "oon", "und", "top1", "unc", "conf2", "conf3", "r100"] \
    + [f"{k}{i}" for k in ("obs", "flo", "fhi", "self") for i in range(len(TAUS))]
IX = {k: i for i, k in enumerate(F)}


def seq_stats(lq, lpt, P, ids64, nxt, top_p, exact=None):
    """Per-token fields [g, len(F)] for one answer: lq / lpt = engine / official log-prob of each sampled token,
    P [g, 64] = the official model's top-64 probabilities (descending), ids64 their token ids, nxt the sampled ids."""
    g = len(lq)
    out = np.zeros((g, len(F)))
    assert g == 0 or (np.diff(P, axis=1) <= 1e-6).all(), "the reference's top tokens are not in descending order"
    cb = np.concatenate([np.zeros((g, 1)), np.cumsum(P, 1)[:, :-1]], 1)     # mass above each stored token
    nuc = cb < top_p
    cum = P.sum(1)
    cov = cum >= top_p
    p64 = P[:, -1]
    R = np.where(cov, 0.0, top_p - cum)                                     # nucleus mass beyond the 64 stored tokens
    Zx = (P * nuc).sum(1)
    Zlo = np.where(cov, Zx, top_p)
    Zhi = np.where(cov, Zx, top_p + p64)
    hit = ids64 == nxt[:, None]
    has = hit.any(1)
    rank = hit.argmax(1)
    above = cb[np.arange(g), rank]
    out[:, IX["n"]] = 1
    out[:, IX["gap"]] = lq - lpt
    out[:, IX["nll"]] = -lpt
    safe = np.log(np.maximum(P, 1e-300))
    out[:, IX["enll"]] = ((P * nuc * -safe).sum(1) + R * -np.log(np.maximum(p64, 1e-300))) / Zlo
    out[:, IX["oon"]] = np.where(has, above >= top_p, cov)
    out[:, IX["und"]] = ~has & ~cov
    out[:, IX["top1"]] = ids64[:, 0] == nxt
    out[:, IX["unc"]] = R
    out[:, IX["conf2"]] = (lq >= math.log(0.5)) & (lpt < math.log(1e-2))
    out[:, IX["conf3"]] = (lq >= math.log(0.5)) & (lpt < math.log(1e-3))
    out[:, IX["r100"]] = (lq - lpt) > math.log(100.0)
    for i, tau in enumerate(TAUS):
        base = (P * nuc * (P < tau)).sum(1)
        out[:, IX[f"obs{i}"]] = lpt < math.log(tau)
        out[:, IX[f"flo{i}"]] = (base + np.where(~cov & (p64 < tau), R, 0.0)) / Zhi
        out[:, IX[f"fhi{i}"]] = np.minimum(1.0, (base + np.where(cov, 0.0, R + p64)) / Zlo)
        out[:, IX[f"self{i}"]] = lq < math.log(tau)
    if exact is not None:                                   # (floor [g, 3], expected NLL [g]) from tailfloor.py
        for i in range(len(TAUS)):
            out[:, IX[f"flo{i}"]] = out[:, IX[f"fhi{i}"]] = exact[0][:, i]
        out[:, IX["enll"]] = exact[1]
    return out


def derive(s):
    """Metrics from a vector of field sums."""
    n = max(s[IX["n"]], 1.0)
    d = {"tokens": int(s[IX["n"]]), "gap": s[IX["gap"]] / n, "official_nll": s[IX["nll"]] / n,
         "excess_surprise": (s[IX["nll"]] - s[IX["enll"]]) / n, "out_of_nucleus_per_1k": 1e3 * s[IX["oon"]] / n,
         "undetermined_per_1k": 1e3 * s[IX["und"]] / n, "official_top1": s[IX["top1"]] / n,
         "uncovered_nucleus_mass": s[IX["unc"]] / n, "confident_p<1%_per_1k": 1e3 * s[IX["conf2"]] / n,
         "confident_p<0.1%_per_1k": 1e3 * s[IX["conf3"]] / n, "ratio>100_per_1k": 1e3 * s[IX["r100"]] / n}
    for i, nm in enumerate(TAU_NAMES):
        obs, lo, hi = (1e3 * s[IX[f"{k}{i}"]] / n for k in ("obs", "flo", "fhi"))
        mid = 0.5 * (lo + hi)
        d[f"{nm}_per_1k"] = obs
        d[f"{nm}_floor_per_1k"] = mid
        d[f"{nm}_floor_lo"] = lo
        d[f"{nm}_floor_hi"] = hi
        d[f"{nm}_excess_per_1k"] = obs - mid
        d[f"{nm}_ratio"] = obs / mid if mid > 0 else float("nan")
        d[f"{nm}_self_per_1k"] = 1e3 * s[IX[f"self{i}"]] / n
    return d


def boot_sums(S, strata, B, rng, pair=None):
    """B resampled field-sum vectors [B, F]: prompts drawn with replacement within each stratum (multinomial counts).
    With `pair` (same shape as S) the same draws are applied to it and both are returned."""
    W = np.zeros((B, S.shape[0]))
    for k in sorted(set(strata)):
        ix = np.array([i for i, x in enumerate(strata) if x == k])
        W[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1.0 / len(ix)), size=B)
    return (W @ S, W @ pair) if pair is not None else W @ S


def with_ci(S, strata, B, rng):
    d = derive(S.sum(0))
    if S.shape[0] < 2 or B <= 0:
        return d
    bs = [derive(x) for x in boot_sums(S, strata, B, rng)]
    d["ci95"] = {}
    for k, v in d.items():
        if k in ("tokens", "ci95") or not isinstance(v, float):
            continue
        a = np.array([b[k] for b in bs], dtype=float)
        if np.isfinite(a).all():
            d["ci95"][k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return d


def precision(S, strata, B, rng):
    """Per count metric: events, Poisson vs bootstrap 95% half-width (relative), design effect, tokens for +-10%."""
    tot = S.sum(0)
    bs = boot_sums(S, strata, B, rng)
    out = {}
    names = {"oon": "out of nucleus", "conf2": "confident p<1%", "r100": "ratio>100"}
    names.update({f"obs{i}": nm for i, nm in enumerate(TAU_NAMES)})
    for f, nm in names.items():
        ev = tot[IX[f]]
        if ev < 1:
            out[nm] = {"events": int(ev)}
            continue
        rate = bs[:, IX[f]] / np.maximum(bs[:, IX["n"]], 1)
        hw = 0.5 * (np.percentile(rate, 97.5) - np.percentile(rate, 2.5)) / (ev / tot[IX["n"]])
        pois = 1.96 / math.sqrt(ev)
        out[nm] = {"events": int(ev), "poisson_halfwidth_rel": pois, "bootstrap_halfwidth_rel": float(hw),
                   "design_effect": float((hw / pois) ** 2), "tokens_for_10pct": int(tot[IX["n"]] * (hw / 0.10) ** 2)}
    g = bs[:, IX["gap"]] / np.maximum(bs[:, IX["n"]], 1)
    hw = 0.5 * (np.percentile(g, 97.5) - np.percentile(g, 2.5)) / abs(tot[IX["gap"]] / tot[IX["n"]])
    out["gap"] = {"bootstrap_halfwidth_rel": float(hw), "tokens_for_10pct": int(tot[IX["n"]] * (hw / 0.10) ** 2)}
    return out


def score(a):
    import torch
    caps, files = {}, []
    for f in a.captures:
        h = hashlib.sha256()
        for line in open(f, "rb"):
            h.update(line)
            r = json.loads(line)
            if not r.get("error") and r.get("ids"):
                caps[r["id"]] = r
        files.append({"file": f.split("/")[-1], "sha256": h.hexdigest()})
    want = None
    if a.ids:
        m = json.load(open(a.ids))
        want = {p["id"] for p in m["prompts"]} if isinstance(m, dict) else set(m)
    info = json.load(open(f"{a.set_dir}/sets.json"))
    ids, kinds, per, ans = [], [], [], {"answers": 0, "at_token_cap": 0, "stopped": 0, "loops": 0,
                                         "loop_tokens_left_out": 0}
    settings, floors = None, set()
    for b in info["main"]["buckets"]:
        c = torch.load(f"{a.set_dir}/{b['name']}.pt", weights_only=False)
        k = torch.load(f"{a.ref_dir}/{b['name']}/kld.pt", weights_only=False)
        nll, lp, rid, tok = k["nll"][0].numpy(), k["ref_lp"].numpy(), k["ref_id"].numpy(), c["tokens"].numpy()
        fp = f"{a.floor_dir or a.ref_dir}/{b['name']}/floor.pt"
        fx = torch.load(fp, weights_only=False) if os.path.exists(fp) else None
        if fx is not None and (fx["ids"] != c["ids"] or abs(fx["top_p"] - a.top_p) > 1e-9 or fx["gen_cap"] < a.gen_cap):
            fx = None
        floors.add("exact" if fx is not None else "bracket")
        for n in range(tok.shape[0]):
            sid = c["ids"][n]
            if want is not None and sid not in want:
                continue
            cp = caps[sid]
            L, G = c["lens"][n], c["prompt"][n]
            assert list(tok[n, G:L]) == cp["ids"][:L - G], f"{sid}: set tokens != capture tokens"
            g = min(L - G, a.gen_cap, len(cp["lp"]))
            loop = cp.get("loop")
            ans["answers"] += 1
            ans["at_token_cap"] += cp.get("finish_reason") == "length"
            ans["stopped"] += cp.get("finish_reason") == "stop"
            if loop:
                ans["loops"] += 1
                ans["loop_tokens_left_out"] += max(0, g - loop["loop_start"])
                g = min(g, loop["loop_start"])
            if cp.get("settings"):
                assert settings in (None, cp["settings"]), f"{sid}: sampling settings differ inside one capture"
                settings = cp["settings"]
            r = np.arange(G - 1, G - 1 + g)
            ex = (fx["floor"][n][:g].double().numpy(), fx["enll"][n][:g].double().numpy()) if fx is not None else None
            st = seq_stats(np.array(cp["lp"][:g], dtype=np.float64), -nll[n, r],
                           np.exp(lp[n, r].astype(np.float64)), rid[n, r], tok[n, r + 1], a.top_p, ex)
            per.append(np.stack([st[lo:hi].sum(0) for lo, hi in BINS]))
            ids.append(sid)
            kinds.append(c["kinds"][n])
    if want is not None and set(ids) != want:
        print(f"WARNING: {len(want - set(ids))} of the manifest's {len(want)} prompts are not in the capture",
              file=sys.stderr)
    S = np.stack(per)                                    # [prompts, bins, fields]
    rng = np.random.default_rng(20261120)
    res = {"label": a.label, "captures": files, "settings": settings or "not recorded", "top_p": a.top_p,
           "gen_cap": a.gen_cap, "prompts": len(ids),
           "floor": {("exact",): "exact (full distribution, tailfloor.py)", ("bracket",): "bracket from 64 tokens"}.get(
               tuple(sorted(floors)), "exact for some buckets, bracket for others"), "answers": ans, "boot": a.boot,
           "all": with_ci(S.sum(1), kinds, a.boot, rng),
           "kinds": {k: with_ci(S[[i for i, x in enumerate(kinds) if x == k]].sum(1), [k] * kinds.count(k), a.boot,
                                rng) for k in sorted(set(kinds))},
           "index_bins": {f"{lo}-{hi if hi < 1 << 30 else 'end'}": with_ci(S[:, j], kinds, a.boot, rng)
                          for j, (lo, hi) in enumerate(BINS) if S[:, j, IX["n"]].sum() > 0},
           "precision": precision(S.sum(1), kinds, max(a.boot, 200), rng),
           "fields": F, "bins": BINS, "per_prompt": {i: {"kind": kd, "sums": S[j].tolist()}
                                                      for j, (i, kd) in enumerate(zip(ids, kinds))}}
    json.dump(res, open(a.out, "w"))
    print(table(res))


def _ci(d, k, fmt):
    c = d.get("ci95", {}).get(k)
    return f"{d[k]:{fmt}}" + (f" [{c[0]:{fmt}}, {c[1]:{fmt}}]" if c else "")


def line(name, d):
    s = f"{name:22s} {d['tokens']:7d} tok | gap {_ci(d, 'gap', '.4f')} | excess surprise {_ci(d, 'excess_surprise', '.4f')}" \
        f" | out of nucleus {_ci(d, 'out_of_nucleus_per_1k', '.2f')}/1k"
    for nm in TAU_NAMES:
        fl = d[nm + "_floor_per_1k"]
        s += f" | {nm} {_ci(d, nm + '_per_1k', '.2f')}, floor {fl:.3f} ({d[nm + '_floor_lo']:.3f}-{d[nm + '_floor_hi']:.3f})" \
             f", excess {_ci(d, nm + '_excess_per_1k', '.2f')}" + (f", {_ci(d, nm + '_ratio', '.1f')}x floor" if fl >= 0.25 else "")
    return s + f" | confident p<1% {_ci(d, 'confident_p<1%_per_1k', '.2f')}/1k | ratio>100 {d['ratio>100_per_1k']:.2f}/1k"


def table(res):
    a = res["answers"]
    out = [f"# {res['label'] or res['captures'][0]['file']}: {res['prompts']} prompts, first {res['gen_cap']} tokens "
           f"of each answer, top-p {res['top_p']}, settings {res['settings']}; official floor: {res['floor']}",
           f"# answers {a['answers']}: {a['stopped']} stopped, {a['at_token_cap']} at the token cap, {a['loops']} "
           f"loops ({a['loop_tokens_left_out']} loop tokens left out); intervals: 95% bootstrap over prompts "
           f"({res['boot']} draws, within kind); tail rates per 1,000 tokens",
           line("ALL", res["all"])]
    out += [line("  " + k, d) for k, d in res["kinds"].items()]
    out += [line("  index " + k, d) for k, d in res["index_bins"].items()]
    d = res["all"]
    out.append("self tail (the engine's own rate of tokens it gave under the threshold): "
               + ", ".join(f"{nm} {d[nm + '_self_per_1k']:.2f}/1k" for nm in TAU_NAMES)
               + f"; official top-1 = token {d['official_top1']:.4f}; nucleus mass beyond the stored 64: "
                 f"{d['uncovered_nucleus_mass']:.5f}; undetermined nucleus membership {d['undetermined_per_1k']:.2f}/1k")
    for nm, p in res["precision"].items():
        if "bootstrap_halfwidth_rel" in p:
            out.append(f"precision {nm:16s} events {p.get('events', '-')!s:>6} | 95% half-width: Poisson "
                       f"{100 * p.get('poisson_halfwidth_rel', float('nan')):5.1f}%, bootstrap over prompts "
                       f"{100 * p['bootstrap_halfwidth_rel']:5.1f}% (design effect {p.get('design_effect', float('nan')):.2f})"
                       f" | tokens for +-10%: {p['tokens_for_10pct']:,}")
    return "\n".join(out)


def compare(a):
    A, Bq = json.load(open(a.a)), json.load(open(a.b))
    sa, sb = A["settings"], Bq["settings"]
    if isinstance(sa, dict) and isinstance(sb, dict):
        diff = {k for k in ("temperature", "top_p", "max_tokens", "reasoning_effort") if sa.get(k) != sb.get(k)}
        if diff and not a.force:
            sys.exit(f"sampling settings differ ({sorted(diff)}): not comparable (--force to compare anyway)")
    if (A["gen_cap"], A["top_p"]) != (Bq["gen_cap"], Bq["top_p"]) and not a.force:
        sys.exit("scores were made with different --gen-cap / --top-p")
    ids = [i for i in A["per_prompt"] if i in Bq["per_prompt"]]
    kinds = [A["per_prompt"][i]["kind"] for i in ids]
    SA = np.array([A["per_prompt"][i]["sums"] for i in ids]).sum(1)
    SB = np.array([Bq["per_prompt"][i]["sums"] for i in ids]).sum(1)
    rng = np.random.default_rng(20261120)
    ba, bb = boot_sums(SA, kinds, a.boot, rng, pair=SB)
    da, db = derive(SA.sum(0)), derive(SB.sum(0))
    ua = boot_sums(SA, kinds, a.boot, np.random.default_rng(1))
    ub = boot_sums(SB, kinds, a.boot, np.random.default_rng(2))
    keys = ["gap", "excess_surprise", "out_of_nucleus_per_1k"] + [nm + "_per_1k" for nm in TAU_NAMES] \
        + ["p<0.1%_excess_per_1k", "confident_p<1%_per_1k", "ratio>100_per_1k"]
    print(f"# A = {A['label'] or A['captures'][0]['file']}, B = {Bq['label'] or Bq['captures'][0]['file']}; "
          f"{len(ids)} common prompts (A {A['prompts']}, B {Bq['prompts']}); tokens A {da['tokens']}, B {db['tokens']}; "
          f"paired bootstrap over prompts, {a.boot} draws")
    print(f"{'metric':26s} {'A':>9s} {'B':>9s} {'B - A':>9s}  95% interval (paired)    P(B > A)  unpaired interval")
    res = {}
    for k in keys:
        d = np.array([derive(y)[k] - derive(x)[k] for x, y in zip(ba, bb)])
        u = np.array([derive(y)[k] - derive(x)[k] for x, y in zip(ua, ub)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        ulo, uhi = np.percentile(u, [2.5, 97.5])
        res[k] = {"a": da[k], "b": db[k], "diff": db[k] - da[k], "ci95": [float(lo), float(hi)],
                  "p_b_gt_a": float((d > 0).mean()), "unpaired_ci95": [float(ulo), float(uhi)]}
        print(f"{k:26s} {da[k]:9.4f} {db[k]:9.4f} {db[k] - da[k]:+9.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{(d > 0).mean():8.3f}  [{ulo:+.4f}, {uhi:+.4f}]{'' if lo <= 0 <= hi else '  *'}")
    if a.out:
        json.dump({"a": a.a, "b": a.b, "common_prompts": len(ids), "metrics": res}, open(a.out, "w"), indent=1)


def selftest(a):
    """Synthetic check of the definitions: a made-up official model over 2,000 tokens, of which the scorer sees the top
    64 and the sampled token's probability (as from kld.pt). (1) An engine that IS the official model, sampling from
    the nucleus: gap 0, excess surprise ~0, observed tail = the floor, nothing out of the nucleus. (2) The floor from
    64 tokens brackets the floor from the full distribution. (3) An engine with 1% of its rows replaced by a
    confident wrong token reads the planted rate."""
    rng = np.random.default_rng(7)
    V, T, top_p = 2000, 60000, a.top_p
    # language-model-like rows: Zipf tails of different steepness (the flattest ones are not covered by 64 tokens)
    logit = -rng.choice([1.2, 2.0, 3.0, 5.0], p=[0.1, 0.3, 0.3, 0.3], size=(T, 1)) * np.log(np.arange(1, V + 1)) \
        + 0.5 * rng.standard_normal((T, V))
    p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    order = np.argsort(-p, 1); ps = np.take_along_axis(p, order, 1)
    cb = np.cumsum(ps, 1) - ps
    nuc = cb < top_p
    Z = (ps * nuc).sum(1)
    u = rng.random(T) * Z
    pick = np.minimum((np.cumsum(ps * nuc, 1) < u[:, None]).sum(1), V - 1)
    x = order[np.arange(T), pick]
    lpt = np.log(ps[np.arange(T), pick])
    st = seq_stats(lpt.copy(), lpt, ps[:, :64], order[:, :64], x, top_p)
    d = derive(st.sum(0))
    ok = True
    exact = [1e3 * ((ps * nuc * (ps < tau)).sum(1) / Z).mean() for tau in TAUS]
    enll = ((ps * nuc * -np.log(ps)).sum(1) / Z).mean()
    print(f"(1) faithful engine: gap {d['gap']:.5f}, excess surprise {d['excess_surprise']:+.4f} (exact expected NLL "
          f"{enll:.4f}, from 64 tokens {d['official_nll'] - d['excess_surprise']:.4f}), out of nucleus "
          f"{d['out_of_nucleus_per_1k']:.2f}/1k, undetermined {d['undetermined_per_1k']:.2f}/1k, nucleus mass beyond 64: "
          f"{d['uncovered_nucleus_mass']:.4f}")
    ok &= d["gap"] == 0 and d["out_of_nucleus_per_1k"] == 0
    for i, nm in enumerate(TAU_NAMES):
        lo, hi, obs = d[nm + "_floor_lo"], d[nm + "_floor_hi"], d[nm + "_per_1k"]
        se = 1e3 * math.sqrt(max(exact[i], 1e-9) / 1e3 / T)
        good = lo - 1e-9 <= exact[i] <= hi + 1e-9 and abs(obs - exact[i]) < 4 * se + 1e-9
        ok &= good
        print(f"(2) {nm}: exact floor {exact[i]:.3f}/1k, from 64 tokens {lo:.3f} .. {hi:.3f}; observed {obs:.3f} "
              f"(+-{2 * se:.3f}) {'ok' if good else 'FAIL'}")
    bad = rng.random(T) < 0.01
    worst = order[:, 300]                                   # a token far down the official ranking
    x2 = np.where(bad, worst, x)
    lpt2 = np.where(bad, np.log(ps[:, 300]), lpt)
    lq2 = np.where(bad, math.log(0.9), lpt)
    d2 = derive(seq_stats(lq2, lpt2, ps[:, :64], order[:, :64], x2, top_p).sum(0))
    planted = 1e3 * (bad & (ps[:, 300] < 1e-3)).mean()
    e_lo, e_hi = d2["p<0.1%_per_1k"] - d2["p<0.1%_floor_hi"], d2["p<0.1%_per_1k"] - d2["p<0.1%_floor_lo"]
    good = e_lo - 1.0 <= planted <= e_hi + 1.0 \
        and abs(d2["confident_p<1%_per_1k"] - 1e3 * (bad & (ps[:, 300] < 1e-2)).mean()) < 1e-9
    ok &= good
    print(f"(3) planted confident errors: {planted:.2f}/1k under 0.1%; excess p<0.1% {e_lo:.2f} .. {e_hi:.2f}/1k, "
          f"confident p<1% {d2['confident_p<1%_per_1k']:.2f}/1k, gap {d2['gap']:.4f}, out of nucleus "
          f"{d2['out_of_nucleus_per_1k']:.2f}/1k {'ok' if good else 'FAIL'}")
    print("TAILSCORE SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("set_dir"); s.add_argument("ref_dir"); s.add_argument("captures", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--gen-cap", type=int, default=1024)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--ids", help="score only these request ids: a set manifest (tail-v1.json) or a JSON list")
    s.add_argument("--label", default="")
    s.add_argument("--floor-dir", help="directory with <bucket>/floor.pt from tailfloor.py (default REF_DIR)")
    s.add_argument("--boot", type=int, default=2000)
    c = sub.add_parser("compare")
    c.add_argument("a"); c.add_argument("b")
    c.add_argument("--boot", type=int, default=2000)
    c.add_argument("--force", action="store_true")
    c.add_argument("--out")
    t = sub.add_parser("selftest")
    t.add_argument("--top-p", type=float, default=0.95)
    a = p.parse_args()
    {"score": score, "compare": compare, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
