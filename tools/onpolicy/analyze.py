"""On-policy KLD report: the engine's own answers scored against the official model, and what happens at loops.

    analyze.py --sets SETS_DIR --ref REF_DIR --results RESULTS_DIR --tag TAG[,TAG2...] --captures A.jsonl [B.jsonl ...]
               --out REPORT.json [--tokenizer DIR] [--loops-tag TAG] [--fixed '{"512 cuts": 0.06, ...}']

SETS_DIR: build_sets.py's buckets and sets.json. REF_DIR/<bucket>/kld.pt: the reference runner's pass over the bucket
(the official model's top-64 log-probabilities per row and its NLL of the next token). RESULTS_DIR/
engine-op-<TAG>-<bucket>.tokens.pt: harness/kld_onpolicy.py's per-row values (exact KL(reference || engine), reverse
KL, the engine's NLL of its own token when re-reading). CAPTURES: capture.py's records (the engine's log-probability
of each token at the moment it sampled it).

Main set (buckets main-*): KL over the on-policy rows, mean with a bootstrap interval over prompts, percentiles, by
kind, by generated-token index, for answers that later loop and those that do not; reverse KL; how often the engine's
token lies outside the official model's top-p 0.95 nucleus (never, if the official model had been sampled at the
capture's settings) or below 1% / 0.1% official probability; and a replay check (the engine's log-probability of its
token when it sampled it against its log-probability when the harness re-read the answer).

Loop set (buckets loop-*): per loop, the KL in windows before the loop's start (did the engine drift first?), inside
it, the official model's probability of the repeated tokens next to the engine's (does the official model also want to
repeat?), the probability of staying on the cycle for one whole period under each model, the largest per-row KLs and
the least likely tokens (by the official model) around the start.
"""
import argparse
import json
import os

import numpy as np
import torch

INDEX = ((0, 64), (64, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768))
TOP_P = 0.95


def boot(vals, rows, blk, n=2000, seed=0):
    """Mean of per-sequence sums / rows and its 95% interval over blocks (prompts)."""
    keys = sorted(set(blk))
    ix = np.array([keys.index(b) for b in blk])
    s = np.bincount(ix, weights=vals, minlength=len(keys))
    c = np.bincount(ix, weights=rows, minlength=len(keys))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n, len(keys)))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1)
    return float(s.sum() / max(c.sum(), 1)), [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def stats(x):
    x = x[~np.isnan(x)]
    if not x.size:
        return None
    q = np.quantile(x, [0.5, 0.9, 0.99, 0.999])
    return {"mean": float(x.mean()), "rows": int(x.size), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "p999": float(q[3]), "max": float(x.max())}


def load_sequences(a, tags, names):
    """One dict per sequence of the named buckets: per generated token j (row prompt - 1 + j) the reference and engine
    quantities, NaN where a source has no value."""
    seqs = []
    for b in names:
        c = torch.load(os.path.join(a.sets, b + ".pt"), weights_only=False)
        kp = os.path.join(a.ref, b, "kld.pt")
        if not os.path.exists(kp):
            continue
        k = torch.load(kp, weights_only=False)
        ref_lp, ref_id = k["ref_lp"].numpy(), k["ref_id"].numpy()
        ref_nll = k["nll"][0].numpy()
        eng = {}
        for tag in tags:
            f = os.path.join(a.results, f"engine-op-{tag}-{b}.tokens.pt")
            if os.path.exists(f):
                eng[tag] = torch.load(f, weights_only=False)
        tok = c["tokens"].numpy()
        for n in range(tok.shape[0]):
            L, G, F = int(c["lens"][n]), int(c["prompt"][n]), int(c["forced_from"][n])
            r = np.arange(G - 1, L - 1)                       # on-policy rows; row r[j] predicts generated token j
            nxt = tok[n, r + 1]
            lp, idd = ref_lp[n, r], ref_id[n, r]
            p = np.exp(lp.astype(np.float64))
            before = np.concatenate([np.zeros((len(r), 1)), np.cumsum(p, 1)[:, :-1]], 1)   # mass ranked above
            hit = idd == nxt[:, None]
            rank = np.where(hit.any(1), hit.argmax(1), -1)
            above = np.where(rank >= 0, before[np.arange(len(r)), np.maximum(rank, 0)], p.sum(1))
            s = {"id": c["ids"][n], "kind": c["kinds"][n], "bucket": b, "row": n, "len": L, "prompt": G,
                 "forced_from": F, "tokens": tok[n, :L], "ref_nll": ref_nll[n, r].astype(np.float64),
                 "ref_top1": idd[:, 0] == nxt, "ref_rank": rank,
                 "out_of_nucleus": above >= TOP_P,           # the official model's top-p 0.95 would not sample it
                 "loop_start": int(c["loop_start"][n]) - G if c.get("loop_start") is not None else None,
                 "period": int(c["period"][n]) if c.get("period") is not None else None, "eng": {}}
            for tag, t in eng.items():
                s["eng"][tag] = {"kl": t["kl"][n, r].numpy().astype(np.float64),
                                 "rev": t["kl_rev"][n, r].numpy().astype(np.float64) if t.get("kl_rev") is not None else None,
                                 "nll": t["nll"][n, r].numpy().astype(np.float64),
                                 "top1": t["top1"][n, r].numpy(), "decode": t["decode"][n, r].numpy()}
            seqs.append(s)
    return seqs


def main_report(seqs, tag, caps, loop_ids):
    use = [s for s in seqs if tag in s["eng"]]
    if not use:
        return None
    blk = [s["id"].split("#")[0] for s in use]
    kl = [s["eng"][tag]["kl"] for s in use]
    allk = np.concatenate(kl)
    out = {"sequences": len(use), "prompts": len(set(blk)), **(stats(allk) or {})}
    out["mean"], out["ci95"] = boot([np.nansum(x) for x in kl], [(~np.isnan(x)).sum() for x in kl], blk)
    out["top1"] = float(np.concatenate([s["eng"][tag]["top1"] for s in use]).mean())
    if use[0]["eng"][tag]["rev"] is not None:
        rv = [s["eng"][tag]["rev"] for s in use]
        m, ci = boot([np.nansum(x) for x in rv], [(~np.isnan(x)).sum() for x in rv], blk, seed=1)
        out["reverse"] = {**(stats(np.concatenate(rv)) or {}), "mean": m, "ci95": ci}
    out["by_kind"] = {}
    for kind in sorted({s["kind"] for s in use}):
        sub = [s for s in use if s["kind"] == kind]
        x = [s["eng"][tag]["kl"] for s in sub]
        m, ci = boot([np.nansum(v) for v in x], [(~np.isnan(v)).sum() for v in x], [s["id"].split("#")[0] for s in sub])
        st = stats(np.concatenate(x)) or {}
        out["by_kind"][kind] = {"mean": m, "ci95": ci, "rows": st.get("rows"), "p99": st.get("p99"), "sequences": len(sub),
                                "out_of_nucleus_per_1k": 1000 * float(np.concatenate([s["out_of_nucleus"] for s in sub]).mean())}
    out["by_index"] = {}
    for a0, a1 in INDEX:
        x = np.concatenate([v[a0:a1] for v in kl])
        if x.size:
            out["by_index"][f"{a0}-{a1}"] = {"mean": float(np.nanmean(x)), "rows": int(x.size)}
    for name, sel in (("answers_with_a_loop", True), ("answers_without_a_loop", False)):
        sub = [s for s in use if (s["id"] in loop_ids) == sel]
        if sub:
            x = np.concatenate([s["eng"][tag]["kl"] for s in sub])
            out[name] = {"sequences": len(sub), "mean": float(np.nanmean(x)), "rows": int(x.size),
                         "by_index": {f"{a0}-{a1}": float(np.nanmean(np.concatenate([s["eng"][tag]["kl"][a0:a1] for s in sub])))
                                      for a0, a1 in INDEX if any(len(s["eng"][tag]["kl"]) > a0 for s in sub)}}
    # the engine's tokens under the official model
    oon = np.concatenate([s["out_of_nucleus"] for s in use])
    rn = np.concatenate([s["ref_nll"] for s in use])
    out["official_view_of_sampled_tokens"] = {
        "out_of_top_p_nucleus_per_1k": 1000 * float(oon.mean()), "below_1pct_per_1k": 1000 * float((rn > -np.log(0.01)).mean()),
        "below_0.1pct_per_1k": 1000 * float((rn > -np.log(0.001)).mean()), "official_nll": float(rn.mean()),
        "official_top1_is_the_token": float(np.concatenate([s["ref_top1"] for s in use]).mean())}
    # replay check and the sample estimate of the reverse KL, from the capture's own log-probabilities
    d, gap, en = [], [], []
    for s in use:
        cp = caps.get(s["id"])
        if cp is None:
            continue
        n = len(s["ref_nll"])
        lp = np.array(cp["lp"][:n], dtype=np.float64)
        d.append(np.abs(-lp - s["eng"][tag]["nll"]))
        gap.append(lp + s["ref_nll"])                           # log q(x) - log p(x) at the sampled token
        en.append(-lp)
    if d:
        d, gap = np.concatenate(d), np.concatenate(gap)
        out["replay_check"] = {"what": "|engine log-prob of its token when sampled - when re-read by the harness|",
                               "mean": float(np.nanmean(d)), "median": float(np.nanmedian(d)),
                               "p99": float(np.nanquantile(d, 0.99)), "rows": int((~np.isnan(d)).sum())}
        out["sampled_token_gap"] = {"what": "mean of log q_engine(x) - log p_official(x) over the engine's sampled tokens "
                                            "(a one-sample estimate of the reverse KL of the top-p-truncated engine)",
                                    "mean": float(gap.mean()), "engine_nll_when_sampled": float(np.concatenate(en).mean())}
    return out


def loop_report(s, tag, cap, base_index, tok):
    e = s["eng"].get(tag)
    LS, P = s["loop_start"], s["period"]
    n = len(s["ref_nll"])
    lp_cap = np.array(cap["lp"][:n], dtype=np.float64) if cap else np.full(n, np.nan)
    out = {"id": s["id"], "kind": s["kind"], "generated_total": len(cap["ids"]) if cap else None,
           "finish_reason": cap["finish_reason"] if cap else None, "loop_start": LS, "period": P,
           "tokens_in_set_after_start": n - LS}
    reg = {"before (all)": (0, LS), "last 4096 before": (max(0, LS - 4096), LS), "last 512 before": (max(0, LS - 512), LS),
           "first period": (LS, min(n, LS + P)), "repeats": (min(n, LS + P), n)}
    out["regions"] = {}
    for name, (a0, a1) in reg.items():
        if a1 <= a0:
            continue
        r = {"tokens": a1 - a0, "official_nll": float(s["ref_nll"][a0:a1].mean()),
             "engine_nll_when_sampled": float(np.nanmean(lp_cap[a0:a1]) * -1),
             "official_top1_is_the_token": float(s["ref_top1"][a0:a1].mean()),
             "official_mean_prob_of_token": float(np.exp(-s["ref_nll"][a0:a1]).mean()),
             "engine_mean_prob_of_token": float(np.nanmean(np.exp(lp_cap[a0:a1]))),
             "out_of_nucleus_per_1k": 1000 * float(s["out_of_nucleus"][a0:a1].mean())}
        if e is not None:
            r["kl"] = float(np.nanmean(e["kl"][a0:a1]))
            r["kl_scored_rows"] = int((~np.isnan(e["kl"][a0:a1])).sum())
            if e["rev"] is not None:
                r["reverse_kl"] = float(np.nanmean(e["rev"][a0:a1]))
            r["engine_top1_is_official_top1"] = float(e["top1"][a0:a1].mean())
        out["regions"][name] = r
    # staying on the cycle for one whole period: exp(sum of log-probs of the period's tokens), per repeat
    cyc = []
    a0 = LS + P
    while a0 + P <= n:
        cyc.append({"official": float(np.exp(-s["ref_nll"][a0:a0 + P].sum())),
                    "engine": float(np.exp(np.nansum(lp_cap[a0:a0 + P])))})
        a0 += P
    out["stay_probability_per_period"] = cyc[:12]
    if e is not None:
        w = 512
        out["kl_windows_before_start"] = [{"from": a1 - w, "to": a1, "kl": float(np.nanmean(e["kl"][max(0, a1 - w):a1])),
                                           "baseline_same_index": base_index(max(0, a1 - w), a1)}
                                          for a1 in range(LS, max(0, LS - 4096), -w) if a1 - w >= 0][::-1]
        a0 = max(0, LS - 4096)
        seg = np.nan_to_num(e["kl"][a0:], nan=-1.0)
        top = np.argsort(-seg)[:8]
        out["largest_kl_rows"] = [{"offset_from_loop_start": int(a0 + i - LS), "kl": float(seg[i]),
                                   "official_prob_of_token": float(np.exp(-s["ref_nll"][a0 + i])),
                                   "engine_prob_when_sampled": float(np.exp(lp_cap[a0 + i])),
                                   "text": context(tok, s, a0 + i)} for i in top]
    a0, a1 = max(0, LS - 1024), min(n, LS + P + 64)
    worst = np.argsort(-s["ref_nll"][a0:a1])[:8]
    out["least_likely_tokens_near_start"] = [{"offset_from_loop_start": int(a0 + i - LS),
                                              "official_prob_of_token": float(np.exp(-s["ref_nll"][a0 + i])),
                                              "engine_prob_when_sampled": float(np.exp(lp_cap[a0 + i])),
                                              "out_of_nucleus": bool(s["out_of_nucleus"][a0 + i]),
                                              "text": context(tok, s, a0 + i)} for i in worst]
    return out


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def looping_error(caps, loop_info, mseqs, lseqs, tag, ltag):
    """The looping error as numbers: how often an answer loops, the KL on rows inside loops against rows outside,
    and the share of all generated tokens / of the KL mass that sits inside loops. Loop rows are scored on each
    loop's first tokens only (the loop set keeps ~2K of them): the mass share extends that per-row value to the
    loop's captured length, an estimate."""
    starts = {q["id"]: q["loop"]["loop_start"] for q in loop_info}
    out = {"answers": len(caps), "loops": sum(i in starts for i in caps), "by_kind": {}}
    out["loop_rate"] = out["loops"] / max(1, out["answers"])
    out["loop_rate_ci95"] = wilson(out["loops"], out["answers"])
    for kind in sorted({r["kind"] for r in caps.values()}):
        ids = [i for i, r in caps.items() if r["kind"] == kind]
        k = sum(i in starts for i in ids)
        out["by_kind"][kind] = {"answers": len(ids), "loops": k, "rate": k / len(ids), "ci95": wilson(k, len(ids)),
                                "at_token_cap": sum(caps[i]["finish_reason"] == "length" for i in ids)}
    tok_all = sum(len(r["ids"]) for r in caps.values())
    tok_loop = sum(len(caps[i]["ids"]) - st for i, st in starts.items() if i in caps)
    out["generated_tokens"] = tok_all
    out["tokens_inside_loops"] = tok_loop
    out["token_share_inside_loops"] = tok_loop / max(1, tok_all)
    non = [s["eng"][tag]["kl"] for s in mseqs if tag in s["eng"] and s["id"] not in starts]
    pre = [s["eng"][ltag]["kl"][:s["loop_start"]] for s in lseqs if ltag in s["eng"]]
    pre += [s["eng"][tag]["kl"][:starts[s["id"]]] for s in mseqs if tag in s["eng"] and s["id"] in starts
            and not any(q["id"] == s["id"] and ltag in q["eng"] for q in lseqs)]
    ins = [s["eng"][ltag]["kl"][s["loop_start"]:] for s in lseqs if ltag in s["eng"]]
    rins = [s["eng"][ltag]["rev"][s["loop_start"]:] for s in lseqs if ltag in s["eng"] and s["eng"][ltag]["rev"] is not None]

    def mean(xs):
        x = np.concatenate(xs) if xs else np.array([])
        x = x[~np.isnan(x)]
        return (float(x.mean()), int(x.size)) if x.size else (None, 0)
    (k_non, n_non), (k_pre, n_pre), (k_in, n_in) = mean(non), mean(pre), mean(ins)
    out["kl_rows_of_answers_without_a_loop"] = {"kld": k_non, "rows": n_non}
    out["kl_rows_before_a_loop_starts"] = {"kld": k_pre, "rows": n_pre}
    out["kl_rows_inside_loops"] = {"kld": k_in, "rows": n_in, "reverse": mean(rins)[0],
                                   "per_loop": [{"id": s["id"], "kld": mean([s["eng"][ltag]["kl"][s["loop_start"]:]])[0]}
                                                for s in lseqs if ltag in s["eng"]]}
    if k_in is not None and k_non is not None:
        out["inside_over_outside"] = k_in / k_non if k_non else None
        mass_loop, mass_rest = k_in * tok_loop, k_non * (tok_all - tok_loop)
        out["kl_mass_share_inside_loops_est"] = mass_loop / max(mass_loop + mass_rest, 1e-30)
        out["kld_all_generated_tokens_est"] = (mass_loop + mass_rest) / max(1, tok_all)
        out["note"] = ("mass share and the all-token KLD are estimates: outside rows use the main set's value (first "
                       "tokens of every answer), inside rows the loop set's value (each loop's first tokens), "
                       "weighted by the captured token counts (answers capped at the capture's max_tokens)")
    return out


def context(tok, s, j, before=24, after=8):
    """Text around generated token j: the tokens before it, [[the token]], the tokens after."""
    if tok is None:
        return None
    g = s["prompt"] + j
    ids = s["tokens"]
    return (tok.decode(ids[max(s["prompt"], g - before):g].tolist()) + "[[" + tok.decode(ids[g:g + 1].tolist()) + "]]"
            + tok.decode(ids[g + 1:g + 1 + after].tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", required=True); p.add_argument("--ref", required=True)
    p.add_argument("--results", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--loops-tag", default="")
    p.add_argument("--captures", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=""); p.add_argument("--fixed", default="")
    a = p.parse_args()
    info = json.load(open(os.path.join(a.sets, "sets.json")))
    caps = {}
    for f in a.captures:
        for line in open(f):
            r = json.loads(line)
            if not r.get("error") and r.get("ids"):
                caps.setdefault(r["id"], r)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    tags = a.tag.split(",")
    ltags = (a.loops_tag or a.tag).split(",")
    main_b = [b["name"] for b in (info.get("main") or {}).get("buckets", [])]
    loop_b = [b["name"] for b in (info.get("loop") or {}).get("buckets", [])]
    mseqs = load_sequences(a, tags, main_b)
    lseqs = load_sequences(a, ltags, loop_b)
    loop_ids = {s["id"] for s in lseqs}
    rep = {"capture": capture_summary(caps), "fixed_text": json.loads(a.fixed) if a.fixed else None, "main": {}, "loops": {}}
    for tag in tags:
        rep["main"][tag] = main_report(mseqs, tag, caps, loop_ids)
    for tag in ltags:
        use = [s for s in mseqs if tag in s["eng"] and s["id"] not in loop_ids] or [s for s in mseqs if tags[0] in s["eng"] and s["id"] not in loop_ids]
        btag = tag if any(tag in s["eng"] for s in mseqs) else tags[0]

        def base_index(a0, a1, use=use, btag=btag):
            x = [s["eng"][btag]["kl"][a0:a1] for s in use if len(s["eng"][btag]["kl"]) > a0]
            return float(np.nanmean(np.concatenate(x))) if x else None
        rep["loops"][tag] = [loop_report(s, tag, caps.get(s["id"]), base_index, tok) for s in lseqs]
        rep.setdefault("looping_error", {})[tag] = looping_error(caps, (info.get("loop") or {}).get("sequences", []),
                                                                  mseqs, lseqs, btag, tag)
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    text(rep)


def capture_summary(caps):
    by = {}
    for r in caps.values():
        k = by.setdefault(r["kind"], {"answers": 0, "tokens": 0, "at_cap": 0, "loops": 0, "lens": []})
        k["answers"] += 1
        k["tokens"] += len(r["ids"])
        k["at_cap"] += r["finish_reason"] == "length"
        k["loops"] += r.get("loop") is not None
        k["lens"].append(len(r["ids"]))
    for k in by.values():
        v = np.array(k.pop("lens"))
        k["median"], k["p90"], k["max"] = int(np.median(v)), int(np.quantile(v, 0.9)), int(v.max())
    return by


def text(rep):
    f = lambda x: "-" if x is None else f"{x:.4f}"                    # noqa: E731
    print("capture:", json.dumps(rep["capture"]))
    if rep["fixed_text"]:
        print("fixed-text numbers:", json.dumps(rep["fixed_text"]))
    for tag, m in rep["main"].items():
        if not m:
            continue
        print(f"\n== main set, {tag}: ON-POLICY KLD {f(m['mean'])} [{f(m['ci95'][0])}, {f(m['ci95'][1])}] over {m['rows']} rows "
              f"({m['sequences']} answers, {m['prompts']} prompts); median {f(m['median'])} p90 {f(m['p90'])} p99 {f(m['p99'])} "
              f"p99.9 {f(m['p999'])}; top-1 {100 * m['top1']:.2f}%")
        if m.get("reverse"):
            r = m["reverse"]
            print(f"   reverse KL(engine || official) {f(r['mean'])} [{f(r['ci95'][0])}, {f(r['ci95'][1])}], p99 {f(r['p99'])}")
        print("   by kind: " + "; ".join(f"{k} {f(v['mean'])} [{f(v['ci95'][0])}, {f(v['ci95'][1])}] ({v['rows']} rows, p99 "
                                         f"{f(v['p99'])})" for k, v in m["by_kind"].items()))
        print("   by generated index: " + " ".join(f"{k} {f(v['mean'])}" for k, v in m["by_index"].items()))
        for k in ("answers_with_a_loop", "answers_without_a_loop"):
            if m.get(k):
                print(f"   {k}: {f(m[k]['mean'])} over {m[k]['rows']} rows ({m[k]['sequences']} answers); by index "
                      + " ".join(f"{i} {f(v)}" for i, v in m[k]["by_index"].items()))
        print("   the engine's tokens under the official model:", json.dumps(m["official_view_of_sampled_tokens"]))
        if m.get("replay_check"):
            print("   replay check:", json.dumps({k: v for k, v in m["replay_check"].items() if k != "what"}),
                  "| sampled-token gap", f(m["sampled_token_gap"]["mean"]))
    for tag, le in (rep.get("looping_error") or {}).items():
        print(f"\n== looping error, {tag}: {le['loops']} of {le['answers']} answers loop ({100 * le['loop_rate']:.1f}%, 95% "
              f"{[round(100 * v, 1) for v in le['loop_rate_ci95']]}); {100 * le['token_share_inside_loops']:.1f}% of generated "
              f"tokens inside loops; KL outside {f(le['kl_rows_of_answers_without_a_loop']['kld'])}, before a loop "
              f"{f(le['kl_rows_before_a_loop_starts']['kld'])}, inside {f(le['kl_rows_inside_loops']['kld'])} "
              f"(x{(le.get('inside_over_outside') or 0):.2f}); KL mass inside loops (est.) "
              f"{100 * (le.get('kl_mass_share_inside_loops_est') or 0):.1f}%; KLD over all generated tokens (est.) "
              f"{f(le.get('kld_all_generated_tokens_est'))}")
        print("   by kind: " + "; ".join(f"{k} {v['loops']}/{v['answers']} (cap {v['at_token_cap']})" for k, v in le["by_kind"].items()))
    for tag, loops in rep["loops"].items():
        print(f"\n== loops, {tag}: {len(loops)}")
        for lp in loops:
            print(f" {lp['id']} ({lp['kind']}): {lp['generated_total']} tokens, {lp['finish_reason']}; loop from "
                  f"{lp['loop_start']}, period {lp['period']}")
            for name, r in lp["regions"].items():
                print(f"    {name:18s} {r['tokens']:6d} tok | KL {f(r.get('kl'))} rev {f(r.get('reverse_kl'))} | official nll "
                      f"{r['official_nll']:.3f} p(tok) {r['official_mean_prob_of_token']:.3f} top-1 hit "
                      f"{r['official_top1_is_the_token']:.3f} | engine nll {r['engine_nll_when_sampled']:.3f} p(tok) "
                      f"{r['engine_mean_prob_of_token']:.3f} | out of nucleus {r['out_of_nucleus_per_1k']:.1f}/1k")
            c = lp["stay_probability_per_period"]
            if c:
                print("    stay on the cycle for a period (official / engine): "
                      + " ".join(f"{x['official']:.2f}/{x['engine']:.2f}" for x in c))
            if lp.get("kl_windows_before_start"):
                print("    KL in 512-token windows before the start (vs non-looping answers at the same index): "
                      + " ".join(f"{f(w['kl'])}({f(w['baseline_same_index'])})" for w in lp["kl_windows_before_start"]))
            for x in (lp.get("largest_kl_rows") or [])[:4]:
                print(f"    KL {x['kl']:.2f} at {x['offset_from_loop_start']:+d}: official p {x['official_prob_of_token']:.4f}, "
                      f"engine p {x['engine_prob_when_sampled']:.4f} | {x['text']!r}")
            for x in lp["least_likely_tokens_near_start"][:4]:
                print(f"    least likely at {x['offset_from_loop_start']:+d}: official p {x['official_prob_of_token']:.5f}, "
                      f"engine p {x['engine_prob_when_sampled']:.4f}, out of nucleus {x['out_of_nucleus']} | {x['text']!r}")


if __name__ == "__main__":
    main()
