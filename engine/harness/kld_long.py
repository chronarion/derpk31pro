"""Long-sequence KLD (kld.py --long; on by itself when the corpus's sequences are longer than 2,048 tokens).

    kld.py URL CORPUS.pt REF_KLD.pt[,HALF2] OUT.json --mode decode --prompt tail --kl-hidden H0.npy[,H1.npy]
           [--kl-rows=A:B] [--prompt-rows exact|estimate] [--prompt-logprobs N] [--block N] [--cuts CUTS.json]
           [--decode-logprobs 1 --pipeline --prefetch]

The L32 protocol (corpus/k31-l32.pt: 32 sequences of 32,768 tokens; `tail` = 4,096 forced rows for agentic and SWE,
2,048 otherwise): each sequence's first P = T - tail tokens go through the prompt path as served, the rest is forced
one decode step at a time (k31_engine/force.py), and the server computes the exact full-vocabulary KL(reference ||
engine) of every row in --kl-rows (default all) from the reference's final hidden states (--kl-hidden; several files
in order are one reference, e.g. the runner's two halves; a file of a row range says so in FILE.json, runner
--dump-rows) or its full log-probabilities (--kl-ref). The request carries `k31_kl_rows`, which selects force.py's
row-range path: the reference rows stay in pinned host memory and only the forced rows sit on the GPU. Prompt rows are
scored exactly too (--prompt-rows estimate: kld.py's top-k estimate there, the 2,048-token protocol's rule), so the
prompt log-probabilities only serve top-1 and NLL: 1 per row by default (the engine's top-1 and the prompt token; 64
per row would be ~50 MB of JSON per 30K-token prompt).

Rows: row t is the engine's distribution after token t (it predicts token t + 1); rows 0 .. T - 2 are scored. Row t is
a forced (decode) row when t >= P - 1 (the prompt's last row is the first sampled one), a generated row when the token
it predicts is assistant output or plain text (the corpus's `labels`, else --roles; role_labels.py's rules:
labels[t + 1] <= 1). Rows outside --kl-rows are not scored (NaN in OUT.tokens.pt).

OUT.json: kld.py's keys with the same meanings, over the scored rows, and `long`: the headline = generated forced rows
(mean, bootstrap CI over blocks of --block sequences, top-1 agreement, NLL, percentiles), generated prompt rows, all
generated rows, forced rows, prompt rows, all rows; by kind; by absolute position; forced rows by distance from the
prompt's end; with --cuts (the same arm's 2,048-token-cut JSON, kld.py with --roles) the headline against the cuts'
generated-row KLD on the same kinds, the cuts' per-kind values weighted by this set's generated forced rows per kind
(proposed gate: headline < 0.08 and <= 1.3 x the cuts).
"""
import json
import os
import sys
import time

import numpy as np
import torch

DIST = ((0, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384))
BOOT = 2000                     # bootstrap resamples (kld.py's)
# bootstrap seeds per group: kld.py's 0 (all rows) and 1 (generated rows), then the long protocol's own
SEEDS = {"all": 0, "generated": 1, "generated_forced": 2, "generated_prompt": 3, "forced": 4, "prompt": 5}


def slice_rows(spec, T):
    """'A:B' (Python slice bounds, negatives from the end; either side may be empty) -> (start, end) within [0, T]."""
    if not spec:
        return 0, T
    a, _, b = spec.partition(":")
    start, end, step = slice(int(a) if a else None, int(b) if b else None).indices(T)
    if step != 1 or end <= start:
        raise SystemExit(f"--kl-rows {spec!r}: no rows for T = {T}")
    return start, end


def pos_buckets(T):
    """kld.py's three buckets, then doubling ranges (the runner's summary buckets for long sequences), clipped to the
    T - 1 scored rows; kld.py's keys for T = 2,048."""
    out, b = [(0, 256), (256, 1024), (1024, 2048)], 2048
    while b < T:
        out.append((b, 2 * b))
        b *= 2
    return [(lo, min(hi, T - 1)) for lo, hi in out if lo < T - 1]


def reference_parts(spec, N, T):
    """The exact-KL reference files as the server reads them, when this container sees the same paths: [(path,
    sequences, row_start, row_end)]; None if they are not visible here (the server checks them then)."""
    parts = []
    for path in spec.split(","):
        if not os.path.exists(path):
            return None
        mm = np.load(path, mmap_mode="r")
        r0, r1 = 0, mm.shape[1]
        if os.path.exists(path + ".json"):
            d = json.load(open(path + ".json"))
            r0, r1 = int(d.get("row_start", 0)), int(d.get("row_end", mm.shape[1]))
            if d.get("seq_len") not in (None, T):
                raise SystemExit(f"{path}: made from {d['seq_len']}-token sequences, the corpus has {T}")
            if r1 - r0 != mm.shape[1]:
                raise SystemExit(f"{path}.json: rows [{r0}, {r1}), the file holds {mm.shape[1]} per sequence")
        elif mm.shape[1] != T:
            raise SystemExit(f"{path}: {mm.shape[1]} rows per sequence and no {path}.json saying which")
        parts.append((path, int(mm.shape[0]), r0, r1))
    if sum(p[1] for p in parts) != N:
        raise SystemExit(f"{spec}: {sum(p[1] for p in parts)} sequences, the corpus has {N}")
    return parts


def role_labels(a, c, n_use, T):
    """Per sequence and row t: the role of token t + 1 ([n_use, T - 1] int64) and the role names: the corpus's own
    `labels` [N, T] (corpus/k31-l32.pt), else --roles (k31-bm-eval-roles.pt: windows of whole cuts)."""
    N = c["tokens"].shape[0]
    if c.get("labels") is not None:
        lab = torch.as_tensor(c["labels"])
        if tuple(lab.shape) != (N, T):
            raise SystemExit(f"corpus labels {tuple(lab.shape)} do not match its tokens {(N, T)}")
        names = list(c.get("label_names") or c.get("names") or [])
        return lab[:n_use, 1:T].long().numpy(), names
    if a.roles and os.path.exists(a.roles):
        r = torch.load(a.roles, weights_only=False)
        lab = r["labels"]
        if lab.dim() != 2 or lab.shape[1] % T or lab.numel() != N * T:     # e.g. 2,048-token cuts' windows for 32K
            raise SystemExit(f"--roles {a.roles}: labels {tuple(lab.shape)} are not whole {T}-token sequences of this "
                             f"corpus ({N} x {T}); give a corpus with its own labels or a matching roles file")
        return lab.reshape(-1, T)[:n_use, 1:T].long().numpy(), list(r["names"])
    return None, None


def parse(pl, ids, T):
    """The response table (T entries; entry t + 1 = the engine's distribution after token t) per row t in [0, T - 1):
    returned token ids and log-probs in response order ([T - 1, W], W = the most any row returned; -1 / -inf padding),
    the engine's top-1 token (the first maximum, as kld.py's max over the dict) and the NLL of token t + 1 (NaN when
    it was not returned)."""
    W = max(len(pl[t + 1]) for t in range(T - 1))
    tid = np.full((T - 1, W), -1, dtype=np.int64)
    tlp = np.full((T - 1, W), -np.inf, dtype=np.float64)
    nll = np.full(T - 1, np.nan)
    for t in range(T - 1):
        ent = pl[t + 1]
        nxt = ids[t + 1]
        for j, (tok, v) in enumerate(ent.items()):
            tok = int(tok)
            lp = float(v["logprob"])
            tid[t, j] = tok
            tlp[t, j] = lp
            if tok == nxt:
                nll[t] = -lp
    top = tid[np.arange(T - 1), tlp.argmax(1)]
    return tid, tlp, top, nll


def estimate(tid, tlp, rid, rlp, block=2048):
    """kld.py's top-k estimate per row: KL(ref || engine) over the reference's top-k tokens the engine returned plus
    one bucket for the rest of both distributions' mass (a lower bound on the full KL); cover = the reference mass on
    those tokens. rid / rlp: the reference's top-k [rows, K] (float64 log-probs)."""
    R = tid.shape[0]
    kl, cover = np.zeros(R), np.zeros(R)
    for a in range(0, R, block):
        b = min(R, a + block)
        m = rid[a:b, :, None] == tid[a:b, None, :]                         # [rows, K, W]
        has = m.any(2)
        e_at = np.where(m, tlp[a:b, None, :], -np.inf).max(2)              # the engine's log-prob of each ref token
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pr = np.where(has, np.exp(rlp[a:b]), 0.0)
            pe = np.where(has, np.exp(e_at), 0.0)
            rest_r = np.maximum(1.0 - pr.sum(1), 1e-300)
            rest_e = np.maximum(1.0 - pe.sum(1), 1e-300)
            v = (pr * (np.log(np.maximum(pr, 1e-300)) - np.log(np.maximum(pe, 1e-300)))).sum(1)
        kl[a:b] = np.maximum(v + rest_r * (np.log(rest_r) - np.log(rest_e)), 0.0)
        cover[a:b] = pr.sum(1)
    return kl, cover


def _q(x, p):
    return float(torch.quantile(torch.from_numpy(np.ascontiguousarray(x)), p))


def group(per, mask, blk, nb, seed, top1=None, nll=None, sample_seed=7):
    """Mean of the scored rows under `mask`, its bootstrap CI over blocks of sequences (ratio of row sums, kld.py's
    generated-row form), rows, median / p90 / p99 / p999 (a 2M-row sample above that), top-1 agreement and NLL."""
    m = mask & ~np.isnan(per)
    rows = int(m.sum())
    if rows == 0:
        return None
    x = per[m]
    s = np.where(m, per, 0.0).sum(1)
    c = m.sum(1).astype(np.float64)
    bs, bc = np.bincount(blk, weights=s, minlength=nb), np.bincount(blk, weights=c, minlength=nb)
    g = torch.Generator().manual_seed(seed)
    boots = []
    for _ in range(BOOT):
        idx = torch.randint(0, nb, (nb,), generator=g).numpy()
        boots.append(float(bs[idx].sum() / max(bc[idx].sum(), 1.0)))
    boots = np.array(boots)
    if x.size > 2_000_000:
        gs = torch.Generator().manual_seed(sample_seed)
        x = x[torch.randperm(x.size, generator=gs)[:2_000_000].numpy()]
    out = {"kld": float(per[m].mean()), "ci95": [_q(boots, 0.025), _q(boots, 0.975)], "rows": rows,
           "median": _q(x, 0.5), "p90": _q(x, 0.9), "p99": _q(x, 0.99), "p999": _q(x, 0.999)}
    if top1 is not None:
        out["top1"] = float(top1[m].mean())
    if nll is not None:
        v = nll[m]
        v = v[~np.isnan(v)]
        out["nll"] = float(v.mean()) if v.size else None
    return out


def _mean(per, mask):
    m = mask & ~np.isnan(per)
    return (float(per[m].mean()), int(m.sum())) if m.any() else (None, 0)


def run(a, c, ask_decode):
    if not isinstance(c, dict):
        raise SystemExit("--long needs a corpus dict (tokens, kinds, labels or --roles, tail for --prompt tail)")
    if a.mode != "decode":
        raise SystemExit("--long scores forced decoding: add --mode decode (prefill everything but the last row: "
                         "--prompt T-1)")
    if a.subset:
        raise SystemExit("--subset selects cuts of the 512-cut set; not with --long")
    tokens = c["tokens"]
    N, T = tokens.shape
    n_use = a.windows or N
    kinds = list(c["kinds"][:n_use]) if c.get("kinds") is not None else None
    if a.prompt == "tail":
        if c.get("tail") is None:
            raise SystemExit("--prompt tail: the corpus has no `tail` (forced rows per sequence)")
        P = [T - int(v) for v in torch.as_tensor(c["tail"])[:n_use].tolist()]
    else:
        P = [int(a.prompt)] * n_use
    if min(P) < 1 or max(P) > T - 1:
        raise SystemExit(f"prompt lengths {min(P)}..{max(P)}: each sequence needs a prompt and a forced row (T = {T})")
    spec = a.kl_hidden or a.kl_ref
    hidden = bool(a.kl_hidden)
    prompt_rows = a.prompt_rows if spec else "estimate"
    prompt_k = a.prompt_logprobs if a.prompt_logprobs is not None else (1 if prompt_rows == "exact" else a.k)
    if prompt_k < 1:
        raise SystemExit("--prompt-logprobs >= 1: top-1 agreement needs the engine's own top-1 on prompt rows")
    if prompt_rows == "estimate" and (prompt_k < a.k or a.ref in ("", "-", "none")):
        raise SystemExit("--prompt-rows estimate scores prompt rows with the top-k estimate: it needs REF_KLD.pt and "
                         "--prompt-logprobs >= --k")
    decode_k = (a.decode_logprobs or a.k) if spec else a.k
    lo, hi = slice_rows(a.kl_rows, T)
    if spec:
        parts = reference_parts(spec, N, T)
        if parts is not None:
            r0, r1 = max(p[2] for p in parts), min(p[3] for p in parts)
            if r0 > lo or r1 < min(hi, T - 1):
                nlo, nhi = max(lo, r0), min(hi, r1)
                if nhi <= nlo:
                    raise SystemExit(f"the reference holds rows [{r0}, {r1}); --kl-rows asks [{lo}, {hi})")
                sys.stderr.write(f"--kl-rows [{lo}, {hi}) clipped to the reference's rows: [{nlo}, {nhi})\n")
                lo, hi = nlo, nhi
        else:
            sys.stderr.write(f"reference {spec}: not visible here; the server reads it\n")
    lab, names = role_labels(a, c, n_use, T)
    ref_lp = ref_id = None
    if not spec and a.ref in ("", "-", "none"):
        raise SystemExit("--long needs an exact reference (--kl-hidden / --kl-ref) or REF_KLD.pt for the estimate")
    if a.ref not in ("", "-", "none"):
        refs = [torch.load(f, weights_only=False) for f in a.ref.split(",")]     # halves of one reference, in order
        ref_lp = torch.cat([r["ref_lp"] for r in refs])
        ref_id = torch.cat([r["ref_id"] for r in refs])
        del refs
        if tuple(ref_lp.shape[:2]) != (N, T):
            raise SystemExit(f"reference {tuple(ref_lp.shape)} does not match the corpus {(N, T)}")
    K = a.k
    rows = np.arange(T - 1)
    per = np.full((n_use, T - 1), np.nan)                 # the per-row KL used for every number below (NaN = unscored)
    est = np.full((n_use, T - 1), np.nan)                 # kld.py's top-k estimate (rows that returned a full top-k)
    cov = np.full((n_use, T - 1), np.nan)
    pex = np.full((n_use, T - 1), np.nan)                 # exact KL of prompt rows when --prompt-rows estimate
    top1 = np.zeros((n_use, T - 1), dtype=bool)
    nll = np.full((n_use, T - 1), np.nan)
    dec = np.zeros((n_use, T - 1), dtype=bool)
    # the engine's own top-k per position (rescore.py), as wide as a row can come back: the asked width + the token
    # itself (2 with prompt log-probs 1 and KLD_FAST's decode log-probs 1: 17 MB for L32 instead of 537 MB)
    KS = min(K, max(prompt_k, decode_k) + 1)
    eng_ids = torch.full((n_use, T, KS), -1, dtype=torch.int32)
    eng_lp = torch.full((n_use, T, KS), float("-inf"))
    tag = os.path.basename(a.out).replace(".json", "")
    t0 = time.time()

    def fetch(n):
        ids = tokens[n].tolist()
        klx = None
        if spec:
            klx = {("k31_kl_hidden" if hidden else "k31_kl_ref"): spec, "k31_kl_index": n,
                   "k31_kl_out": f"{a.kl_server_dir}/{tag}-{n:03d}.npy", "k31_kl_rows": [lo, hi]}
            if a.prefetch:
                klx["k31_kl_next"] = n + 1 if n + 1 < n_use else -1
        t_post = time.time()
        pl = ask_decode(a.url, ids, P[n], K, kl=klx, lp_k=a.decode_logprobs if klx else 0, prompt_k=prompt_k)
        exact = None
        if klx:
            f = f"{a.kl_dir}/{tag}-{n:03d}.npy"
            if os.path.getmtime(f) < t_post - 2:
                raise RuntimeError(f"sequence {n}: {f} is older than the request (the server did not write it)")
            exact = np.load(f).astype(np.float64)
        return ids, pl, exact, time.time() - t_post

    pool = fut = None
    if a.pipeline:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        fut = pool.submit(fetch, 0)
    for n in range(n_use):
        if pool is not None:
            ids, pl, exact, secs = fut.result()
            if n + 1 < n_use:
                fut = pool.submit(fetch, n + 1)
        else:
            ids, pl, exact, secs = fetch(n)
        if len(pl) != T:
            raise SystemExit(f"sequence {n}: {len(pl)} positions, expected {T}")
        tid, tlp, top, nll[n] = parse(pl, ids, T)
        order = np.argsort(-tlp, axis=1, kind="stable")[:, :KS]            # kld.py's kept top-k, best first
        w = min(KS, tid.shape[1])
        eng_ids[n, :T - 1, :w] = torch.from_numpy(np.take_along_axis(tid, order, 1)[:, :w].astype(np.int32))
        eng_lp[n, :T - 1, :w] = torch.from_numpy(np.take_along_axis(tlp, order, 1)[:, :w].astype(np.float32))
        dec[n] = rows >= P[n] - 1
        if ref_id is not None:
            rid = ref_id[n, :T - 1].long().numpy()
            top1[n] = top == rid[:, 0]
            full = np.where(dec[n], decode_k >= K, prompt_k >= K)          # rows that returned a whole top-k
            if full.any():
                e, cv = estimate(tid[full], tlp[full], rid[full], ref_lp[n, :T - 1].double().numpy()[full])
                est[n, full], cov[n, full] = e, cv
        scored = (rows >= lo) & (rows < hi)
        if exact is not None:
            ex = exact[:T - 1]
            need = scored & (dec[n] | (prompt_rows == "exact"))
            bad = need & np.isnan(ex)
            if bad.any():
                raise RuntimeError(f"sequence {n}: {int(bad.sum())} rows without an exact KL (first row "
                                   f"{int(np.argmax(bad))}; prompt {P[n]}, rows [{lo}, {hi}))")
            per[n, need] = np.maximum(ex[need], 0.0)
            if prompt_rows == "estimate":
                pr = scored & ~dec[n]
                per[n, pr] = est[n, pr]
                ok = pr & ~np.isnan(ex)
                pex[n, ok] = np.maximum(ex[ok], 0.0)
        else:
            per[n, scored] = est[n, scored]
        f_mean = _mean(per[n:n + 1], dec[n:n + 1])[0]
        p_mean = _mean(per[n:n + 1], ~dec[n:n + 1])[0]
        run_mean = _mean(per[:n + 1], np.ones((n + 1, T - 1), dtype=bool))[0]
        fmt = lambda v: "-" if v is None else f"{v:.4f}"                          # noqa: E731
        t1s = f"{top1[n].mean():.3f}" if ref_id is not None else "-"
        sys.stderr.write(f"sequence {n} ({kinds[n] if kinds else '?'}, prompt {P[n]}): KLD forced {fmt(f_mean)} "
                         f"prompt {fmt(p_mean)} top-1 {t1s} | running KLD {fmt(run_mean)} over {n + 1} sequences | "
                         f"{secs:.0f} s, {time.time() - t0:.0f} s in all\n")
    if pool is not None:
        pool.shutdown()
    res = summarize(a, per, est, cov, pex, top1 if ref_id is not None else None, nll, dec, lab, names, kinds, P, T,
                    n_use, lo, hi, spec, prompt_rows, prompt_k, decode_k)
    torch.save({"kl": torch.from_numpy(per).float(), "kl_topk": torch.from_numpy(est).float(),
                "top1": torch.from_numpy(top1), "cover": torch.from_numpy(cov), "nll": torch.from_numpy(nll),
                "prompt_exact": torch.from_numpy(pex).float() if prompt_rows == "estimate" else None,
                "exact": bool(spec), "eng_ids": eng_ids, "eng_lp": eng_lp, "sel": list(range(n_use)), "mode": a.mode,
                "prompt": P, "kinds": kinds, "long": True, "kl_rows": [lo, hi], "prompt_rows": prompt_rows},
               a.out.replace(".json", "") + ".tokens.pt")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def summarize(a, per, est, cov, pex, top1, nll, dec, lab, names, kinds, P, T, n_use, lo, hi, spec, prompt_rows,
              prompt_k, decode_k):
    """OUT.json: kld.py's keys over the scored rows, and the long protocol's groups under `long`."""
    block = a.block or 1
    blk = np.arange(n_use) // block
    nb = int(blk.max()) + 1
    allr = np.ones_like(dec)
    gen = lab <= 1 if lab is not None else None
    G = {"all": group(per, allr, blk, nb, SEEDS["all"], top1, nll),
         "forced": group(per, dec, blk, nb, SEEDS["forced"], top1, nll),
         "prompt": group(per, ~dec, blk, nb, SEEDS["prompt"], top1, nll)}
    if gen is not None:
        G["generated"] = group(per, gen, blk, nb, SEEDS["generated"], top1, nll)
        G["generated_forced"] = group(per, gen & dec, blk, nb, SEEDS["generated_forced"], top1, nll)
        G["generated_prompt"] = group(per, gen & ~dec, blk, nb, SEEDS["generated_prompt"], top1, nll)
    A = G["all"] or {}
    nl = nll[~np.isnan(per) & ~np.isnan(nll)]
    est_v = est[~np.isnan(est)]
    res = {"kld": A.get("kld"), "kld_ci95": A.get("ci95"), "median": A.get("median"), "p90": A.get("p90"),
           "p99": A.get("p99"), "p999": A.get("p999"), "top1": A.get("top1"),
           "cover": float(np.nanmean(cov)) if (~np.isnan(cov)).any() else None,
           "nll_in_topk": float(nl.mean()) if nl.size else None, "windows": n_use, "positions": A.get("rows", 0),
           "by_position": {f"{b0}-{b1}": _mean(per[:, b0:b1], allr[:, b0:b1])[0] for b0, b1 in pos_buckets(T)},
           "kld_topk_estimate": float(est_v.mean()) if est_v.size else None,
           "kld_decode_rows": (G["forced"] or {}).get("kld"),
           "note": ("long protocol (kld_long.py): exact full-vocabulary KL computed in the server on "
                    + ("every scored row" if prompt_rows == "exact" else "forced rows; prompt rows: top-k estimate")
                    if spec else "long protocol (kld_long.py) without an exact reference: top-k estimate on every row "
                    "(a lower bound on the full KL)")}
    if kinds is not None:
        res["by_domain"] = {k: _mean(per, allr & (np.array(kinds) == k)[:, None])[0] for k in sorted(set(kinds))}
    if gen is not None:
        Gg = G["generated"] or {}
        res["kld_generated"] = Gg.get("kld")
        res["kld_generated_ci95"] = Gg.get("ci95")
        res["generated_rows"] = Gg.get("rows", 0)
        res["generated_median"] = Gg.get("median")
        res["generated_top1"] = Gg.get("top1")
        res["by_role"] = {}
        for i in range(int(lab.max()) + 1):
            v, r = _mean(per, lab == i)
            if r:
                res["by_role"][names[i] if i < len(names) else str(i)] = v
        res["kld_generated_decode_rows"] = (G["generated_forced"] or {}).get("kld")
        res["kld_generated_decode_rows_ci95"] = (G["generated_forced"] or {}).get("ci95")
        if prompt_rows == "exact" or (~np.isnan(pex)).any():
            src = per if prompt_rows == "exact" else pex
            res["kld_prompt_rows_exact"] = _mean(src, ~dec)[0]
            res["kld_generated_prompt_rows_exact"], res["generated_prompt_rows"] = _mean(src, gen & ~dec)
        if kinds is not None:
            res["generated_by_domain"] = {k: _mean(per, gen & (np.array(kinds) == k)[:, None])[0]
                                          for k in sorted(set(kinds))}
    # the long protocol's own report
    L = {"protocol": {"sequences": n_use, "seq_len": T, "prompt": {"min": min(P), "max": max(P),
                                                                    "by_sequence": P},
                      "kl_rows": [lo, hi], "prompt_rows": prompt_rows, "prompt_logprobs": prompt_k,
                      "decode_logprobs": decode_k, "block": block, "reference": spec or None,
                      "top1_reference": a.ref if top1 is not None else None},
         "groups": G}
    if gen is not None:
        L["headline"] = {"what": "generated forced rows: exact KL(reference || engine), rows whose predicted token is "
                                 "assistant output or plain text, after the prompt", **(G["generated_forced"] or {})}
    dist = {}
    off = np.arange(T - 1)[None, :] - (np.array(P)[:, None] - 1)            # distance from the first forced row
    for d0, d1 in DIST:
        m = dec & (off >= d0) & (off < d1)
        if not m.any():
            continue
        v, r = _mean(per, m)
        e = {"kld": v, "rows": r}
        if gen is not None:
            e["generated"], e["generated_rows"] = _mean(per, m & gen)
        dist[f"{d0}-{d1}"] = e
    L["by_distance"] = dist
    pos = {}
    for b0, b1 in pos_buckets(T):
        m = np.zeros_like(dec)
        m[:, b0:b1] = True
        v, r = _mean(per, m)
        e = {"kld": v, "rows": r, "forced_rows": int((m & dec & ~np.isnan(per)).sum())}
        if gen is not None:
            e["generated"], e["generated_rows"] = _mean(per, m & gen)
        pos[f"{b0}-{b1}"] = e
    L["by_position"] = pos
    if kinds is not None:
        bk = {}
        ka = np.array(kinds)
        for k in sorted(set(kinds)):
            km = (ka == k)[:, None] & allr
            e = {"sequences": int((ka == k).sum())}
            e["forced"], e["forced_rows"] = _mean(per, km & dec)
            e["prompt"], e["prompt_rows"] = _mean(per, km & ~dec)
            if gen is not None:
                e["generated_forced"], e["generated_forced_rows"] = _mean(per, km & gen & dec)
                e["generated_prompt"], e["generated_prompt_rows"] = _mean(per, km & gen & ~dec)
                e["generated"], e["generated_rows"] = _mean(per, km & gen)
            bk[k] = e
        L["by_kind"] = bk
    if a.cuts and gen is not None and kinds is not None:
        L["cuts"] = against_cuts(a.cuts, per, gen & dec, kinds, (G["generated_forced"] or {}).get("kld"))
    res["long"] = L
    return res


def against_cuts(path, per, mask, kinds, headline):
    """The headline against the same arm's 2,048-token cuts (kld.py's generated_by_domain) on the kinds both have,
    the cuts' per-kind values weighted by this set's generated forced rows per kind."""
    cj = json.load(open(path))
    cg = cj.get("generated_by_domain") or {}
    ka = np.array(kinds)
    w = {k: int((mask & (ka == k)[:, None] & ~np.isnan(per)).sum()) for k in sorted(set(kinds))}
    shared = [k for k, v in w.items() if v > 0 and cg.get(k) is not None]
    out = {"file": path, "kinds": shared, "missing": [k for k, v in w.items() if v > 0 and k not in shared],
           "weights": {k: w[k] for k in shared}, "cuts_kld_generated": cj.get("kld_generated")}
    if not shared:
        return out
    base = sum(w[k] * cg[k] for k in shared) / sum(w[k] for k in shared)
    mine, _ = _mean(per, mask & np.isin(ka, shared)[:, None])
    out.update({"cuts_generated_same_kinds": base, "headline_same_kinds": mine,
                "ratio": mine / base if base else None,
                "proposed_gate": {"headline_below_0.08": headline is not None and headline < 0.08,
                                  "ratio_at_most_1.3": base > 0 and mine / base <= 1.3}})
    return out
