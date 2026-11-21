"""On-policy KLD (kld.py on a corpus with per-sequence lengths, i.e. a bucket from tools/onpolicy/build_sets.py).

    kld.py URL BUCKET.pt REF_KLD.pt OUT.json --mode decode --kl-hidden HIDDEN.npy [--decode-logprobs 1 --pipeline]
           [--no-rev] [--conc N]

The fixed-text protocols feed the engine text it did not write. Here each sequence is a prompt plus what the served
engine itself sampled for it (tools/onpolicy/capture.py), and the reference is the official model run over exactly
that text (the reference runner's final hidden states, HIDDEN.npy [N, T, H], rows past a sequence's length are
padding). The engine re-reads its own answer as it serves: tokens [0, forced_from) through the prompt path, every later
token forced one decode step at a time (k31_engine/force.py, row-range path; one sequence at a time, or --conc N in
flight as a loaded server reads them), and the server computes the exact
full-vocabulary KL(reference || engine) of every row, and with k31_kl_rev the reverse KL(engine || reference).

Rows: row t is the distribution after token t (it predicts token t + 1). For a sequence of `len` tokens with `prompt`
prompt tokens, the on-policy rows are t in [prompt - 1, len - 2]: one per generated token, each scored with the
engine's own earlier tokens as context. Rows before that belong to the prompt (chat template and user text, scored
for reference; a prompt row the server took from its prefix cache has no value). Rows t >= forced_from - 1 are decode
rows; for the loop set forced_from > prompt, so the first on-policy rows are read through the prompt path.

OUT.json: `onpolicy` (all on-policy rows: mean, bootstrap CI over prompts (ids up to '#'), percentiles, top-1
agreement, the engine's NLL of its own tokens), `onpolicy_decode` / `onpolicy_prompt_path` (by the path that read the
row), `onpolicy_generated` (rows whose predicted token is labelled assistant or text), `reverse` (KL(engine ||
reference), same groups), `by_kind`, `by_index` (by generated-token index), `prompt_rows`. OUT.tokens.pt keeps the
per-row values ("kl", "kl_rev", "top1", "nll", masks, lengths, ids) for tools/onpolicy/analyze.py.
"""
import json
import os
import sys
import time

import numpy as np
import torch

import kld_long

INDEX = ((0, 64), (64, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768))
SEEDS = {"onpolicy": 11, "onpolicy_decode": 12, "onpolicy_prompt_path": 13, "onpolicy_generated": 14,
         "prompt_rows": 15, "all": 0}


def run(a, c, ask_decode):
    if a.mode != "decode":
        raise SystemExit("the on-policy protocol scores forced decoding: add --mode decode")
    if not a.kl_hidden and not a.kl_ref:
        raise SystemExit("the on-policy protocol needs the exact reference (--kl-hidden)")
    tokens = c["tokens"]
    N, T = tokens.shape
    n_use = a.windows or N
    lens = [int(v) for v in c["lens"]][:n_use]
    G = [int(v) for v in c["prompt"]][:n_use]                  # index of the first generated token
    F = [int(v) for v in (c.get("forced_from") or c["prompt"])][:n_use]   # index of the first forced token
    kinds = list(c["kinds"][:n_use])
    ids_req = list(c.get("ids") or [str(i) for i in range(N)])[:n_use]
    for n in range(n_use):
        if not (1 <= G[n] <= F[n] <= lens[n] - 1 <= T - 1):
            raise SystemExit(f"sequence {n}: prompt {G[n]}, forced_from {F[n]}, len {lens[n]}, T {T}")
    spec, hidden = (a.kl_hidden or a.kl_ref), bool(a.kl_hidden)
    kld_long.reference_parts(spec, N, T)
    ref_id = ref_nll = None
    if a.ref not in ("", "-", "none"):
        refs = [torch.load(f, weights_only=False) for f in a.ref.split(",")]
        ref_id = torch.cat([r["ref_id"][..., 0] for r in refs]).long().numpy()           # [N, T] the reference's top-1
        if all(r.get("nll") is not None for r in refs):
            ref_nll = torch.cat([r["nll"][0] for r in refs]).double().numpy()            # [N, T - 1]
        del refs
        if ref_id.shape != (N, T):
            raise SystemExit(f"reference {ref_id.shape} does not match the bucket {(N, T)}")
    rev_on = not a.no_rev
    rows = np.arange(T - 1)
    per = np.full((n_use, T - 1), np.nan)
    rev = np.full((n_use, T - 1), np.nan)
    nll = np.full((n_use, T - 1), np.nan)
    top1 = np.zeros((n_use, T - 1), dtype=bool)
    eng_top = np.full((n_use, T - 1), -1, dtype=np.int64)
    valid = np.zeros((n_use, T - 1), dtype=bool)               # rows of the real sequence
    samp = np.zeros((n_use, T - 1), dtype=bool)                # on-policy rows
    dec = np.zeros((n_use, T - 1), dtype=bool)                 # read by forced decode steps
    tag = os.path.basename(a.out).replace(".json", "")
    t0 = time.time()

    def fetch(n):
        ids = tokens[n, :lens[n]].tolist()
        klx = {("k31_kl_hidden" if hidden else "k31_kl_ref"): spec, "k31_kl_index": n,
               "k31_kl_out": f"{a.kl_server_dir}/{tag}-{n:03d}.npy", "k31_kl_rows": [0, lens[n]]}
        if rev_on:
            klx["k31_kl_rev"] = 1
        t_post = time.time()
        pl = ask_decode(a.url, ids, F[n], a.k, kl=klx, lp_k=a.decode_logprobs, prompt_k=1)
        f = f"{a.kl_dir}/{tag}-{n:03d}.npy"
        if os.path.getmtime(f) < t_post - 2:
            raise RuntimeError(f"sequence {n}: {f} is older than the request (the server did not write it)")
        exact = np.load(f).astype(np.float64)
        rv = None
        if rev_on:
            fr = f[:-4] + ".rev.npy"
            if not os.path.exists(fr) or os.path.getmtime(fr) < t_post - 2:
                raise RuntimeError(f"sequence {n}: no reverse KL file {fr} (the server's force.py has no k31_kl_rev?)")
            rv = np.load(fr).astype(np.float64)
        return ids, pl, exact, rv, time.time() - t_post

    pool = fut = futs = None
    conc = getattr(a, "conc", 1) or 1
    if conc > 1:                                # `conc` sequences in flight at once (the server shares its tiers)
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(conc)
        futs = [pool.submit(fetch, n) for n in range(n_use)]
    elif a.pipeline:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        fut = pool.submit(fetch, 0)
    for n in range(n_use):
        if futs is not None:
            ids, pl, exact, rv, secs = futs[n].result()
        elif pool is not None:
            ids, pl, exact, rv, secs = fut.result()
            if n + 1 < n_use:
                fut = pool.submit(fetch, n + 1)
        else:
            ids, pl, exact, rv, secs = fetch(n)
        Ln = lens[n]
        if len(pl) != Ln or exact.shape[0] != Ln:
            raise SystemExit(f"sequence {n}: {len(pl)} positions and {exact.shape[0]} KL rows, expected {Ln}")
        _, _, top, nl = kld_long.parse(pl, ids, Ln)
        m = rows < Ln - 1
        valid[n] = m
        samp[n] = m & (rows >= G[n] - 1)
        dec[n] = m & (rows >= F[n] - 1)
        ex = exact[:Ln - 1]
        bad = np.isnan(ex) & dec[n, :Ln - 1]
        if bad.any():
            raise RuntimeError(f"sequence {n}: {int(bad.sum())} decode rows without an exact KL (first {int(np.argmax(bad))})")
        per[n, :Ln - 1] = np.where(np.isnan(ex), np.nan, np.maximum(ex, 0.0))
        if rv is not None:
            r = rv[:Ln - 1]
            rev[n, :Ln - 1] = np.where(np.isnan(r), np.nan, np.maximum(r, 0.0))
        nll[n, :Ln - 1] = nl
        eng_top[n, :Ln - 1] = top
        if ref_id is not None:
            top1[n, :Ln - 1] = top == ref_id[n, :Ln - 1]
        s = samp[n]
        miss = int((samp[n] & np.isnan(per[n])).sum())
        sys.stderr.write(f"sequence {n} ({ids_req[n]}, prompt {G[n]}, forced from {F[n]}, len {Ln}): on-policy KLD "
                         f"{np.nanmean(per[n][s]):.4f} reverse {np.nanmean(rev[n][s]) if rev_on else float('nan'):.4f} "
                         f"over {int(s.sum())} rows ({miss} unscored) | running "
                         f"{np.nanmean(per[:n + 1][samp[:n + 1]]):.4f} over {n + 1} sequences | {secs:.0f} s, "
                         f"{time.time() - t0:.0f} s in all\n")
    if pool is not None:
        pool.shutdown()
    lab = torch.as_tensor(c["labels"])[:n_use, 1:T].long().numpy() if c.get("labels") is not None else None
    res = summarize(per, rev if rev_on else None, top1 if ref_id is not None else None, nll,
                    ref_nll[:n_use] if ref_nll is not None else None, valid, samp, dec, lab, kinds, ids_req, G, T)
    res["protocol"] = {"sequences": n_use, "T": T, "in_flight": conc, "reference": spec, "top1_reference": a.ref, "bucket": a.corpus,
                       "tokens": int(sum(lens)), "what": (c.get("protocol") or "")}
    torch.save({"kl": torch.from_numpy(per).float(), "kl_rev": torch.from_numpy(rev).float() if rev_on else None,
                "top1": torch.from_numpy(top1), "nll": torch.from_numpy(nll).float(),
                "eng_top": torch.from_numpy(eng_top).int(), "valid": torch.from_numpy(valid),
                "onpolicy": torch.from_numpy(samp), "decode": torch.from_numpy(dec), "lens": lens, "prompt": G,
                "forced_from": F, "ids": ids_req, "kinds": kinds, "exact": True, "mode": "onpolicy"},
               a.out.replace(".json", "") + ".tokens.pt")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def summarize(per, rev, top1, nll, ref_nll, valid, samp, dec, lab, kinds, ids_req, G, T):
    n_use = per.shape[0]
    keys = sorted({i.split("#")[0] for i in ids_req})
    blk = np.array([keys.index(i.split("#")[0]) for i in ids_req])       # bootstrap unit: the prompt
    nb = len(keys)
    gen = (lab <= 1) if lab is not None else np.ones_like(samp)
    masks = {"onpolicy": samp, "onpolicy_decode": samp & dec, "onpolicy_prompt_path": samp & ~dec,
             "onpolicy_generated": samp & gen, "prompt_rows": valid & ~samp, "all": valid}
    res = {}
    for k, m in masks.items():
        res[k] = kld_long.group(per, m, blk, nb, SEEDS[k], top1, nll)
        if res[k] is not None:
            res[k]["unscored"] = int((m & np.isnan(per)).sum())
    if rev is not None:
        res["reverse"] = {k: kld_long.group(rev, m, blk, nb, SEEDS[k] + 100) for k, m in masks.items()}
    if ref_nll is not None:                 # the official model's NLL of the engine's own tokens, beside the engine's
        m = samp & ~np.isnan(nll) & ~np.isnan(ref_nll[:, :T - 1])
        if m.any():
            res["nll_of_sampled_tokens"] = {"engine": float(nll[m].mean()), "reference": float(ref_nll[:, :T - 1][m].mean()),
                                            "rows": int(m.sum())}
    ka = np.array(kinds)
    res["by_kind"] = {}
    for k in sorted(set(kinds)):
        m = samp & (ka == k)[:, None]
        e = kld_long.group(per, m, blk, nb, 21, top1, nll) or {}
        if rev is not None:
            e["reverse"] = kld_long._mean(rev, m)[0]
        e["sequences"] = int((ka == k).sum())
        res["by_kind"][k] = e
    off = np.arange(T - 1)[None, :] - (np.array(G)[:, None] - 1)            # generated-token index of the row
    res["by_index"] = {}
    for a0, a1 in INDEX:
        m = samp & (off >= a0) & (off < a1)
        v, r = kld_long._mean(per, m)
        if r:
            res["by_index"][f"{a0}-{a1}"] = {"kld": v, "rows": r,
                                             "reverse": kld_long._mean(rev, m)[0] if rev is not None else None}
    o = res["onpolicy"] or {}
    res["kld"], res["kld_ci95"] = o.get("kld"), o.get("ci95")
    return res
