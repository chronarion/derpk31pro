"""KLD of a served engine against the published model, on the runner's corpus windows.

    kld.py URL CORPUS.pt REF_KLD.pt OUT.json [--windows N] [--k 64]

CORPUS.pt: the runner's token windows (`tokens` [N, T] int, or a bare tensor). REF_KLD.pt: a runner pass's kld.pt
whose universe 0 is the published model; it holds that model's top-64 log-probabilities per position (`ref_lp`,
`ref_id` [N, T, 64]). Each window goes to the server as one prompt of token ids with `prompt_logprobs = k`
(server flag --max-logprobs >= k), which returns, per position, the engine's own top-k log-probabilities (exact: vLLM
takes them from the full-vocabulary log-softmax).

Per position t (the distribution after token t, predicting t + 1; the last position is skipped):
  KL(ref || engine), coarse-grained over the tokens in both top-k lists plus one bucket for everything else. By the
  data-processing inequality this is a lower bound on the full-vocabulary KL, tight when the shared tokens hold most
  of both distributions' mass (`cover` reports the reference mass they hold; the runner's own universe KLDs are full
  vocabulary). Also: top-1 agreement (exact) and the engine's NLL of the next token where it is in its top-k.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import torch


def ask(url, ids, k, timeout=900):
    body = {"model": "derpk31pro", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
            "logprobs": k}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    pl = ch.get("prompt_logprobs") or d.get("prompt_logprobs")
    if pl is None:
        raise RuntimeError(f"no prompt_logprobs in the response: {list(d)} / {list(ch)}")
    return pl


def ask_decode(url, ids, P, k, timeout=7200, kl=None, lp_k=0):
    """Forced decoding (k31_engine/force.py on the server): the first P tokens as the prompt, the rest generated one
    decode step at a time with each step's token forced to the reference; returns the per-position table `ask` gives
    (index t = the distribution after token t - 1), from prompt_logprobs (t < P) and the raw generated log-probs."""
    body = {"model": "derpk31pro", "prompt": ids[:P], "max_tokens": len(ids) - P, "temperature": 0, "logprobs": lp_k or k,
            "prompt_logprobs": k, "return_tokens_as_token_ids": True, "ignore_eos": True,
            "vllm_xargs": {"k31_force": ids[P:], "k31_start": P, **(kl or {})}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    pl = list(ch.get("prompt_logprobs") or d.get("prompt_logprobs"))
    lp = ch["logprobs"]
    toks = [int(t.split(":")[1]) for t in lp["tokens"]]
    if toks != ids[P:P + len(toks)] or len(toks) != len(ids) - P:
        raise RuntimeError(f"forced decoding did not follow the reference ({len(toks)} tokens; first mismatch at "
                           f"{next((i for i, (a, b) in enumerate(zip(toks, ids[P:])) if a != b), None)})")
    for dct in lp["top_logprobs"]:
        pl.append({int(t.split(":")[1]): {"logprob": v} for t, v in dct.items()})
    return pl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url"); p.add_argument("corpus"); p.add_argument("ref"); p.add_argument("out")
    p.add_argument("--windows", type=int, default=0, help="first N windows only (0 = all)")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--mode", default="prefill", choices=("prefill", "decode"),
                   help="prefill: each cut teacher-forced in one prompt (brandonmusic's protocol); decode: a short "
                        "prompt (--prompt tokens), the rest forced one decode step at a time")
    p.add_argument("--prompt", type=int, default=64)
    p.add_argument("--decode-logprobs", type=int, default=0, help="decode mode: top-k log-probs per generated row "
                   "(0 = --k); with --kl-ref the generated rows' KL is the server's exact one, so 1 keeps top-1 and NLL "
                   "and drops only the top-k estimate and `cover` there (the 64-wide response was most "
                   "of the ~4 s between cuts)")
    p.add_argument("--pipeline", action="store_true", help="post cut n + 1 when cut n's response arrives and score n "
                   "while it runs (the server never waits for the harness)")
    p.add_argument("--prefetch", action="store_true", help="with --kl-ref: name the next cut's block (k31_kl_next) so "
                   "the server reads it into pinned memory while the current cut decodes")
    p.add_argument("--kl-ref", default="", help="decode mode: the reference's full log-probabilities as the SERVER sees "
                   "them (runner --dump-logits over the same cuts, in the harness's order): exact full-vocabulary KL "
                   "computed in the server (k31_engine/force.py) for every decode position")
    p.add_argument("--roles", default="/s/corpus/k31-bm-eval-roles.pt",
                   help="role labels of the 512 cuts (runner/tools/role_labels.py: 0 assistant, 1 plain text, 2 system, "
                   "3 user, 4 tool, 5 header, 6 before any header); the headline counts generated rows = rows whose "
                   "predicted token is assistant output or plain text; '' = all rows only")
    p.add_argument("--kl-hidden", default="", help="decode mode: the reference's final hidden states as the SERVER sees "
                   "them (runner --dump-hidden over the same cuts, in the harness's order): exact KL with the "
                   "reference logits rebuilt in the server from its LM head (15 GB for all 512 cuts)")
    p.add_argument("--kl-dir", default="/s/kl/out", help="where the harness reads the server's exact-KL files")
    p.add_argument("--kl-server-dir", default="/kl/out", help="the same directory as the server sees it")
    p.add_argument("--subset", default="", help="rot128: one cut per window of the bm-eval set, the cut offset "
                   "rotating with the window (cut 4w + w % 4): a quick screen with every window and offset; rot32 (NB: "
                   "w % 4 is 0 for every 4th window, so these are the first cuts only): "
                   "every 4th of those (all domains)")
    p.add_argument("--dump-ref", action="store_true",
                   help="write this engine's own top-k as a reference file at REF_KLD.pt instead of scoring (to compare "
                        "two engines, or to check the harness: an engine scored against itself gives KLD 0)")
    a = p.parse_args()
    c = torch.load(a.corpus, weights_only=False)
    tokens = c["tokens"] if isinstance(c, dict) else c
    if a.dump_ref:
        N, T = tokens.shape
        n_use = a.windows or N
        ref_lp = torch.full((N, T, a.k), -1e4, dtype=torch.float32)
        ref_id = torch.zeros((N, T, a.k), dtype=torch.int32)
        for n in range(n_use):
            pl = ask(a.url, tokens[n].tolist(), a.k)
            for t in range(T - 1):
                top = sorted(((int(tok), v["logprob"]) for tok, v in pl[t + 1].items()), key=lambda x: -x[1])[:a.k]
                ref_id[n, t, :len(top)] = torch.tensor([x[0] for x in top], dtype=torch.int32)
                ref_lp[n, t, :len(top)] = torch.tensor([x[1] for x in top], dtype=torch.float32)
            sys.stderr.write(f"window {n}: reference dumped\n")
        torch.save({"ref_lp": ref_lp, "ref_id": ref_id, "source": a.url, "windows": n_use}, a.ref)
        return
    refs = [torch.load(f, weights_only=False) for f in a.ref.split(",")]     # halves of one reference, in order
    ref_lp = torch.cat([r["ref_lp"] for r in refs]).double()
    ref_id = torch.cat([r["ref_id"] for r in refs]).long()
    kinds = c.get("kinds") if isinstance(c, dict) else None
    N, T = tokens.shape
    if ref_lp.shape[:2] != (N, T):
        raise SystemExit(f"reference {tuple(ref_lp.shape)} does not match the corpus {tuple(tokens.shape)}")
    sel_list = list(range(N))
    if a.subset in ("rot128", "rot32"):
        step = 1 if a.subset == "rot128" else 4               # rot32: every 4th window, all domains, all offsets
        sel = torch.tensor([4 * w + w % 4 for w in range(0, N // 4, step)])
        tokens, ref_lp, ref_id = tokens[sel], ref_lp[sel], ref_id[sel]
        sel_list = sel.tolist()
        if kinds is not None:
            kinds = [kinds[i] for i in sel.tolist()]
        N = tokens.shape[0]
    n_use = a.windows or N
    prompt_exact_all = []
    kl_all, top1_all, cover_all, nll_all, by_pos, coarse_all = [], [], [], [], {}, []
    eng_ids = torch.full((n_use, T, a.k), -1, dtype=torch.int32)        # the engine's own top-k per position, kept so
    eng_lp = torch.full((n_use, T, a.k), float("-inf"))                 # it can be re-scored offline (rescore.py)
    t0 = time.time()

    def fetch(n):
        ids = tokens[n].tolist()
        exact = None
        if a.mode == "prefill":
            pl = ask(a.url, ids, a.k)
        else:
            tag = os.path.basename(a.out).replace(".json", "")
            klx = ({"k31_kl_hidden": a.kl_hidden} if a.kl_hidden else {"k31_kl_ref": a.kl_ref} if a.kl_ref else None)
            if klx is not None:
                klx.update({"k31_kl_index": n, "k31_kl_out": f"{a.kl_server_dir}/{tag}-{n:03d}.npy"})
                if a.prefetch:
                    klx["k31_kl_next"] = n + 1 if n + 1 < n_use else -1
            pl = ask_decode(a.url, ids, a.prompt, a.k, kl=klx, lp_k=a.decode_logprobs if klx else 0)
            if klx:
                exact = torch.from_numpy(np.load(f"{a.kl_dir}/{tag}-{n:03d}.npy")).double()
        return ids, pl, exact

    pool = fut = None
    if a.pipeline:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        fut = pool.submit(fetch, 0)
    for n in range(n_use):
        if pool is not None:
            ids, pl, exact = fut.result()
            if n + 1 < n_use:
                fut = pool.submit(fetch, n + 1)
        else:
            ids, pl, exact = fetch(n)
        if len(pl) != T:
            raise SystemExit(f"window {n}: {len(pl)} prompt positions, expected {T}")
        kl = torch.zeros(T - 1, dtype=torch.float64)
        top1 = torch.zeros(T - 1, dtype=torch.bool)
        cover = torch.zeros(T - 1, dtype=torch.float64)
        nll = []
        for t in range(T - 1):
            ent = pl[t + 1]                      # the engine's distribution after token t (it scores token t + 1)
            eng = {int(tok): v["logprob"] for tok, v in ent.items()}
            top = sorted(eng.items(), key=lambda kv: -kv[1])[:a.k]
            eng_ids[n, t, :len(top)] = torch.tensor([x[0] for x in top], dtype=torch.int32)
            eng_lp[n, t, :len(top)] = torch.tensor([x[1] for x in top], dtype=torch.float32)
            rid, rlp = ref_id[n, t].tolist(), ref_lp[n, t]
            shared = [(j, tok) for j, tok in enumerate(rid) if tok in eng]
            pr = torch.tensor([math.exp(rlp[j]) for j, _ in shared], dtype=torch.float64)
            pe = torch.tensor([math.exp(eng[tok]) for _, tok in shared], dtype=torch.float64)
            rest_r = max(1.0 - float(pr.sum()), 1e-300)
            rest_e = max(1.0 - float(pe.sum()), 1e-300)
            v = float((pr * (pr.clamp_min(1e-300).log() - pe.clamp_min(1e-300).log())).sum())
            v += rest_r * (math.log(rest_r) - math.log(rest_e))
            kl[t] = max(v, 0.0)
            cover[t] = float(pr.sum())
            eng_top = max(eng.items(), key=lambda kv: kv[1])[0]
            top1[t] = eng_top == rid[0]
            nxt = ids[t + 1]
            if nxt in eng:
                nll.append(-eng[nxt])
        coarse_all.append(kl.clone())
        if exact is not None:                    # decode rows from the server's exact KL, prompt rows estimated
            P0 = a.prompt - 1
            ex = exact[P0:T - 1]
            if torch.isnan(ex).any():
                raise RuntimeError(f"window {n}: {int(torch.isnan(ex).sum())} decode rows without an exact KL")
            kl[P0:] = ex.clamp_min(0)
            pe = exact[:P0]                      # the prompt rows' exact KL (force.py, prompt path), when the server
            if P0 > 0 and not torch.isnan(pe).any():   # computed it: reported apart, the headline keeps its estimate
                prompt_exact_all.append(pe.clamp_min(0))
        kl_all.append(kl); top1_all.append(top1); cover_all.append(cover); nll_all += nll
        for lo, hi in ((0, 256), (256, 1024), (1024, T - 1)):
            by_pos.setdefault(f"{lo}-{hi}", []).append(kl[lo:hi])
        done = n + 1
        k_mean = torch.cat(kl_all).mean().item()
        sys.stderr.write(f"window {n}: KLD {kl.mean():.4f} top-1 {top1.float().mean():.3f} cover {cover.mean():.3f} | "
                         f"running KLD {k_mean:.4f} over {done} windows, {time.time() - t0:.0f} s\n")
    kl, top1, cover = torch.cat(kl_all), torch.cat(top1_all), torch.cat(cover_all)
    per = torch.stack(kl_all)                                            # [sequences, T - 1]
    # brandonmusic's reporting: mean with a window-block bootstrap (a cut belongs to window c // 4; percentile
    # interval, 2000 resamples), median and tail percentiles of per-row KLD, top-1, by domain
    seq_mean = per.mean(1)
    win = torch.arange(n_use) // (1 if a.subset in ("rot128", "rot32") else 4)
    nw = int(win.max()) + 1
    wsum = torch.zeros(nw, dtype=torch.float64).index_add_(0, win, seq_mean)
    wcnt = torch.zeros(nw, dtype=torch.float64).index_add_(0, win, torch.ones(n_use, dtype=torch.float64))
    g = torch.Generator().manual_seed(0)
    boots = []
    for _ in range(2000):
        idx = torch.randint(0, nw, (nw,), generator=g)
        boots.append(float(wsum[idx].sum() / wcnt[idx].sum()))
    boots = torch.tensor(boots)
    q = lambda t, x: float(torch.quantile(t, x))                       # noqa: E731
    flat = per.flatten().float()
    sample = flat[torch.randperm(flat.numel(), generator=g)[:min(flat.numel(), 2_000_000)]]
    res = {"kld": kl.mean().item(), "kld_ci95": [q(boots, 0.025), q(boots, 0.975)],
           "median": q(sample, 0.5), "p90": q(sample, 0.9), "p99": q(sample, 0.99), "p999": q(sample, 0.999),
           "top1": top1.float().mean().item(), "cover": cover.mean().item(),
           "nll_in_topk": sum(nll_all) / max(len(nll_all), 1), "windows": n_use, "positions": int(kl.numel()),
           "by_position": {k: torch.cat(v).mean().item() for k, v in by_pos.items()},
           "kld_topk_estimate": torch.stack(coarse_all).mean().item(),
           "kld_decode_rows": per[:, a.prompt - 1:].mean().item() if a.mode == "decode" else None,
           "note": ("decode rows: exact full-vocabulary KL computed in the server; prompt rows: top-k estimate"
                    if (a.kl_ref or a.kl_hidden) else "KL(ref || engine) coarse-grained over shared top-k tokens + rest (a lower bound "
                    "on full KL: 2.5x low on rot32 for act-mxfp8)")}
    if kinds is not None:
        dom = {}
        for i in range(n_use):
            dom.setdefault(kinds[i], []).append(seq_mean[i])
        res["by_domain"] = {k: float(torch.stack(v).mean()) for k, v in sorted(dom.items())}
    if a.roles and os.path.exists(a.roles):
        # the headline: KL on generated rows, i.e. rows whose predicted token is assistant output
        # or plain text; user / system / tool / header rows measure K3.1's instability on other roles' text (any
        # implementation change moves them ~0.25 on rot32) and are reported beside it
        lab = torch.load(a.roles, weights_only=False)
        names = lab["names"]
        L = lab["labels"].view(-1, 2048)[torch.tensor(sel_list[:n_use])][:, 1:T].long()
        gen = L <= 1
        gsum, gcnt = (per * gen).sum(1), gen.double().sum(1)
        wg = torch.zeros(nw, dtype=torch.float64).index_add_(0, win, gsum)
        wc = torch.zeros(nw, dtype=torch.float64).index_add_(0, win, gcnt)
        gb = []
        g2 = torch.Generator().manual_seed(1)
        for _ in range(2000):
            idx = torch.randint(0, nw, (nw,), generator=g2)
            gb.append(float(wg[idx].sum() / wc[idx].sum().clamp_min(1)))
        gb = torch.tensor(gb)
        res["kld_generated"] = float(per[gen].mean())
        res["kld_generated_ci95"] = [q(gb, 0.025), q(gb, 0.975)]
        res["generated_rows"] = int(gen.sum())
        xg = per[gen].float()
        res["generated_median"] = q(xg[torch.randperm(xg.numel(), generator=g)[:2_000_000]], 0.5)
        res["generated_top1"] = float(torch.stack(top1_all)[gen].float().mean())
        res["by_role"] = {names[i]: float(per[L == i].mean()) for i in range(len(names)) if (L == i).any()}
        if a.mode == "decode":
            P0 = a.prompt - 1
            gd = gen.clone()
            gd[:, :P0] = False
            res["kld_generated_decode_rows"] = float(per[gd].mean()) if gd.any() else None
            if len(prompt_exact_all) == n_use and P0 > 0:   # exact KL of the prompt rows (the prompt path's quality)
                pp = torch.stack(prompt_exact_all)          # [sequences, P0]
                res["kld_prompt_rows_exact"] = float(pp.mean())
                gp = gen[:, :P0]
                res["kld_generated_prompt_rows_exact"] = float(pp[gp].mean()) if gp.any() else None
                res["generated_prompt_rows"] = int(gp.sum())
        if kinds is not None:
            gdom = {}
            for i in range(n_use):
                d0 = gdom.setdefault(kinds[i], [0.0, 0.0])
                d0[0] += float(gsum[i]); d0[1] += float(gcnt[i])
            res["generated_by_domain"] = {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(gdom.items())}
    torch.save({"kl": per.float(), "kl_topk": torch.stack(coarse_all).float(), "top1": torch.stack(top1_all),
                "cover": torch.stack(cover_all), "exact": bool(a.kl_ref or a.kl_hidden),
                "eng_ids": eng_ids, "eng_lp": eng_lp, "sel": sel_list, "mode": a.mode, "prompt": a.prompt,
                "kinds": kinds[:n_use] if kinds is not None else None},
               a.out.replace(".json", "") + ".tokens.pt")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
