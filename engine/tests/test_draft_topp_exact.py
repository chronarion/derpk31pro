"""Exactness of speculative decoding with a truncated draft proposal (K31_DRAFT_TOPP / K31_DRAFT_ARGMAX), CPU only.

Run in the engine image (from engine/; about 4 minutes on 2 cores):
    docker run --rm --network none --cpuset-cpus 50-51 --user $(id -u):$(id -g) -e HOME=/tmp -e TRITON_INTERPRET=1 \
        -e CUDA_VISIBLE_DEVICES= -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /engine/tests/test_draft_topp_exact.py
Exit 0 = PASS.

A toy order-1 Markov "language" over 256 tokens: the target's next-token logits per previous token, processed as
vLLM's sampler does (temperature, then top-p 0.9: -inf outside the nucleus); the drafter's logits per previous token
are the target's plus noise (bf16). The draft proposal is the drafter's row at the request's temperature: unmasked
(stock vLLM), masked by drafttopp's kernels (K31_DRAFT_TOPP: the request's top-p, here also top-k 40 in one case), or
reduced to its argmax (K31_DRAFT_ARGMAX). 4 drafts per step.

1. vLLM's own verifier: rejection_sample (rejection_sampler_utils, block and standard paths) run under Triton's
   interpreter on drafts sampled by vLLM's gumbel_sample from the masked rows (the draft logits cache written by it),
   against verify_ref below (a torch transcription of the same kernels' arithmetic) fed the same random numbers: the
   uniforms (vLLM's tl_rand32 with spec_step_seed) and the resampling noise (vLLM's murmur3 Gumbel draw) computed by
   vLLM's own Triton functions. Every accepted length and every emitted token must agree (a row may differ only where a
   float32 sum lands within rounding of a decision; none expected). The interpreter has no libdevice log1p: the
   residual uses log(1 + x) there (stated; same on both sides).
2. Statistics with verify_ref (validated by part 1) and torch random numbers, 2^18 sequences per case: the first three
   emitted tokens' marginals equal the target chain's (chi-square, cells pooled to >= 20 expected; p > 1e-4 passes),
   no token outside the target's nucleus is ever emitted, for the stock, top-p, top-k + top-p and argmax proposals
   under block and standard verification; the accepted length per step is reported and must be higher with the
   truncated proposal than with the stock one (block verification, top-p 0.9). Negative control: drafts drawn from the
   masked rows but verified against the unmasked q must FAIL the chi-square (the test can see a q mismatch).
"""
import math
import os
import sys

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _log1p(x):
    return tl.log(1.0 + x)


libdevice.log1p = _log1p                      # the interpreter has none (vLLM's resample kernel calls it)
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import drafttopp as dt  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import (  # noqa: E402
    _log1p_neg_stable, gumbel_sample, murmur3_uniform32, spec_step_seed, tl_rand32)
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402

V, N, TOPP, TOPK = 256, 4, 0.9, 40
fails = []
g = torch.Generator().manual_seed(2024)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


# the toy language
LT = torch.randn(V, V, generator=g) * 2.2
LT[torch.arange(V), torch.randint(0, V, (V,), generator=g)] += 6.0 * torch.rand(V, generator=g)   # peaked rows
LD = (LT + torch.randn(V, V, generator=g) * 1.2).to(torch.bfloat16)                            # the drafter


def target_processed(T, top_p=TOPP, top_k=V):
    """vLLM's processed target logits per previous token: x / T, then top-k / top-p (-inf outside)."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    x = (LT / T).float()
    return apply_top_k_top_p_pytorch(x.clone(), torch.full((V,), top_k) if top_k < V else None,
                                     torch.full((V,), top_p), allow_cpu_sync=True)


def proposal(T, mode, top_p=TOPP, top_k=V):
    """The drafter's cached rows per previous token: 'stock' (unmasked), 'topp' or 'argmax' (drafttopp kernels)."""
    rows = LD.clone()
    if mode != "stock":
        idx = torch.arange(V, dtype=torch.int64)
        dt.apply(rows, idx, torch.full((V,), T), torch.full((V,), top_p), torch.full((V,), top_k, dtype=torch.int32),
                 dt.Scratch(V, V, "cpu"), argmax=mode == "argmax")
    return rows


def verify_ref(tl_rows, cache_rows, T, drafts, u, noise, block):
    """vLLM's rejection_sample (non-greedy rows) in torch: tl_rows [B, n+1, V] processed target logits, cache_rows
    [B, n, V] cached pre-temperature draft logits, T [B], drafts [B, n], u [B, n], noise [B, n+1, V] (the resampling
    Gumbel draw at each position). Returns (tokens [B, n+1], -1 past the end; count [B])."""
    B, n = drafts.shape
    lpt = tl_rows - torch.logsumexp(tl_rows, -1, keepdim=True)
    qs = cache_rows.float() / T[:, None, None]
    lpq = qs - torch.logsumexp(qs, -1, keepdim=True)
    lt_d = lpt[:, :n].gather(-1, drafts[..., None])[..., 0]
    lq_d = lpq.gather(-1, drafts[..., None])[..., 0]
    ar = torch.arange(n)
    if block:
        cum, lp = [], torch.zeros(B)
        for i in range(n):                                       # the running joint ratio, log p_{i+1}
            lp = torch.clamp(lp + (lt_d[:, i] - lq_d[:, i]), max=0.0)
            cum.append(lp)
        cum = torch.stack(cum, 1)
        h = torch.empty(B, n)
        for i in range(n):
            pn = torch.exp(cum[:, i])
            if i < n - 1:                                        # residual mass one position ahead
                rm = torch.clamp(pn[:, None] * torch.exp(lpt[:, i + 1]) - torch.exp(lpq[:, i + 1]), min=0.0).sum(-1)
                den = rm + 1.0 - pn
                h[:, i] = torch.where(den > 0, rm / den, torch.ones(B))
            else:
                h[:, i] = pn
        tau = ((u <= h) * (ar + 1)).max(1).values               # the last position whose test passes
    else:
        acc = lt_d > torch.log(u) + lq_d
        tau = torch.where(acc.all(1), torch.full((B,), n), (~acc).float().argmax(1))
    rows = torch.arange(B)
    t_tau = tl_rows[rows, tau]
    resid = t_tau.clone()                                        # the bonus row when all n drafts are accepted
    rej = tau < n
    tc = tau.clamp(max=n - 1)
    tlp = lpt[rows, tc]
    if block:
        tlp = tlp + torch.where(tc > 0, cum[rows, (tc - 1).clamp(min=0)], torch.zeros(B))[:, None]
    ratio = torch.exp(lpq[rows, tc] - tlp)
    r_log = torch.where(ratio < 1.0, tlp + torch.log1p(-ratio), torch.full_like(tlp, -float("inf")))
    resid = torch.where(rej[:, None], r_log, resid)
    tok = torch.argmax(resid + noise[rows, tau], -1)
    out = torch.full((B, n + 1), -1, dtype=torch.int64)
    out[:, :n] = torch.where(ar[None, :] < tau[:, None], drafts, torch.full_like(drafts, -1))
    out[rows, tau] = tok
    return out, tau + 1


@triton.jit
def _uniforms_kernel(u_ptr, seed_ptr, pos_ptr, N: tl.constexpr, NL: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    seed = tl.load(seed_ptr + c)
    for i in tl.static_range(N):
        pos = tl.load(pos_ptr + c * NL + i)
        s = spec_step_seed(seed, i) if BLOCK else seed
        tl.store(u_ptr + c * N + i, tl_rand32(s, pos, includes_zero=False))


@triton.jit
def _noise_kernel(n_ptr, seed_ptr, pos_ptr, V, NL: tl.constexpr, BV: tl.constexpr):
    c = tl.program_id(0)
    j = tl.program_id(1)
    seed = tl.load(seed_ptr + c)
    pos = tl.load(pos_ptr + c * NL + j)
    keys = tl.arange(0, BV)
    u = murmur3_uniform32(seed, pos, keys)
    tl.store(n_ptr + (c * NL + j) * BV + keys, -tl.log(-_log1p_neg_stable(u)), mask=keys < V)


def vllm_step(ctx, T, mode, block, seeds, top_p=TOPP, top_k=V):
    """One step through vLLM's own kernels: drafts by gumbel_sample from the (masked) drafter rows (writing the
    draft logits cache), then rejection_sample. Also returns everything verify_ref needs to replay it."""
    B = ctx.shape[0]
    temp = torch.full((B,), T)
    cache = DraftLogitsCache(B, N, V, dtype=torch.bfloat16, fill=0.0, device="cpu")
    idx = torch.arange(B, dtype=torch.int64)
    scratch = dt.Scratch(B, V, "cpu")
    prev, drafts = ctx, []
    for i in range(N):
        rows = LD[prev].clone()
        if mode != "stock":
            dt.apply(rows, idx, temp, torch.full((B,), top_p), torch.full((B,), top_k, dtype=torch.int32), scratch,
                     argmax=mode == "argmax")
        d = gumbel_sample(rows, idx, temp, seeds, torch.full((B,), 1000 + i, dtype=torch.int64), True, True,
                          logits_cache=cache.values, logits_cache_col=torch.tensor(i, dtype=torch.int32))
        drafts.append(d)
        prev = d
    drafts = torch.stack(drafts, 1)
    tp = target_processed(T, top_p, top_k)
    ctxs = torch.cat([ctx[:, None], drafts], 1)                 # row i: the target after the i-th token
    tl_rows = tp[ctxs]                                          # [B, n+1, V]
    nl = N + 1
    ds = torch.zeros(B, nl, dtype=torch.int64)
    ds[:, 1:] = drafts
    cu = torch.arange(0, B * nl + 1, nl, dtype=torch.int32)
    pos = (torch.arange(B * nl) % nl + 999).to(torch.int64)
    exp_idx = torch.repeat_interleave(idx, nl)
    loc = (torch.arange(B * nl) % nl).to(torch.int32)
    out, num = rejection_sample(tl_rows.reshape(B * nl, V).contiguous(), cache, ds.view(-1), cu, pos, idx, exp_idx,
                                loc, temp, seeds, N, use_block_verification=block)
    u = torch.empty(B, N)
    _uniforms_kernel[(B,)](u, seeds, pos, N=N, NL=nl, BLOCK=block)
    noise = torch.empty(B, nl, 256)
    _noise_kernel[(B, nl)](noise, seeds, pos, V, NL=nl, BV=256)
    return out, num, (tl_rows, cache.values.float(), temp, drafts, u, noise)


# 1. replay: vLLM's interpreted verifier against verify_ref on the same inputs and random numbers
for mode, block, T, top_k in (("topp", True, 1.0, V), ("argmax", True, 0.8, V), ("topp", False, 0.8, TOPK),
                              ("stock", True, 1.0, V)):
    B = 96
    ctx = torch.randint(0, V, (B,), generator=g)
    seeds = torch.randint(-2 ** 62, 2 ** 62, (B,), generator=g, dtype=torch.int64)
    out, num, args = vllm_step(ctx, T, mode, block, seeds, top_k=top_k)
    ref_out, ref_num = verify_ref(*args, block=block)
    same_num = num.long() == ref_num
    same_tok = torch.tensor([torch.equal(out[b, :int(num[b])], ref_out[b, :int(num[b])]) for b in range(B)])
    ok = bool((same_num & same_tok).all())
    hist = torch.bincount(num.long() - 1, minlength=N + 1).tolist()
    check(f"vLLM's rejection_sample = verify_ref, same randomness ({mode}, {'block' if block else 'standard'}, T {T}"
          f"{', top-k 40' if top_k < V else ''})", ok,
          f"({int((same_num & same_tok).sum())}/{B} rows equal; accepted drafts 0-{N}: {hist})")


# 2. statistics with verify_ref
def chi2_pvalue(counts, probs, n):
    exp = probs * n
    order = torch.argsort(exp, descending=True)
    c, e = counts[order].double(), exp[order].double()
    big = e >= 20
    cs = torch.cat([c[big], c[~big].sum()[None]])
    es = torch.cat([e[big], e[~big].sum()[None]])
    keep = es > 0
    cs, es = cs[keep], es[keep]
    stat = float(((cs - es) ** 2 / es).sum())
    df = max(int(keep.sum()) - 1, 1)
    return stat, df, float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64),
                                                   torch.tensor(stat / 2.0, dtype=torch.float64)))


def simulate(T, mode, block, n_seq, top_p=TOPP, top_k=V, verify_mode=None, x0=7, chunk=1 << 15):
    """n_seq sequences from context x0 until each has 3 tokens; returns the first three tokens and the mean accepted
    drafts of the first step. verify_mode: the q the verifier is given (default: the one the drafts came from)."""
    tp = target_processed(T, top_p, top_k)
    prop = proposal(T, mode, top_p, top_k)
    prop_v = prop if verify_mode is None else proposal(T, verify_mode, top_p, top_k)
    qprob = torch.softmax(prop.float() / T, -1)
    toks, accepted = [], 0
    for c0 in range(0, n_seq, chunk):
        B = min(chunk, n_seq - c0)
        seq = torch.full((B, 3), -1, dtype=torch.int64)
        have = torch.zeros(B, dtype=torch.int64)
        last = torch.full((B,), x0, dtype=torch.int64)
        first = True
        while bool((have < 3).any()):
            act = (have < 3).nonzero()[:, 0]
            b = act.shape[0]
            prev, drafts = last[act], []
            for i in range(N):
                d = torch.multinomial(qprob[prev], 1, generator=g)[:, 0]
                drafts.append(d)
                prev = d
            drafts = torch.stack(drafts, 1)
            ctxs = torch.cat([last[act, None], drafts], 1)
            tl_rows = tp[ctxs]
            prevs = torch.cat([last[act, None], drafts[:, :-1]], 1)
            cache_rows = torch.stack([prop_v[prevs[:, i]] for i in range(N)], 1)
            u = torch.rand(b, N, generator=g).clamp(min=4.6566127342e-10)
            noise = -torch.log(-torch.log(torch.rand(b, N + 1, V, generator=g).clamp(min=1e-12)))
            out, num = verify_ref(tl_rows, cache_rows, torch.full((b,), T), drafts, u, noise, block)
            if first:
                accepted += int((num - 1).sum())
                first = False
            for k in range(N + 1):                                    # append the emitted tokens
                emit = (k < num)
                pos_ok = emit & (have[act] < 3)
                rows = act[pos_ok]
                seq[rows, have[rows]] = out[pos_ok, k]
                have[rows] += 1
            last[act] = out[torch.arange(b), num - 1]
        toks.append(seq)
    return torch.cat(toks), accepted / n_seq


n_seq = 1 << 18
cases = [("stock", True, 1.0, V), ("topp", True, 1.0, V), ("argmax", True, 1.0, V), ("topp", False, 0.8, V),
         ("topp", True, 0.8, TOPK), ("argmax", False, 1.0, V)]
acc_len = {}
for mode, block, T, top_k in cases:
    seqs, acc = simulate(T, mode, block, n_seq, top_k=top_k)
    acc_len[(mode, block, T, top_k)] = acc
    P = torch.softmax(target_processed(T, TOPP, top_k), -1)
    dist = P[7]
    worst, outside = 1.0, 0
    for k in range(3):
        cnt = torch.bincount(seqs[:, k], minlength=V).double()
        outside += int(cnt[dist == 0].sum())
        stat, df, pv = chi2_pvalue(cnt, dist, n_seq)
        worst = min(worst, pv)
        dist = dist @ P
    name = f"{mode}{' + top-k 40' if top_k < V else ''}, {'block' if block else 'standard'}, T {T}"
    check(f"emitted tokens follow the target chain ({name})", worst > 1e-4 and outside == 0,
          f"(worst chi-square p over positions 1-3 = {worst:.3g}; tokens outside the nucleus {outside}; "
          f"accepted drafts per step {acc:.3f})")
check("truncated proposals accept more than the stock one (block, top-p 0.9, T 1)",
      acc_len[("topp", True, 1.0, V)] > acc_len[("stock", True, 1.0, V)] + 0.02,
      f"(stock {acc_len[('stock', True, 1.0, V)]:.3f}, top-p {acc_len[('topp', True, 1.0, V)]:.3f}, "
      f"argmax {acc_len[('argmax', True, 1.0, V)]:.3f} drafts per step)")
seqs, _ = simulate(1.0, "topp", True, n_seq, verify_mode="stock")
P = torch.softmax(target_processed(1.0), -1)
dist, worst = P[7], 1.0
for k in range(3):
    worst = min(worst, chi2_pvalue(torch.bincount(seqs[:, k], minlength=V).double(), dist, n_seq)[2])
    dist = dist @ P
check("negative control: masked drafts verified against the unmasked q are detected", worst < 1e-6,
      f"(worst chi-square p = {worst:.3g})")

print("DRAFT TOPP EXACT:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
