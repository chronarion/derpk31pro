"""K31_DRAFT_TOPP / K31_DRAFT_ARGMAX: the DSpark drafter's proposal truncated per request (start-up flags).

vLLM drafts with the request's temperature only (speculator.py: "we only consider the temperature and ignore the other
sampling parameters such as top_k and top_p"): each draft position samples from q = softmax(logits / T) over the whole
vocabulary (gumbel_sample), which also stores the pre-temperature logits row in the draft logits cache; the verifier
(rejection_sampler_utils, standard and block paths) reads that row back and divides it by the same T, so the q it
verifies against is the q the draft was drawn from, bit for bit. The target's own logits are truncated by the request's
top-k / top-p before verification, so a draft from the drafter's tail outside the target's nucleus is a certain
rejection.

K31_DRAFT_TOPP=1: before vLLM's draft sampling, each draft logits row of a request with temperature > 0 and top_p < 1
or top_k < vocabulary is masked in place (-inf) outside the drafter's own approximate nucleus under that request's
top-k / top-p, computed on the drafter's temperature-scaled distribution as vLLM's sampler orders it (top-k first,
then top-p on what is left). vLLM's gumbel_sample then samples from the masked row and stores the masked row in the
cache, so sampling and verification agree on the truncated q bit for bit and speculative sampling stays exact (it is
exact for any proposal distribution the verifier knows). Rows of other requests (top_p = 1 and no top-k, temperature
0, padding) are not touched: their drafts are bit-identical to the stock path.
K31_DRAFT_ARGMAX=1: every row of a request with temperature > 0 keeps only its largest logit (ties all kept): the draft
is the drafter's argmax and the cached q a point mass on it (uniform over exact ties), which the verifier treats the
same way. vLLM's own draft_sample_method 'greedy' is the production form of this (argmax without the cache, one-hot q
in the verifier, no masking kernels; k31.sh DRAFT_SAMPLE=greedy).

The nucleus (2 launches per draft position, CUDA-graph safe: fixed grids, device-side parameters, no host sync):
  1. stats: each program owns SLICE vocabulary entries of one row: y = x * (1 / T) (IEEE), the absolute bin
     b = floor(17 y) (bins of 1/17 nat), the slice's top bin a_s = max b, the counts of the H = 256 bins below a_s,
     its largest y and sum(exp(y - max y)), and its largest raw logit.
  2. mask: each program recomputes the row's cut from the slice summaries (the bins are aligned: slice s's bin j is
     the row's bin (A - a_s) + j, A = max a_s), estimates each bin's mass as count x exp(bin centre - max y) with
     each slice's largest token at its exact mass, finds the smallest top set of bins whose estimated mass reaches
     top_p x Z (Z exact) after the top-k cut (the bin that holds the k-th largest logit: counts are exact), and
     writes -inf over its own slice below it. The row's largest raw logit is never masked. A cut beyond the 256 bins
     (15 nats below the maximum) masks nothing.
The kept set is a top set of the drafter's logits (ties kept or dropped together) that matches the exact sorted
nucleus to about one bin (a token's probability is within e^(1/17) of the others in its bin); the cut is approximate
by design (any proposal is exact, a better match only raises acceptance), the bit-exact agreement of sampling and
verification is what exactness needs.

Paths: DSparkSpeculator._sample_logits (the generic Markov loop and the dspark_draft_topk loop of the speculator both
sample through it) is wrapped by `patch` (hook.PATCHES, at import); `bind` (runner.load_model) attaches the state
and refuses every configuration whose drafts or cached q bypass it or that this was not checked for: another
speculator, greedy drafting, a drafter with its own fused Markov sampler (decode_backend=latency_specialized), a
draft watermarker, a reduced draft vocabulary, adaptive verification, synthetic / fly rejection, a custom rejection
sampler. Forced drafts (K31_FORCE_DRAFTS, the KLD harness) are reference tokens, not draws from q: the harness's
requests are greedy (temperature 0), which these flags never touch; bind says so in its line.
"""
from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

H = 256                  # bins per slice (a power of two)
# bins per nat of the temperature-scaled logits: odd, so the bins do not line up with the bf16 lattice (dyadic steps;
# with 16 bins per nat every token of |x| in [8, 32) at T = 1 sits on a bin's lower edge and the centre estimate is
# 3.2% high in every bin)
INV_W = 17.0
W = 1.0 / INV_W
SLICE = int(os.environ.get("K31_DRAFT_TOPP_SLICE", "8192"))   # vocabulary entries per program (a power of two)
WARPS = int(os.environ.get("K31_DRAFT_TOPP_WARPS", "8"))
# K31_DRAFT_TEMP=tau (start-up, with K31_DRAFT_TOPP; default 1): the drafter's logits rows are multiplied by 1 / tau in
# place before the nucleus mask, so a sampled request's drafts come from q ~ softmax(logits / (tau T)) (tau < 1 sharper,
# > 1 flatter). The scaled row is what vLLM samples from and caches, and the verifier divides that cached row by the
# same T, so speculative sampling stays exact for any tau; only the acceptance changes. Rows of temperature-0 requests
# keep their argmax (a positive scale does not move it).
DRAFT_TEMP = float(os.environ.get("K31_DRAFT_TEMP", "1") or 1)
assert DRAFT_TEMP > 0, "K31_DRAFT_TEMP must be > 0"
NEG = -(1 << 30)         # the top bin of a slice without a finite logit
CLAMP = float(1 << 29)   # |17 y| is clamped here so the integer conversion is defined (never reached at T >= 0.01)
SPEC_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
assert SLICE & (SLICE - 1) == 0 and SLICE >= 1024, "K31_DRAFT_TOPP_SLICE must be a power of two >= 1024"


@triton.jit
def _row_params(r, idx_ptr, idx_stride, temp_ptr, topp_ptr, topk_ptr, V, ARGMAX: tl.constexpr):
    """The row's request (-1 = padding), temperature, top_p, top_k, and whether the row is truncated."""
    req = tl.load(idx_ptr + r * idx_stride).to(tl.int64)
    ok = req >= 0
    reqc = tl.where(ok, req, 0)
    t = tl.load(temp_ptr + reqc).to(tl.float32)
    p = tl.load(topp_ptr + reqc).to(tl.float32)
    k = tl.load(topk_ptr + reqc).to(tl.int32)
    if ARGMAX:
        active = ok & (t > 0.0)
    else:
        active = ok & (t > 0.0) & ((p < 1.0) | (k < V))
    return active, t, p, k


@triton.jit
def _bins(y, INV_W: tl.constexpr, CLAMP: tl.constexpr):
    """Absolute bin floor(17 y) of the scaled logits y = x * (1 / T) (an IEEE reciprocal and product, as torch's)."""
    v = tl.minimum(tl.maximum(y * INV_W, -CLAMP), CLAMP)
    return tl.floor(v).to(tl.int32)


@triton.jit(do_not_specialize=["x_stride", "idx_stride", "V"])
def _stats_kernel(x_ptr, x_stride, idx_ptr, idx_stride, temp_ptr, topp_ptr, topk_ptr,
                  anc_ptr, cnt_ptr, mx_ptr, ym_ptr, z_ptr, V,
                  S: tl.constexpr, SLICE: tl.constexpr, H: tl.constexpr, INV_W: tl.constexpr,
                  CLAMP: tl.constexpr, NEG: tl.constexpr, ARGMAX: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    active, t, p, k = _row_params(r, idx_ptr, idx_stride, temp_ptr, topp_ptr, topk_ptr, V, ARGMAX)
    if not active:
        return
    offs = s * SLICE + tl.arange(0, SLICE)
    inb = offs < V
    xf = tl.load(x_ptr + r.to(tl.int64) * x_stride + offs, mask=inb, other=float("-inf")).to(tl.float32)
    fin = (xf > float("-inf")) & (xf < float("inf"))
    base = r * S + s
    tl.store(mx_ptr + base, tl.max(tl.where(fin, xf, float("-inf")), axis=0))
    if ARGMAX:
        return
    y = xf * tl.div_rn(1.0, t)
    b = tl.where(fin, _bins(y, INV_W, CLAMP), NEG)
    a = tl.max(b, axis=0)
    ym = tl.max(tl.where(fin, y, float("-inf")), axis=0)
    j = a - b
    inh = fin & (j < H)
    hist = tl.histogram(tl.where(inh, j, 0), H, mask=inh)
    z = tl.sum(tl.where(fin, tl.exp(y - ym), 0.0), axis=0)
    tl.store(anc_ptr + base, a)
    tl.store(ym_ptr + base, ym)
    tl.store(z_ptr + base, z)
    tl.store(cnt_ptr + base.to(tl.int64) * H + tl.arange(0, H), hist)


@triton.jit
def _cut(r, t, p, k, anc_ptr, cnt_ptr, ym_ptr, z_ptr,
         S: tl.constexpr, SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, NEG: tl.constexpr):
    """The row's cut from the slice summaries: (A, gs); tokens of bin >= A - gs stay, gs >= H keeps everything."""
    sid = tl.arange(0, SP)
    sm = sid < S
    rb = r * S
    anc = tl.load(anc_ptr + rb + sid, mask=sm, other=NEG)
    A = tl.max(anc, axis=0)
    ymv = tl.load(ym_ptr + rb + sid, mask=sm, other=float("-inf"))
    ym = tl.max(ymv, axis=0)
    zv = tl.load(z_ptr + rb + sid, mask=sm, other=0.0)
    Z = tl.sum(tl.where(ymv > float("-inf"), zv * tl.exp(ymv - ym), 0.0), axis=0)
    g = tl.arange(0, H)
    jj = g[None, :] - (A - anc)[:, None]                          # slice s's bin for the row's bin g
    ok = sm[:, None] & (jj >= 0) & (anc[:, None] > NEG)
    c = tl.load(cnt_ptr + (rb + sid[:, None]).to(tl.int64) * H + jj, mask=ok, other=0)
    n = tl.sum(c, axis=0)                                          # tokens per bin of the row, from the top
    e = tl.minimum(((A - g).to(tl.float32) + 0.5) * W - ym, 1.0)   # the bin's centre against the row's max
    # each slice's largest token at its exact mass instead of its bin's centre (a dominant token decides peaked rows)
    top = tl.where(anc > NEG, tl.exp(ymv - ym) - tl.exp(tl.minimum((anc.to(tl.float32) + 0.5) * W - ym, 1.0)), 0.0)
    corr = tl.sum(tl.where((A - anc)[:, None] == g[None, :], top[:, None], 0.0), axis=0)
    mass = n.to(tl.float32) * tl.exp(e) + corr
    cm = tl.cumsum(mass, axis=0)
    cn = tl.cumsum(n, axis=0)
    gk = tl.min(tl.where(cn >= k, g, H), axis=0)                   # top-k: the bin of the k-th largest logit
    zk = tl.where(gk < H, tl.sum(tl.where(g == gk, cm, 0.0), axis=0), Z)
    gp = tl.min(tl.where((cm >= p * zk) & (g <= gk), g, H), axis=0)
    gp = tl.where(p < 1.0, gp, H)               # top-k alone: float32 sums saturate below the k-th token's bin
    return A, tl.minimum(gk, gp)


@triton.jit(do_not_specialize=["x_stride", "idx_stride", "V"])
def _mask_kernel(x_ptr, x_stride, idx_ptr, idx_stride, temp_ptr, topp_ptr, topk_ptr,
                 anc_ptr, cnt_ptr, mx_ptr, ym_ptr, z_ptr, V,
                 S: tl.constexpr, SP: tl.constexpr, SLICE: tl.constexpr, H: tl.constexpr, INV_W: tl.constexpr,
                 W: tl.constexpr, CLAMP: tl.constexpr, NEG: tl.constexpr, ARGMAX: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    active, t, p, k = _row_params(r, idx_ptr, idx_stride, temp_ptr, topp_ptr, topk_ptr, V, ARGMAX)
    if not active:
        return
    sid = tl.arange(0, SP)
    xmax = tl.max(tl.load(mx_ptr + r * S + sid, mask=sid < S, other=float("-inf")), axis=0)
    if xmax == float("-inf"):
        return
    offs = s * SLICE + tl.arange(0, SLICE)
    inb = offs < V
    ptrs = x_ptr + r.to(tl.int64) * x_stride + offs
    x = tl.load(ptrs, mask=inb, other=float("-inf"))
    xf = x.to(tl.float32)
    fin = (xf > float("-inf")) & (xf < float("inf"))
    if ARGMAX:
        drop = fin & (xf != xmax)
    else:
        A, gs = _cut(r, t, p, k, anc_ptr, cnt_ptr, ym_ptr, z_ptr, S, SP, H, W, NEG)
        if gs >= H:
            return
        drop = fin & (_bins(xf * tl.div_rn(1.0, t), INV_W, CLAMP) < A - gs) & (xf != xmax)
    tl.store(ptrs, tl.full([SLICE], float("-inf"), tl.float32).to(x.dtype), mask=inb & drop)


class Scratch:
    """Per-slice summaries for up to `rows` rows of `vocab` logits (reused by every draft position: stream order)."""

    def __init__(self, rows: int, vocab: int, device):
        self.rows, self.vocab = rows, vocab
        self.S = triton.cdiv(vocab, SLICE)
        n = rows * self.S
        self.anc = torch.zeros(n, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(n * H, dtype=torch.int32, device=device)
        self.mx = torch.zeros(n, dtype=torch.float32, device=device)
        self.ym = torch.zeros(n, dtype=torch.float32, device=device)
        self.z = torch.zeros(n, dtype=torch.float32, device=device)


def apply(logits: torch.Tensor, idx: torch.Tensor, temperature: torch.Tensor, topp: torch.Tensor, topk: torch.Tensor,
          scratch: Scratch, argmax: bool = False) -> None:
    """Mask `logits` [R, V] (bf16 / fp16 / fp32, unit stride over V) in place: row r belongs to request idx[r] (-1 =
    padding: untouched) with temperature[req], topp[req], topk[req] (device tensors indexed by request slot)."""
    R, V = logits.shape
    if R == 0:
        return
    assert logits.stride(1) == 1, "draft logits rows must be contiguous over the vocabulary"
    assert R <= scratch.rows and V == scratch.vocab, (R, V, scratch.rows, scratch.vocab)
    S = scratch.S
    grid = (R, S)
    _stats_kernel[grid](logits, logits.stride(0), idx, idx.stride(0), temperature, topp, topk,
                        scratch.anc, scratch.cnt, scratch.mx, scratch.ym, scratch.z, V,
                        S=S, SLICE=SLICE, H=H, INV_W=INV_W, CLAMP=CLAMP, NEG=NEG, ARGMAX=argmax, num_warps=WARPS)
    _mask_kernel[grid](logits, logits.stride(0), idx, idx.stride(0), temperature, topp, topk,
                       scratch.anc, scratch.cnt, scratch.mx, scratch.ym, scratch.z, V,
                       S=S, SP=triton.next_power_of_2(S), SLICE=SLICE, H=H, INV_W=INV_W, W=W, CLAMP=CLAMP, NEG=NEG,
                       ARGMAX=argmax, num_warps=WARPS)


def reference(logits: torch.Tensor, idx, temperature, topp, topk, argmax: bool = False, slice_: int = SLICE):
    """The same cut computed with torch, row by row (tests). Returns (masked copy of logits, per-row cut info: None for
    an untouched row, else a dict with A, gs, the bins and the float32 scaled logits)."""
    out = logits.clone()
    R, V = logits.shape
    S = -(-V // slice_)
    info = []
    for r in range(R):
        req = int(idx[r])
        if req < 0:
            info.append(None)
            continue
        t = temperature[req].float().cpu()
        p = float(topp[req])
        k = int(topk[req])
        active = float(t) > 0 and (argmax or p < 1.0 or k < V)
        x = logits[r].float().cpu()
        fin = torch.isfinite(x)
        if not active or not bool(fin.any()):
            info.append(None)
            continue
        xmax = x[fin].max()
        if argmax:
            drop = fin & (x != xmax)
            out[r][drop.to(out.device)] = float("-inf")
            info.append({"argmax": True})
            continue
        y = x * (torch.tensor(1.0, dtype=torch.float32) / t)
        b = torch.floor((y * INV_W).clamp(-CLAMP, CLAMP)).to(torch.int64)
        anc, ymax, zs, cnts = [], [], [], []
        for s in range(S):
            sl = slice(s * slice_, min((s + 1) * slice_, V))
            f = fin[sl]
            if not bool(f.any()):
                anc.append(NEG), ymax.append(float("-inf")), zs.append(0.0)
                cnts.append(torch.zeros(H, dtype=torch.int64))
                continue
            bs, ys = b[sl][f], y[sl][f]
            a = int(bs.max())
            m = ys.max()
            j = a - bs
            cnts.append(torch.bincount(j[j < H], minlength=H)[:H])
            anc.append(a), ymax.append(float(m)), zs.append(float(torch.exp(ys - m).sum()))
        A = max(anc)
        ym = max(ymax)
        Z = sum(z * float(torch.exp(torch.tensor(m - ym))) for z, m in zip(zs, ymax) if m > float("-inf"))
        n = torch.zeros(H, dtype=torch.int64)
        for a, c in zip(anc, cnts):
            if a > NEG and A - a < H:
                n[A - a:] += c[:H - (A - a)]
        g = torch.arange(H)
        ym32 = torch.tensor(ym, dtype=torch.float32)
        e = (((A - g).to(torch.float32) + 0.5) * W - ym32).clamp(max=1.0)
        mass = n.to(torch.float32) * torch.exp(e)
        for a, m in zip(anc, ymax):
            if a > NEG and A - a < H:
                c = torch.exp(torch.tensor(m, dtype=torch.float32) - ym32)
                mass[A - a] += c - torch.exp(((torch.tensor(float(a)) + 0.5) * W - ym32).clamp(max=1.0))
        cm = torch.cumsum(mass, 0)
        cn = torch.cumsum(n, 0)
        hit = (cn >= k).nonzero()
        gk = int(hit[0]) if len(hit) else H
        zk = float(cm[gk]) if gk < H else Z
        hit = ((cm >= torch.tensor(p, dtype=torch.float32) * zk) & (g <= gk)).nonzero()
        gp = int(hit[0]) if len(hit) and p < 1.0 else H
        gs = min(gk, gp)
        info.append({"A": A, "gs": gs, "bins": b, "y": y, "Z": Z})
        if gs >= H:
            continue
        drop = fin & (b < A - gs) & (x != xmax)
        out[r][drop.to(out.device)] = float("-inf")
    return out, info


class State:
    """What the wrapped speculator methods use: the request slots' top_p / top_k in persistent device buffers (refreshed
    from the sampler's per-step UVA views before every propose, as vLLM copies the temperature) and the scratch."""

    def __init__(self, spec, sampling_states, argmax: bool):
        dev = spec.device
        n = spec.max_num_reqs
        self.ss = sampling_states
        self.argmax = argmax
        self.temp = spec.temperature
        self.topp = torch.ones(n, dtype=torch.float32, device=dev)
        self.topk = torch.full((n,), 1 << 30, dtype=torch.int32, device=dev)
        self.scratch = Scratch(n, spec.vocab_size, dev)

    def refresh(self) -> None:
        if self.argmax:
            return
        n = self.topp.shape[0]
        self.topp.copy_(self.ss.top_p.gpu[:n], non_blocking=True)
        self.topk.copy_(self.ss.top_k.gpu[:n], non_blocking=True)

    def apply(self, logits: torch.Tensor, idx: torch.Tensor) -> None:
        sc = self.scratch
        if logits.shape[0] > sc.rows or logits.shape[1] != sc.vocab:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"K31_DRAFT_TOPP: draft logits {tuple(logits.shape)} outside the scratch "
                                   f"({sc.rows} x {sc.vocab}) during graph capture")
            self.scratch = sc = Scratch(max(logits.shape[0], sc.rows), logits.shape[1], logits.device)
        if DRAFT_TEMP != 1.0 and not self.argmax:
            logits.mul_(1.0 / DRAFT_TEMP)            # in place: the sampled and cached row (K31_DRAFT_TEMP)
        apply(logits, idx, self.temp, self.topp, self.topk, sc, self.argmax)


def patch(module) -> None:
    """hook.PATCHES (only when a flag is on): wrap DSparkSpeculator's draft sampling and propose. Instances without a
    bound State (bind not run) take the stock methods unchanged."""
    cls = module.DSparkSpeculator
    if getattr(cls, "_k31_dtopp", False):
        return
    orig_sample = cls._sample_logits
    orig_propose = cls.propose

    def _sample_logits(self, logits, idx_map, sample_pos, step, _orig=orig_sample):
        st = self.__dict__.get("_k31_dtopp_state")
        if st is not None:
            st.apply(logits, idx_map)            # in place: vLLM samples from and caches the masked row
        return _orig(self, logits, idx_map, sample_pos, step)

    def propose(self, *args, _orig=orig_propose, **kwargs):
        st = self.__dict__.get("_k31_dtopp_state")
        if st is not None:
            st.refresh()                         # before the draft graph replays (it reads the buffers)
        return _orig(self, *args, **kwargs)

    cls._sample_logits = _sample_logits
    cls.propose = propose
    cls._k31_dtopp = True


def bind(runner, argmax: bool, forced: bool, log) -> None:
    """runner.load_model: attach the State to the DSpark speculator, or stop with the reason this path is not served."""
    name = "K31_DRAFT_ARGMAX" if argmax else "K31_DRAFT_TOPP"

    def refuse(why: str):
        raise RuntimeError(f"K31: {name}=1 refused: {why}; unset {name}")

    spec = getattr(runner, "speculator", None)
    mod = sys.modules.get(SPEC_MODULE)
    if spec is None or mod is None or not isinstance(spec, mod.DSparkSpeculator):
        refuse(f"it covers the DSpark speculator only (speculative method dspark), this runner has "
               f"{type(spec).__name__ if spec is not None else 'no speculator'}")
    if not getattr(mod.DSparkSpeculator, "_k31_dtopp", False):
        refuse("DSparkSpeculator was not patched (hook.PATCHES)")
    if spec.draft_logits is None:
        refuse("the drafts are greedy (draft_sample_method 'greedy': argmax drafts, no cached q to truncate); "
               "k31.sh sets 'probabilistic'")
    if spec.draft_watermarker is not None:
        refuse("a draft watermarker samples the drafts")
    if getattr(spec, "_d2t_scatter_index", None) is not None:
        refuse("a reduced draft vocabulary (the cached q is scattered into the target vocabulary)")
    if spec.acceptance_estimator is not None:
        # vLLM's adaptive verification without a trained confidence head: OnlineAcceptanceEstimator.predict reads the
        # draft logits rows inside _sample_logits (after this mask); not checked
        refuse("adaptive verification without a confidence head (its acceptance estimator reads the draft logits; "
               "not checked)")
    # Adaptive verification with the drafter's confidence head (the final DSpark drafter has one): the per-position
    # confidence comes from compute_confidence(hidden, markov embeddings), never from the draft logits, and trimming a
    # verify block only drops trailing drafts, whose cached q the verifier then does not read: sampling stays exact.
    av = bool(spec.enable_adaptive_verification and spec.use_confidence_head)
    if getattr(spec.model, "sample_sequential", None) is not None:
        refuse("the drafter samples in its own fused Markov kernels (decode_backend=latency_specialized)")
    if spec.draft_logits.row_mapping is not None:
        refuse("batch-sharded draft logits (TP > 1; not checked)")
    rs = getattr(runner, "rejection_sampler", None)
    if rs is None or type(rs).__name__ != "RejectionSampler":
        refuse(f"the verifier is {type(rs).__name__ if rs is not None else 'missing'}, not vLLM's RejectionSampler")
    if rs.synthetic_conditional_rates is not None or rs.fly_window_size or rs.watermark_key is not None:
        refuse("rejection_sample_method synthetic / fly or a watermark (standard and block are exact with any q)")
    ss = runner.sampler.sampling_states
    if ss.top_p.gpu.shape[0] < spec.max_num_reqs or ss.top_k.gpu.shape[0] < spec.max_num_reqs:
        refuse("the sampler's top_p / top_k views are shorter than the speculator's request slots")
    spec._k31_dtopp_state = State(spec, ss, argmax)
    method = "block" if rs.use_block_verification else "standard"
    what = ("drafts of sampled requests (temperature > 0) are the drafter's argmax; the cached q is a point mass"
            if argmax else "each request's top-k / top-p also truncate the drafter's q (drafts and cached q)")
    log(f"{name}: {what}; {spec.vocab_size} logits per row in {spec._k31_dtopp_state.scratch.S} slices of {SLICE}, "
        f"{H} bins of 1/{INV_W:g} nat; verifier {method}; request slots {spec.max_num_reqs}"
        + ("; forced drafts (K31_FORCE_DRAFTS) are reference tokens, exact only for temperature-0 requests, "
           "which this flag leaves alone" if forced else "")
        + ("; adaptive verification sizes the verify blocks from the drafter's confidence head" if av else "")
        + (f"; draft temperature x {DRAFT_TEMP:g} (K31_DRAFT_TEMP)" if DRAFT_TEMP != 1.0 and not argmax else ""))
