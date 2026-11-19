"""K31_MEGAMOE_NOPREP on the GB300 (k31_engine/noprep.py, csrc/noprep_plan.cu): MegaMoE gets the same bytes without
vLLM's prep kernel. Every check is bit-for-bit unless it says otherwise.

Run from the repository root (GB = the GB300's CDI UUID; the plan kernel builds on first use, ~1-2 min):
    docker run --rm --network none --device nvidia.com/gpu=$GB --ipc host --cpuset-cpus 62-63 --ulimit core=0 \
        --user $(id -u):$(id -g) -e HOME=/tmp -e K31_PDL=1 -v "$PWD"/engine:/engine:ro \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_noprep.py
To run part 6 on the engine's MegaMoE (MEGAMOE_FORK=1 in k31.sh), mount the overlay's two headers as k31.sh does:
    DG=/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm
    O=engine/overlay/megamoe_k31/include/deep_gemm
and add to the docker command:
    -v "$PWD/$O/impls/sm100_fp8_fp4_mega_moe_situ.cuh:$DG/impls/sm100_fp8_fp4_mega_moe_situ.cuh:ro" \
    -v "$PWD/$O/scheduler/mega_moe.cuh:$DG/scheduler/mega_moe.cuh:ro"
Exit 0 and "NOPREP: PASS" = every part passed; the plan kernel's time per call is printed (meaningful only on an idle
GB300).

1. x half: noprep.linear_mx (the latent down 3584 x 7168 with the MXFP8 epilogue) against w8a16.linear followed by
   vLLM's prepare_megamoe_inputs, T = 1..16, the engine's config (BN 32, split 4) and three others: the bf16 rows, the
   FP8 codes and the scale words equal. The epilogue alone on rows built for the scale's edges (zeros, group maxima of
   exactly 448 x 2^k and one bf16 step either side, tiny and huge rows) against prep.
2. top-k: the plan kernel's top-k against vLLM's grouped_topk (fused_grouped_topk, sigmoid, one group), 400 calls of
   T = 1..16 (3,400 rows): logits at several scales, quantized logits (ties), saturated logits (equal scores), random
   and zero bias, renormalize on / off, scales 1.0 / 2.5, rows with NaN / +-inf: ids and weights equal.
3. unitk: the plan kernel's rows / stage list (with and without the top-k in the kernel) against unitk.plan on the
   same picks; pick counts and pending rows equal; residency scores equal where no expert repeats in a call (repeats
   add in atomics' order, as unitk's do: within 1e-6 there). Also with ALLROWS (K31_STAGE_ALL): against
   unitk.plan(allrows=True) where this tree's unitk has the mode, else against the rule applied to unitk's output.
4. index half: the plan kernel's topk_idx (int64) / topk_weights (float32) against prep's, with padding rows.
5. CUDA graph: the plan kernel and linear_mx captured on two streams as the engine runs them, replayed on new inputs:
   equal to eager calls.
6. End to end: a DeepGemmMegaMoEExperts built as KimiMoE builds it (MXFP4, SiTU 4 / 25; 64 rows), a symmetric buffer,
   calls of T = 1..16: the four inputs written by linear_mx + the plan kernel equal prep's bytes, and MegaMoE's
   output equals the stock call's (after checking that two stock calls agree).
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import noprep, unitk, w8a16  # noqa: E402
from k31_engine.noprep import _mx_store  # noqa: E402
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk  # noqa: E402

dev = torch.device("cuda")
E, K, LAT, HID = 896, 16, 3584, 7168
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def nbad(a, b):
    """Differing bytes."""
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def fp8_buf(rows, fill=0x38):
    """A float8_e4m3fn buffer [rows, LAT] filled with one byte (0x38 = 1.0)."""
    return torch.full((rows, LAT), fill, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)


def prep_ref(y, w, rows, pad=None):
    """vLLM's prep on fresh buffers: (x, x_sf, topk_idx, topk_weights) of T rows."""
    T = y.shape[0]
    x = fp8_buf(16)
    xs = torch.full((16, LAT // 128), -1, dtype=torch.int32, device=dev)
    ti = torch.full((16, K), -9, dtype=torch.int64, device=dev)
    tw = torch.full((16, K), -9.0, dtype=torch.float32, device=dev)
    prepare_megamoe_inputs(y, w, rows, x[:T], xs[:T], ti[:T], tw[:T], is_padding=pad)
    return x[:T], xs[:T], ti[:T], tw[:T]


g = torch.Generator(device="cpu").manual_seed(7)
q, s = w8a16.quantize((torch.randn(LAT, HID, generator=g) * 0.02).to(dev))

# ---------------------------------------------------------------------------------------------------- 1. x half
key = f"{LAT}x{HID}"
keep = w8a16._OVR.get(key)
bad = n = 0
for cfg in ((32, 1792, 4, 4, 3), (64, 1792, 4, 4, 3), (128, 3584, 2, 4, 3), (32, 7168, 1, 4, 3)):
    w8a16._OVR[key] = cfg
    for T in range(1, 17):
        for sc in (0.5, 3.0, 30.0):
            h = (torch.randn(T, HID, generator=g) * sc).to(dev, torch.bfloat16)
            if T >= 3:
                h[1] = 0
                h[2] *= 1e-5
            xq = fp8_buf(16, 0x30)
            xs = torch.full((16, LAT // 128), -1, dtype=torch.int32, device=dev)
            y = noprep.linear_mx(h, q, s, xq, xs)
            y_ref = w8a16.linear(h, q, s)
            xr, xsr, _, _ = prep_ref(y_ref, torch.zeros(T, K, device=dev), torch.zeros(T, K, dtype=torch.int32,
                                                                                      device=dev))
            d = nbad(y, y_ref) + nbad(xq[:T], xr) + nbad(xs[:T], xsr)
            n += 1
            if d:
                bad += 1
                if bad <= 3:
                    print(f"  cfg {cfg} T={T} x{sc}: y {nbad(y, y_ref)}, codes {nbad(xq[:T], xr)}, "
                          f"scales {nbad(xs[:T], xsr)} bytes differ")
if keep is None:
    w8a16._OVR.pop(key, None)
else:
    w8a16._OVR[key] = keep
check("1a. x half: latent down + MXFP8 epilogue = w8a16.linear + vLLM's prep", bad == 0, f"({n} calls, {bad} differ)")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _epilogue_only(y_ptr, xq_ptr, xs_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pn = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    out = tl.load(y_ptr + rm[:, None] * N + rn[None, :], mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    _mx_store(out, rm, rn, pn, M, N, xq_ptr, N, xs_ptr, N // 128, 1, BM, BN)


rows = torch.randn(16, LAT, generator=g)
rows[0] = 0
for gi in range(LAT // 32):
    k = gi % 40 - 20                                     # group gi's maximum: 448 x 2^k, or one bf16 step from it
    for r, v in ((1, 448.0 * 2.0 ** k), (3, 448.0 * 2.0 ** k + 2.0 ** (k + 1)), (4, 448.0 * 2.0 ** k - 2.0 ** (k + 1))):
        rows[r, 32 * gi:32 * gi + 32] *= 2.0 ** (k + 2)
        rows[r, 32 * gi + gi % 32] = v
    rows[2, 32 * gi:32 * gi + 32] = -1.75 * 2.0 ** k
rows[5] *= 1e-30
rows[6] *= 1e37
rows[7] = rows[7].abs() * 448.0
rows = rows.to(dev, torch.bfloat16)
for BN in (32, 64, 128):
    xq = fp8_buf(16, 0)
    xs = torch.zeros(16, LAT // 128, dtype=torch.int32, device=dev)
    _epilogue_only[(LAT // BN,)](rows, xq, xs, 16, LAT, BM=16, BN=BN)
    xr, xsr, _, _ = prep_ref(rows, torch.zeros(16, K, device=dev), torch.zeros(16, K, dtype=torch.int32, device=dev))
    check(f"1b. epilogue on the scale's edges (BN {BN}) = vLLM's prep", nbad(xq, xr) + nbad(xs, xsr) == 0,
          f"({nbad(xq, xr)} code bytes, {nbad(xs, xsr)} scale bytes differ)")

# ---------------------------------------------------------------------------------------------------- 2. top-k
bias_r = (torch.randn(E, generator=g) * 0.05).to(dev)
bias_0 = torch.zeros(E, device=dev)
rm0 = torch.full((E,), -1, dtype=torch.int32, device=dev)
bad = n = 0
for trial in range(400):
    T = 1 + trial % 16
    kind = trial % 8
    if kind == 0:
        lg = torch.randn(T, E, generator=g) * 0.5
    elif kind == 1:
        lg = torch.randn(T, E, generator=g) * 2.0
    elif kind == 2:
        lg = torch.randn(T, E, generator=g) * 8.0
    elif kind == 3:
        lg = (torch.randn(T, E, generator=g) * 2.0 / 0.25).round() * 0.25        # ties
    elif kind == 4:
        lg = torch.randn(T, E, generator=g) * 80.0                                 # saturated: equal scores
    elif kind == 5:
        lg = torch.randn(T, E, generator=g) * 2.0
        lg[0, :5] = float("nan")
        lg[0, 5:9] = float("inf")
        lg[-1, ::3] = float("-inf")
    elif kind == 6:
        lg = torch.full((T, E), 1.25)                                              # every logit equal
    else:
        lg = torch.randn(T, E, generator=g) * 3.0
    lg = lg.to(dev).contiguous()
    bias = bias_0 if trial % 5 == 4 else bias_r
    renorm = trial % 3 != 2
    scale = 2.5 if trial % 4 == 1 else 1.0
    w_ref, ids_ref = fused_grouped_topk(lg, lg, K, renorm, bias, 1, 1, "sigmoid", scale)
    ti = torch.zeros(16, K, dtype=torch.int64, device=dev)
    tw = torch.zeros(16, K, dtype=torch.float32, device=dev)
    w, ids, _, _ = noprep.plan_kernel(T, rm0, rm0, 3, 0, 16, ti, tw, logits=lg, bias=bias, renorm=renorm, scale=scale)
    d = nbad(w, w_ref) + nbad(ids, ids_ref)
    n += T
    if d:
        bad += 1
        if bad <= 3:
            print(f"  trial {trial} kind {kind} T={T}: ids differ {int((ids != ids_ref).sum())}, weights differ "
                  f"{int((w.view(torch.int32) != w_ref.view(torch.int32)).sum())}")
check("2. top-k in the plan kernel = vLLM's grouped_topk (ids, weights)", bad == 0, f"({n} rows, {bad} calls differ)")

# ------------------------------------------------------------------------------------------------ 3 + 4. unitk, index
def allrows_ref(rows_off, stage, ids, row_map, bank_map, S):
    """unitk's ALLROWS rule (K31_STAGE_ALL) applied to its rows without it: every valid non-resident pick of an expert
    that holds a slot -> S + that slot; every other pick as it was."""
    inv = {int(j): e for e, j in enumerate(bank_map.tolist()) if j >= 0}
    slot_of = torch.full((E,), -1, dtype=torch.long, device=dev)
    for s_, b in enumerate(stage.tolist()):
        if b >= 0:
            slot_of[inv[b]] = s_
    valid = (ids >= 0) & (ids < E)
    e = torch.where(valid, ids, 0).long()
    hit = valid & (row_map[e] < 0) & (slot_of[e] >= 0)
    return torch.where(hit, S + slot_of[e], rows_off.long()).to(torch.int32)


def counts(rows_, ids, row_map, S, resmax_rank_ok):
    """unitk's pick counts [resident, staged, cut, all] of a call's rows."""
    valid = (ids >= 0) & (ids < E)
    e = torch.where(valid, ids, 0).long()
    native = torch.where(valid, row_map[e], -1)
    n_all = int(valid.sum())
    n_res = int((valid & (native >= 0) & resmax_rank_ok).sum())
    n_stg = int(((rows_ >= S) & (rows_ < S + 16) & (native < 0) & valid).sum())
    return [n_res, n_stg, n_all - n_res - n_stg, n_all]


with_allrows = noprep.unitk_has_allrows()
bad3 = bad4 = n = 0
newly = 0
score_close = True
for trial in range(300):
    T = 1 + trial % 16
    strong = 1 + trial % 4
    resmax = (16, 16, 12, 9)[(trial // 4) % 4]
    S = 16 * (trial % 8)
    perm = torch.randperm(E, generator=g)
    nres, nbank = int(torch.randint(0, 400, (1,), generator=g)), int(torch.randint(0, 450, (1,), generator=g))
    row_map = torch.full((E,), -1, dtype=torch.int32)
    bank_map = torch.full((E,), -1, dtype=torch.int32)
    row_map[perm[:nres]] = torch.arange(128, 128 + nres, dtype=torch.int32)
    bank_map[perm[nres:nres + nbank]] = torch.arange(nbank, dtype=torch.int32)
    row_map, bank_map = row_map.to(dev), bank_map.to(dev)
    lg = torch.randn(T, E, generator=g) * 2.0
    if trial % 3 == 0:                                   # few distinct experts: repeats across rows
        lg[:, 64:] -= 30.0
    lg = lg.to(dev)
    w_ref, ids_ref = fused_grouped_topk(lg, lg, K, True, bias_r, 1, 1, "sigmoid", 1.0)
    if trial % 2:
        w_ref = (w_ref * 8).round() / 8                  # tied weights within a row (ranks by position)
    pad = None if trial % 4 else (torch.arange(T, device=dev) % 3 == 1)
    mode = trial % 3                                     # 0 plain, 1 scores + counts, 2 pending rows
    sc0 = torch.rand(E, generator=g).to(dev)
    lam = 0.97
    rank = torch.argsort(torch.argsort(-w_ref, dim=1, stable=True), dim=1)        # by weight, ties by position
    off_rows = off_stage = None
    for allrows in (False, True):
        outs = []
        for who in ("unitk", "plan", "plan_topk"):
            kw = {}
            if mode:
                kw.update(score=sc0.clone(), stats=torch.zeros(4, dtype=torch.int32, device=dev), lam=lam)
            if mode == 2:
                kw["pend"] = (torch.full((16, 16), -7, dtype=torch.int32, device=dev),
                              torch.full((16, 16), -7.0, device=dev), torch.zeros(1, dtype=torch.int32, device=dev))
            ti = torch.full((16, K), -9, dtype=torch.int64, device=dev)
            tw = torch.full((16, K), -9.0, dtype=torch.float32, device=dev)
            if who == "unitk":
                if allrows and not with_allrows:         # this tree's unitk has no ALLROWS: the rule on its output
                    rows_ = allrows_ref(off_rows, off_stage, ids_ref, row_map, bank_map, S)
                    stage_ = off_stage
                    kw = dict(off_kw)
                    if mode:
                        kw["stats"] = torch.tensor(counts(rows_, ids_ref, row_map, S, rank < resmax), device=dev,
                                                   dtype=torch.int32)
                else:
                    rows_, stage_ = noprep._unitk_plan(ids_ref, w_ref, row_map, bank_map, strong, S, resmax=resmax,
                                                       allrows=allrows, **kw)
                if not allrows:
                    off_rows, off_stage, off_kw = rows_, stage_, kw
            elif who == "plan":
                _, _, rows_, stage_ = noprep.plan_kernel(T, row_map, bank_map, strong, S, resmax, ti, tw, ids=ids_ref,
                                                         w=w_ref, pad=pad, allrows=allrows, **kw)
            else:
                if trial % 2:
                    continue                             # the in-kernel top-k cannot produce the rounded weights
                _, _, rows_, stage_ = noprep.plan_kernel(T, row_map, bank_map, strong, S, resmax, ti, tw, logits=lg,
                                                         bias=bias_r, renorm=True, scale=1.0, pad=pad,
                                                         allrows=allrows, **kw)
            outs.append((who, rows_, stage_, kw, ti, tw))
        ref = outs[0]
        if allrows:
            newly += int((ref[1] != off_rows).sum())
        valid_ids = ids_ref.reshape(-1)
        repeats = valid_ids.numel() != torch.unique(valid_ids).numel()
        for who, rows_, stage_, kw, ti, tw in outs[1:]:
            d = nbad(rows_, ref[1]) + nbad(stage_, ref[2])
            if mode:
                d += int(kw["stats"].tolist() != ref[3]["stats"].tolist())
                if repeats:
                    score_close = score_close and torch.allclose(kw["score"], ref[3]["score"], rtol=1e-6, atol=0)
                else:
                    d += nbad(kw["score"], ref[3]["score"])
            if mode == 2:
                d += nbad(kw["pend"][0][:T], ref[3]["pend"][0][:T]) + \
                    nbad(kw["pend"][1][:T], ref[3]["pend"][1][:T]) + int(int(kw["pend"][2]) != int(ref[3]["pend"][2]))
            n += 1
            if d:
                bad3 += 1
                if bad3 <= 3:
                    print(f"  trial {trial} {who} allrows {allrows} T={T} strong={strong} mode={mode}: rows differ "
                          f"{int((rows_ != ref[1]).sum())}, stage {stage_.tolist()} vs {ref[2].tolist()}")
            _, _, ir, wr = prep_ref(torch.zeros(T, LAT, dtype=torch.bfloat16, device=dev), w_ref, ref[1], pad)
            d4 = nbad(ti[:T], ir) + nbad(tw[:T], wr)
            if d4:
                bad4 += 1
                if bad4 <= 3:
                    print(f"  trial {trial} {who} T={T}: index half differs ({d4} bytes; padding {pad is not None})")
check("3. unitk in the plan kernel = unitk.plan (rows, stage, counts, pending rows, scores), ALLROWS off and on",
      bad3 == 0 and score_close and newly > 0,
      f"({n} calls, {bad3} differ; scores with repeats within 1e-6: {score_close}; ALLROWS reference "
      f"{'unitk.plan(allrows=True)' if with_allrows else 'the rule on unitk.plan'}, {newly} picks newly on a slot)")
check("4. index half (topk_idx int64, topk_weights float32, padding) = vLLM's prep", bad4 == 0, f"({bad4} differ)")

# ---------------------------------------------------------------------------------------------------- 5. graphs
T = 5
h_in = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
lg_in = torch.zeros(T, E, device=dev)
row_map = torch.full((E,), -1, dtype=torch.int32, device=dev)
row_map[:300] = torch.arange(128, 428, dtype=torch.int32, device=dev)
bank_map = torch.full((E,), -1, dtype=torch.int32, device=dev)
bank_map[300:700] = torch.arange(400, dtype=torch.int32, device=dev)
score = torch.rand(E, device=dev)
stats = torch.zeros(4, dtype=torch.int32, device=dev)
xq_g = fp8_buf(16, 0)
xs_g = torch.zeros(16, LAT // 128, dtype=torch.int32, device=dev)
ti_g = torch.zeros(16, K, dtype=torch.int64, device=dev)
tw_g = torch.zeros(16, K, dtype=torch.float32, device=dev)
side = torch.cuda.Stream()


def step(score_t, stats_t):
    main = torch.cuda.current_stream()
    ev0 = torch.cuda.Event()
    ev0.record(main)
    out = noprep.plan_kernel(T, row_map, bank_map, 3, 16, 16, ti_g, tw_g, logits=lg_in, bias=bias_r, renorm=True,
                             scale=1.0, score=score_t, lam=0.97, stats=stats_t)
    with torch.cuda.stream(side):
        side.wait_event(ev0)
        y = noprep.linear_mx(h_in, q, s, xq_g, xs_g)
        ev1 = torch.cuda.Event()
        ev1.record(side)
    main.wait_event(ev1)
    return out + (y,)


for _ in range(2):                                   # warm-up (compiles) outside the capture
    step(score.clone(), stats.clone())
torch.cuda.synchronize()
sc_g, st_g = score.clone(), torch.zeros(4, dtype=torch.int32, device=dev)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    gout = step(sc_g, st_g)
same = True
for it in range(4):
    h_new = (torch.randn(T, HID, generator=g) * 2).to(dev, torch.bfloat16)
    lg_new = (torch.randn(T, E, generator=g) * 2).to(dev)
    h_in.copy_(h_new)
    lg_in.copy_(lg_new)
    sc_g.copy_(score)
    st_g.zero_()
    graph.replay()
    torch.cuda.synchronize()
    snap = [t.clone() for t in gout] + [xq_g[:T].clone(), xs_g[:T].clone(), ti_g[:T].clone(), tw_g[:T].clone(),
                                        sc_g.clone(), st_g.clone()]
    sc_e, st_e = score.clone(), torch.zeros(4, dtype=torch.int32, device=dev)
    eout = step(sc_e, st_e)
    torch.cuda.synchronize()
    eager = list(eout) + [xq_g[:T], xs_g[:T], ti_g[:T], tw_g[:T], sc_e, st_e]
    for a, b in zip(snap, eager):
        if a.dtype == torch.float32 and a.numel() == E:      # scores: atomics' order
            same = same and torch.allclose(a, b, rtol=1e-6, atol=0)
        else:
            same = same and nbad(a, b) == 0
check("5. CUDA graph replays (plan kernel + latent down on two streams) = eager calls", same)

# timing of the plan kernel alone (an idle GB300 only)
for Tt in (1, 5, 8, 16):
    lg_t = torch.randn(Tt, E, device=dev)
    args = (Tt, row_map, bank_map, 3, 16, 16, ti_g, tw_g)
    ids_t = torch.randint(0, E, (Tt, K), device=dev, dtype=torch.int32)
    w_t = torch.rand(Tt, K, device=dev)
    noprep.plan_kernel(*args, logits=lg_t, bias=bias_r, renorm=True, scale=1.0)      # warm-ups outside capture
    unitk.plan(ids_t, w_t, row_map, bank_map, 3, 16)
    fused_grouped_topk(lg_t, lg_t, K, True, bias_r, 1, 1, "sigmoid", 1.0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(20):
            noprep.plan_kernel(*args, logits=lg_t, bias=bias_r, renorm=True, scale=1.0)
    gr2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr2):
        for _ in range(20):
            unitk.plan(ids_t, w_t, row_map, bank_map, 3, 16)
    gr3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr3):
        for _ in range(20):
            fused_grouped_topk(lg_t, lg_t, K, True, bias_r, 1, 1, "sigmoid", 1.0)
    res = []
    for gx in (gr, gr2, gr3):
        best = 1e9
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gx.replay()
            b.record()
            b.synchronize()
            best = min(best, a.elapsed_time(b) * 1e3 / 20)
        res.append(best)
    print(f"  T={Tt}: plan kernel with top-k {res[0]:.1f} us per call; unitk {res[1]:.1f}, vLLM top-k {res[2]:.1f} "
          f"(back to back in a graph)", flush=True)

# ---------------------------------------------------------------------------------------------------- 6. end to end
try:
    import torch.distributed as dist
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (MEGA_MOE_MXFP4, DeepGemmMegaMoEExperts,
                                                                         MegaMoEActivation)
    from vllm.utils.deep_gemm import _import_deep_gemm
    dg = _import_deep_gemm()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29633")
    dist.init_process_group("nccl", rank=0, world_size=1)
    group = dist.group.WORLD
    NR, TMAX, I = 64, 16, 3072
    cfgv = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=TMAX),
                                 compilation_config=types.SimpleNamespace(static_forward_context={}))
    with torch.device(dev):
        mod = DeepGemmMegaMoEExperts(cfgv, num_experts=NR, num_local_experts=NR, experts_start_idx=0, top_k=K,
                                     hidden_size=LAT, intermediate_size=I,
                                     activation=MegaMoEActivation("situ", alpha=4.0, beta=25.0),
                                     weight_format=MEGA_MOE_MXFP4, max_num_tokens=TMAX, prefix="noprep.test.experts")
    gd = torch.Generator(device=dev).manual_seed(3)
    params = dict(mod.named_parameters())
    for e in range(NR):
        for shard, pp, r_, c_ in (("w1", "w13", I, LAT), ("w3", "w13", I, LAT), ("w2", "w2", LAT, I)):
            wq = torch.randint(0, 256, (r_, c_ // 2), dtype=torch.uint8, device=dev, generator=gd)
            sq = torch.randint(119, 124, (r_, c_ // 32), dtype=torch.uint8, device=dev, generator=gd)
            for suffix, t in (("weight", wq), ("weight_scale", sq)):
                p = params[f"{pp}_{suffix}"]
                p.weight_loader(p, t, f"layers.0.block_sparse_moe.experts.{pp}_{suffix}", expert_id=e, shard_id=shard)
    symm = dg.get_symm_buffer_for_mega_moe(group, NR, TMAX, K, LAT, I, activation="situ", mma_type="fp8xfp4")
    kw = dict(mod._kernel_kwargs)
    # rows: 16 staging slots (S = 0) then 48 resident experts; the bank holds 300 others
    row_map = torch.full((E,), -1, dtype=torch.int32, device=dev)
    row_map[:48] = torch.arange(16, 64, dtype=torch.int32, device=dev)
    bank_map = torch.full((E,), -1, dtype=torch.int32, device=dev)
    bank_map[48:348] = torch.arange(300, dtype=torch.int32, device=dev)
    det = True
    bad6 = 0
    for T in range(1, 17):
        h = (torch.randn(T, HID, generator=g) * 2).to(dev, torch.bfloat16)
        lg = (torch.randn(T, E, generator=g) * 2).to(dev)
        lg[:, :48] += 2.0                                # most picks resident, some staged, some cut
        pad = (torch.arange(T, device=dev) == T - 1) if T > 2 else None
        # stock: w8a16 latent down, vLLM's top-k, unitk, prep, MegaMoE
        y_lat = w8a16.linear(h, q, s)
        w_ref, ids_ref = fused_grouped_topk(lg, lg, K, True, bias_r, 1, 1, "sigmoid", 1.0)
        rows_ref, _ = unitk.plan(ids_ref, w_ref, row_map, bank_map, 3, 0)
        prepare_megamoe_inputs(y_lat, w_ref, rows_ref, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T],
                               symm.topk_weights[:T], is_padding=pad)
        snap = [symm.x[:T].clone(), symm.x_sf[:T].clone(), symm.topk_idx[:T].clone(), symm.topk_weights[:T].clone()]
        out_ref = torch.empty(T, LAT, dtype=torch.bfloat16, device=dev)
        dg.fp8_fp4_mega_moe(out_ref, mod.l1_weights, mod.l2_weights, symm, fast_math=mod.fast_math, **kw)
        out_ref2 = torch.empty_like(out_ref)
        prepare_megamoe_inputs(y_lat, w_ref, rows_ref, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T],
                               symm.topk_weights[:T], is_padding=pad)
        dg.fp8_fp4_mega_moe(out_ref2, mod.l1_weights, mod.l2_weights, symm, fast_math=mod.fast_math, **kw)
        det = det and nbad(out_ref, out_ref2) == 0
        # no prep: scribble over the inputs, then linear_mx + the plan kernel write them, then MegaMoE
        symm.x[:16].view(torch.uint8).fill_(0x38)
        symm.x_sf[:16].fill_(0x7f7f7f7f)
        symm.topk_idx[:16].fill_(3)
        symm.topk_weights[:16].fill_(0.25)
        y2 = noprep.linear_mx(h, q, s, symm.x, symm.x_sf)
        _, _, rows2, _ = noprep.plan_kernel(T, row_map, bank_map, 3, 0, 16, symm.topk_idx, symm.topk_weights,
                                            logits=lg, bias=bias_r, renorm=True, scale=1.0, pad=pad)
        got = [symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T]]
        d_in = sum(nbad(a, b) for a, b in zip(got, snap)) + nbad(y2, y_lat) + nbad(rows2, rows_ref)
        out = torch.empty(T, LAT, dtype=torch.bfloat16, device=dev)
        dg.fp8_fp4_mega_moe(out, mod.l1_weights, mod.l2_weights, symm, fast_math=mod.fast_math, **kw)
        torch.cuda.synchronize()
        d_out = nbad(out, out_ref)
        if d_in or (det and d_out):
            bad6 += 1
            if bad6 <= 3:
                print(f"  T={T}: input bytes differ {d_in}, output bytes differ {d_out}")
    check("6. end to end: MegaMoE's four inputs = prep's bytes, and its output = the stock call's", bad6 == 0,
          f"(T 1..16; stock calls deterministic: {det}{'' if det else ' (outputs not compared)'})")
    dist.destroy_process_group()
except Exception as ex:                                  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("6. end to end (MegaMoE)", False, f"({type(ex).__name__}: {ex})")

print(f"NOPREP: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}", flush=True)
sys.exit(1 if fails else 0)
