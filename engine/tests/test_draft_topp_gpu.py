"""K31_DRAFT_TOPP / K31_DRAFT_ARGMAX on the GB300: drafttopp.py with vLLM's own draft sampling and verifier kernels.

Run from engine/ (GB = the GB300's UUID; a few minutes, mostly the CPU reference):
    docker run --rm --device nvidia.com/gpu=$GB --network none --cpuset-cpus 50-51 --user $(id -u):$(id -g) \
        -e HOME=/tmp -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /engine/tests/test_draft_topp_gpu.py
Exit 0 = PASS (parts 1-3); part 4 prints the timing and a TIMING line (target: masking < 10 us per draft position).
DTOPP_CPU=1 (with TRITON_INTERPRET=1 and an empty CUDA_VISIBLE_DEVICES) runs parts 1 and 3 on the CPU at smaller
sizes, under Triton's interpreter, to check this script itself.

1. The kernels against drafttopp.reference (torch on the CPU) on 163,840-wide bf16 rows of mixed requests: untouched
   rows (top_p 1 and no top-k, temperature 0, padding) bit-identical; masks equal (one row in 32 may differ by one bin
   where a cumulative mass is within float rounding of top_p x Z: the GPU's exp is not the CPU's).
2. One draft position as the DSpark speculator runs it (a fresh logits row, the mask, vLLM's gumbel_sample writing the
   draft logits cache) captured in a CUDA graph at 1, 4 and 8 rows, both modes, replayed on new logits, requests,
   seeds and positions: drafts, cache rows and masked rows equal eager bit for bit.
3. What the verifier reads: (a) every cache row = its masked row bit for bit and every draft lies in the kept set;
   (b) vLLM's rejection_sample (standard, one draft, 4,096 rows) with the target = the drafter's UNMASKED
   distribution accepts with probability sum_x min(p, q') = the kept mass Z_q (it would accept every draft if it
   read the unmasked q); (c) block verification over 4 drafts with the target = the masked q accepts all 4 in every
   row (any disagreement between the sampled and the verified q rejects some).
4. Timing in CUDA graphs (20 positions per graph, 50 replays) at 1, 4, 8 rows: the masking (stats + mask) per draft
   position, vLLM's gumbel_sample alone, and both; argmax mode; the per-step refresh of top_p / top_k.
"""
import os
import sys

import torch

CPU = os.environ.get("DTOPP_CPU", "0") == "1"
if CPU:                                         # Triton's interpreter has no libdevice log1p (the verifier's residual)
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as libdevice

    @triton.jit
    def _log1p(x):
        return tl.log(1.0 + x)
    libdevice.log1p = _log1p
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import drafttopp as dt  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402

dev = torch.device("cpu" if CPU else "cuda")
V = 163840
fails = []
g = torch.Generator().manual_seed(23)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def llm_rows(n, v=V):
    """bf16 rows shaped like LLM next-token logits: a few strong tokens over a broad bulk, varied sharpness."""
    x = torch.randn(n, v, generator=g) * (1.0 + 2.0 * torch.rand(n, 1, generator=g))
    top = torch.randint(0, v, (n, 16), generator=g)
    x.scatter_(1, top, 6.0 + 10.0 * torch.rand(n, 16, generator=g))
    return x.to(torch.bfloat16)


def requests(n_slots):
    """Request slots: temperature, top_p, top_k (mixed: untouched, top-p, top-k, both, greedy)."""
    T = torch.tensor([1.0, 1.0, 0.7, 1.0, 0.0, 1.3, 1.0, 0.9] * ((n_slots + 7) // 8))[:n_slots]
    P = torch.tensor([0.95, 1.0, 0.9, 0.95, 0.95, 0.8, 0.5, 1.0] * ((n_slots + 7) // 8))[:n_slots]
    K = torch.tensor([V, V, V, 50, V, V, 20, 1000] * ((n_slots + 7) // 8), dtype=torch.int32)[:n_slots]
    return T, P, K


# 1. kernels against the reference
diff_rows, far_rows, untouched_ok, n_rows = 0, 0, True, 0
for batch in range(2 if CPU else 8):
    R = 8
    x = llm_rows(R)
    T, P, K = requests(8)
    idx = torch.randperm(8, generator=g)[:R].to(torch.int64)
    if batch % 2:
        idx[3] = -1
    xg = x.clone().to(dev)                     # (a copy on the CPU too: the reference needs the unmasked rows)
    dt.apply(xg, idx.to(dev), T.to(dev), P.to(dev), K.to(dev), dt.Scratch(8, V, dev))
    y = xg.cpu()
    ref, info = dt.reference(x, idx, T, P, K)
    for r in range(R):
        n_rows += 1
        slot = int(idx[r])
        touched = slot >= 0 and float(T[slot]) > 0 and (float(P[slot]) < 1 or int(K[slot]) < V)
        if not touched:
            untouched_ok &= torch.equal(y[r].view(torch.int16), x[r].view(torch.int16))
            continue
        if not torch.equal(y[r].view(torch.int16), ref[r].view(torch.int16)):
            diff_rows += 1
            yv = x[r].float() * (torch.tensor(1.0) / T[slot])
            ba = torch.floor(yv[torch.isfinite(y[r].float())].min() * dt.INV_W)
            br = torch.floor(yv[torch.isfinite(ref[r].float())].min() * dt.INV_W)
            far_rows += int(abs(float(ba - br)) > 1)
check("untouched rows bit-identical on the GPU", untouched_ok)
check("GPU masks = reference (<= 1 in 32 rows one bin apart at a float tie)",
      far_rows == 0 and diff_rows * 32 <= n_rows, f"({diff_rows} of {n_rows} rows differ)")


# 2. one draft position in a CUDA graph = eager
def position(src, idx, temp, topp, topk, seeds, pos, cache, col, scratch, argmax):
    logits = src * 1                           # a fresh row, as base_logits[:, i] + bias in the speculator's loop
    dt.apply(logits, idx, temp, topp, topk, scratch, argmax=argmax)
    d = gumbel_sample(logits, idx, temp, seeds, pos, apply_temperature=True, is_drafting=True,
                      logits_cache=cache.values, logits_cache_col=col)
    return logits, d


for argmax in (() if CPU else (False, True)):
    for R in (1, 4, 8):
        slots = 8
        st = {"src": llm_rows(R).to(dev), "idx": torch.arange(R, dtype=torch.int64, device=dev)}
        T, P, K = requests(slots)
        st.update(temp=T.to(dev), topp=P.to(dev), topk=K.to(dev),
                  seeds=torch.randint(-2 ** 62, 2 ** 62, (slots,), generator=g, dtype=torch.int64).to(dev),
                  pos=(torch.arange(R, dtype=torch.int64) * 101 + 500).to(dev),
                  col=torch.tensor(1, dtype=torch.int32, device=dev))
        cache = DraftLogitsCache(slots, 2, V, dtype=torch.bfloat16, fill=0.0, device=dev)
        scratch = dt.Scratch(slots, V, dev)
        args = lambda: (st["src"], st["idx"], st["temp"], st["topp"], st["topk"], st["seeds"], st["pos"],  # noqa
                        cache, st["col"], scratch, argmax)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                position(*args())
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_logits, g_d = position(*args())
        ok = True
        for trial in range(3):
            st["src"].copy_(llm_rows(R).to(dev))
            st["idx"].copy_(torch.randperm(slots, generator=g)[:R].to(dev))
            T, P, K = requests(slots)
            perm = torch.randperm(slots, generator=g)
            st["temp"].copy_(T[perm].to(dev)), st["topp"].copy_(P[perm].to(dev)), st["topk"].copy_(K[perm].to(dev))
            st["seeds"].copy_(torch.randint(-2 ** 62, 2 ** 62, (slots,), generator=g, dtype=torch.int64).to(dev))
            st["pos"].add_(7)
            cache.values.zero_()
            graph.replay()
            torch.cuda.synchronize()
            gl, gd, gc = g_logits.clone(), g_d.clone(), cache.values[:, 1].clone()
            cache.values.zero_()
            el, ed = position(*args())
            torch.cuda.synchronize()
            ok &= torch.equal(gl.view(torch.int16), el.view(torch.int16)) and torch.equal(gd, ed) and \
                torch.equal(gc.view(torch.int16), cache.values[:, 1].view(torch.int16))
        check(f"graph replay = eager at {R} rows ({'argmax' if argmax else 'top-k/top-p'}): masked rows, drafts, cache",
              ok)

# 3a. cache = masked rows; drafts in the kept set (8 rows, two positions)
slots, R, NS = 8, 8, 2
T, P, K = requests(slots)
temp, topp, topk = T.to(dev), P.to(dev), K.to(dev)
seeds = torch.randint(-2 ** 62, 2 ** 62, (slots,), generator=g, dtype=torch.int64).to(dev)
cache = DraftLogitsCache(slots, NS, V, dtype=torch.bfloat16, fill=0.0, device=dev)
scratch = dt.Scratch(slots, V, dev)
idx = torch.randperm(slots, generator=g).to(dev)
ok = True
for col in range(NS):
    logits, d = position(llm_rows(R).to(dev), idx, temp, topp, topk, seeds, torch.arange(R, device=dev) + 900 + col,
                         cache, torch.tensor(col, dtype=torch.int32, device=dev), scratch, False)
    for r in range(R):
        ok &= torch.equal(cache.values[int(idx[r]), col].view(torch.int16), logits[r].view(torch.int16))
        ok &= bool(torch.isfinite(logits[r, d[r]].float()))
check("the draft logits cache holds each masked row bit for bit; every draft lies in its kept set", ok)


# 3b. standard verification, one draft, target = the unmasked drafter distribution: acceptance = kept mass
def verify_batch(B, v, top_p, block, n_drafts, target_masked):
    xs = [llm_rows(B, v).to(dev) for _ in range(n_drafts)]
    temp = torch.ones(B, device=dev)
    topp = torch.full((B,), top_p, device=dev)
    topk = torch.full((B,), v, dtype=torch.int32, device=dev)
    seeds = torch.randint(-2 ** 62, 2 ** 62, (B,), generator=g, dtype=torch.int64).to(dev)
    cache = DraftLogitsCache(B, n_drafts, v, dtype=torch.bfloat16, fill=0.0, device=dev)
    scratch = dt.Scratch(B, v, dev)
    idx = torch.arange(B, dtype=torch.int64, device=dev)
    drafts, kept_mass, masked = [], [], []
    for i in range(n_drafts):
        lg, d = position(xs[i], idx, temp, topp, topk, seeds, torch.full((B,), 300 + i, device=dev, dtype=torch.int64),
                         cache, torch.tensor(i, dtype=torch.int32, device=dev), scratch, False)
        drafts.append(d)
        masked.append(lg)
        q = torch.softmax(xs[i].float(), 1)
        kept_mass.append((q * torch.isfinite(lg.float())).sum(1))
    nl = n_drafts + 1
    tgt = torch.empty(B, nl, v, device=dev)
    for i in range(n_drafts):
        tgt[:, i] = masked[i].float() if target_masked else xs[i].float()
    tgt[:, n_drafts] = llm_rows(B, v).to(dev).float()
    ds = torch.zeros(B, nl, dtype=torch.int64, device=dev)
    ds[:, 1:] = torch.stack(drafts, 1)
    cu = torch.arange(0, B * nl + 1, nl, dtype=torch.int32, device=dev)
    pos = (torch.arange(B * nl, device=dev) % nl + 299).to(torch.int64)
    exp_idx = torch.repeat_interleave(idx, nl)
    loc = (torch.arange(B * nl, device=dev) % nl).to(torch.int32)
    _, num = rejection_sample(tgt.view(B * nl, v), cache, ds.view(-1), cu, pos, idx, exp_idx, loc, temp, seeds,
                              n_drafts, use_block_verification=block)
    return (num.long() - 1).cpu(), torch.stack(kept_mass, 1).cpu()


acc, zq = [], []
for _ in range(4):
    a, z = verify_batch(128 if CPU else 1024, 8192 if CPU else 32768, 0.9, False, 1, False)
    acc.append(a.float()), zq.append(z[:, 0])
acc, zq = torch.cat(acc), torch.cat(zq)
m, e = float(acc.mean()), float(zq.mean())
sd = float((zq * (1 - zq)).mean().sqrt()) / len(acc) ** 0.5
check("standard verifier reads the masked q: acceptance = mean kept mass Z_q", abs(m - e) < 4 * sd + 1e-3 and m < 0.99,
      f"(accepted {m:.4f}, mean Z_q {e:.4f} +- {sd:.4f}; reading the unmasked q would give 1)")
a, _ = verify_batch(32 if CPU else 1024, 8192 if CPU else 32768, 0.9, True, 4, True)
check("block verification, target = masked q, 4 drafts: all accepted in every row", bool((a == 4).all()),
      f"(accepted lengths {torch.bincount(a, minlength=5).tolist()})")


if CPU:
    print("DRAFT TOPP GPU (CPU dry run, parts 1 and 3):", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
    sys.exit(1 if fails else 0)


# 4. timing
def graph_time(fn, reps=20, replays=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(reps):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(replays):
        gr.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (reps * replays)


slots = 8
T, P, K = requests(slots)
temp = torch.ones(slots, device=dev)
topp = torch.full((slots,), 0.95, device=dev)
topk = torch.full((slots,), V, dtype=torch.int32, device=dev)
seeds = torch.randint(-2 ** 62, 2 ** 62, (slots,), generator=g, dtype=torch.int64).to(dev)
cache = DraftLogitsCache(slots, 4, V, dtype=torch.bfloat16, fill=0.0, device=dev)
scratch = dt.Scratch(slots, V, dev)
col = torch.tensor(0, dtype=torch.int32, device=dev)
worst = 0.0
print("rows | copy   mask   argmax  gumbel  mask+gumbel  (us per draft position, mask = stats + mask kernels)")
for R in (1, 4, 8):
    src = llm_rows(R).to(dev)
    buf = torch.empty_like(src)
    idx = torch.arange(R, dtype=torch.int64, device=dev)
    pos = torch.arange(R, dtype=torch.int64, device=dev) + 77
    t_copy = graph_time(lambda: buf.copy_(src))
    t_mask = graph_time(lambda: (buf.copy_(src), dt.apply(buf, idx, temp, topp, topk, scratch)))
    t_arg = graph_time(lambda: (buf.copy_(src), dt.apply(buf, idx, temp, topp, topk, scratch, argmax=True)))
    t_gum = graph_time(lambda: (buf.copy_(src), gumbel_sample(buf, idx, temp, seeds, pos, True, True,
                                                              logits_cache=cache.values, logits_cache_col=col)))
    t_both = graph_time(lambda: (buf.copy_(src), dt.apply(buf, idx, temp, topp, topk, scratch),
                                 gumbel_sample(buf, idx, temp, seeds, pos, True, True, logits_cache=cache.values,
                                               logits_cache_col=col)))
    print(f"{R:4d} | {t_copy:5.1f}  {t_mask - t_copy:5.1f}  {t_arg - t_copy:5.1f}   {t_gum - t_copy:5.1f}   "
          f"{t_both - t_copy:5.1f}")
    worst = max(worst, t_mask - t_copy)
ss = type("SS", (), {})()
ss.top_p = type("U", (), {"gpu": topp})()
ss.top_k = type("U", (), {"gpu": topk})()
st = dt.State(type("Spec", (), {"device": dev, "max_num_reqs": slots, "temperature": temp, "vocab_size": V})(), ss,
              False)
print(f"per-step refresh of top_p / top_k (2 copies): {graph_time(st.refresh):.1f} us")
print(f"TIMING: masking {worst:.1f} us per draft position at <= 8 rows (target < 10): "
      f"{'met' if worst < 10 else 'NOT met'}")

print("DRAFT TOPP GPU:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
