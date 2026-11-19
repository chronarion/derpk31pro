"""dsa_fused (K31_DSA_FUSED) on the GB300: each fused kernel against the vLLM calls it replaces, bytes equal.

Run (GB300 only, no other GPU work; from the repo root; GB = the GB300's UUID):
    docker run --rm --network none --device nvidia.com/gpu=$GB --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD/engine":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_dsa_fused.py
Exit 0 and "DSA FUSED GPU: PASS" = every strict check equal. Lines starting "INFO" are reported, not judged.

1. Part A + M (dsa_fused.prologue) against vLLM's indexer_k_norm_quant_insert, indexer_q_quant_fold_weights,
   mqa_logits_mx_inputs (the op's BF16 weights) and ops.concat_and_cache_mla(kv_cache_dtype "fp8", k_scale 1,
   skip_scale), at 1-64 rows: index cache, top-k buffer, MLA cache (whole buffers, strided page views with spare
   bytes), FP8 query and BF16 weights. Strict on finite data incl. |x| > 448; INFO for +-inf / NaN rows.
2. Part B (dsa_fused.qpad) against q_pe copied into mqa_q_cat + ops.scaled_fp8_quant(per tensor, scale 1, as
   QuantFP8.forward_cuda calls it) + _pad_query_heads (96 -> 128 heads): bytes of the padded FP8 query. Strict on
   finite data incl. |x| > 448; INFO for +-inf / NaN.
3. Part C (dsa_fused.gate) against attn_out * gate.sigmoid(): every BF16 gate bit pattern x several attn_out
   distributions (normal, powers of two, tiny values whose products are subnormal, zeros), strict, NaN payloads
   included; and the production shape [T, 12288].
4. CUDA graphs: each launcher captured, inputs changed in place, replayed: equal to eager.
5. INFO timing in CUDA graphs at 5 rows: vLLM's kernels against the fused one, per part (meaningful only on an
   otherwise idle GB300).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from vllm import _custom_ops as ops                                             # noqa: E402
from vllm.models.kimi_k3.nvidia.ops import dsa_indexer as dix                   # noqa: E402
from vllm.utils.deep_gemm import mqa_logits_mx_inputs, mqa_logits_use_mx_scales  # noqa: E402
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fis         # noqa: E402

from k31_engine import dsa_fused as F                                           # noqa: E402

dev = torch.device("cuda")
FP8 = torch.float8_e4m3fn
bf16 = torch.bfloat16
fails = []
H, D, TOPK, BS, MBS, NH, L, R = 32, 128, 2048, 64, 64, 96, 512, 64
WS = D ** -0.5 * H ** -0.5
ONE = torch.tensor(1.0, device=dev)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def info(name, detail):
    print(f"INFO {name} {detail}", flush=True)


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def ndiff(a, b):
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def strided_cache(g, blocks, block, width, pad):
    buf = torch.randint(0, 256, (blocks, block * width + pad), generator=g, device=dev, dtype=torch.int32)
    buf = buf.to(torch.uint8)
    return buf.as_strided((blocks, block, width), (block * width + pad, width, 1)), buf


def wide(g, *shape, scale=3.0):
    """BF16 values over a wide range: normal * scale, with ~2% of entries times 2^(4..10) (beyond E4M3's 448)."""
    x = torch.randn(*shape, generator=g, device=dev) * scale
    big = torch.rand(*shape, generator=g, device=dev) < 0.02
    x = torch.where(big, x * 2.0 ** torch.randint(4, 11, shape, generator=g, device=dev).float(), x)
    return x.to(bf16)


def specials(x):
    """Entries at both ends of row 0 set to +-inf / NaN."""
    x = x.clone()
    flat = x[0].view(-1)
    for i, v in ((0, "inf"), (1, "-inf"), (2, "nan"), (7, "nan"), (-1, "inf"), (-2, "-inf"), (-3, "nan")):
        flat[i] = float(v)
    return x


assert torch.cuda.get_device_capability() == (10, 3), "run on the GB300 (sm_103): --device nvidia.com/gpu=$GB"
assert mqa_logits_use_mx_scales(), "the GB300 runs the MQA logits with MX scales (SM100 family)"
g = torch.Generator(device=dev).manual_seed(11)

# ---------------------------------------------------------------------------------------------------------------------
# 1. A + M
# ---------------------------------------------------------------------------------------------------------------------


def stock_am(index_k, kn_w, kn_b, islot, icache, topk, index_q, index_w, kv, k_pe, mslot, mcache, T):
    dix.indexer_k_norm_quant_insert(index_k, kn_w, kn_b, 1e-6, islot, icache, topk, T)
    q8, w32 = dix.indexer_q_quant_fold_weights(index_q, index_w, WS, H)
    _, wb = mqa_logits_mx_inputs(q8, None, w32)
    ops.concat_and_cache_mla(kv, k_pe, mcache, mslot.flatten(), kv_cache_dtype="fp8", scale=ONE, skip_scale=True)
    return q8, wb


def case_am(T, nslots, special):
    wkv = wide(g, T, D + H, scale=2.0)
    index_k, index_w = wkv[:, :D], wkv[:, D:]
    qs = torch.exp(torch.randn(T, H, 1, generator=g, device=dev) * 3)
    index_q = (torch.randn(T, H, D, generator=g, device=dev) * qs).to(bf16)
    index_q[0, 3] = 0
    kn_w = torch.randn(D, generator=g, device=dev)
    kn_b = torch.randn(D, generator=g, device=dev) * 0.1
    islot = torch.randperm(8 * BS, generator=g, device=dev)[:nslots].long()
    mslot = torch.randperm(8 * MBS, generator=g, device=dev)[:nslots].long()
    if nslots > 2:
        islot[1], mslot[-1] = -1, -1
    kv = wide(g, T, L)
    qkv = wide(g, T, 1536 + L + R)
    if special:
        kv, qkv = specials(kv), specials(qkv)
    k_pe = qkv[:, 1536 + L:]
    icache, ibuf = strided_cache(g, 8, BS, D + 4, 256)
    mcache, mbuf = strided_cache(g, 8, MBS, L + R, 512)
    topk = torch.randint(0, 1 << 20, (80, TOPK), generator=g, device=dev, dtype=torch.int32)
    ibuf_r, mbuf_r, topk_r = ibuf.clone(), mbuf.clone(), topk.clone()
    q_r, w_r = stock_am(index_k, kn_w, kn_b, islot, ibuf_r.as_strided(icache.shape, icache.stride()), topk_r, index_q,
                        index_w, kv, k_pe, mslot, mbuf_r.as_strided(mcache.shape, mcache.stride()), T)
    q_f, w_f = F.prologue(index_k, kn_w, kn_b, 1e-6, islot, icache, topk, index_q, index_w, WS, H, True, kv, k_pe,
                          mslot, mcache)
    torch.cuda.synchronize()
    a_ok = same(ibuf, ibuf_r) and torch.equal(topk, topk_r) and same(q_f, q_r) and same(w_f, w_r)
    m_ok = same(mbuf, mbuf_r)
    detail = (f"(index cache {ndiff(ibuf, ibuf_r)}, top-k {int((topk != topk_r).sum())}, query {ndiff(q_f, q_r)}, "
              f"weights {ndiff(w_f, w_r)}, MLA cache {ndiff(mbuf, mbuf_r)} bytes differ)")
    return a_ok, m_ok, detail


for T, nslots in ((1, 1), (2, 2), (5, 5), (5, 3), (8, 8), (10, 10), (16, 16), (20, 20), (40, 37), (64, 64)):
    ok_a, ok_m, det = case_am(T, nslots, False)
    check(f"A+M T={T} slots={nslots}: index cache, top-k, FP8 query, BF16 weights / MLA cache", ok_a and ok_m, det)
for T in (1, 5):
    ok_a, ok_m, det = case_am(T, T, True)
    info(f"A+M T={T} with +-inf / NaN in kv_c_normed and k_pe", f"index part equal {ok_a}, MLA cache equal {ok_m} "
         f"{det}")

# ---------------------------------------------------------------------------------------------------------------------
# 2. B
# ---------------------------------------------------------------------------------------------------------------------


def stock_b(cat, q_pe, inplace=False):
    """MLAAttention.forward_impl's q_pe copy into mqa_q_cat, the static FP8 quant, the backend's head pad."""
    T = cat.shape[0]
    c = cat if inplace else cat.clone()
    c[..., L:].copy_(q_pe)
    q8, _ = ops.scaled_fp8_quant(c.reshape(T, -1), ONE, group_shape=(-1, -1))
    return fis._pad_query_heads(q8.view(T, NH, L + R), fis.trtllm_gen_query_heads(NH))


def case_b(T, special):
    cat = wide(g, T, NH, L + R, scale=40.0)
    q = wide(g, T, NH, 192, scale=40.0)
    if special:
        cat, q = specials(cat), specials(q)
        q[0, 0, 130], q[0, 1, 140], q[0, 2, 150] = float("inf"), float("-inf"), float("nan")
    q_pe = q[..., 128:]
    ref = stock_b(cat, q_pe)
    out = F.qpad(cat, q_pe, L, fis.trtllm_gen_query_heads(NH))
    torch.cuda.synchronize()
    return same(out, ref), ndiff(out, ref)


for T in (1, 2, 5, 8, 10, 16, 20, 40, 64):
    ok, nd = case_b(T, False)
    check(f"B T={T}: padded FP8 query", ok, f"({nd} bytes differ)")
ok, nd = case_b(5, True)
info("B T=5 with +-inf / NaN in the query", f"equal {ok} ({nd} bytes differ)")

# ---------------------------------------------------------------------------------------------------------------------
# 3. C
# ---------------------------------------------------------------------------------------------------------------------
lut = F.sigmoid_table(dev)
gall = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.int16)).to(dev).view(bf16).view(16, 4096)
dists = {
    "normal": (torch.randn(16, 4096, generator=g, device=dev) * 2).to(bf16),
    "powers of two": (2.0 ** torch.randint(-20, 21, (16, 4096), generator=g, device=dev).float()).to(bf16),
    "tiny (subnormal products)": (torch.randn(16, 4096, generator=g, device=dev) * 1e-30).to(bf16),
    "zeros and signs": torch.where(torch.rand(16, 4096, generator=g, device=dev) < 0.5, 0.0, -0.0).to(bf16),
}
def c_diff(ref, out):
    """(entries whose bits differ and that are not NaN in both, entries NaN in both with different payloads)"""
    rb, ob = ref.view(torch.int16), out.view(torch.int16)
    nan = torch.isnan(ref) & torch.isnan(out)
    return int(((rb != ob) & ~nan).sum()), int(((rb != ob) & nan).sum())


nan_payload = 0
for name, a in dists.items():
    for perm in range(2):
        gg = gall if perm == 0 else gall.view(-1)[torch.randperm(65536, generator=g, device=dev)].view(16, 4096)
        nd, npay = c_diff(a * gg.sigmoid(), F.gate(a, gg, lut))
        nan_payload += npay
        check(f"C {name} x every gate value ({'ordered' if perm == 0 else 'shuffled'})", nd == 0,
              f"({nd} of 65,536 differ)")
for T in (1, 5, 20, 64):
    a = (torch.randn(T, NH * 128, generator=g, device=dev) * 0.5).to(bf16)
    gt = (torch.randn(T, NH * 128, generator=g, device=dev) * 4).to(bf16)
    nd, npay = c_diff(a * gt.sigmoid(), F.gate(a, gt, lut))
    check(f"C T={T} [T, 12288]", nd == 0 and npay == 0, f"({nd} differ)")
a = specials((torch.randn(2, 4096, generator=g, device=dev)).to(bf16))
nd, npay = c_diff(a * gall[:2].sigmoid(), F.gate(a, gall[:2], lut))
check("C with +-inf / NaN in attn_out", nd == 0, f"({nd} differ)")
info("C NaN results with another NaN bit pattern than torch's (not judged)", f"{nan_payload + npay}")

# ---------------------------------------------------------------------------------------------------------------------
# 4. CUDA graphs
# ---------------------------------------------------------------------------------------------------------------------
T = 5
wkv = wide(g, T, D + H)
index_q = (torch.randn(T, H, D, generator=g, device=dev) * 3).to(bf16)
kn_w, kn_b = torch.randn(D, generator=g, device=dev), torch.randn(D, generator=g, device=dev) * 0.1
islot = torch.arange(T, device=dev, dtype=torch.long) * 3
mslot = torch.arange(T, device=dev, dtype=torch.long) * 5
kv, qkv = wide(g, T, L), wide(g, T, 1536 + L + R)
icache, ibuf = strided_cache(g, 8, BS, D + 4, 256)
mcache, mbuf = strided_cache(g, 8, MBS, L + R, 512)
topk = torch.zeros(80, TOPK, device=dev, dtype=torch.int32)
cat, qq = wide(g, T, NH, L + R, scale=40.0), wide(g, T, NH, 192, scale=40.0)
att, gat = (torch.randn(T, NH * 128, generator=g, device=dev)).to(bf16), (torch.randn(T, NH * 128, generator=g,
                                                                                         device=dev) * 4).to(bf16)


def run_all():
    q8, wb = F.prologue(wkv[:, :D], kn_w, kn_b, 1e-6, islot, icache, topk, index_q, wkv[:, D:], WS, H, True, kv,
                        qkv[:, 1536 + L:], mslot, mcache)
    return q8, wb, F.qpad(cat, qq[..., 128:], L, 128), F.gate(att, gat, lut)


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    run_all()
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    outs = run_all()
for t in (wkv, index_q, kv, qkv, cat, qq, att, gat):
    t.copy_((torch.randn(t.shape, generator=g, device=dev) * 3).to(bf16))
islot.copy_(islot + 1)
mslot.copy_(mslot + 2)
graph.replay()
torch.cuda.synchronize()
gi, gm, gt_ = ibuf.clone(), mbuf.clone(), topk.clone()
outs_g = [o.clone() for o in outs]
outs_e = run_all()
torch.cuda.synchronize()
check("CUDA graph replay = eager (A+M caches, top-k, query, weights, padded query, gate)",
      same(gi, ibuf) and same(gm, mbuf) and torch.equal(gt_, topk) and all(same(a, b) for a, b in zip(outs_g, outs_e)))

# ---------------------------------------------------------------------------------------------------------------------
# 5. timing (INFO)
# ---------------------------------------------------------------------------------------------------------------------


def timed(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(reps):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / (10 * reps))
    return best


ik, iw = wkv[:, :D], wkv[:, D:]
kpe = qkv[:, 1536 + L:]
t_am_s = timed(lambda: stock_am(ik, kn_w, kn_b, islot, icache, topk, index_q, iw, kv, kpe, mslot, mcache, T))
t_am_f = timed(lambda: F.prologue(ik, kn_w, kn_b, 1e-6, islot, icache, topk, index_q, iw, WS, H, True, kv, kpe, mslot,
                                  mcache))
t_b_s = timed(lambda: stock_b(cat, qq[..., 128:], inplace=True))
t_b_f = timed(lambda: F.qpad(cat, qq[..., 128:], L, 128))
t_c_s = timed(lambda: att * gat.sigmoid())
t_c_f = timed(lambda: F.gate(att, gat, lut))
info("timing at 5 rows, us per DSA layer (CUDA graph)",
     f"A+M {t_am_s:.1f} -> {t_am_f:.1f}; B {t_b_s:.1f} -> {t_b_f:.1f}; "
     f"C {t_c_s:.1f} -> {t_c_f:.1f}; sum {t_am_s + t_b_s + t_c_s:.1f} -> {t_am_f + t_b_f + t_c_f:.1f}")

print(f"DSA FUSED GPU: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
