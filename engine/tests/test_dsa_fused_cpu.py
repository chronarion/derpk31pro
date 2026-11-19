"""dsa_fused (K31_DSA_FUSED) on CPU, no GPU: the three kernels in Triton's interpreter against vLLM's own kernels (same
interpreter) and torch references, and the engine path's wiring against vLLM's own code on stand-in layers.

Run in the engine image, CPU only (from the repo root):
    docker run --rm --network none --cpuset-cpus 64-65 --user $(id -u):$(id -g) -e HOME=/tmp \
        -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -v "$PWD/engine":/engine:ro \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_dsa_fused_cpu.py
(CUDA_VISIBLE_DEVICES empty keeps vLLM on the real triton module; TRITON_INTERPRET=1 runs the kernels on numpy.)
Exit 0 and "DSA FUSED CPU: PASS" = every check equal.

The interpreter does not round like the GPU: its fp32 -> E4M3 / BF16 conversions truncate (and E4M3 overflows), where
the compiled kernels use cvt.rn.satfinite.e4m3x2.f32 / cvt.rn.bf16.f32. Every value that this test converts in a
kernel is therefore exactly representable in the target format (E4M3 grid values, powers of two times BF16 values),
so the checks are about indexing, masks, strides and wiring; tests/test_dsa_fused.py checks the rounding on the GB300.
Part A runs vLLM's own index kernels in the same interpreter, so its arithmetic is compared with any values.

1. Kernels: part A against vLLM's indexer_k_norm_quant_insert + indexer_q_quant_fold_weights and the op's
   .to(bfloat16) of the folded weights (bytes of the index cache, top-k buffer, FP8 query, BF16 weights); part M
   against concat_and_cache_mla's arithmetic in torch; part B against the q_pe copy + static FP8 quant in torch +
   vLLM's _pad_query_heads (interpreted); part C against torch's a * g.sigmoid() over all 65,536 BF16 gate values.
   Strided cache page views, padded rows (slot -1) and slot mappings shorter than the batch included.
2. Wiring: vLLM's MultiHeadLatentAttentionWrapper.forward (with KimiK3DSAIndexer.forward / score / _fused_k_insert,
   vLLM's code) against dsa_fused's wrapper path on a stand-in DSA layer (small linears and norms; the indexer op
   and the MLA attention replaced by recording stand-ins): same output, same op inputs, same caches and top-k rows;
   vLLM's MLAAttention.forward_impl against dsa_fused's on a stand-in MLA layer (the backend's forward_mqa replaced
   by a stand-in that pads through the real _pad_query_heads): same output, same padded query; the fallbacks.
"""
import os
import sys
import types

import numpy as np
import torch

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (see the docstring)"
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
import vllm.forward_context as vfc                                              # noqa: E402
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401  (registers the op before it is replaced)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention     # noqa: E402
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper     # noqa: E402
from vllm.models.kimi_k3.nvidia import dsa                                      # noqa: E402
from vllm.models.kimi_k3.nvidia.ops import dsa_indexer as dix                   # noqa: E402
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fis         # noqa: E402
from vllm.v1.attention.backends.mla import indexer as v32_indexer               # noqa: E402

from k31_engine import dsa_fused as F                                           # noqa: E402

FP8 = torch.float8_e4m3fn
bf16 = torch.bfloat16
fails = []
dix.mqa_logits_use_mx_scales = lambda: True          # SM100's MX scales, as on the GB300
F._STATE["mx"] = True


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def e4m3_grid(g, *shape, big=False):
    """BF16 values on the E4M3 grid (finite codes), optionally with |x| in 512..2048 at ~3% (beyond 448: the clamp)."""
    codes = torch.randint(0, 256, shape, generator=g, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes).to(torch.uint8)      # no NaN codes
    x = codes.view(FP8).float().to(bf16)
    if big:
        sel = torch.rand(shape, generator=g) < 0.03
        x = torch.where(sel, (2.0 ** torch.randint(9, 12, shape, generator=g)).to(bf16) * x.sign(), x)
    return x


def to_grid(x):
    """A tensor's values moved onto the E4M3 grid (stand-in layers whose outputs a kernel converts to FP8)."""
    return x.float().clamp(-448, 448).to(FP8).float().to(x.dtype)


def strided_cache(g, blocks, block, width, pad):
    """[blocks, block, width] uint8 page view with `pad` spare bytes per block (as a packed KV pool's view), random."""
    buf = torch.randint(0, 256, (blocks, block * width + pad), generator=g, dtype=torch.int32).to(torch.uint8)
    return buf.as_strided((blocks, block, width), (block * width + pad, width, 1)), buf


def ref_mla_insert(kv, pe, cache, slots):
    """concat_and_cache_mla (kFp8E4M3, skip_scale): rows < len(slots) with slot >= 0, E4M3 RNE."""
    bs = cache.shape[1]
    for t, s in enumerate(slots.tolist()):
        if t >= kv.shape[0] or s < 0:
            continue
        cache[s // bs, s % bs] = torch.cat([kv[t], pe[t]]).float().to(FP8).view(torch.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# 1. kernels
# ---------------------------------------------------------------------------------------------------------------------
g = torch.Generator().manual_seed(7)
H, D, TOPK, BS, MBS = 32, 128, 2048, 16, 16
for T, nslots in ((1, 1), (5, 5), (5, 3), (8, 8)):
    wkv = (torch.randn(T, 160, generator=g) * 2).to(bf16)
    index_k, index_w = wkv[:, :D], wkv[:, D:]
    qscale = torch.exp(torch.randn(T, H, 1, generator=g) * 4)              # per head amax from ~1e-5 to ~1e5
    index_q = (torch.randn(T, H, D, generator=g) * qscale).to(bf16)
    index_q[0, 3] = 0                                                      # amax 0: the 1e-10 floor
    kn_w, kn_b = torch.randn(D, generator=g), torch.randn(D, generator=g) * 0.1
    islot = torch.randperm(4 * BS, generator=g)[:nslots].long()
    islot[nslots // 2] = -1                                                # a padded row
    icache, ibuf = strided_cache(g, 4, BS, D + 4, 64)
    topk = torch.randint(0, 1000, (16, TOPK), generator=g, dtype=torch.int32)
    kv = e4m3_grid(g, T, 512)
    qkv = e4m3_grid(g, T, 2112)
    k_pe = qkv[:, 2048:]                                                   # a strided slice, as in the wrapper
    mslot = torch.randperm(4 * MBS, generator=g)[:nslots].long()
    if nslots > 1:
        mslot[-1] = -1
    mcache, mbuf = strided_cache(g, 4, MBS, 576, 128)
    ws = 128 ** -0.5 * 32 ** -0.5
    # vLLM's path
    ibuf_r, topk_r, mbuf_r = ibuf.clone(), topk.clone(), mbuf.clone()
    icache_r = ibuf_r.as_strided(icache.shape, icache.stride())
    mcache_r = mbuf_r.as_strided(mcache.shape, mcache.stride())
    dix.indexer_k_norm_quant_insert(index_k, kn_w, kn_b, 1e-6, islot, icache_r, topk_r, T)
    q_r, w_r = dix.indexer_q_quant_fold_weights(index_q, index_w, ws, H)
    w_r = w_r.to(bf16)
    ref_mla_insert(kv, k_pe, mcache_r, mslot)
    # fused
    q_f, w_f = F.prologue(index_k, kn_w, kn_b, 1e-6, islot, icache, topk, index_q, index_w, ws, H, True, kv, k_pe,
                          mslot, mcache)
    check(f"A T={T} slots={nslots}: index cache, top-k buffer, FP8 query, BF16 weights",
          same(ibuf, ibuf_r) and torch.equal(topk, topk_r) and same(q_f, q_r) and same(w_f, w_r))
    check(f"M T={T} slots={nslots}: MLA cache bytes (incl. untouched rows and block padding)", same(mbuf, mbuf_r))
    mbuf2 = mbuf_r.clone()                                                 # part A without M: the MLA cache alone
    F.prologue(index_k, kn_w, kn_b, 1e-6, islot, icache, topk, index_q, index_w, ws, H, True)
    check(f"A without M T={T}: same outputs, MLA cache untouched",
          same(ibuf, ibuf_r) and torch.equal(topk, topk_r) and same(mbuf2, mbuf_r))

for T in (1, 5):
    cat = e4m3_grid(g, T, 96, 576, big=True)
    q = e4m3_grid(g, T, 96, 192, big=True)
    q_pe = q[..., 128:]
    ref = cat.clone()
    ref[..., 512:].copy_(q_pe)
    ref = ref.reshape(T, -1).float().clamp(-448, 448).to(FP8).view(T, 96, 576)
    ref = fis._pad_query_heads(ref, 128)
    out = F.qpad(cat, q_pe, 512, 128)
    check(f"B T={T}: padded FP8 query = copy + static quant + _pad_query_heads", same(out, ref))

gb = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.int16)).view(bf16).view(16, 4096)
a = ((2.0 ** torch.randint(0, 9, (16, 4096), generator=g)) * (torch.randint(0, 2, (16, 4096), generator=g) * 2 - 1))
a = a.to(bf16)
lut = F.sigmoid_table("cpu")
ref = a * gb.sigmoid()
out = F.gate(a, gb, lut)
rb, ob = ref.view(torch.int16), out.view(torch.int16)
nan = torch.isnan(ref) & torch.isnan(out)
sub = (ref.float().abs() < 2.0 ** -126) & (ref != 0)          # subnormal results: the interpreter mis-converts them
check("C: a * sigmoid(g) over all 65,536 BF16 gate values (subnormal results: GPU test only)",
      bool(((rb == ob) | nan | sub).all()),
      f"({int(((rb != ob) & ~nan & ~sub).sum())} bit patterns differ; {int(nan.sum())} both NaN, {int(sub.sum())} "
      f"subnormal skipped)")
check("C: table = torch.sigmoid by bit pattern", same(lut[gb.view(torch.int16).long() & 0xFFFF], gb.sigmoid()))

# ---------------------------------------------------------------------------------------------------------------------
# 2. wiring
# ---------------------------------------------------------------------------------------------------------------------
REC = []


class Lin:
    def __init__(self, k, n, grid=False):
        self.w = (torch.randn(n, k, generator=g) * k ** -0.5).to(bf16)
        self.grid = grid

    def __call__(self, x):
        y = torch.nn.functional.linear(x, self.w)
        return (to_grid(y) if self.grid else y), None


class Norm:
    def __init__(self, n, grid=False):
        self.weight = (1 + 0.1 * torch.randn(n, generator=g)).to(bf16)
        self.grid = grid

    def __call__(self, x):
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight.float()).to(bf16)
        return to_grid(y) if self.grid else y


class FakeIndexer:
    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class FakeMLA:
    """MLAAttention.forward's custom-op path: update_kv_cache, then forward_impl (a deterministic stand-in whose
    output is a power of two per element, so the gate's product is exact in any rounding)."""
    layer_name = "model.layers.3.self_attn.attn"

    def update_kv_cache(self, kv_c_normed, k_pe, kv_cache, slot_mapping, attn_metadata, kv_cache_dtype, k_scale):
        REC.append(("kv_update", kv_c_normed.shape[0]))
        ref_mla_insert(kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping)

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, **kw):
        n = q.shape[0]
        e = (q.float().sum(-1).repeat_interleave(output.shape[1] // q.shape[1], dim=1) * 7).round().remainder(9) - 4
        output[:n] = (2.0 ** e).to(bf16)

    def __call__(self, q, kv_c_normed, k_pe, output_shape=None, q_dcp_replicated=None):
        fc = vfc.get_forward_context()
        slot = fc.slot_mapping.get(self.layer_name)
        if slot is not None:
            self.update_kv_cache(kv_c_normed, k_pe, self.kv_cache, slot, None, "fp8", None)
        output = torch.empty(output_shape, dtype=q.dtype)
        self.forward_impl(q, kv_c_normed, k_pe, self.kv_cache, None, output=output)
        return output


class FakeWrapper:
    forward = MultiHeadLatentAttentionWrapper.forward

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


def fake_indexer_op(hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights, *args, **kwargs):
    """Records what the op would read (q bytes, the BF16 weights it converts to) and writes deterministic top-k."""
    REC.append(("op", q_quant.view(torch.uint8).clone(), weights.to(bf16).view(torch.int16).clone(), q_scale is None,
                k is None, kwargs.get("skip_topk_buffer_clear")))
    topk_buf = args[6]
    T = q_quant.shape[0]
    s = (q_quant.view(torch.uint8).float().sum(-1) * weights.to(bf16).float()).sum(-1)
    topk_buf[:T, :8] = ((s.abs() * 1000).long() % 97).to(torch.int32)[:, None] + torch.arange(8, dtype=torch.int32)
    return topk_buf


torch.ops.vllm.sparse_attn_indexer = fake_indexer_op
HID, QL, KVL, RP, NH, NOPE, VH = 64, 32, 32, 16, 96, 16, 16
ix = FakeIndexer()
ix.wq_index, ix.wkv_index = Lin(QL, H * D), Lin(HID, D + H)
ix.index_num_heads, ix.index_head_dim, ix.padded_index_num_heads = H, D, H
ix.index_softmax_scale, ix.indexer_weight_scale = D ** -0.5, H ** -0.5
ix.k_index_norm = types.SimpleNamespace(weight=torch.randn(D, generator=g), bias=torch.randn(D, generator=g) * 0.1,
                                        eps=1e-6)
ix.k_cache = types.SimpleNamespace(prefix="model.layers.3.self_attn.dsa_indexer.k_cache")
ix.use_fused_k_insert, ix.use_fp4_cache, ix.use_pcp, ix.dcp_world_size, ix.dcp_rank = True, False, False, 1, 0
ix.shardq_layer_name, ix.index_rope_dim, ix.quant_block_size, ix.scale_fmt, ix.topk_tokens = "", 0, D, "ue8m0", TOPK
ix.max_model_len = ix.max_total_seq_len = 4096
ix._parallel_config = types.SimpleNamespace(pcp_shard_decode_requests=False)
ix.dense_mha_metadata_layer_name, ix.cp_kv_cache_interleave_size, ix.topk_backend, ix.dcp_local_topk = "", 1, "auto", 0
ix._shardq_indexer_rows = lambda: None
for fn in ("forward", "score", "_fused_k_insert"):
    setattr(ix, fn, types.MethodType(getattr(dsa.KimiK3DSAIndexer, fn), ix))
mla = FakeMLA()
mla.impl = types.SimpleNamespace(record_logical_topk_ready=lambda: REC.append(("ready",)))
w = FakeWrapper()
w.q_lora_rank, w.kv_lora_rank, w.qk_rope_head_dim, w.qk_nope_head_dim = QL, KVL, RP, NOPE
w.qk_head_dim, w.v_head_dim, w.num_heads = NOPE + RP, VH, NH
w.fused_qkv_a_proj, w.q_a_layernorm, w.kv_a_layernorm = Lin(HID, QL + KVL + RP, grid=True), Norm(QL), Norm(KVL, True)
w.q_b_proj, w.g_proj, w.o_proj = Lin(QL, NH * (NOPE + RP)), Lin(HID, NH * VH), Lin(NH * VH, HID)
w.fuse_qkv_rmsnorm, w.rotary_emb, w.indexer, w.indexer_rope_emb = False, None, ix, None
w.is_sparse, w.skip_topk, w.dcp_q_replicate, w.mla_attn = True, False, False, mla
F.install(w)
F.install_hooks()
F._STATE["lut"] = F.sigmoid_table("cpu")
check("install: per-instance wrapper forward / update_kv_cache / forward_impl, module hooks",
      isinstance(w.__dict__.get("forward"), types.MethodType) and "update_kv_cache" in mla.__dict__
      and "forward_impl" in mla.__dict__ and dsa.indexer_q_quant_fold_weights is F._q_quant_hook
      and fis._pad_query_heads is F._pad_hook)

for T in (1, 5, 8):
    icache, ibuf = strided_cache(g, 4, BS, D + 4, 64)
    mcache, mbuf = strided_cache(g, 4, MBS, KVL + RP, 32)
    topk = torch.randint(0, 1000, (16, TOPK), generator=g, dtype=torch.int32)
    islot = torch.randperm(4 * BS, generator=g)[:T].long()
    mslot = torch.randperm(4 * MBS, generator=g)[:T].long()
    if T > 1:
        islot[1], mslot[-1] = -1, -1
    imeta = object.__new__(v32_indexer.DeepseekV32IndexerMetadata)
    imeta.slot_mapping = islot
    fc = types.SimpleNamespace(attn_metadata={ix.k_cache.prefix: imeta, mla.layer_name: None},
                               slot_mapping={mla.layer_name: mslot})
    vfc.get_forward_context = dsa.get_forward_context = lambda fc=fc: fc
    h = (torch.randn(T, HID, generator=g) * 2).to(bf16)
    runs = {}
    for mode in ("stock", "fused"):
        ibuf_m, mbuf_m, topk_m = ibuf.clone(), mbuf.clone(), topk.clone()
        ix.k_cache.kv_cache = ibuf_m.as_strided(icache.shape, icache.stride())
        mla.kv_cache = mbuf_m.as_strided(mcache.shape, mcache.stride())
        ix.topk_indices_buffer = topk_m
        REC.clear()
        out = FakeWrapper.forward(w, None, h) if mode == "stock" else w.forward(None, h)
        runs[mode] = (out, ibuf_m, mbuf_m, topk_m, list(REC))
    (o_s, i_s, m_s, t_s, r_s), (o_f, i_f, m_f, t_f, r_f) = runs["stock"], runs["fused"]
    op_s = [r for r in r_s if r[0] == "op"]
    op_f = [r for r in r_f if r[0] == "op"]
    ops_same = len(op_s) == len(op_f) == 1 and same(op_s[0][1], op_f[0][1]) and same(op_s[0][2], op_f[0][2]) \
        and op_s[0][3:] == op_f[0][3:]
    check(f"wrapper T={T}: output, op inputs (q bytes, BF16 weights, flags), index / MLA caches, top-k rows",
          same(o_s, o_f) and ops_same and same(i_s, i_f) and same(m_s, m_f) and torch.equal(t_s, t_f),
          f"(output {same(o_s, o_f)}, op {ops_same}, index cache {same(i_s, i_f)}, MLA cache {same(m_s, m_f)}, "
          f"top-k {torch.equal(t_s, t_f)})")
    check(f"wrapper T={T}: the MLA layer's own insert ran in vLLM's path only, ready marker in both",
          ("kv_update", T) in r_s and not any(r[0] == "kv_update" for r in r_f)
          and ("ready",) in r_s and ("ready",) in r_f)
    check(f"wrapper T={T}: no hand-off left armed", F._PEND["q"] is None and mla._k31_kv_skip is None)
logs = []                                             # the in-engine self-check (K31_DSA_FUSED_CHECK) on the same layer
F._STATE["log"] = logs.append
torch.cuda.is_current_stream_capturing = lambda: False
F.CHECK = 2
for _ in range(2):
    ix.k_cache.kv_cache = ibuf.clone().as_strided(icache.shape, icache.stride())
    mla.kv_cache = mbuf.clone().as_strided(mcache.shape, mcache.stride())
    ix.topk_indices_buffer = topk.clone()
    w.forward(None, h)
check("self-check: vLLM's path then the fused path, 2 calls compared, no mismatch",
      F._STATE["checked"] == 2 and F._STATE["bad"] == 0 and any("2 calls, 0 mismatches" in m for m in logs), str(logs))
F.CHECK, lut_ok = 3, F._STATE["lut"]
F._STATE["lut"] = torch.zeros_like(lut_ok)            # a wrong gate table: the check must see it
w.forward(None, h)
F._STATE["lut"], F.CHECK = lut_ok, 0
check("self-check reports a wrong fused output", F._STATE["bad"] == 1 and any("MISMATCH" in m for m in logs),
      str(logs[-2:]))
fc = types.SimpleNamespace(attn_metadata=None, slot_mapping=None)       # dummy run, then too many rows
vfc.get_forward_context = dsa.get_forward_context = lambda: fc
called = []
orig_fwd = FakeWrapper.forward
FakeWrapper.forward = lambda self, *a, **k: called.append(a[1].shape[0]) or "stock"
r1 = w.forward(None, torch.zeros(3, HID, dtype=bf16))
fc.attn_metadata, fc.slot_mapping = {}, {}
r2 = w.forward(None, torch.zeros(F.MAX_ROWS + 1, HID, dtype=bf16))
FakeWrapper.forward = orig_fwd
check("wrapper fallbacks: dummy run and > MAX_ROWS rows go to vLLM's forward", r1 == r2 == "stock" and len(called) == 2)


class FakeAttn:
    forward_impl = MLAAttention.forward_impl
    _v_up_proj = MLAAttention._v_up_proj


def fake_quant(ql_nope, q_pe, scale, scale_ub=None):
    """_DecodeConcatQuantFP8.forward_cuda: static per-tensor FP8 (x * (1 / scale), clamp, E4M3 RNE)."""
    q0 = ql_nope if q_pe.shape[-1] == 0 and ql_nope.is_contiguous() else torch.cat((ql_nope, q_pe), dim=-1)
    flat = q0.reshape(q0.shape[0], -1)
    return (flat.float() * (1.0 / scale.float())).clamp(-448, 448).to(FP8).view(q0.shape)


def fake_forward_mqa(q, kv_cache, attn_metadata, layer):
    """_forward_mqa_kernel's query handling (pad through the module's _pad_query_heads), then a stand-in output."""
    num_heads = q.shape[1]
    padded = fis.trtllm_gen_query_heads(num_heads)
    if padded != num_heads:
        q = fis._pad_query_heads(q, padded)
    REC.append(("q", q.view(torch.uint8).clone(), kv_cache.dtype))
    out = (q.view(torch.uint8)[..., :layer.kv_lora_rank].float() * 0.01 - 1.0).to(bf16)
    return out[:, :num_heads], None


L2 = FakeAttn()
L2.num_heads, L2.v_head_dim, L2.qk_nope_head_dim, L2.qk_rope_head_dim, L2.kv_lora_rank = NH, VH, 128, 64, 512
L2.kv_cache_dtype, L2.use_pcp, L2.use_shardq, L2.q_pad_num_heads, L2.prefill_backend = "fp8", False, False, None, None
L2.is_aiter_triton_fp4_bmm_enabled = L2.is_aiter_triton_fp8_bmm_enabled = L2.is_amx_bmm_enabled = False
wuk = torch.zeros(NH, 128, 512)                       # the bmm picks q_nope's columns: an exact product in any order
wuk[:, torch.arange(128), torch.arange(128) * 3] = 1.0
L2.W_UK_T = wuk.to(bf16)
L2.W_UK_T_dcp_qrep = None
L2.W_UV = (torch.randn(NH, 512, VH, generator=g) * 0.05).to(bf16)
L2._q_scale, L2._k_scale = torch.tensor(1.0), torch.tensor(1.0)
L2._decode_concat_quant_fp8_op = fake_quant
L2.impl = types.SimpleNamespace(dcp_world_size=1, is_sparse=True, supports_quant_query_input=True,
                                forward_mqa=fake_forward_mqa)
L2.forward_impl = types.MethodType(F._forward_impl, L2)
kvc = torch.zeros(4, 16, 576, dtype=torch.uint8)
for T in (1, 5):
    q = e4m3_grid(g, T, NH, 192, big=True)
    kc, kp = torch.zeros(T + 2, 512, dtype=bf16), torch.zeros(T + 2, 1, 64, dtype=bf16)
    md = types.SimpleNamespace(num_actual_tokens=T, num_decodes=T, num_prefills=0, num_decode_tokens=T)
    outs = {}
    for mode in ("stock", "fused"):
        REC.clear()
        out = torch.full((T + 2, NH * VH), 7.0, dtype=bf16)                 # 2 graph-padding rows
        fn = FakeAttn.forward_impl if mode == "stock" else F._forward_impl
        fn(L2, torch.cat([q, q[:2]]), kc, kp, kvc, md, output=out)
        outs[mode] = (out, [r for r in REC if r[0] == "q"])
    (o_s, q_s), (o_f, q_f) = outs["stock"], outs["fused"]
    check(f"forward_impl T={T}: output (padding rows untouched) and the padded FP8 query the kernel reads",
          same(o_s, o_f) and len(q_s) == len(q_f) == 1 and same(q_s[0][1], q_f[0][1]) and q_s[0][2] == q_f[0][2]
          and bool((o_f[T:] == 7.0).all()))
    check(f"forward_impl T={T}: no hand-off left armed", F._PEND["pad"] is None)
called = []                                           # a mixed batch and a quantized-output call: vLLM's forward_impl
orig_fi = FakeAttn.forward_impl
FakeAttn.forward_impl = lambda self, *a, **k: called.append(1)
md = types.SimpleNamespace(num_actual_tokens=5, num_decodes=4, num_prefills=1, num_decode_tokens=4)
F._forward_impl(L2, q, kc, kp, kvc, md, output=torch.empty(5, NH * VH, dtype=bf16))
md = types.SimpleNamespace(num_actual_tokens=5, num_decodes=5, num_prefills=0, num_decode_tokens=5)
F._forward_impl(L2, q, kc, kp, kvc, md, output=torch.empty(5, NH * VH, dtype=bf16), output_scale=torch.ones(1))
FakeAttn.forward_impl = orig_fi
check("forward_impl fallbacks: mixed batch and fused output quant go to vLLM's forward_impl", len(called) == 2)

print(f"DSA FUSED CPU: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
