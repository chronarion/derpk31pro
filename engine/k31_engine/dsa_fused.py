"""K31_DSA_FUSED=1: the DSA layers' small decode kernels in three launches, bit-exact against vLLM's path.

vLLM's unpacked DSA forward (MultiHeadLatentAttentionWrapper with KimiK3DSAIndexer, FP8 KV cache, the FlashInfer
sparse MLA backend) runs these small kernels per DSA layer and decode step (means of the 11-18 R3 trace, 5 rows):
  idxk      _indexer_k_norm_quant_insert_kernel: index key LayerNorm + FP8 + cache insert, top-k rows -1   2.5 us
  idxq      _indexer_q_quant_fold_weights_kernel: index query FP8 per head, scale folded into the weights  2.2 us
  bf16copy  the folded weights fp32 -> bf16 (mqa_logits_mx_inputs on SM100)                               1.5 us
  cachemla  concat_and_cache_mla: [kv_c_normed | k_pe] -> FP8 MLA cache (k_scale 1: skip_scale)            2.9 us
  elemw     q_pe copied beside the absorbed query (mqa_q_cat[..., 512:])                                   3.9 us
  fp8quant  static per-tensor FP8 quant of the [T, 96, 576] query (one CTA per row)                        6.3 us
  padq      _pad_query_heads_kernel: 96 -> 128 query heads for trtllm-gen, zero heads                      1.4 us
  sigmoid + mul   attn_out * sigmoid(gate)                                                           2.3 + 1.6 us
With the flag (parts, K31_DSA_FUSED_PARTS, default "AMBC"):
  A  one launch for idxk + idxq, the folded weights written in BF16 (bf16copy becomes a no-op .to());
  M  concat_and_cache_mla folded into A's launch (needs A; the MLA layer's own insert is then skipped);
  B  one launch for elemw + fp8quant + padq: the padded FP8 query written directly from the bmm's output and q_pe;
  C  one launch for sigmoid + mul: sigmoid read from a 65,536-entry table built once with torch.sigmoid.
upcast + cooperative_topk stay (an exact top-k on BF16 logits would need cooperative_topk's source: its index order
and tie rules decide the attention's summation order). The GEMMs, norms, the indexer op's MQA logits and top-k, the
attention kernel and the bmms are the same calls as vLLM's.

Exactness (what each part must reproduce, and how):
  A  the same Triton code: vLLM's _index_k_insert and _fill_topk are called from this kernel, and the query quant is
     _indexer_q_quant_fold_weights_kernel's body; num_warps 1 as there. Compiled for sm_103 the reductions keep vLLM's
     layout (sizePerThread 4, one warp: the LayerNorm's two sums in the same order) and the floating-point instruction
     counts are exactly idxk's + idxq's plus the MLA insert's conversions and one cvt.rn.bf16.f32. The fp32 folded
     weight w * scale * weight_scale is the same expression, rounded to BF16 here (RNE) where vLLM rounds the stored
     fp32 with .to(torch.bfloat16) (the same RNE) inside the op; in this model weight_scale = 128^-0.5 * 32^-0.5 =
     2^-6 and scale is a power of two, so the value is a BF16 value times a power of two and the rounding is exact.
  M  concat_and_cache_mla with skip_scale (kv_cache_dtype fp8, k_scale 1.0): each bf16 value -> fp32 -> E4M3 with
     round to nearest even and satfinite (cvt.rn.satfinite.e4m3x2.f32, as __nv_cvt_float_to_fp8 with SATFINITE).
  B  the static per-tensor quant with scale 1.0 (bind checks _q_scale == 1): x * (1 / 1) = x, then
     fmaxf(-448, fminf(x, 448)) (NaN -> 448 as vLLM's clamp; compiled as NaN-dropping min.bf16 / max.bf16 against
     448, which BF16 holds exactly) and the satfinite RNE conversion; heads 96..127 zero as _pad_query_heads writes
     them. The bmm into mqa_q_cat is vLLM's call with the same output strides.
  C  out = bf16(float(a) * float(table[bits(g)])) with table = torch.sigmoid over all 65,536 BF16 bit patterns: the
     same sigmoid values by construction, and one fp32 multiply (mul.f32x2, subnormals kept) + RNE (cvt.rn.bf16x2.f32)
     as PyTorch's BinaryFunctor<bf16, MulFunctor<float>> (see _gate_kernel on why the inputs are shifted, not cast).
Checked by engine/tests/test_dsa_fused.py (GPU, kernels against vLLM's functions, bytes equal), by
engine/tests/test_dsa_fused_cpu.py (Triton's interpreter: indexing and wiring against vLLM's own code), and in a
running engine by K31_DSA_FUSED_CHECK=N: the first N eligible calls that run outside graph capture go through vLLM's
path and then the fused path on the same inputs; the layer output, the top-k rows and the cache bytes of the step's
slots must be equal (one log line per mismatch, a summary every 24 checked calls). With --enforce-eager (k31.sh
EAGER=1) that is every decode step.

Scope: calls of 1..K31_DSA_FUSED_ROWS rows (default 64, every CUDA graph size) with attention metadata and bound caches
(profile / dummy runs keep vLLM's path); B also needs a decode-only batch, decided per call inside the attention op
(which runs eagerly in a breakable capture), so mixed piecewise batches keep vLLM's attention while A, M and C (row-wise,
independent of the batch's composition) stay fused. Layers whose configuration differs from the checked one (packed
forward, rope, MXFP4 keys, PCP / DCP / ShardQ, q or k scale != 1, another MLA backend, prims_ts, HiSparse) keep vLLM's
path; the bind line names them.
"""
from __future__ import annotations

import os
import types

import numpy as np
import torch
import triton
import triton.language as tl

from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (_FP8_DTYPE, _FP8_MAX, _USE_FNUZ, _fill_topk, _index_k_insert,
                                                         _ue8m0_scale)

MAX_ROWS = int(os.environ.get("K31_DSA_FUSED_ROWS", "64"))
PARTS = os.environ.get("K31_DSA_FUSED_PARTS", "AMBC").upper()
CHECK = int(os.environ.get("K31_DSA_FUSED_CHECK", "0") or 0)
_ON = {p: p in PARTS for p in "AMBC"}
_ON["M"] = _ON["M"] and _ON["A"]
_STATE: dict = {"lut": None, "mx": True, "stock": False, "log": None, "checked": 0, "bad": 0}
_PEND: dict = {"q": None, "pad": None}           # one-shot hand-offs into vLLM's indexer score and the MLA backend
_ORIG: dict = {}


# ---------------------------------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------------------------------

@triton.jit(do_not_specialize=["num_islots", "num_mslots"])
def _prologue_kernel(k_ptr, k_stride, kn_w_ptr, kn_b_ptr, kn_eps, islot_ptr, num_islots, icache_ptr, iscale_ptr,
                     iblock_size, iblock_stride, topk_ptr, topk_stride,
                     q_ptr, q_stride0, q_stride1, w_ptr, w_stride, qf_ptr, qf_stride0, qf_stride1, wo_ptr, wo_stride,
                     weight_scale,
                     kvc_ptr, kvc_stride, kpe_ptr, kpe_stride, mslot_ptr, num_mslots, mcache_ptr, mblock_size,
                     mblock_stride, mentry_stride,
                     HEAD_DIM: tl.constexpr, TOPK: tl.constexpr, TOPK_BLOCK: tl.constexpr, MX_SCALE: tl.constexpr,
                     FP8_MAX: tl.constexpr, USE_FNUZ: tl.constexpr, KV_DIM: tl.constexpr, PE_DIM: tl.constexpr,
                     MLA: tl.constexpr):
    """Grid (T, 3 + H): program (t, 0) = vLLM's index key insert of row t, (t, 1) = the -1 fill of top-k row t,
    (t, 2) = the MLA cache insert of row t (MLA), (t, 3 + h) = the index query quant of row t, head h."""
    tok = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if pid == 0:
        _index_k_insert(tok, k_ptr + tok * k_stride, kn_w_ptr, kn_b_ptr, kn_eps, islot_ptr, num_islots, icache_ptr,
                        iscale_ptr, iblock_size, iblock_stride, HEAD_DIM, MX_SCALE, FP8_MAX, USE_FNUZ)
    elif pid == 1:
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)
    elif pid == 2:
        if MLA:
            # concat_and_cache_mla, kFp8E4M3 with skip_scale: rows past the slot mapping or with slot < 0 are skipped
            slot = tl.load(mslot_ptr + tok, mask=tok < num_mslots, other=-1)
            if slot >= 0:
                base = (slot // mblock_size) * mblock_stride + (slot % mblock_size) * mentry_stride
                c = tl.arange(0, KV_DIM)
                x = tl.load(kvc_ptr + tok * kvc_stride + c).to(tl.float32)
                tl.store(mcache_ptr + base + c, x.to(tl.float8e4nv))
                p = tl.arange(0, PE_DIM)
                y = tl.load(kpe_ptr + tok * kpe_stride + p).to(tl.float32)
                tl.store(mcache_ptr + base + KV_DIM + p, y.to(tl.float8e4nv))
    else:
        # _indexer_q_quant_fold_weights_kernel's body for (tok, head); the folded weight is stored as BF16
        head = pid - 3
        offs = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + tok * q_stride0 + head * q_stride1 + offs).to(tl.float32)
        scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, FP8_MAX)
        fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
        q_fp8 = tl.clamp(tl.div_rn(q, scale), -FP8_MAX, FP8_MAX).to(fp8_dtype)
        tl.store(qf_ptr + tok * qf_stride0 + head * qf_stride1 + offs, q_fp8)
        w = tl.load(w_ptr + tok * w_stride + head).to(tl.float32)
        tl.store(wo_ptr + tok * wo_stride + head, (w * scale * weight_scale).to(tl.bfloat16))


@triton.jit
def _qpad_kernel(cat_ptr, cat_s0, cat_s1, pe_ptr, pe_s0, pe_s1, out_ptr, out_s0, out_s1, NH,
                 L: tl.constexpr, R: tl.constexpr, FP8_MAX: tl.constexpr):
    """Grid (T, padded heads): head h < NH = FP8 of [mqa_q_cat[t, h, :L] | q_pe[t, h, :R]] (scale 1, vLLM's clamp
    then satfinite RNE); heads >= NH = zero bytes."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    dst = out_ptr + b * out_s0 + h * out_s1
    cl = tl.arange(0, L)
    cr = tl.arange(0, R)
    if h < NH:
        x = tl.load(cat_ptr + b * cat_s0 + h * cat_s1 + cl).to(tl.float32)
        x = tl.maximum(tl.minimum(x, FP8_MAX), -FP8_MAX)
        tl.store(dst + cl, x.to(tl.float8e4nv))
        y = tl.load(pe_ptr + b * pe_s0 + h * pe_s1 + cr).to(tl.float32)
        y = tl.maximum(tl.minimum(y, FP8_MAX), -FP8_MAX)
        tl.store(dst + L + cr, y.to(tl.float8e4nv))
    else:
        tl.store(dst + cl, tl.zeros([L], dtype=tl.float32).to(tl.float8e4nv))
        tl.store(dst + L + cr, tl.zeros([R], dtype=tl.float32).to(tl.float8e4nv))


@triton.jit
def _gate_kernel(a_ptr, a_stride, g_ptr, g_stride, lut_ptr, o_ptr, o_stride, N, BLOCK: tl.constexpr):
    """out[t, c] = bf16_rn(fp32(a[t, c]) * fp32(table[bits of g[t, c]])), all inputs read as int16 bits. BF16 -> fp32
    by a 16-bit shift (as c10::BFloat16 does), one fp32 multiply (mul.f32: subnormal results kept), one RNE rounding:
    PyTorch's BinaryFunctor<bf16, MulFunctor<float>>. Written with .to(tl.float32) the compiler turns the pair into a
    native BF16 multiply (mul.bf16x2: one rounding of the exact product), which differs when the fp32 product is
    subnormal."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    ab = tl.load(a_ptr + row * a_stride + cols, mask=m, other=0).to(tl.int32) & 0xFFFF
    gb = tl.load(g_ptr + row * g_stride + cols, mask=m, other=0).to(tl.int32) & 0xFFFF
    sb = tl.load(lut_ptr + gb, mask=m, other=0).to(tl.int32) & 0xFFFF
    a = (ab << 16).to(tl.float32, bitcast=True)
    s = (sb << 16).to(tl.float32, bitcast=True)
    tl.store(o_ptr + row * o_stride + cols, (a * s).to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------------------------------------------------------------
# launchers (explicit tensors: the engine path below and engine/tests/test_dsa_fused.py)
# ---------------------------------------------------------------------------------------------------------------------

def prologue(index_k, kn_weight, kn_bias, kn_eps, islot, icache, topk_buf, index_q, index_weights, weight_scale,
             padded_heads, mx_scale, kv_c_normed=None, k_pe=None, mslot=None, mcache=None):
    """Part A (+ M when mcache is given). index_k [T, D] and index_weights [T, H] (row slices of the wkv_index output),
    index_q [T, H, D]; islot / icache = the indexer's slot mapping and [blocks, block, D + 4] uint8 cache; topk_buf the
    model's top-k buffer (rows [:T] filled with -1); kv_c_normed [T, kv], k_pe [T, pe] (2-D), mslot / mcache = the
    MLA layer's slot mapping and [blocks, block, kv + pe] cache. Returns (q_fp8 [T, P, D], weights bf16 [T, P]) as
    indexer_q_quant_fold_weights would return them, the weights already in the dtype the op converts them to."""
    T, H, D = index_q.shape
    assert index_k.shape == (T, D) and index_weights.shape == (T, H) and T > 0
    assert index_k.stride(-1) == 1 and index_q.stride(-1) == 1 and triton.next_power_of_2(D) == D
    assert icache.dim() == 3 and icache.shape[-1] == D + 4
    alloc = torch.empty if padded_heads == H else torch.zeros
    q_fp8 = alloc((T, padded_heads, D), dtype=_FP8_DTYPE, device=index_q.device)
    w_out = alloc((T, padded_heads), dtype=torch.bfloat16, device=index_q.device)
    u8 = icache.view(torch.uint8)
    topk = topk_buf.shape[-1]
    mla = mcache is not None
    if mla:
        assert kv_c_normed.dim() == 2 and k_pe.dim() == 2 and kv_c_normed.shape[0] == T and k_pe.shape[0] == T
        assert kv_c_normed.stride(-1) == 1 and k_pe.stride(-1) == 1 and mcache.dim() == 3 and mcache.stride(-1) == 1
        assert mcache.shape[-1] == kv_c_normed.shape[1] + k_pe.shape[1]
        assert all(triton.next_power_of_2(x) == x for x in (kv_c_normed.shape[1], k_pe.shape[1]))
        mc = mcache.view(_FP8_DTYPE)
        mslot = mslot.flatten()
        kv_dim, pe_dim = kv_c_normed.shape[1], k_pe.shape[1]
        margs = (kv_c_normed, kv_c_normed.stride(0), k_pe, k_pe.stride(0), mslot, mslot.shape[0], mc, mc.shape[1],
                 mc.stride(0), mc.stride(1))
    else:
        kv_dim, pe_dim = 16, 16
        margs = (index_k, 0, index_k, 0, islot, 0, q_fp8, 1, 0, 0)          # unused (MLA = False)
    _prologue_kernel[(T, 3 + H)](
        index_k, index_k.stride(0), kn_weight, kn_bias, kn_eps, islot, islot.shape[0], u8.view(_FP8_DTYPE),
        u8.view(torch.int32 if mx_scale else torch.float32), u8.shape[1], u8.stride(0), topk_buf, topk_buf.stride(0),
        index_q, index_q.stride(0), index_q.stride(1), index_weights, index_weights.stride(0), q_fp8, q_fp8.stride(0),
        q_fp8.stride(1), w_out, w_out.stride(0), weight_scale, *margs,
        HEAD_DIM=D, TOPK=topk, TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)), MX_SCALE=mx_scale,
        FP8_MAX=_FP8_MAX, USE_FNUZ=_USE_FNUZ, KV_DIM=kv_dim, PE_DIM=pe_dim, MLA=mla, num_warps=1)
    return q_fp8, w_out


def qpad(mqa_q_cat, q_pe, L, heads):
    """Part B. mqa_q_cat [T, N, L + R] bf16 (columns :L written by the W_UK bmm), q_pe [T, N, R] bf16 (any strides,
    last dim contiguous). Returns the [T, heads, L + R] FP8 query that _pad_query_heads would return."""
    T, N, LR = mqa_q_cat.shape
    R = q_pe.shape[-1]
    assert LR == L + R and q_pe.shape[:2] == (T, N) and heads >= N
    assert triton.next_power_of_2(L) == L and triton.next_power_of_2(R) == R
    assert mqa_q_cat.stride(-1) == 1 and q_pe.stride(-1) == 1
    out = torch.empty((T, heads, L + R), dtype=_FP8_DTYPE, device=mqa_q_cat.device)
    _qpad_kernel[(T, heads)](mqa_q_cat, mqa_q_cat.stride(0), mqa_q_cat.stride(1), q_pe, q_pe.stride(0), q_pe.stride(1),
                             out, out.stride(0), out.stride(1), N, L=L, R=R, FP8_MAX=_FP8_MAX, num_warps=2)
    return out


def sigmoid_table(device) -> torch.Tensor:
    """torch.sigmoid of every BF16 bit pattern, indexed by the pattern (uint16)."""
    bits = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.int16)).to(device)
    return torch.sigmoid(bits.view(torch.bfloat16)).contiguous()


def gate(attn_out, g, lut):
    """Part C: attn_out * g.sigmoid() (both bf16 [T, N], unit column stride) in one launch."""
    T, N = attn_out.shape
    out = torch.empty((T, N), dtype=torch.bfloat16, device=attn_out.device)
    block = 1024
    _gate_kernel[(T, triton.cdiv(N, block))](attn_out.view(torch.int16), attn_out.stride(0), g.view(torch.int16),
                                             g.stride(0), lut.view(torch.int16), out, out.stride(0), N, BLOCK=block,
                                             num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# engine path
# ---------------------------------------------------------------------------------------------------------------------

def _why_not(attn) -> str | None:
    """None if this KimiK3DSAAttention runs the configuration the fused path reproduces, else the reason."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAIndexerKNorm
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
    w = attn.mla_attn
    ix = getattr(w, "indexer", None)
    L = getattr(w, "mla_attn", None)
    if getattr(attn, "_packed", None) is not None:
        return "packed forward"
    checks = [
        (w.q_lora_rank is not None and w.fused_qkv_a_proj is not None and w.q_a_layernorm is not None
         and w.q_b_proj is not None, "no q LoRA"),
        (not getattr(w, "fuse_qkv_rmsnorm", False), "fused q/kv norm"),
        (w.rotary_emb is None, "rope"),
        (ix is not None and w.is_sparse and not getattr(w, "skip_topk", False), "no indexer"),
        (not getattr(w, "dcp_q_replicate", False), "DCP query replication"),
        (w.g_proj is not None, "no output gate"),
        (ix is not None and ix.use_fused_k_insert and not ix.use_fp4_cache and not ix.use_pcp
         and ix.dcp_world_size == 1 and not ix.shardq_layer_name and ix.index_rope_dim == 0
         and isinstance(ix.k_index_norm, KimiK3DSAIndexerKNorm) and ix.topk_indices_buffer is not None,
         "indexer variant (FP4 keys, PCP, DCP, ShardQ or rope)"),
        (mqa_logits_use_mx_scales(), "not SM100 (fp32 MQA weights)"),
        (L is not None and L.kv_cache_dtype == "fp8", "KV cache dtype not fp8"),
        (L is not None and float(L._k_scale_float) == 1.0 and float(L._q_scale_float) == 1.0
         and float(L._q_scale.float().max()) == 1.0 and float(L._q_scale.float().min()) == 1.0, "q / k scale != 1"),
        (L is not None and getattr(L, "q_pad_num_heads", None) is None and not L.is_aiter_triton_fp4_bmm_enabled
         and not L.is_aiter_triton_fp8_bmm_enabled and not getattr(L, "is_amx_bmm_enabled", False)
         and getattr(L, "W_UK_T", None) is not None, "MLA query path variant"),
        (L is not None and type(L.impl).__name__ == "FlashInferMLASparseImpl" and not L.impl.use_prims_ts
         and L.impl.supports_quant_query_input and L.impl.dcp_world_size == 1 and L.hisparse_cache is None
         and not L.use_pcp and L.dcp_manager is None and L.qk_rope_head_dim > 0
         and type(getattr(L.impl, "index_group", None)).__name__ != "HiSparseMLAIndexGroup",
         "MLA backend variant"),
    ]
    for ok, why in checks:
        if not ok:
            return why
    return None


def bind(model, log) -> None:
    """After load_model: per-instance forwards on the eligible DSA layers, the two one-shot module hooks, the table."""
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
    _STATE["log"] = log
    _STATE["mx"] = bool(mqa_logits_use_mx_scales())
    if not any(_ON.values()):
        log("DSA fused (K31_DSA_FUSED): no part enabled (K31_DSA_FUSED_PARTS); vLLM's path")
        return
    dev, n, skipped = None, 0, {}
    for name, m in model.named_modules():
        if type(m).__name__ != "KimiK3DSAAttention":
            continue
        why = _why_not(m)
        if why is not None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        install(m.mla_attn)
        dev = m.mla_attn.mla_attn.W_UK_T.device
        n += 1
    if n:
        install_hooks()
        if _ON["C"]:
            _STATE["lut"] = sigmoid_table(dev)
    log(f"DSA fused (K31_DSA_FUSED): {n} DSA layers, parts {''.join(p for p in 'AMBC' if _ON[p])}, calls of 1-{MAX_ROWS}"
        f" rows (A: index key + query quant + top-k fill{' + MLA cache insert' if _ON['M'] else ''} in one launch; "
        f"B: FP8 query + head pad; C: gate by a sigmoid table)"
        + (f"; vLLM's path kept for {skipped}" if skipped else "")
        + (f"; self-check of the first {CHECK} eager calls" if CHECK else ""))


def install(w) -> None:
    """The fused path on one MultiHeadLatentAttentionWrapper (and its MLAAttention): instance attributes only."""
    layer = w.mla_attn
    w.forward = types.MethodType(_wrapper_forward, w)
    layer._k31_kv_skip = None
    layer.update_kv_cache = types.MethodType(_update_kv_cache, layer)
    if _ON["B"]:
        layer.forward_impl = types.MethodType(_forward_impl, layer)


def install_hooks() -> None:
    """The two module-level one-shot hand-offs (pass-through unless the fused path armed them for its own tensor)."""
    import importlib
    if _ON["A"] and "q" not in _ORIG:
        dsa = importlib.import_module("vllm.models.kimi_k3.nvidia.dsa")
        _ORIG["q"] = dsa.indexer_q_quant_fold_weights
        dsa.indexer_q_quant_fold_weights = _q_quant_hook
    fis = importlib.import_module("vllm.v1.attention.backends.mla.flashinfer_mla_sparse")
    if _ON["B"] and "pad" not in _ORIG:
        _ORIG["pad"] = fis._pad_query_heads
        fis._pad_query_heads = _pad_hook
    _ORIG["heads"] = fis.trtllm_gen_query_heads


def _q_quant_hook(index_q, index_weights, weight_scale, padded_num_heads):
    """vllm.models.kimi_k3.nvidia.dsa.indexer_q_quant_fold_weights: the fused prologue's outputs for its own query."""
    pend = _PEND["q"]
    if pend is not None and index_q.data_ptr() == pend[0] and tuple(index_q.shape) == pend[1] \
            and padded_num_heads == pend[2].shape[1]:
        _PEND["q"] = None
        return pend[2], pend[3]
    return _ORIG["q"](index_q, index_weights, weight_scale, padded_num_heads)


def _pad_hook(query, heads):
    """flashinfer_mla_sparse._pad_query_heads: the padded FP8 query part B wrote, for the view it handed over."""
    pend = _PEND["pad"]
    if pend is not None and query.data_ptr() == pend[0] and tuple(query.shape) == pend[1] \
            and heads == pend[2].shape[1]:
        _PEND["pad"] = None
        return pend[2]
    return _ORIG["pad"](query, heads)


def _update_kv_cache(self, kv_c_normed, k_pe, kv_cache, slot_mapping, attn_metadata, kv_cache_dtype, k_scale):
    """MLAAttention.update_kv_cache: skipped for the rows part M already inserted (same kv_c_normed)."""
    skip = self._k31_kv_skip
    if skip is not None and kv_c_normed.data_ptr() == skip[0] and tuple(kv_c_normed.shape) == skip[1]:
        self._k31_kv_skip = None
        return None
    return type(self).update_kv_cache(self, kv_c_normed, k_pe, kv_cache, slot_mapping, attn_metadata, kv_cache_dtype,
                                      k_scale)


def _ready(w, hidden_states):
    """(forward context, indexer metadata, MLA slot mapping) if this call can take the fused path, else None."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.mla import indexer as v32_indexer
    T = hidden_states.shape[0]
    if not (0 < T <= MAX_ROWS and hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16):
        return None
    fc = get_forward_context()
    am = fc.attn_metadata
    ix, L = w.indexer, w.mla_attn
    if not isinstance(am, dict) or not isinstance(fc.slot_mapping, dict):
        return None
    icache = ix.k_cache.kv_cache
    if icache.numel() == 0 or L.kv_cache.numel() == 0 or icache.dim() != 3 or L.kv_cache.dim() != 3:
        return None
    imeta = am.get(ix.k_cache.prefix)
    mslot = fc.slot_mapping.get(L.layer_name)
    if not isinstance(imeta, v32_indexer.DeepseekV32IndexerMetadata) or mslot is None:
        return None
    return fc, imeta, mslot


def _wrapper_forward(self, positions, hidden_states, llama_4_scaling=None, skip_o_proj=False):
    """MultiHeadLatentAttentionWrapper.forward of an eligible DSA layer (q LoRA, NoPE, sparse, output gate)."""
    ready = None if (llama_4_scaling is not None or skip_o_proj or _STATE["stock"]) else _ready(self, hidden_states)
    if ready is None:
        return type(self).forward(self, positions, hidden_states, llama_4_scaling, skip_o_proj)
    if CHECK and _STATE["checked"] < CHECK and not torch.cuda.is_current_stream_capturing():
        return _selfcheck(self, positions, hidden_states, ready)
    return _fused_forward(self, positions, hidden_states, ready)


def _fused_forward(self, positions, hidden_states, ready):
    """vLLM's wrapper forward for this configuration, call for call, with parts A / M / C in place of the small
    kernels (B sits in the MLA layer's forward_impl)."""
    fc, imeta, mslot = ready
    ix, layer = self.indexer, self.mla_attn
    qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
    q_c = self.q_a_layernorm(q_c)
    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)
    q = self.q_b_proj(q_c)[0]
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    if _ON["A"]:
        # KimiK3DSAIndexer.forward with the key insert and the query quant in one launch, then its score()
        index_q = ix.wq_index(q_c)[0].view(-1, ix.index_num_heads, ix.index_head_dim)
        index_k, index_weights = ix.wkv_index(hidden_states)[0].split([ix.index_head_dim, ix.index_num_heads], dim=-1)
        norm = ix.k_index_norm
        weight_scale = ix.index_softmax_scale * ix.indexer_weight_scale
        mla = _ON["M"]
        q_fp8, w_bf16 = prologue(index_k, norm.weight, norm.bias, norm.eps, imeta.slot_mapping, ix.k_cache.kv_cache,
                                 ix.topk_indices_buffer, index_q, index_weights, weight_scale,
                                 ix.padded_index_num_heads, _STATE["mx"],
                                 kv_c_normed if mla else None, k_pe.squeeze(1) if mla else None,
                                 mslot if mla else None, layer.kv_cache if mla else None)
        if mla:
            layer._k31_kv_skip = (kv_c_normed.data_ptr(), tuple(kv_c_normed.shape))
        _PEND["q"] = (index_q.data_ptr(), tuple(index_q.shape), q_fp8, w_bf16)
        try:
            ix.score(hidden_states, index_q, index_weights, None)
        finally:
            _PEND["q"] = None
    else:
        self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
    layer.impl.record_logical_topk_ready()
    try:
        attn_out = layer(q, kv_c_normed, k_pe, output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim))
    finally:
        layer._k31_kv_skip = None
    g = self.g_proj(hidden_states)[0]
    lut = _STATE["lut"]
    if lut is not None and attn_out.dtype == torch.bfloat16 and g.dtype == torch.bfloat16 and attn_out.dim() == 2 \
            and attn_out.shape == g.shape and attn_out.stride(1) == 1 and g.stride(1) == 1:
        attn_out = gate(attn_out, g, lut)
    else:
        attn_out = attn_out * g.sigmoid()
    return self.o_proj(attn_out)[0]


def _forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, output_scale=None,
                  output_block_scale=None, quant_group_size=None, quant_scale_ue8m0=None, quant_col_major=None,
                  quant_tma_aligned=None, q_dcp_replicated=None):
    """MLAAttention.forward_impl of an eligible DSA layer: a decode-only batch of <= MAX_ROWS rows runs vLLM's MQA
    branch for this configuration (sparse, fp8 KV, FP8 query, no q padding at the layer, no DCP / ShardQ / output
    quant) with part B in place of the q_pe copy, the FP8 quant and the head pad; anything else is vLLM's own."""
    n = getattr(attn_metadata, "num_actual_tokens", 0) if attn_metadata is not None else 0
    if _STATE["stock"] or not (
            output_scale is None and output_block_scale is None and q_dcp_replicated is None and 0 < n <= MAX_ROWS
            and attn_metadata.num_decode_tokens == n and q.dim() == 3 and q.dtype == torch.bfloat16
            and q.shape[1] == self.num_heads and output.dtype == q.dtype
            and (not self.use_shardq or getattr(attn_metadata, "shardq", None) is None)):
        return type(self).forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output=output,
                                       output_scale=output_scale, output_block_scale=output_block_scale,
                                       quant_group_size=quant_group_size, quant_scale_ue8m0=quant_scale_ue8m0,
                                       quant_col_major=quant_col_major, quant_tma_aligned=quant_tma_aligned,
                                       q_dcp_replicated=q_dcp_replicated)
    output_padded = output
    output = output[:n, ...]
    q = q[:n, ...]
    kv_cache = kv_cache.view(_FP8_DTYPE)
    mqa_q_nope, mqa_q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    mqa_q_nope = mqa_q_nope.transpose(0, 1)
    N, B, P = mqa_q_nope.shape
    W_UK_T = self.W_UK_T
    L = W_UK_T.shape[2]
    R = mqa_q_pe.shape[-1]
    mqa_q_cat = mqa_q_nope.new_empty((B, N, L + R))
    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_q_cat[..., :L].transpose(0, 1))
    heads = _ORIG["heads"](N)
    padded = qpad(mqa_q_cat, mqa_q_pe, L, heads)
    mqa_q = padded[:, :N] if heads != N else padded
    if heads != N:
        _PEND["pad"] = (mqa_q.data_ptr(), tuple(mqa_q.shape), padded)
    try:
        attn_out, _ = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)
    finally:
        _PEND["pad"] = None
    self._v_up_proj(attn_out, out=output)
    return output_padded


# ---------------------------------------------------------------------------------------------------------------------
# K31_DSA_FUSED_CHECK: vLLM's path, then the fused path, on the same inputs
# ---------------------------------------------------------------------------------------------------------------------

def _snapshot(w, T, imeta, mslot):
    """Top-k rows [:T], and per row with a slot: the indexer cache's key bytes + scale bytes, the MLA cache entry."""
    ix, L = w.indexer, w.mla_attn
    out = [ix.topk_indices_buffer[:T].clone()]
    u8 = ix.k_cache.kv_cache.view(torch.uint8)
    bs, D = u8.shape[1], ix.index_head_dim
    mc = L.kv_cache.view(torch.uint8)
    for slots, kind in ((imeta.slot_mapping, "i"), (mslot.flatten(), "m")):
        for t, s in enumerate(slots[:T].tolist()):
            if s < 0:
                continue
            if kind == "i":
                blk = u8[s // bs].reshape(-1)
                o = s % bs
                out.append(torch.cat([blk[o * D:(o + 1) * D], blk[bs * D + o * 4:bs * D + o * 4 + 4]]).clone())
            else:
                out.append(mc[s // mc.shape[1], s % mc.shape[1]].clone())
    return out


def _selfcheck(w, positions, hidden_states, ready):
    fc, imeta, mslot = ready
    T = hidden_states.shape[0]
    _STATE["stock"] = True
    try:
        ref = type(w).forward(w, positions, hidden_states)
    finally:
        _STATE["stock"] = False
    snap_ref = _snapshot(w, T, imeta, mslot)
    out = _fused_forward(w, positions, hidden_states, ready)
    snap = _snapshot(w, T, imeta, mslot)
    _STATE["checked"] += 1
    same_out = ref.shape == out.shape and torch.equal(ref.view(torch.int16), out.view(torch.int16))
    same_state = len(snap) == len(snap_ref) and all(torch.equal(a, b) for a, b in zip(snap_ref, snap))
    log = _STATE["log"] or (lambda m: None)
    if not (same_out and same_state):
        _STATE["bad"] += 1
        d = (ref.float() - out.float()).abs().max().item() if ref.shape == out.shape else float("nan")
        nd = sum(int(not torch.equal(a, b)) for a, b in zip(snap_ref, snap))
        log(f"DSA fused self-check MISMATCH ({w.mla_attn.layer_name}, {T} rows): output equal {same_out} (max |diff| "
            f"{d:.3g}), cache / top-k pieces differing {nd} of {len(snap)}")
    if _STATE["checked"] % 24 == 0 or _STATE["checked"] == CHECK:
        log(f"DSA fused self-check: {_STATE['checked']} calls, {_STATE['bad']} mismatches"
            + (" (all equal)" if _STATE["bad"] == 0 else ""))
    return out
