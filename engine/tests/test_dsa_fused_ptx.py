"""dsa_fused (K31_DSA_FUSED), static: the fused kernels and vLLM's originals compiled for sm_103 on CPU (Triton's
ahead-of-time path, no GPU), with the arguments specialized as a GB300 launch would (16-byte aligned pointers, strides
that are multiples of 16), and their generated code compared.

Run in the engine image, CPU only (from the repo root):
    docker run --rm --network none --cpuset-cpus 64-65 --user $(id -u):$(id -g) -e HOME=/tmp \
        -e CUDA_VISIBLE_DEVICES= -v "$PWD/engine":/engine:ro \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_dsa_fused_ptx.py
Exit 0 and "DSA FUSED PTX: PASS" = every check holds.

1. Part A (+ M): every tt.reduce of the fused prologue has the layout of vLLM's (idxk's LayerNorm sums and max, idxq's
   max), so the sums keep their order; the PTX's floating-point arithmetic opcodes are exactly idxk's + idxq's (same
   counts per opcode), the only additions being the MLA insert's conversions and one cvt.rn.bf16.f32 (the BF16 store
   of the folded weight).
2. Part B: the clamp is NaN-dropping min / max (no .NaN modifier) and the conversion cvt.rn.satfinite.e4m3x2.f32.
3. Part C: one fp32 multiply (mul.f32 / mul.f32x2, no .ftz) and cvt.rn.bf16*.f32; no native BF16 multiply.
"""
import collections
import os
import re
import sys

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from vllm.models.kimi_k3.nvidia.ops import dsa_indexer as dix   # noqa: E402

from k31_engine import dsa_fused as F                           # noqa: E402

TGT = GPUTarget("cuda", 103, 32)
TY = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.int64: "i64", torch.int32: "i32",
      torch.float8_e4m3fn: "fp8e4nv", torch.int16: "i16"}
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def compile_fn(fn, args, consts, num_warps):
    """Triton's launch-time specialization by hand: a dtype = an aligned tensor (tt.divisibility 16), an int == 1 a
    constant, an int % 16 == 0 divisible (unless do_not_specialize), a float fp32."""
    sig, cx, attrs = {}, dict(consts), {}
    dns = set(getattr(fn, "do_not_specialize", None) or [])
    dns |= {fn.arg_names[i] for i in dns if isinstance(i, int)}
    for i, n in enumerate(fn.arg_names):
        if n in consts:
            sig[n] = "constexpr"
            continue
        v = args[n]
        if isinstance(v, torch.dtype):
            sig[n] = "*" + TY[v]
            attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, float):
            sig[n] = "fp32"
        elif v == 1 and n not in dns:
            sig[n], cx[n] = "constexpr", 1
        else:
            sig[n] = "i32" if -2 ** 31 <= v < 2 ** 31 else "i64"
            if v % 16 == 0 and n not in dns:
                attrs[(i,)] = [["tt.divisibility", 16]]
    return triton.compile(ASTSource(fn, sig, cx, attrs), target=TGT, options={"num_warps": num_warps})


def reduce_layouts(ttgir):
    defs = dict(re.findall(r"^(#\w+) = (#ttg\.\w+<\{.*?\}>)", ttgir, re.M))
    out = []
    for m in re.finditer(r'"tt\.reduce"\(([^)]*)\).*?\}\) : \((tensor<[^>]*>)', ttgir, re.S):
        lay = re.search(r", (#\w+)>$", m.group(2))
        out.append((m.group(2).split(",")[0], defs.get(lay.group(1)) if lay else None))
    return out


def opcodes(ptx):
    c = collections.Counter()
    for line in ptx.splitlines():
        m = re.match(r"\s*(?:@!?%p\d+\s+)?([a-z][\w.]*)", line)
        if m:
            c[m.group(1)] += 1
    return c


FP_ARITH = re.compile(r"^(abs|add|sub|mul|fma|div|rcp|rsqrt|sqrt|ex2|lg2|max|min|neg|shfl)\.")
bf, f32, i64, i32, f8 = torch.bfloat16, torch.float32, torch.int64, torch.int32, torch.float8_e4m3fn
BLK, ISTR, MBLK, MSTR, TOPK, T = 16384, 325582848, 16384, 325582848, 2048, 5    # the packed KV row: one stride
idxk = compile_fn(dix._indexer_k_norm_quant_insert_kernel,
                  dict(k_ptr=bf, k_stride=160, norm_w_ptr=f32, norm_b_ptr=f32, norm_eps=1e-6, slot_mapping_ptr=i64,
                       num_slots=T, cache_ptr=f8, cache_scale_ptr=i32, cache_block_size=BLK, cache_block_stride=ISTR,
                       topk_ptr=i32, topk_stride=TOPK),
                  dict(HEAD_DIM=128, TOPK=TOPK, TOPK_BLOCK=1024, MX_SCALE=True, FP8_MAX=448.0, USE_FNUZ=False), 1)
idxq = compile_fn(dix._indexer_q_quant_fold_weights_kernel,
                  dict(q_ptr=bf, q_stride0=4096, q_stride1=128, w_ptr=bf, w_stride=160, q_fp8_ptr=f8,
                       q_fp8_stride0=4096, q_fp8_stride1=128, w_out_ptr=f32, w_out_stride=32, weight_scale=2.0 ** -6),
                  dict(HEAD_DIM=128, FP8_MAX=448.0, USE_FNUZ=False), 1)
pro = compile_fn(F._prologue_kernel,
                 dict(k_ptr=bf, k_stride=160, kn_w_ptr=f32, kn_b_ptr=f32, kn_eps=1e-6, islot_ptr=i64, num_islots=T,
                      icache_ptr=f8, iscale_ptr=i32, iblock_size=BLK, iblock_stride=ISTR, topk_ptr=i32,
                      topk_stride=TOPK, q_ptr=bf, q_stride0=4096, q_stride1=128, w_ptr=bf, w_stride=160, qf_ptr=f8,
                      qf_stride0=4096, qf_stride1=128, wo_ptr=bf, wo_stride=32, weight_scale=2.0 ** -6, kvc_ptr=bf,
                      kvc_stride=512, kpe_ptr=bf, kpe_stride=2112, mslot_ptr=i64, num_mslots=T, mcache_ptr=f8,
                      mblock_size=MBLK, mblock_stride=MSTR, mentry_stride=576),
                 dict(HEAD_DIM=128, TOPK=TOPK, TOPK_BLOCK=1024, MX_SCALE=True, FP8_MAX=448.0, USE_FNUZ=False,
                      KV_DIM=512, PE_DIM=64, MLA=True), 1)
r_ref = reduce_layouts(idxk.asm["ttgir"]) + reduce_layouts(idxq.asm["ttgir"])
r_fus = reduce_layouts(pro.asm["ttgir"])
check("A: reductions (count, operand type, layout) = idxk's + idxq's",
      len(r_fus) == len(r_ref) == 4 and sorted(map(str, r_fus)) == sorted(map(str, r_ref)), f"{r_fus}")
ck, cq, cf = opcodes(idxk.asm["ptx"]), opcodes(idxq.asm["ptx"]), opcodes(pro.asm["ptx"])
arith = sorted(op for op in set(ck) | set(cq) | set(cf) if FP_ARITH.match(op) and (".f" in op or "shfl" in op
                                                                                  or ".bf16" in op))
diff = {op: (ck[op] + cq[op], cf[op]) for op in arith if ck[op] + cq[op] != cf[op]}
check("A: floating-point arithmetic opcodes of the fused kernel = idxk's + idxq's, count for count", not diff,
      f"(differing {diff}; compared {len(arith)} opcodes)")
extra = {op: cf[op] - ck[op] - cq[op] for op in cf if op.startswith("cvt.") and cf[op] != ck[op] + cq[op]
         and re.search(r"\.(f32|bf16|f16|f64|e4m3|e5m2)", op)}                  # floating-point conversions only
check("A: added conversions are the MLA insert's (bf16 -> f32 -> E4M3) and one cvt.rn.bf16.f32",
      set(extra) <= {"cvt.f32.bf16", "cvt.rn.satfinite.e4m3x2.f32", "cvt.rn.bf16.f32"} and extra.get(
          "cvt.rn.bf16.f32") == 1, f"{extra}")
qp = compile_fn(F._qpad_kernel, dict(cat_ptr=bf, cat_s0=96 * 576, cat_s1=576, pe_ptr=bf, pe_s0=96 * 192, pe_s1=192,
                                     out_ptr=f8, out_s0=128 * 576, out_s1=576, NH=96), dict(L=512, R=64, FP8_MAX=448.0),
                2)
cb = opcodes(qp.asm["ptx"])
mm = [op for op in cb if re.match(r"^(min|max)\.", op)]
check("B: NaN-dropping min / max (no .NaN), satfinite RNE to E4M3, no other float arithmetic",
      mm and all(".NaN" not in op for op in mm) and cb["cvt.rn.satfinite.e4m3x2.f32"] > 0
      and not [op for op in cb if FP_ARITH.match(op) and op not in mm and (".f" in op or ".bf16" in op)], f"{dict(cb)}")
gt = compile_fn(F._gate_kernel, dict(a_ptr=torch.int16, a_stride=12288, g_ptr=torch.int16, g_stride=12288,
                                     lut_ptr=torch.int16, o_ptr=bf, o_stride=12288, N=12288), dict(BLOCK=1024), 4)
cg = opcodes(gt.asm["ptx"])
fmul = [op for op in cg if re.match(r"^mul\.(rn\.)?f32", op)]
check("C: fp32 multiply without .ftz, RNE to BF16, no native BF16 multiply",
      fmul and all(".ftz" not in op for op in fmul) and any(op.startswith("cvt.rn.bf16") for op in cg)
      and not [op for op in cg if re.match(r"^(mul|fma)\..*bf16", op)], f"{dict(cg)}")
print(f"DSA FUSED PTX: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
