"""Dense linears as W4A8 through DeepGEMM's FP8 x FP4 GEMM (hook format `mxfp4dg`, fp6.py): MXFP4 weights (E2M1, one
UE8M0 scale per 32 along K), MXFP8 activations (E4M3, one UE8M0 scale per 32: the smallest power of two with
amax / scale <= 448, round to nearest even), BF16 output.

Meant for the KDA input projection (49,376 x 7,168, 69 layers: ~40% of the dense bytes): at that width DeepGEMM has
386 N tiles, so its decode-size occupancy problem (w8a16.py) does not apply, and FP4 halves the bytes of W8A16.
Runner (gen32, weights only): MXFP4 on the KDA input projection +0.0044. NVFP4 weights are not accepted by
DeepGEMM 2.8.1's fp8_fp4 GEMM ("Unknown SF transformation").

The weights are cast and their scales packed into DeepGEMM's layout once at load. `quant` writes the activation codes
and the packed scales (four UE8M0 bytes per int32, groups 4j .. 4j + 3 of a row; MN-major with the row count padded
to 4: int32 [K / 128, M4], viewed as [M, K / 128] with strides (1, M4)) in one launch, so the GEMM call needs no
transform.
"""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/third_party")


def _dg():
    import deep_gemm as dg
    return dg


# K31_MXFP4_PAD=1: weights whose N is not a multiple of 128 (the KDA in_proj, 49,376) get zero rows up to the next
# multiple (49,408): DeepGEMM then runs whole N tiles; linear() returns the first N columns (a view)
PAD = os.environ.get("K31_MXFP4_PAD", "0") == "1"


def prepare(w: torch.Tensor):
    """BF16 weight [N, K] -> (codes int8 [Np, K / 2], packed scales in DeepGEMM's layout); Np = N, or N rounded up to
    128 with K31_MXFP4_PAD."""
    from deep_gemm.utils.math import per_token_cast_to_fp4
    N, K = w.shape
    assert K % 128 == 0
    if PAD and N % 128:
        w = torch.cat([w, w.new_zeros(128 - N % 128, K)])
        N = w.shape[0]
    q, sf = per_token_cast_to_fp4(w.to(torch.bfloat16).contiguous(), use_ue8m0=True, gran_k=32)
    sft = _dg().transform_sf_into_required_layout(sf, N, K, [1, 32])
    return q.contiguous(), sft


@triton.jit
def _mx8(x_ptr, q_ptr, s_ptr, K, M4, NG4: tl.constexpr, NG4P: tl.constexpr):
    tl.extra.cuda.gdc_wait()                     # PDL: the previous kernel's x is visible
    tl.extra.cuda.gdc_launch_dependents()        # DeepGEMM's GEMM (PDL in vLLM) may launch now
    m = tl.program_id(0)
    j = tl.arange(0, NG4P)                       # int32 word j: groups 4j .. 4j + 3
    b = tl.arange(0, 4)
    c = tl.arange(0, 32)
    jm = j < NG4
    off = m * K + (j[:, None, None] * 4 + b[None, :, None]) * 32 + c[None, None, :]
    x = tl.load(x_ptr + off, mask=jm[:, None, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=2)                                                 # [NG4P, 4]
    code = tl.maximum(((amax.to(tl.int32, bitcast=True) + 0x1FFFFF) >> 23) - 8, 0)
    inv = ((254 - code) << 23).to(tl.float32, bitcast=True)
    y = tl.minimum(tl.maximum(x * inv[:, :, None], -448.0), 448.0)
    tl.store(q_ptr + off, y.to(tl.float8e4nv), mask=jm[:, None, None])
    word = tl.sum(code << (8 * b[None, :]), axis=1)
    tl.store(s_ptr + j * M4 + m, word, mask=jm)


_BUF: dict = {}


def quant(x: torch.Tensor):
    """x [M, K] bf16 -> (codes float8_e4m3fn [M, K], packed scales int32 viewed [M, K / 128] with strides (1, M4))."""
    M, K = x.shape
    M4 = (M + 3) // 4 * 4
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((K // 128) * M4, dtype=torch.int32, device=x.device)
    NG4 = K // 128
    _mx8[(M,)](x, q, s, K, M4, NG4=NG4, NG4P=triton.next_power_of_2(NG4), num_warps=4,
               launch_pdl=os.environ.get("K31_PDL", "0") == "1")
    return q, torch.as_strided(s, (M, NG4), (1, M4))


def linear(x: torch.Tensor, q: torch.Tensor, sft: torch.Tensor, N: int, out: torch.Tensor = None) -> torch.Tensor:
    M = x.shape[0]
    Np = q.shape[0]                              # > N for a padded weight (K31_MXFP4_PAD)
    if out is None or Np != N:
        out = torch.empty((M, Np), dtype=torch.bfloat16, device=x.device)
    xa = quant(x.contiguous())
    _dg().fp8_fp4_gemm_nt(xa, (q, sft), out, recipe_a=(1, 32), recipe_b=(1, 32), disable_ue8m0_cast=True)
    return out if Np == N else out[:, :N]


class MarlinW4:
    """K31_DENSE_FMT ...:marlin4: a BF16 weight [N, K] as MXFP4 (the codes and UE8M0 scales prepare() would give
    DeepGEMM: per_token_cast_to_fp4, gran_k 32) repacked for vLLM's Marlin GEMM, which takes BF16 inputs as they come
    (W4A16: no MXFP8 activation step). N is padded to Marlin's 64 inside; the output is the first N columns (a view)."""

    def __init__(self, w: torch.Tensor):
        from deep_gemm.utils.math import per_token_cast_to_fp4
        from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mf4
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (marlin_make_workspace_new,
                                                                                 marlin_padded_nk)
        from vllm.scalar_type import scalar_types
        N, K = w.shape
        q, sf = per_token_cast_to_fp4(w.to(torch.bfloat16).contiguous(), use_ue8m0=True, gran_k=32)
        e8 = (torch.log2(sf.float()).round() + 127).clamp(0, 254).to(torch.uint8).contiguous()
        L = torch.nn.Module()
        L.weight = torch.nn.Parameter(q.view(torch.uint8).contiguous(), requires_grad=False)
        L.weight_scale = torch.nn.Parameter(e8, requires_grad=False)
        L.output_size_per_partition, L.input_size_per_partition = N, K
        L.params_dtype = torch.bfloat16
        mf4.prepare_fp4_layer_for_marlin(L)
        self.w, self.s = L.weight.data, L.weight_scale.data
        self.N, self.K = N, K
        self.pn, self.pk = marlin_padded_nk(N, K, 32)
        assert self.pk == K
        self.ws = marlin_make_workspace_new(w.device, 4)
        self.tid = scalar_types.float4_e2m1f.id
        self.nbytes = self.w.numel() * self.w.element_size() + self.s.numel() * self.s.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.stride(-1) != 1 or (x.shape[0] > 1 and x.stride(0) != x.shape[1]):
            x = x.contiguous()
        y = torch.ops.vllm.marlin_gemm(a=x, b_q_weight=self.w, b_bias=None, b_scales=self.s, a_scales=None,
                                       global_scale=None, b_zeros=None, workspace=self.ws, b_q_type_id=self.tid,
                                       size_m=x.shape[0], size_n=self.pn, size_k=self.K, use_atomic_add=False,
                                       use_fp32_reduce=True)
        return y if self.pn == self.N else y[:, :self.N]
