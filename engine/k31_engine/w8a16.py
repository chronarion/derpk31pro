"""Decode-size dense GEMMs (M <= 16 rows) as W8A16: FP8 E4M3 weights with one float32 scale per row and 128-wide K
group, BF16 activations as they come (no activation quantization kernel, no activation rounding), split over K so
that every shape fills the GB300's 152 SMs.

DeepGEMM's FP8 GEMM tiles N by 128 at decode size, so a 7168-wide output runs on 56 CTAs: the engine's dense GEMMs
reach 0.9-4.5 TB/s (3-row decode profile 11-17). Here a program owns a BN x KCH slice (BN output rows, KCH of K), and
the last program of an N tile to finish sums the tile's partials in K order (deterministic: the result does not depend
on which program finishes when) and writes BF16. One launch; a workspace and a counter row per CUDA stream.

y [M, N] = x [M, K] @ (q [N, K] * s [N, K / 128] repeated over each group)^T, accumulated in float32: the FP8 values are
exact in BF16, so the only roundings are the float32 sums and the final BF16 output. Scales are BF16 (amax / 448 rounded;
the group's largest value lands within 0.4% of 448, clamped): 1.6% on top of the FP8 bytes. Power-of-two (UE8M0)
scales would cost 0.8% but put the largest value anywhere in (224, 448]: +3% weight error on Gaussian weights, +21% on
heavy-tailed ones (the group maximum carries much of a heavy-tailed group's energy).
"""
import os

import torch
import triton
import triton.language as tl

from . import flags

GK = 128                                         # scale group along K
# programmatic dependent launch (K31_PDL=1, opt-in): the grid may start while the previous kernel in the stream ends
# (vLLM runs DeepGEMM with PDL on SM90+, so a non-PDL kernel between two DeepGEMM calls would break their overlap);
# every program waits for the previous kernel's writes first (griddepcontrol.wait), so results never depend on timing
PDL = os.environ.get("K31_PDL", "0") == "1"
# K31_W8_A8: 0 = BF16 activations (weights converted per element: ALU-bound, ~3 TB/s); 1 = activations cast to FP8 in
# the kernel per row and 128-group (FP8 x FP8 tensor cores, stock-like activation rounding); 2 = 1 plus the residual as
# a second FP8 term (near-BF16 activations)
A8 = int(os.environ.get("K31_W8_A8", "0"))


@triton.jit
def _w8a16(x_ptr, q_ptr, s_ptr, ws_ptr, cnt_ptr, y_ptr, M, N, K, NG, g_ptr, eps, res_ptr, sxm, rs_ptr,
           BM: tl.constexpr, BN: tl.constexpr, KCH: tl.constexpr, SPLIT: tl.constexpr, W16: tl.constexpr,
           NORM: tl.constexpr, RES: tl.constexpr, A8: tl.constexpr = 0):
    tl.extra.cuda.gdc_wait()                     # the previous kernel's writes (x, a reused workspace) are visible
    tl.extra.cuda.gdc_launch_dependents()        # the next kernel may launch and run up to its own wait
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rm = tl.program_id(2) * BM + tl.arange(0, BM)
    rg = tl.arange(0, 128)
    nm = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    if NORM == 2:                                # K31_TAIL1: the statistic comes from the v4 band finish (rs_ptr)
        rstd = tl.load(rs_ptr + rm, mask=rm < M, other=0.0)
    elif NORM:                                   # RMSNorm of the input rows first (vLLM's rms_norm: fp32 statistics,
        ss = tl.zeros([BM], dtype=tl.float32)    # bf16(x * rstd * weight))
        for k1 in tl.range(0, K, 512):
            kc = k1 + tl.arange(0, 512)
            xx = tl.load(x_ptr + rm[:, None] * sxm + kc[None, :], mask=(rm[:, None] < M) & (kc[None, :] < K),
                         other=0.0).to(tl.float32)
            ss += tl.sum(xx * xx, axis=1)
        rstd = 1.0 / tl.sqrt(ss / K + eps)
    for g in tl.range(0, KCH // 128):
        k0 = pk * KCH + g * 128
        km = (k0 + rg) < K
        x = tl.load(x_ptr + rm[:, None] * sxm + k0 + rg[None, :], mask=(rm[:, None] < M) & km[None, :], other=0.0)
        if NORM:                                 # one rounding, as vLLM's rms_norm (checked: 0 of 28,672 differ)
            gw = tl.load(g_ptr + k0 + rg, mask=km, other=0.0).to(tl.float32)
            x = (x.to(tl.float32) * rstd[:, None] * gw[None, :]).to(tl.bfloat16)
        w = tl.load(q_ptr + rn[:, None] * K + k0 + rg[None, :], mask=nm[:, None] & km[None, :], other=0.0)
        if W16:                                                           # BF16 weights as they are (the router)
            acc += tl.dot(x, tl.trans(w))
        elif A8 == 0:
            part = tl.dot(x, tl.trans(w.to(tl.bfloat16)))                 # [BM, BN] float32
            sc = tl.load(s_ptr + rn * NG + (k0 // 128), mask=nm & ((k0 // 128) < NG), other=0.0).to(tl.float32)
            acc += part * sc[None, :]
        else:
            # K31_W8_A8: the FP8 weights go to the tensor cores as they are (no per-element conversion: the BF16 path is
            # ALU-bound at ~3 TB/s); the small activation tile is scaled per row and 128-group and cast to FP8 here.
            # A8 == 2 adds the rounding residual as a second FP8 term (x 16): ~7 bits of the activation, near BF16.
            xf = x.to(tl.float32)
            amax = tl.max(tl.abs(xf), axis=1)
            sx = tl.where(amax > 0, amax / 448.0, 1.0)
            xs = xf / sx[:, None]
            xh = xs.to(tl.float8e4nv)
            wt = tl.trans(w)
            part = tl.dot(xh, wt)
            if A8 == 2:
                xl = ((xs - xh.to(tl.float32)) * 16.0).to(tl.float8e4nv)
                part += tl.dot(xl, wt) * 0.0625
            sc = tl.load(s_ptr + rn * NG + (k0 // 128), mask=nm & ((k0 // 128) < NG), other=0.0).to(tl.float32)
            acc += part * sx[:, None] * sc[None, :]
    if SPLIT == 1:
        out = acc.to(y_ptr.dtype.element_ty)
        if RES:                                  # y = residual + bf16(result), as residual.add_(result)
            r = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=(rm[:, None] < M) & nm[None, :], other=0.0)
            out = (r.to(tl.float32) + out.to(tl.float32)).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & nm[None, :])
    else:
        tl.store(ws_ptr + (pk * BM + rm[:, None]) * N + rn[None, :], acc, mask=(rm[:, None] < M) & nm[None, :])
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
        if old == SPLIT - 1:                                              # the last slice of this N tile: sum in order
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for j in tl.range(0, SPLIT):
                tot += tl.load(ws_ptr + (j * BM + rm[:, None]) * N + rn[None, :],
                               mask=(rm[:, None] < M) & nm[None, :], other=0.0, cache_modifier=".cg")
            out = tot.to(y_ptr.dtype.element_ty)
            if RES:
                r = tl.load(res_ptr + rm[:, None] * N + rn[None, :], mask=(rm[:, None] < M) & nm[None, :],
                            other=0.0)
                out = (r.to(tl.float32) + out.to(tl.float32)).to(y_ptr.dtype.element_ty)
            tl.store(y_ptr + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & nm[None, :])
            tl.atomic_xchg(cnt_ptr + pn, 0, sem="relaxed", scope="gpu")


def quantize(w: torch.Tensor):
    """BF16 / float weight [N, K] (K % 128 == 0) -> (q float8_e4m3fn [N, K], s bfloat16 [N, K / 128]): one scale per
    row and 128-wide group, amax / 448 in BF16."""
    N, K = w.shape
    assert K % GK == 0
    wf = w.float().view(N, K // GK, GK)
    s = (wf.abs().amax(-1) / 448.0).clamp_min(1e-30).to(torch.bfloat16)
    q = (wf / s.float()[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(N, K)
    return q.contiguous(), s.contiguous()


_WS: dict = {}


def _workspace(dev, stream, split: int, bm: int, n: int, n_tiles: int):
    key = (dev, stream)
    ws, cnt = _WS.get(key, (None, None))
    if ws is None or ws.numel() < split * bm * n or cnt.numel() < n_tiles:
        ws = torch.empty(max(split * bm * n, ws.numel() if ws is not None else 0), dtype=torch.float32, device=dev)
        cnt = torch.zeros(max(n_tiles, cnt.numel() if cnt is not None else 0), dtype=torch.int32, device=dev)
        _WS[key] = (ws, cnt)
    return ws, cnt


_TABLE: dict = {}


def _table() -> dict:
    """K31_W8A16_CFG: the dense benchmark's JSON (tests/bench_dense_dg.py): best configs per shape at 3 rows, and the
    times of DeepGEMM's FP8 GEMM against this kernel's."""
    if not _TABLE:
        import json
        import os
        p = os.environ.get("K31_W8A16_CFG", "")
        _TABLE["t"] = json.load(open(p)) if p and os.path.exists(p) else {}
    return _TABLE["t"]


def _best_us(N: int, K: int):
    """This module's best time for the shape at 3 rows: the split-K kernel's, or Marlin's where it will serve it."""
    a = _table().get("w8a16_best_us", {}).get(f"{N}x{K}", {}).get("3")
    return marlin_us(N, K) if marlin_preferred(N, K) else a


def preferred(N: int, K: int) -> bool:
    """False when the benchmark measured DeepGEMM's FP8 GEMM faster than this kernel for the shape at 3 rows."""
    a = _best_us(N, K)
    b = _table().get("fp8_default", {}).get(f"{N}x{K}", {}).get("3")
    return a is None or b is None or a < b


def faster_than_bf16(N: int, K: int) -> bool:
    """True only when the benchmark (K31_W8A16_CFG) measured this kernel's best config faster than BF16 cuBLAS for
    the shape at 3 rows (the drafter's linears and LM head run BF16 today; no measurement: keep BF16)."""
    a = _best_us(N, K)
    b = _table().get("bf16_us", {}).get(f"{N}x{K}", {}).get("3")
    return a is not None and b is not None and a < b


# K31_W8_MARLIN=1: the FP8 weights of the shapes where tests/bench_marlin.py measured vLLM's Marlin GEMM faster than
# the split-K kernel at 3 rows (K31_MARLIN_TABLE, that benchmark's JSON) are repacked for Marlin at load and served by
# it (the same FP8 codes and group-128 scales: only the summation order differs); "all" = every shape. linear()
# dispatches on the weight's type, so every caller (dense layers, drafter linears, LM head copy) follows.
MARLIN = os.environ.get("K31_W8_MARLIN", "")
_MTABLE: dict = {}


def marlin_us(N: int, K: int, M: int = 3):
    if "t" not in _MTABLE:
        import json
        p = os.environ.get("K31_MARLIN_TABLE", "")
        _MTABLE["t"] = json.load(open(p)) if p and os.path.exists(p) else {}
    return (_MTABLE["t"].get("marlin8_us", {}).get(f"{N}x{K}") or {}).get(str(M))


def marlin_preferred(N: int, K: int) -> bool:
    if not MARLIN or N % 64 or K % GK:
        return False
    if MARLIN == "all":
        return True
    m = marlin_us(N, K)
    a = _table().get("w8a16_best_us", {}).get(f"{N}x{K}", {}).get("3")
    return m is not None and (a is None or m < 0.95 * a)


class MarlinW:
    """An FP8 W8A16 weight (quantize's q, s) repacked for vLLM's Marlin GEMM: FP8 E4M3 codes in Marlin's fragment
    order, the group-128 scales permuted with the BF16 exponent bias (2^120) folded in. Its own lock workspace:
    side-stream linears may run a Marlin call beside another."""

    def __init__(self, q: torch.Tensor, sc: torch.Tensor):
        from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mf8
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.scalar_type import scalar_types
        N, K = q.shape
        L = torch.nn.Module()
        L.weight = torch.nn.Parameter(q, requires_grad=False)
        L.weight_scale = torch.nn.Parameter(sc, requires_grad=False)
        L.weight_block_size = [1, GK]
        L.output_size_per_partition, L.input_size_per_partition = N, K
        L.orig_dtype = torch.bfloat16
        mf8.prepare_fp8_layer_for_marlin(L, size_k_first=False)
        self.w, self.s = L.weight.data, L.weight_scale.data
        self.N, self.K = N, K
        self.pn, pk = mf8.marlin_repacked_nk(self.w, num_bits=8)
        assert pk == K and self.pn == N, f"Marlin would pad {N} x {K} to {self.pn} x {pk}"
        self.ws = marlin_make_workspace_new(q.device, 4)
        self.tid = scalar_types.float8_e4m3fn.id
        self.shape = (N, K)
        self.nbytes = self.w.numel() * self.w.element_size() + self.s.numel() * self.s.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.stride(-1) != 1 or (x.shape[0] > 1 and x.stride(0) != x.shape[1]):
            x = x.contiguous()
        return torch.ops.vllm.marlin_gemm(a=x, b_q_weight=self.w, b_bias=None, b_scales=self.s, a_scales=None,
                                          global_scale=None, b_zeros=None, workspace=self.ws, b_q_type_id=self.tid,
                                          size_m=x.shape[0], size_n=self.N, size_k=self.K, use_atomic_add=False,
                                          use_fp32_reduce=True)


def maybe_marlin(q: torch.Tensor, sc: torch.Tensor):
    """(q, s) as they are, or (MarlinW, None) when K31_W8_MARLIN serves the shape."""
    N, K = q.shape
    if not marlin_preferred(N, K):
        return q, sc
    return MarlinW(q, sc), None


# K31_W8A16_OVERRIDE="NxK=BN,KCH,SPLIT,warps,stages;...": configs that win over the benchmark table, for kernels whose
# best standalone config is wrong in the engine: a side-stream GEMM must fit one wave (<= ~4 x 152 CTAs) or its pending
# CTAs stall the main stream's next launches (the latent down's 1,120-CTA config costs 5.6 us per layer)
_OVR: dict = {}
for _item in [v for v in os.environ.get("K31_W8A16_OVERRIDE", "").split(";") if v]:
    _shape, _, _cfg = _item.partition("=")
    _OVR[_shape.strip()] = tuple(int(v) for v in _cfg.split(","))


def config(M: int, N: int, K: int, sms: int = 152):
    """(BN, KCH, SPLIT[, warps, stages]): the benchmark's best for the shape if K31_W8A16_CFG has it, else BN 64 or 128
    rows per program, K slices of a whole number of groups, about 3 programs per SM."""
    ovr = _OVR.get(f"{N}x{K}")
    if ovr is not None:
        return ovr
    best = _table().get("best", {}).get(f"{N}x{K}", {}).get("3")
    if best is not None:
        return tuple(best)
    BN = 128 if N >= 8192 else 64
    nt = triton.cdiv(N, BN)
    groups = K // GK
    if nt >= 2 * sms:                            # wide outputs fill the SMs alone: no split (49376 x 7168, 11-18
        return BN, groups * GK, 1, 4, 4          # contended smoke: split 2 229 us, no split + 4 stages 111 us)
    want = max(1, (3 * sms + nt - 1) // nt)
    split = min(groups, want)
    per = triton.cdiv(groups, split)
    split = triton.cdiv(groups, per)
    return BN, per * GK, split


def _big_m(M: int, K: int) -> str:
    """K31_W8A16_BIG_M (runtime flag, read per eager call; default 0 = today's tiles): the bigm.py implementation for
    a call of M rows ("1" / "triton", "hilo", "cublas"), or "" for today's path. Only calls of at least
    K31_W8A16_BIG_M_ROWS rows (default 1,024) with K >= K31_W8A16_BIG_M_MIN_K (default 1,024: f_b_proj's K = 128
    stays), never under CUDA-graph capture."""
    if M <= 16:
        return ""
    impl = flags.get("K31_W8A16_BIG_M", "0")
    if impl in ("", "0", "off") or M < flags.num("K31_W8A16_BIG_M_ROWS", 1024) or \
            K < flags.num("K31_W8A16_BIG_M_MIN_K", 1024) or torch.cuda.is_current_stream_capturing():
        return ""
    return impl


def linear(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, out: torch.Tensor = None, cfg=None) -> torch.Tensor:
    """y = x @ dequant(q, s)^T for x [M, K] bf16. M <= 16: split over K (decode); larger M (prompt chunks): 64-row tiles,
    no split."""
    if not isinstance(q, torch.Tensor) and not isinstance(q, MarlinW):   # another kernel's weight (w8i.I8W)
        return q(x, out)
    if isinstance(q, MarlinW):                       # K31_W8_MARLIN: this weight lives in Marlin's layout
        impl = _big_m(x.shape[0], x.shape[1])
        if impl:                                     # prompt chunks: Marlin's layout read by bigm.py's GEMM
            from . import bigm
            return bigm.marlin_linear(x, q, out, impl=impl)
        y = q(x)
        if out is not None:
            out.copy_(y)
            return out
        return y
    M, K = x.shape
    N = q.shape[0]
    assert q.shape[1] == K and s.shape == (N, K // GK) and x.stride(1) == 1 and (M == 1 or x.stride(0) >= K)
    sxm = x.stride(0) if M > 1 else K    # rows may be strided (a column slice of a wider output): no copy
    if M > 16:
        impl = _big_m(M, K) if A8 == 0 else ""      # K31_W8A16_BIG_M: prompt chunks through bigm.py
        if impl:
            from . import bigm
            return bigm.w8_linear(x, q, s, out, impl=impl)
        if out is None:
            out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        nt = triton.cdiv(N, 128)
        _w8a16[(nt, 1, triton.cdiv(M, 64))](x, q, s, out, out, out, M, N, K, K // GK, x, 0.0, out, sxm, x, BM=64, BN=128,
                                            KCH=triton.cdiv(K, GK) * GK, SPLIT=1, W16=False, NORM=False,
                                            RES=False, A8=A8, num_warps=8, num_stages=3, launch_pdl=PDL)
        return out
    cfg = cfg or config(M, N, K)
    BN, KCH, SPLIT = cfg[:3]
    nw, ns = (cfg[3], cfg[4]) if len(cfg) > 3 else (4, 3)
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    nt = triton.cdiv(N, BN)
    ws, cnt = _workspace(x.device, torch.cuda.current_stream().cuda_stream, SPLIT, 16, N, nt)
    _w8a16[(nt, SPLIT)](x, q, s, ws, cnt, out, M, N, K, K // GK, x, 0.0, out, sxm, x, BM=16, BN=BN, KCH=KCH, SPLIT=SPLIT,
                        W16=False, NORM=False, RES=False, A8=A8, num_warps=nw, num_stages=ns, launch_pdl=PDL)
    return out


def linear_norm_res(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, g: torch.Tensor, eps: float,
                    res: torch.Tensor, cfg=None, rstd: torch.Tensor = None) -> torch.Tensor:
    """res += dequant(q, s) @ rmsnorm(x; g, eps) in place, for x [M, K] bf16 (M <= 16), res [M, N] bf16: the routed
    latent's RMSNorm, the up-projection and the shared-expert residual add of KimiRoutedOutputTransform in one launch
    (K31_FUSE_UP). Rounding as the three separate ops: bf16(x * rstd * weight), GEMM -> bf16, then the add."""
    M, K = x.shape
    N = q.shape[0]
    assert M <= 16 and q.shape[1] == K and s.shape == (N, K // GK) and x.is_contiguous() and res.is_contiguous()
    assert res.shape == (M, N) and res.dtype == torch.bfloat16 and g.numel() == K
    # rstd (K31_TAIL1): [>= M] float32 rsqrt(mean(x^2) + eps) per row from the v4 band finish: no statistic prologue
    # in each program (that prologue made the fused kernel 2.2x the unfused trio, 11-18)
    assert rstd is None or (rstd.dtype == torch.float32 and rstd.numel() >= M)
    cfg = cfg or config(M, N, K)
    BN, KCH, SPLIT = cfg[:3]
    nw, ns = (cfg[3], cfg[4]) if len(cfg) > 3 else (4, 3)
    nt = triton.cdiv(N, BN)
    ws, cnt = _workspace(x.device, torch.cuda.current_stream().cuda_stream, SPLIT, 16, N, nt)
    _w8a16[(nt, SPLIT)](x, q, s, ws, cnt, res, M, N, K, K // GK, g, float(eps), res, K,
                        rstd if rstd is not None else x, BM=16, BN=BN, KCH=KCH, SPLIT=SPLIT, W16=False,
                        NORM=2 if rstd is not None else True, RES=True, A8=A8, num_warps=nw, num_stages=ns,
                        launch_pdl=PDL)
    return res


def linear16(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """y (float32) = x @ w^T for x [M, K] bf16 (M <= 16) and BF16 weights w [N, K]: the router's gate GEMV, split over
    K with the same in-order reduction (K31_ROUTER_GEMV)."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 16 and w.shape[1] == K and K % GK == 0 and x.is_contiguous() and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    BN = 64
    nt = triton.cdiv(N, BN)
    groups = K // GK
    split = min(groups, max(1, (3 * 152 + nt - 1) // nt))
    per = triton.cdiv(groups, split)
    split = triton.cdiv(groups, per)
    ws, cnt = _workspace(x.device, torch.cuda.current_stream().cuda_stream, split, 16, N, nt)
    _w8a16[(nt, split)](x, w, w, ws, cnt, out, M, N, K, groups, x, 0.0, out, K, x, BM=16, BN=BN, KCH=per * GK,
                        SPLIT=split, W16=True, NORM=False, RES=False, num_warps=4, num_stages=3, launch_pdl=PDL)
    return out
