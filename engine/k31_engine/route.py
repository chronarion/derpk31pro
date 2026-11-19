"""K3.1's router for decode-size calls (T <= 16) in one launch: the gate GEMV (BF16 weights [896, 7168], float32
logits) split over N and K across the SMs, and in the last program to finish the selection vLLM's grouped_topk kernel
makes for one group and sigmoid scoring:

  scores = sigmoid(logits); biased = scores + e_score_correction_bias; top-16 of biased in descending order (ties: lower
  expert id first); weights = scores of the chosen / their sum (if renormalize) x routed_scaling_factor.

The engine's chain is the cute_dsl gate GEMV (7.2-7.9 us at 3 rows, 1.7 TB/s) and the top-k kernel (4.7-6.3 us) on
the main stream (11-17 trace). v1 (11-17) did the selection row by row with three reductions per pick (53.8 us); v2
selects all rows at once with one reduction per pick: each candidate is a 64-bit key (order-preserving bits of the
biased score << 10 | 1023 - expert), so the row maximum gives the score and the lowest id among ties together, and the
chosen scores are gathered from a scratch row afterwards. Partial logits are summed in K order by the last program
(deterministic). One workspace and counter per CUDA stream. PDL (K31_PDL=1): waits at entry, releases its dependents
at once (they wait for its completion before reading).
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

PDL = os.environ.get("K31_PDL", "0") == "1"


@triton.jit
def _route(x_ptr, w_ptr, b_ptr, ws_ptr, sb_ptr, cnt_ptr, ids_ptr, wt_ptr, T, K, scale,
           BM: tl.constexpr, BN: tl.constexpr, KCH: tl.constexpr, SPLIT: tl.constexpr, NE: tl.constexpr,
           NEP: tl.constexpr, TOPK: tl.constexpr, NPROG: tl.constexpr, TB: tl.constexpr, RENORM: tl.constexpr,
           TAIL: tl.constexpr = 2):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    rg = tl.arange(0, 128)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in tl.range(0, KCH // 128):
        k0 = pk * KCH + g * 128
        km = (k0 + rg) < K
        x = tl.load(x_ptr + rm[:, None] * K + k0 + rg[None, :], mask=(rm[:, None] < T) & km[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + k0 + rg[None, :], mask=(rn[:, None] < NE) & km[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(ws_ptr + (pk * BM + rm[:, None]) * NE + rn[None, :], acc, mask=(rm[:, None] < T) & (rn[None, :] < NE))
    tl.debug_barrier()
    old = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if old == NPROG - 1 and TAIL > 0:            # every partial is in: logits, scores, the top-k of all rows at once
        e = tl.arange(0, NEP)
        r = tl.arange(0, TB)
        em = e < NE
        m2 = (r[:, None] < T) & em[None, :]
        logit = tl.zeros([TB, NEP], dtype=tl.float32)
        for j in tl.static_range(SPLIT):
            logit += tl.load(ws_ptr + (j * BM + r[:, None]) * NE + e[None, :], mask=m2, other=0.0,
                             cache_modifier=".cg")
        s = 1.0 / (1.0 + libdevice.exp(-logit))
        bias = tl.load(b_ptr + e, mask=em, other=0.0)
        biased = s + bias[None, :]
        bits = biased.to(tl.int32, bitcast=True)
        ordered = tl.where(bits >= 0, bits.to(tl.int64) + 2147483648, (~bits).to(tl.int64))
        key = tl.where(m2, (ordered << 10) | (1023 - e[None, :]).to(tl.int64), -1)
        kk = tl.arange(0, TOPK)
        ids = tl.zeros([TB, TOPK], dtype=tl.int32)
        for k in tl.static_range(TOPK if TAIL > 1 else 0):
            mx = tl.max(key, axis=1)                                        # [TB]
            idx = (1023 - (mx & 1023)).to(tl.int32)
            ids = tl.where(kk[None, :] == k, idx[:, None], ids)
            key = tl.where(key == mx[:, None], -1, key)
        rv = r[:, None] < T
        tl.store(sb_ptr + r[:, None] * NEP + e[None, :], s, mask=m2)
        tl.debug_barrier()
        sv = tl.load(sb_ptr + r[:, None] * NEP + ids, mask=rv, other=0.0)  # the chosen experts' unbiased scores
        if RENORM:
            sv = sv / tl.sum(sv, axis=1)[:, None]
        sv = sv * scale
        tl.store(ids_ptr + r[:, None] * TOPK + kk[None, :], ids, mask=rv)
        tl.store(wt_ptr + r[:, None] * TOPK + kk[None, :], sv, mask=rv)
    if old == NPROG - 1:
        tl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")


_WS: dict = {}
BN, SPLIT = 64, 8


def route(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, topk: int = 16, scale: float = 1.0,
          renormalize: bool = True, bn: int = None, split: int = None, warps: int = 4, tail: int = 2):
    """x [T, K] bf16 (T <= 16), w [E, K] bf16, bias [E] float32 -> (weights float32 [T, topk], ids int32 [T, topk])."""
    T, K = x.shape
    E = w.shape[0]
    assert T <= 16 and w.shape[1] == K and K % 128 == 0 and x.is_contiguous() and w.is_contiguous() and E <= 1024
    bn = bn or BN
    split = split or SPLIT
    key = (x.device, torch.cuda.current_stream().cuda_stream)
    if key not in _WS:
        _WS[key] = (torch.empty(16 * 16 * E, dtype=torch.float32, device=x.device),
                    torch.empty(16 * 1024, dtype=torch.float32, device=x.device),
                    torch.zeros(1, dtype=torch.int32, device=x.device))
    ws, sb, cnt = _WS[key]
    ids = torch.empty((T, topk), dtype=torch.int32, device=x.device)
    wt = torch.empty((T, topk), dtype=torch.float32, device=x.device)
    groups = K // 128
    per = triton.cdiv(groups, split)
    sp = triton.cdiv(groups, per)
    nt = triton.cdiv(E, bn)
    _route[(nt, sp)](x, w, bias, ws, sb, cnt, ids, wt, T, K, float(scale), BM=16, BN=bn, KCH=per * 128, SPLIT=sp,
                     NE=E, NEP=triton.next_power_of_2(E), TOPK=topk, NPROG=nt * sp, TB=max(2, triton.next_power_of_2(T)),
                     RENORM=bool(renormalize), TAIL=tail, num_warps=warps, launch_pdl=PDL)
    return wt, ids


# ----- CUDA router (csrc/k31route.cu -> band/libk31route.so): the same selection, one warp per row -----
_LIB: dict = {}


class _Args(__import__("ctypes").Structure):
    import ctypes as _c
    _fields_ = [("x", _c.c_void_p), ("w", _c.c_void_p), ("bias", _c.c_void_p), ("logits", _c.c_void_p),
                ("counter", _c.c_void_p), ("ids", _c.c_void_p), ("wt", _c.c_void_p),
                ("T", _c.c_int), ("K", _c.c_int), ("E", _c.c_int), ("renorm", _c.c_int), ("scale", _c.c_float)]


def _lib():
    if "lib" not in _LIB:
        import ctypes
        lib = ctypes.CDLL(os.environ.get("K31_ROUTE_LIB", "/engine/band/libk31route.so"))
        lib.k31_route.restype = ctypes.c_int
        lib.k31_route.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.k31_route_pdl(1 if PDL else 0)
        _LIB["lib"] = lib
    return _LIB["lib"]


_CWS: dict = {}


def route_cuda(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, topk: int = 16, scale: float = 1.0,
               renormalize: bool = True):
    """x [T, K] bf16 (T <= 8), w [E, K] bf16 (E <= 1024), bias [E] float32 -> (weights float32 [T, 16], ids int32
    [T, 16]). One launch (k31route.cu)."""
    import ctypes
    T, K = x.shape
    E = w.shape[0]
    assert topk == 16 and 1 <= T <= 8 and E <= 1024 and K % 8 == 0 and x.dtype == torch.bfloat16 and \
        w.dtype == torch.bfloat16 and bias.dtype == torch.float32 and x.is_contiguous() and w.is_contiguous()
    key = (x.device, torch.cuda.current_stream().cuda_stream)
    if key not in _CWS:
        _CWS[key] = (torch.empty(8 * 1024, dtype=torch.float32, device=x.device),
                     torch.zeros(1, dtype=torch.int32, device=x.device))
    lg, cnt = _CWS[key]
    ids = torch.empty((T, 16), dtype=torch.int32, device=x.device)
    wt = torch.empty((T, 16), dtype=torch.float32, device=x.device)
    a = _Args(x.data_ptr(), w.data_ptr(), bias.data_ptr(), lg.data_ptr(), cnt.data_ptr(), ids.data_ptr(),
              wt.data_ptr(), T, K, E, 1 if renormalize else 0, float(scale))
    rc = _lib().k31_route(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"k31_route failed: {rc}")
    return wt, ids
