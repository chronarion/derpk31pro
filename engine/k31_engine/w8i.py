"""Decode-size dense GEMMs on Hecate's int8 GEMV (engine/csrc/k31i8: dense_i8.cu, BF16 x converted to f16 pairs on
load, BF16 / float32 y): K31_DENSE_FMT ...:w8i.

Weights: int8 codes (code - 128) with one float32 scale per output row, from the BF16 checkpoint weight, the clip per
row chosen from {1.0, 0.95, 0.9, 0.85, 0.8} x the row's absmax by squared error (light-tailed rows keep absmax,
heavy-tailed ones trade their outliers for finer steps), packed in the kernel's mma fragment order (Hecate's
to_i8_copy). The kernel takes 1-8 rows; longer calls (prompt chunks) run in groups of 8 rows.
Quiet-GPU timing (tests/bench_i8.py, 4 rows, fp16 build): 12288 x 7168 18.7 us, 7168 x 12288 19.7, 7168 x 3584 9.2,
3584 x 7168 9.3, 2112 x 7168 7.4, the 163840 x 7168 head 199 us (5.9 TB/s).
"""
import importlib.util
import os

import torch

_LIB: dict = {}
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8)


def _lib():
    if "m" not in _LIB:
        # K31_I8_PDL=1: the kernel's programmatic dependent launch (dense_i8.cu reads EXL3_I8_PDL at its first launch;
        # k31.sh forwards only K31_* variables into the container)
        os.environ.setdefault("EXL3_I8_PDL", os.environ.get("K31_I8_PDL", "0"))
        p = os.environ.get("K31_I8_LIB", "/engine/band/k31i8.so")
        spec = importlib.util.spec_from_file_location("k31i8", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _LIB["m"] = m
    return _LIB["m"]


def supported(N: int, K: int) -> bool:
    return N % 16 == 0 and K % 32 == 0


def quantize(w: torch.Tensor):
    """BF16 / float [N, K] -> (packed uint8 codes [N K], float32 scales [N], spw)."""
    N, K = w.shape
    assert supported(N, K), f"w8i: {N} x {K} not a multiple of 16 x 32"
    spw = 4 if (N // 16) % 4 == 0 else 2 if (N // 16) % 2 == 0 else 1
    q = torch.empty(N, K, dtype=torch.uint8, device=w.device)
    s = torch.empty(N, dtype=torch.float32, device=w.device)
    for i in range(0, N, 4096):                               # rows in chunks: float32 temporaries stay small
        W = w[i:i + 4096].float()
        amax = W.abs().amax(1).clamp_min(1e-12)
        best_err = best_s = None
        for c in CLIPS:
            sc = amax * (c / 127.0)
            qq = torch.round(W / sc[:, None]).clamp_(-128, 127)
            err = (qq * sc[:, None] - W).pow(2).sum(1)
            if best_err is None:
                best_err, best_s = err, sc
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, sc, best_s)
        q[i:i + 4096] = (torch.round(W / best_s[:, None]).clamp_(-128, 127) + 128).to(torch.uint8)
        s[i:i + 4096] = best_s
        del W
    q = q.view(N // (16 * spw), spw, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 8, 2).contiguous().view(-1)
    return q, s.contiguous(), spw


def dequant_ref(w: torch.Tensor) -> torch.Tensor:
    """The float32 weights quantize() represents (same clip choice), for tests."""
    W = w.float()
    amax = W.abs().amax(1).clamp_min(1e-12)
    best_err = best_s = None
    for c in CLIPS:
        sc = amax * (c / 127.0)
        err = (torch.round(W / sc[:, None]).clamp_(-128, 127) * sc[:, None] - W).pow(2).sum(1)
        if best_err is None:
            best_err, best_s = err, sc
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, sc, best_s)
    return torch.round(W / best_s[:, None]).clamp_(-128, 127) * best_s[:, None]


# K31_I8_BIG_M: calls with more rows than this (prompt chunks) run as one BF16 GEMM on the int8 codes instead of
# GEMV launches of 8 rows (each of which re-reads the whole weight): the codes are unpacked from the kernel's fragment
# order to row-major BF16 integers (exact: -128..127), multiplied with x in float32 accumulation, and each output
# column is scaled by its row scale in float32. Same arithmetic as the GEMV up to the summation order (the GEMV
# rounds x to float16 first). 0 = always the GEMV.
BIG_M = int(os.environ.get("K31_I8_BIG_M", "16")) or 1 << 30


def unpack(q: torch.Tensor, N: int, K: int, spw: int) -> torch.Tensor:
    """quantize()'s packed codes -> row-major uint8 codes [N, K] (the inverse of its fragment-order permutation)."""
    v = q.view(N // (16 * spw), K // 32, spw, 8, 4, 2, 2, 2, 2)
    return v.permute(0, 2, 8, 3, 1, 5, 6, 4, 7).reshape(N, K)


def big_m(x: torch.Tensor, w: "I8W") -> torch.Tensor:
    """y [M, N] float32 = x [M, K] bf16 @ (codes - 128)^T * s, for eager calls of M > BIG_M (one GEMM, no 8-row
    launches). The BF16 codes are a temporary (N x K x 2 bytes, freed on return)."""
    codes = unpack(w.q, w.N, w.K, w.spw).to(torch.bfloat16).sub_(128.0)   # integers: exact in bfloat16
    if _MM32[0] is not False:
        try:
            y = torch.mm(x, codes.t(), out_dtype=torch.float32)            # float32 output, scaled before rounding
            _MM32[0] = True
            return y.mul_(w.s)
        except (TypeError, RuntimeError):
            _MM32[0] = False
    return torch.mm(x, codes.t()).float().mul_(w.s)


_MM32 = [None]                       # torch.mm(..., out_dtype=float32) for bf16 inputs works here (probed at first use)


class I8W:
    """An int8 weight for the k31i8 GEMV: codes, per-row scales, and its reduction scratch (one per linear: each runs
    on one stream at a time)."""

    def __init__(self, w: torch.Tensor):
        self.N, self.K = w.shape
        self.q, self.s, self.spw = quantize(w)
        self.y32 = torch.zeros(8 * self.N, dtype=torch.float32, device=w.device)
        self.ctr = torch.zeros(max(1, self.N // 16), dtype=torch.int32, device=w.device)
        self.shape = (self.N, self.K)
        self.nbytes = self.q.numel() + 4 * self.s.numel()

    def __call__(self, x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        """y [M, N] bf16 (or out's dtype: bf16 / float32) = x [M, K] bf16 @ W^T."""
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        M = x.shape[0]
        if M > BIG_M and not torch.cuda.is_current_stream_capturing():   # prompt chunks: one GEMM (big_m)
            y = big_m(x, self)
            if out is None:
                return y.to(torch.bfloat16)
            out.copy_(y)
            return out
        if out is None:
            out = torch.empty((M, self.N), dtype=torch.bfloat16, device=x.device)
        m = _lib()
        for r in range(0, M, 8):                              # 1-8 rows per launch
            r1 = min(M, r + 8)
            y = out[r:r1] if r1 - r == M else out[r:r1]
            m.dense_i8(x[r:r1], self.q, self.s, y, self.y32, self.ctr, self.spw)
        return out


def linear(x: torch.Tensor, w: I8W, out: torch.Tensor = None) -> torch.Tensor:
    return w(x, out)
