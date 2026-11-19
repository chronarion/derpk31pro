"""Prompt-size dense GEMMs (calls of many rows: prompt chunks) for the engine's 8-bit dense formats, behind two flags
that are off by default (the default path stays today's). Both are runtime flags (flags.py: the control file, else the
environment), read per eager call by the call sites in w8i.py and w8a16.py; calls under CUDA-graph capture never come
here.

K31_I8_BIG_M_KERNEL (w8i.I8W, eager calls of more than K31_I8_BIG_M rows; today: the int8 codes unpacked from the GEMV's
fragment order with torch ops per call, ~0.5 ms per o_proj, then cuBLAS):
  1 | triton   one Triton GEMM on the packed codes as they are (no unpack, no temporary): the codes are read in their
               fragment order, rearranged in registers to [K, N] tiles (a bit permutation of the index), converted to
               BF16 integers (exact), multiplied with x on the tensor cores in float32, and each output column scaled by
               its row scale in float32 before the single rounding to the output dtype. Same arithmetic as today's
               path up to the order of the float32 sums.
  unpack       one Triton kernel unpacks the codes to row-major BF16 integers, cuBLAS multiplies (float32 out), a second
               Triton kernel scales the columns and rounds: today's path with the torch permute replaced.
K31_W8A16_BIG_M (w8a16.linear, eager calls of >= K31_W8A16_BIG_M_ROWS rows, default 1,024; today: 64-row tiles of the
split-K kernel, whose loop waits for every MMA and every scale load in turn: ~165 TFLOP/s; Marlin for the shapes it
serves at decode, which is built for small M):
  1 | triton   FP8 split-K-format weights (q [N, K], s [N, K / 128]): a Triton GEMM with 128-row tiles and per-group
               promotion: for each 128-wide K group the tensor cores sum x * q (exact products, float32), the partial is
               scaled by the group's scale and added in float32. The same arithmetic as the 64-row tiles (both sum the
               same exact products per group, scale, and add the groups in order).
               Marlin-format weights (MarlinW): a Triton GEMM that reads Marlin's FP8 layout and its permuted scales
               as they are, with Marlin's own arithmetic: each weight is the BF16 product q * s (one rounding), then
               x * w summed in float32.
  hilo         split-K format: each weight q * s (exact in float32: <= 12 significant bits) split into two BF16 terms
               hi + lo (exact), both against x in one MMA chain (hi and lo joined along K): no per-group promotion,
               twice the MMA work. Same exact products, float32 sums in another order. (Marlin weights: as 1.)
  cublas       split-K format: one Triton kernel writes [hi | lo] as BF16 [N, 2K], x is duplicated to [M, 2K], cuBLAS
               multiplies (BF16 out). Marlin weights: one Triton kernel writes BF16(q * s) [N, K] (Marlin's rounding),
               cuBLAS multiplies. Transient buffers of N * 2K * 2 (+ M * 2K * 2) bytes per call.
K31_BIGM_CFG=FILE (runtime flag): the timing script's JSON (tests/bench_bigm.py): per kind, shape and row count, the
implementation and tile config it measured fastest; with it, "1" means "the table's choice" (else the defaults above).

Tiles: BM x BN output tiles over a 1-D grid in groups of GROUP_M row tiles (L2 reuse of the weight tiles), K in BK
steps, loads pipelined by Triton (cp.async stages); the converted weight tile goes through shared memory to tcgen05.
All kernels are deterministic (no atomics, fixed summation order per config). PDL as the other engine kernels
(K31_PDL=1: griddepcontrol.wait at entry, launch_dependents after it).
"""
import json
import os

import torch
import triton
import triton.language as tl

from . import flags

PDL = os.environ.get("K31_PDL", "0") == "1"
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"   # CPU tests: the interpreter's BF16 dot is not usable
_INV_MARLIN_BIAS = 2.0 ** -120     # Marlin folds 2^120 into BF16 scales for its FP8 -> BF16 exponent shift


def on(v: str) -> bool:
    return v not in ("", "0", "off")


# ------------------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _tile(pid, M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    """Program id -> (row tile, column tile), GROUP_M row tiles per column sweep (the weight tile stays in L2)."""
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    per = GROUP_M * npn
    first = (pid // per) * GROUP_M
    gm = tl.minimum(npm - first, GROUP_M)
    return first + (pid % per) % gm, (pid % per) // gm


@triton.jit
def _mma(a, b, acc, INTERP: tl.constexpr):
    if INTERP:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _mma0(a, b, INTERP: tl.constexpr):
    """A fresh product (no accumulator input: the MMA starts from zero)."""
    if INTERP:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
    else:
        return tl.dot(a, b)


@triton.jit
def _i8_bf16(w, INTERP: tl.constexpr):
    """uint8 codes (code = value + 128) -> BF16 integers -128..127 (exact). The XOR + int8 bitcast form compiles to
    Triton's packed int8 -> BF16 conversion (prmt + bf16x2 subtract, 4 codes per 5 instructions)."""
    if INTERP:
        return (w.to(tl.float32) - 128.0).to(tl.bfloat16)
    else:
        return (w ^ 0x80).to(tl.int8, bitcast=True).to(tl.bfloat16)


@triton.jit
def _i8_tile(q_ptr, nb0, kb0, NB, K, SPW: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EVEN_N: tl.constexpr):
    """A [BK, BN] tile of the packed int8 codes (uint8), rows k0 = 32 kb0 .., columns n0 = 16 SPW nb0 ..

    w8i.quantize's order: byte index (((((nb * K/32 + kb) * SPW + sw) * 8 + r) * 4 + c) * 4 + ab) * 4 + d * 2 + h for
    n = 16 SPW nb + 16 sw + 8 h + r, k = 32 kb + 8 ab + 2 c + d. For one nb the K/32 blocks are contiguous, so the tile
    is BN / (16 SPW) runs of BK * 16 SPW bytes; the reshape + permute below is a bit permutation of the index."""
    NBT: tl.constexpr = BN // (16 * SPW)
    KBT: tl.constexpr = BK // 32
    RUN: tl.constexpr = KBT * 512 * SPW
    nbs = nb0 + tl.arange(0, NBT)
    off = nbs[:, None].to(tl.int64) * (K * 16 * SPW) + kb0 * (512 * SPW) + tl.arange(0, RUN)[None, :]
    if EVEN_N:
        wq = tl.load(q_ptr + off)
    else:
        wq = tl.load(q_ptr + off, mask=nbs[:, None] < NB)
    w = tl.reshape(wq, [NBT, KBT, SPW, 8, 4, 4, 2, 2])         # nb, kb, sw, r, c, ab, d, h
    w = tl.permute(w, [1, 5, 4, 6, 0, 2, 7, 3])                 # k: kb ab c d | n: nb sw h r
    return tl.reshape(w, [BK, BN])


@triton.jit
def _i8_mm(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, sxm, syn,
           SPW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
           EVEN_N: tl.constexpr, PDL: tl.constexpr, INTERP: tl.constexpr):
    """y [M, N] = (x [M, K] bf16 @ (codes - 128)^T) * s[n], codes packed as w8i.quantize; float32 sums, one rounding."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid_m, pid_n = _tile(tl.program_id(0), M, N, BM, BN, GROUP_M)
    NBT: tl.constexpr = BN // (16 * SPW)
    rm = pid_m * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    xp = x_ptr + rm[:, None].to(tl.int64) * sxm + rk[None, :]
    xm = rm[:, None] < M
    NB = N // (16 * SPW)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(xp + k0, mask=xm, other=0.0)
        w = _i8_tile(q_ptr, pid_n * NBT, k0 // 32, NB, K, SPW, BN, BK, EVEN_N)
        acc = _mma(a, _i8_bf16(w, INTERP), acc, INTERP)
    rn = pid_n * BN + tl.arange(0, BN)
    sc = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    y = acc * sc[None, :]
    tl.store(y_ptr + rm[:, None].to(tl.int64) * syn + rn[None, :], y.to(y_ptr.dtype.element_ty),
             mask=xm & (rn[None, :] < N))


@triton.jit
def _i8_unpack(q_ptr, out_ptr, N, K, SPW: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EVEN_N: tl.constexpr,
               INTERP: tl.constexpr):
    """Packed codes -> row-major BF16 integers [N, K] (code - 128), one [BN, BK] tile per program."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    NBT: tl.constexpr = BN // (16 * SPW)
    w = _i8_tile(q_ptr, pid_n * NBT, pid_k * (BK // 32), N // (16 * SPW), K, SPW, BN, BK, EVEN_N)
    w = tl.trans(_i8_bf16(w, INTERP))                          # [BN, BK]
    rn = pid_n * BN + tl.arange(0, BN)
    rk = pid_k * BK + tl.arange(0, BK)
    tl.store(out_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], w, mask=rn[:, None] < N)


@triton.jit
def _scale_cols(y_ptr, s_ptr, out_ptr, M, N, syn, BM: tl.constexpr, BN: tl.constexpr):
    """out [M, N] = y [M, N] (float32) * s[n] (float32), one rounding to out's dtype."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    y = tl.load(y_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], mask=m, other=0.0)
    sc = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    tl.store(out_ptr + rm[:, None].to(tl.int64) * syn + rn[None, :], (y * sc[None, :]).to(out_ptr.dtype.element_ty),
             mask=m)


@triton.jit
def _w8_mm(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, NG, sxm, syn,
           BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, EVEN_N: tl.constexpr, PDL: tl.constexpr,
           INTERP: tl.constexpr):
    """y = x @ (q * s)^T for split-K-format FP8 weights, per-group promotion: for each 128-wide K group g,
    acc += (x[:, g] @ q[:, g]^T) * s[:, g] in float32 (the 64-row tiles' arithmetic), then one rounding.
    s_ptr: the scales transposed and widened, float32 [NG, N] (a group's scales of a column tile are contiguous: the
    promotion reads one per accumulator column, and [N, NG] BF16 would make that one 2-byte load per column per
    thread). The group's scales are loaded at the top of its step and first used after the MMA has completed, so
    the load's latency hides behind the conversion and the MMA."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid_m, pid_n = _tile(tl.program_id(0), M, N, BM, BN, GROUP_M)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, 128)
    xm = rm[:, None] < M
    nm = rn < N
    xp = x_ptr + rm[:, None].to(tl.int64) * sxm + rk[None, :]
    wp = q_ptr + rn[:, None].to(tl.int64) * K + rk[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in range(0, NG):
        if EVEN_N:
            sc = tl.load(s_ptr + g * N + rn)
            w = tl.load(wp + g * 128)
        else:
            sc = tl.load(s_ptr + g * N + rn, mask=nm, other=0.0)
            w = tl.load(wp + g * 128, mask=nm[:, None], other=0.0)
        a = tl.load(xp + g * 128, mask=xm, other=0.0)
        part = _mma0(a, tl.trans(w.to(tl.bfloat16)), INTERP)
        acc += part * sc[None, :]
    tl.store(y_ptr + rm[:, None].to(tl.int64) * syn + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=xm & nm[None, :])


@triton.jit
def _w8_hilo_mm(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, NG, sxm, syn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, EVEN_N: tl.constexpr,
                PDL: tl.constexpr, INTERP: tl.constexpr):
    """y = x @ (q * s)^T for split-K-format FP8 weights without promotion: wf = q * s (exact in float32), hi =
    bf16(wf), lo = bf16(wf - hi) (exact), [x | x] @ [hi | lo]^T in one MMA chain over 2 BK per step."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid_m, pid_n = _tile(tl.program_id(0), M, N, BM, BN, GROUP_M)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xm = rm[:, None] < M
    nm = rn < N
    xp = x_ptr + rm[:, None].to(tl.int64) * sxm + rk[None, :]
    wp = q_ptr + rn[:, None].to(tl.int64) * K + rk[None, :]
    sp = s_ptr + rn.to(tl.int64) * NG
    if EVEN_N:
        sc = tl.load(sp).to(tl.float32)
    else:
        sc = tl.load(sp, mask=nm, other=0.0).to(tl.float32)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(xp + k0, mask=xm, other=0.0)
        gn = tl.minimum(k0 + BK, K - BK) // 128                  # the next step's scales, loaded one step ahead
        if EVEN_N:
            w = tl.load(wp + k0)
            sc_n = tl.load(sp + gn).to(tl.float32)
        else:
            w = tl.load(wp + k0, mask=nm[:, None], other=0.0)
            sc_n = tl.load(sp + gn, mask=nm, other=0.0).to(tl.float32)
        wf = w.to(tl.float32) * sc[:, None]
        hi = wf.to(tl.bfloat16)
        lo = (wf - hi.to(tl.float32)).to(tl.bfloat16)
        w2 = tl.reshape(tl.join(hi, lo), [BN, 2 * BK])
        a2 = tl.reshape(tl.join(a, a), [BM, 2 * BK])
        acc = _mma(a2, tl.trans(w2), acc, INTERP)
        sc = sc_n
    tl.store(y_ptr + rm[:, None].to(tl.int64) * syn + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=xm & nm[None, :])


@triton.jit
def _w8_dequant(q_ptr, s_ptr, w2_ptr, N, K, NG, BN: tl.constexpr, BK: tl.constexpr):
    """Split-K-format FP8 weights -> [hi | lo] BF16 [N, 2K]: hi = bf16(q * s), lo = bf16(q * s - hi), both exact."""
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    nm = rn[:, None] < N
    w = tl.load(q_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], mask=nm, other=0.0)
    sc = tl.load(s_ptr + rn.to(tl.int64) * NG + tl.program_id(1) * BK // 128, mask=rn < N, other=0.0).to(tl.float32)
    wf = w.to(tl.float32) * sc[:, None]
    hi = wf.to(tl.bfloat16)
    lo = (wf - hi.to(tl.float32)).to(tl.bfloat16)
    o = w2_ptr + rn[:, None].to(tl.int64) * (2 * K) + rk[None, :]
    tl.store(o, hi, mask=nm)
    tl.store(o + K, lo, mask=nm)


@triton.jit
def _m8_codes(q_ptr, k0, n0, N, BN: tl.constexpr, BK: tl.constexpr):
    """A [BK, BN] tile of a Marlin FP8 weight's codes, rows k0 .., columns n0 ..

    Marlin's 8-bit layout (gptq_marlin_repack, vLLM marlin_utils_test.marlin_permute_weights + get_weight_perm(8)):
    bytes [K / 16, 16 N]; row kt holds k = 16 kt .. 16 kt + 15, in groups of 1,024 bytes per 64 columns; inside a
    group, byte p = (p9..p0) holds n = 64 ng + 16 (p4 p3) + 8 p2 + (p9 p8 p7), k = 16 kt + 8 p0 + 2 (p6 p5) + p1."""
    KTT: tl.constexpr = BK // 16
    NGT: tl.constexpr = BN // 64
    off = (k0 // 16 + tl.arange(0, KTT))[:, None].to(tl.int64) * (N * 16) + n0 * 16 + \
        tl.arange(0, NGT * 1024)[None, :]
    wq = tl.load(q_ptr + off)
    w = tl.reshape(wq, [KTT, NGT, 8, 4, 4, 2, 2, 2])            # kt, ng, (p9 p8 p7), (p6 p5), (p4 p3), p2, p1, p0
    w = tl.permute(w, [0, 7, 3, 6, 1, 4, 5, 2])                 # k: kt p0 (p6 p5) p1 | n: ng (p4 p3) p2 (p9 p8 p7)
    return tl.reshape(w, [BK, BN])


@triton.jit
def _m8_scales(s_ptr, k0, n0, N, BN: tl.constexpr):
    """Group scales of columns n0 .. n0 + BN - 1 for the 128-wide K group holding k0, as the BF16 scales themselves
    (exact). Marlin's scales are [K / 128, N] BF16 with 2^120 folded in; n = 64 ng + 8 b + a sits at column
    64 ng + 8 a + b."""
    rn = n0 + tl.arange(0, BN)
    sidx = (rn // 64) * 64 + (rn % 8) * 8 + (rn % 64) // 8
    sc = tl.load(s_ptr + (k0 // 128) * N + sidx).to(tl.float32) * 7.52316384526264e-37   # 2^-120
    return sc.to(tl.bfloat16)


@triton.jit
def _m8_weights(w, sc, INTERP: tl.constexpr):
    """BF16(q * s) as Marlin computes it: the FP8 code widened to BF16 (exact) times the BF16 scale, one rounding to
    nearest (mul.rn.bf16: a third of the instructions of the float32 multiply with its two conversions, the same
    result). The interpreter's BF16 arithmetic is not usable: float32 there."""
    if INTERP:
        return (w.to(tl.float32) * sc.to(tl.float32)[None, :]).to(tl.bfloat16)
    else:
        return w.to(tl.bfloat16) * sc[None, :]


@triton.jit
def _m8_mm(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, sxm, syn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, PDL: tl.constexpr,
           INTERP: tl.constexpr):
    """y = x @ w^T with w = BF16(q * s) read from Marlin's FP8 layout (N % BN == 0, BK <= 128, K % BK == 0); the
    next step's scales are loaded one step ahead."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid_m, pid_n = _tile(tl.program_id(0), M, N, BM, BN, GROUP_M)
    n0 = pid_n * BN
    rm = pid_m * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    xm = rm[:, None] < M
    xp = x_ptr + rm[:, None].to(tl.int64) * sxm + rk[None, :]
    sc = _m8_scales(s_ptr, 0, n0, N, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(xp + k0, mask=xm, other=0.0)
        w = _m8_codes(q_ptr, k0, n0, N, BN, BK)
        sc_n = _m8_scales(s_ptr, tl.minimum(k0 + BK, K - BK), n0, N, BN)
        acc = _mma(a, _m8_weights(w, sc, INTERP), acc, INTERP)
        sc = sc_n
    rn = n0 + tl.arange(0, BN)
    tl.store(y_ptr + rm[:, None].to(tl.int64) * syn + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=xm)


@triton.jit
def _m8_dequant(q_ptr, s_ptr, w_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr, INTERP: tl.constexpr):
    """Marlin FP8 weight -> BF16(q * s) [N, K] row-major (Marlin's rounding), one [BN, BK] tile per program."""
    n0 = tl.program_id(0) * BN
    k0 = tl.program_id(1) * BK
    w = _m8_codes(q_ptr, k0, n0, N, BN, BK)
    sc = _m8_scales(s_ptr, k0, n0, N, BN)
    w = tl.trans(_m8_weights(w, sc, INTERP))                     # [BN, BK]
    rn = n0 + tl.arange(0, BN)
    rk = k0 + tl.arange(0, BK)
    tl.store(w_ptr + rn[:, None].to(tl.int64) * K + rk[None, :], w)


# ------------------------------------------------------------------------------------------------------- configs
# (BM, BN, BK, GROUP_M, warps, stages); first-guess defaults, unmeasured: the timing script's table replaces them
DEFAULT = {
    "i8": (128, 128, 64, 8, 4, 3),
    "w8": (128, 128, 128, 8, 8, 3),       # BK is the 128-wide scale group
    "hilo": (128, 128, 64, 8, 4, 3),
    "m8": (128, 128, 64, 8, 4, 3),
}
_TABLE: dict = {}


def table() -> dict:
    """K31_BIGM_CFG (runtime flag): the timing script's JSON, loaded once per path."""
    p = flags.get("K31_BIGM_CFG", "")
    if p not in _TABLE:
        _TABLE[p] = json.load(open(p)) if p and os.path.exists(p) else {}
    return _TABLE[p]


def choice(kind: str, N: int, K: int, M: int, impl: str):
    """(impl, cfg) for a call: an explicit impl keeps the default config of that impl (or the table's config for the
    same impl); "1" takes the table's fastest entry for the shape at the nearest measured row count <= M (else the
    smallest), or the kind's default."""
    rows = table().get(kind, {}).get(f"{N}x{K}", {})
    ent = None
    if rows:
        ms = sorted(int(m) for m in rows)
        below = [m for m in ms if m <= M]
        ent = rows[str(below[-1] if below else ms[0])]
    if impl in ("1", "on"):
        if ent:
            return ent["impl"], tuple(ent["cfg"]) if ent.get("cfg") else None
        impl = "triton"
    if ent and ent.get("impl") == impl and ent.get("cfg"):
        return impl, tuple(ent["cfg"])
    return impl, None


def _even_bn(N: int, BN: int, step: int) -> int:
    """BN, or the largest power-of-two tile below it (>= step) that divides N (masked weight tiles cost a select per
    byte); BN itself (masked) if none does."""
    b = BN
    while b >= step:
        if N % b == 0:
            return b
        b //= 2
    return BN


def _mm_into(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> None:
    """y = a @ b^T by cuBLAS (BF16 inputs, float32 sums), written into y (BF16 or float32, rows may be strided)."""
    if y.dtype == torch.bfloat16 and y.is_contiguous():
        torch.mm(a, b.t(), out=y)
    else:
        y.copy_(torch.mm(a, b.t()) if y.dtype == torch.bfloat16 else torch.mm(a, b.t(), out_dtype=torch.float32))


def _grid(M: int, N: int, BM: int, BN: int):
    return (triton.cdiv(M, BM) * triton.cdiv(N, BN),)


def _out(x: torch.Tensor, N: int, out: torch.Tensor, dtype=torch.bfloat16):
    if out is None:
        return torch.empty((x.shape[0], N), dtype=dtype, device=x.device), None
    if out.stride(-1) == 1 and (out.shape[0] == 1 or out.stride(0) >= N):
        return out, None
    return torch.empty((x.shape[0], N), dtype=out.dtype, device=x.device), out     # copied back at the end


# ------------------------------------------------------------------------------------------------------ int8 (I8W)
def i8_linear(x: torch.Tensor, w, out: torch.Tensor = None, impl: str = None, cfg=None) -> torch.Tensor:
    """y [M, N] (bf16, or out's dtype: bf16 / float32) = x [M, K] bf16 @ (codes - 128)^T * s for a w8i.I8W."""
    M, K = x.shape
    N = w.N
    assert K == w.K and x.dtype == torch.bfloat16 and x.stride(1) == 1
    impl, tcfg = choice("i8", N, K, M, impl or "1")
    cfg = cfg or tcfg
    y, back = _out(x, N, out)
    if impl == "unpack":
        codes = i8_unpack(w)
        try:
            y32 = torch.mm(x, codes.t(), out_dtype=torch.float32)          # as w8i.big_m: float32 out, then scaled
        except (TypeError, RuntimeError):
            y32 = torch.mm(x, codes.t()).float()
        del codes
        BMs, BNs = 32, 128
        _scale_cols[(triton.cdiv(M, BMs), triton.cdiv(N, BNs))](y32, w.s, y, M, N, y.stride(0), BM=BMs, BN=BNs,
                                                                num_warps=4)
    else:
        BM, BN, BK, GM, nw, ns = cfg or DEFAULT["i8"]
        BN = max(16 * w.spw, _even_bn(N, BN, 16 * w.spw))
        assert BN % (16 * w.spw) == 0 and BK % 32 == 0 and K % BK == 0, (BN, BK, K, w.spw)
        kw = {} if INTERP else dict(num_warps=nw, num_stages=ns, launch_pdl=PDL)
        _i8_mm[_grid(M, N, BM, BN)](x, w.q, w.s, y, M, N, K, x.stride(0), y.stride(0), SPW=w.spw, BM=BM, BN=BN,
                                    BK=BK, GROUP_M=GM, EVEN_N=N % BN == 0, PDL=PDL and not INTERP, INTERP=INTERP,
                                    **kw)
    if back is not None:
        back.copy_(y)
        return back
    return y


def i8_unpack(w) -> torch.Tensor:
    """w8i.I8W -> row-major BF16 codes - 128 [N, K] (the inverse of w8i.quantize's packing, as w8i.unpack)."""
    N, K = w.N, w.K
    BN = _even_bn(N, 128, 16 * w.spw)
    BN = max(BN, 16 * w.spw)
    BK = 64 if K % 64 == 0 else 32
    codes = torch.empty((N, K), dtype=torch.bfloat16, device=w.q.device)
    _i8_unpack[(triton.cdiv(N, BN), K // BK)](w.q, codes, N, K, SPW=w.spw, BN=BN, BK=BK, EVEN_N=N % BN == 0,
                                              INTERP=INTERP)
    return codes


# ------------------------------------------------------------------------------------ W8A16, split-K format (q, s)
def w8_linear(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, out: torch.Tensor = None, impl: str = None,
              cfg=None) -> torch.Tensor:
    """y [M, N] bf16 = x [M, K] bf16 (rows may be strided) @ dequant(q, s)^T, q FP8 [N, K], s BF16 [N, K / 128]."""
    M, K = x.shape
    N = q.shape[0]
    NG = K // 128
    assert K % 128 == 0 and s.shape == (N, NG) and x.stride(1) == 1 and q.is_contiguous() and s.is_contiguous()
    impl, tcfg = choice("w8", N, K, M, impl or "1")
    cfg = cfg or tcfg
    y, back = _out(x, N, out)
    if impl == "cublas":
        w2 = w8_dequant(q, s)
        x2 = torch.cat([x, x], dim=1)
        _mm_into(x2, w2, y)
        del w2, x2
    elif impl == "hilo":
        BM, BN, BK, GM, nw, ns = cfg or DEFAULT["hilo"]
        BN = _even_bn(N, BN, 16)
        assert 128 % BK == 0 and K % BK == 0
        kw = {} if INTERP else dict(num_warps=nw, num_stages=ns, launch_pdl=PDL)
        _w8_hilo_mm[_grid(M, N, BM, BN)](x, q, s, y, M, N, K, NG, x.stride(0), y.stride(0), BM=BM, BN=BN, BK=BK,
                                         GROUP_M=GM, EVEN_N=N % BN == 0, PDL=PDL and not INTERP, INTERP=INTERP, **kw)
    else:
        BM, BN, _, GM, nw, ns = cfg or DEFAULT["w8"]
        BN = _even_bn(N, BN, 16)
        sT = s.t().float().contiguous()             # [NG, N] float32: one small copy per call (N * K / 32 bytes)
        kw = {} if INTERP else dict(num_warps=nw, num_stages=ns, launch_pdl=PDL)
        _w8_mm[_grid(M, N, BM, BN)](x, q, sT, y, M, N, K, NG, x.stride(0), y.stride(0), BM=BM, BN=BN, GROUP_M=GM,
                                    EVEN_N=N % BN == 0, PDL=PDL and not INTERP, INTERP=INTERP, **kw)
    if back is not None:
        back.copy_(y)
        return back
    return y


def w8_dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """(q FP8 [N, K], s BF16 [N, K / 128]) -> [hi | lo] BF16 [N, 2K], hi + lo = q * s exactly."""
    N, K = q.shape
    w2 = torch.empty((N, 2 * K), dtype=torch.bfloat16, device=q.device)
    BN, BK = 64, 128
    _w8_dequant[(triton.cdiv(N, BN), K // BK)](q, s, w2, N, K, K // 128, BN=BN, BK=BK)
    return w2


# ------------------------------------------------------------------------------------------- W8A16, Marlin format
def marlin_linear(x: torch.Tensor, mw, out: torch.Tensor = None, impl: str = None, cfg=None) -> torch.Tensor:
    """y [M, N] bf16 = x [M, K] bf16 @ w^T for a w8a16.MarlinW (Marlin's FP8 layout and permuted scales as they are),
    w = BF16(q * s) as Marlin computes it."""
    M, K = x.shape
    N = mw.N
    assert K == mw.K and N % 64 == 0 and K % 128 == 0 and x.stride(1) == 1
    impl, tcfg = choice("m8", N, K, M, impl or "1")
    cfg = cfg or tcfg
    y, back = _out(x, N, out)
    qb = mw.w.view(torch.uint8)
    if impl == "cublas":
        w = marlin_dequant(mw)
        _mm_into(x, w, y)
        del w
    else:
        BM, BN, BK, GM, nw, ns = cfg or DEFAULT["m8"]
        BN = _even_bn(N, BN, 64)
        assert N % BN == 0 and BN % 64 == 0 and 128 % BK == 0 and BK % 16 == 0
        kw = {} if INTERP else dict(num_warps=nw, num_stages=ns, launch_pdl=PDL)
        _m8_mm[_grid(M, N, BM, BN)](x, qb.view(torch.float8_e4m3fn), mw.s, y, M, N, K, x.stride(0), y.stride(0),
                                    BM=BM, BN=BN, BK=BK, GROUP_M=GM, PDL=PDL and not INTERP, INTERP=INTERP, **kw)
    if back is not None:
        back.copy_(y)
        return back
    return y


def marlin_dequant(mw) -> torch.Tensor:
    """w8a16.MarlinW -> BF16(q * s) [N, K] row-major."""
    N, K = mw.N, mw.K
    w = torch.empty((N, K), dtype=torch.bfloat16, device=mw.w.device)
    BN, BK = 64, 64
    _m8_dequant[(N // BN, K // BK)](mw.w.view(torch.uint8).view(torch.float8_e4m3fn), mw.s, w, N, K, BN=BN, BK=BK,
                                    INTERP=INTERP)
    return w


# ------------------------------------------------------------------------------------------ host-side references
def marlin_byte_index(K: int, N: int) -> torch.Tensor:
    """int64 [K, N]: the byte offset of weight (k, n) in Marlin's 8-bit layout, from the bit map _m8_codes reads
    (tests compare it with a direct implementation of vLLM's permutation; the GPU test checks the dequant against
    gptq_marlin_repack)."""
    k = torch.arange(K).view(K, 1)
    n = torch.arange(N).view(1, N)
    kt, kk = k // 16, k % 16
    ng, nl = n // 64, n % 64
    p0, p65, p1 = (kk >> 3) & 1, (kk >> 1) & 3, kk & 1
    p43, p2, p987 = (nl >> 4) & 3, (nl >> 3) & 1, nl & 7
    p = (p987 << 7) | (p65 << 5) | (p43 << 3) | (p2 << 2) | (p1 << 1) | p0
    return kt * (N * 16) + ng * 1024 + p


def marlin_scale_index(N: int) -> torch.Tensor:
    """int64 [N]: the column of output n in Marlin's permuted group scales (one row per 128-wide K group)."""
    n = torch.arange(N)
    return (n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8
