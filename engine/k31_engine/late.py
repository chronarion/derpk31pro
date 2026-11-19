"""The late tier's Triton kernels and their references (K31_LATE; the engine side is latetier.py, the CUDA kernels
csrc/late.cu, the pool engine/pool).

plan(ids, w, row_map, bank_map, rows, lfrom, lto, D, lslot, lw, lmask, count, band_map=None)
    Which picks of a MoE call the Grace pool computes late. A candidate is a pick with an expert id, no resident row,
    a bank record, unitk's row id < 0 (nothing serves it on time), a rank (by weight within its token, ties: lower
    position) in [lfrom, lto) and, with the RTX band, an expert that is not in the band (band picks are the band's,
    on time or with the mean past its cap). The candidates' distinct experts are ordered by their summed weight^2 over
    the call's candidate picks (ties: lower expert id); the first D take late slots 0, 1, ... Outputs: lslot [8] (the
    bank record of slot p, -1 past the last), lw [8, 8] (lw[p, t] = row t's weight for slot p's expert, 0 = unused),
    lmask [T, 16] (each pick's slot or -1); count (int64) += late picks. T <= 8, D <= 8. Nothing on the main stream
    reads these: the on-time path (unitk, stager, cut-mean, MegaMoE) is unchanged, so the late picks get their mean
    on time like any cut pick, and the landing adds (late rows - that mean).

mean(out, lmask, ids, w, mean_table)
    out [T, 3584] float32 = sum over the late picks (lmask >= 0) of weight x the expert's mean output: the share of the
    on-time cut-mean the late picks received (the landing subtracts it from the pool's rows).

add(dst, z, y, coef, out=None)
    out = dst + z + cb * y per row with a late share (coef[t, 0] > 0), float32, one rounding to dst's dtype; rows
    without a share are copied bit for bit. dst is the next layer's attention output (half-layer landing) or the
    stored AttnRes block state's row res[:, b, :] (row-strided). z = up(xl) (float32 or bf16), y = the layer's on-time
    routed output, coef from late_land: (has, cb, ra, rb).

References (torch, CPU or GPU; the tests and the bind's self-check): plan_reference, mean_reference,
quant_rows_reference (the send's int8 rows), coef_reference / land_reference (the landing), delta_reference (the
definition up(norm(u + v)) - up(norm(u))), decode_hi2 and pool_reference (the pool's arithmetic in float64).
"""
import os

import torch
import triton
import triton.language as tl

E, K, H, I = 896, 16, 3584, 3072
PMAX, MAXT = 8, 8                   # cpustage_seg2.h: records per job, rows per call
PDL = os.environ.get("K31_PDL", "0") == "1"


# ------------------------------------------------------------------------------------------------ plan
@triton.jit
def _lplan(ids_ptr, w_ptr, row_ptr, bank_ptr, band_ptr, rows_ptr, lslot_ptr, lw_ptr, lmask_ptr, cnt_ptr, T, lfrom,
           lto, D, NE: tl.constexpr, TB: tl.constexpr, KC: tl.constexpr, PM: tl.constexpr, HAS_BAND: tl.constexpr,
           USE_PDL: tl.constexpr):
    if USE_PDL:                              # unitk's row ids and the router's picks are visible
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.arange(0, TB)
    k = tl.arange(0, KC)
    m2 = (t[:, None] < T) & (k[None, :] < KC)
    off = t[:, None] * KC + k[None, :]
    ids = tl.load(ids_ptr + off, mask=m2, other=-1)
    w = tl.load(w_ptr + off, mask=m2, other=float("-inf")).to(tl.float32)
    valid = m2 & (ids >= 0) & (ids < NE)
    e = tl.where(valid, ids, 0)
    native = tl.load(row_ptr + e, mask=valid, other=-1)
    bank = tl.load(bank_ptr + e, mask=valid, other=-1)
    rows = tl.load(rows_ptr + off, mask=m2, other=0)                 # unitk's row ids: >= 0 = served on time
    # rank within the token: picks with a larger weight, or an equal weight at a lower position (unitk's rule)
    wi = w[:, :, None]
    wj = w[:, None, :]
    kj = k[None, None, :]
    ki = k[None, :, None]
    beats = (wj > wi) | ((wj == wi) & (kj < ki))
    rank = tl.sum(beats.to(tl.int32), axis=2)                         # [TB, KC]
    cand = valid & (native < 0) & (bank >= 0) & (rows < 0) & (rank >= lfrom) & (rank < lto)
    if HAS_BAND:
        cand = cand & (tl.load(band_ptr + e, mask=valid, other=-1) < 0)
    N: tl.constexpr = TB * KC
    fe = tl.reshape(e, (N,))
    fw = tl.reshape(tl.where(cand, w, 0.0), (N,))
    fc = tl.reshape(cand, (N,))
    fb = tl.reshape(bank, (N,))
    n_ar = tl.arange(0, N)
    same = fc[:, None] & fc[None, :] & (fe[:, None] == fe[None, :])                  # [N, N]
    first = fc & (tl.sum((same & (n_ar[None, :] < n_ar[:, None])).to(tl.int32), axis=1) == 0)
    energy = tl.sum(tl.where(same, (fw * fw)[None, :], 0.0), axis=1)                # the expert's summed weight^2
    better = first[None, :] & ((energy[None, :] > energy[:, None]) |
                               ((energy[None, :] == energy[:, None]) & (fe[None, :] < fe[:, None])))
    slot = tl.sum(better.to(tl.int32), axis=1)                                       # place among distinct experts
    ok = first & (slot < D)
    full = tl.sum(tl.where(same & ok[None, :], slot[None, :] + 1, 0), axis=1) - 1   # every entry's slot, or -1
    p_ar = tl.arange(0, PM)
    lslot = tl.sum(tl.where(ok[None, :] & (slot[None, :] == p_ar[:, None]), fb[None, :] + 1, 0), axis=1) - 1
    tl.store(lslot_ptr + p_ar, lslot)
    full2 = tl.reshape(full, (TB, KC))
    tl.store(lmask_ptr + off, full2, mask=m2)
    # lw[p, t]: row t's weight for slot p's expert (a row holds an expert at most once); rows t >= T get 0
    wp = tl.sum(tl.where(full2[:, :, None] == p_ar[None, None, :], tl.where(cand, w, 0.0)[:, :, None], 0.0), axis=1)
    tt = tl.arange(0, TB)
    tl.store(lw_ptr + p_ar[None, :] * TB + tt[:, None], wp)
    tl.atomic_add(cnt_ptr, tl.sum((full2 >= 0).to(tl.int64)))


def plan(ids, w, row_map, bank_map, rows, lfrom: int, lto: int, D: int, lslot, lw, lmask, count, band_map=None,
         pdl: bool = None):
    """ids [T, 16] (int32 or int64), w [T, 16] (float32 or bf16), rows [T, 16] int32 (unitk's), row_map / bank_map /
    band_map int32 [896]; lslot int32 [8], lw float32 [8, 8], lmask int32 [>= T, 16], count int64 [1] (persistent
    buffers: the send and the mean read them on the same stream)."""
    T, Kc = ids.shape
    assert Kc == K and 1 <= T <= MAXT and 0 <= D <= PMAX and lw.shape == (PMAX, MAXT) and lw.dtype == torch.float32
    assert lslot.dtype == torch.int32 and lmask.dtype == torch.int32 and rows.dtype == torch.int32
    _lplan[(1,)](ids if ids.dtype == torch.int32 else ids.int(), w.contiguous(), row_map, bank_map,
                 band_map if band_map is not None else row_map, rows.contiguous(), lslot, lw, lmask, count, T,
                 int(lfrom), int(lto), int(D), NE=E, TB=MAXT, KC=K, PM=PMAX, HAS_BAND=band_map is not None,
                 USE_PDL=PDL if pdl is None else pdl, num_warps=8)
    return lslot, lw, lmask


def plan_reference(ids, w, row_map, bank_map, rows, lfrom, lto, D, band_map=None):
    """The plan with torch ops (tests): -> lslot [8], lw [8, 8], lmask [T, 16], number of late picks."""
    T, Kc = ids.shape
    dev = ids.device
    valid = (ids >= 0) & (ids < E)
    e = ids.long().clamp(0, E - 1)
    wf = w.float()
    order = wf.argsort(dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(Kc, device=dev).expand(T, Kc))
    cand = valid & (row_map[e] < 0) & (bank_map[e] >= 0) & (rows < 0) & (rank >= lfrom) & (rank < lto)
    if band_map is not None:
        cand &= band_map[e] < 0
    energy = torch.zeros(E, dtype=torch.float32, device=dev)
    energy.index_add_(0, e.reshape(-1), torch.where(cand, wf * wf, torch.zeros_like(wf)).reshape(-1))
    has = torch.zeros(E, dtype=torch.bool, device=dev)
    has[e[cand]] = True
    cid = has.nonzero().flatten().tolist()
    cid.sort(key=lambda x: (-float(energy[x]), x))
    cid = cid[:D]
    lslot = torch.full((PMAX,), -1, dtype=torch.int32, device=dev)
    lw = torch.zeros((PMAX, MAXT), dtype=torch.float32, device=dev)
    lmask = torch.full((T, Kc), -1, dtype=torch.int32, device=dev)
    for p, x in enumerate(cid):
        lslot[p] = bank_map[x]
        sel = cand & (e == x)
        lmask[sel] = p
        for t in range(T):
            if bool(sel[t].any()):
                lw[p, t] = wf[t][sel[t]][0]
    return lslot, lw, lmask, int((lmask >= 0).sum())


# ------------------------------------------------------------------------------------------------ mean
@triton.jit
def _lmean(out_ptr, lmask_ptr, ids_ptr, w_ptr, mean_ptr, D, NE, KC: tl.constexpr, BD: tl.constexpr,
           USE_PDL: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.arange(0, KC)
    d = c * BD + tl.arange(0, BD)
    dm = d < D
    e = tl.load(ids_ptr + t * KC + k).to(tl.int32)
    lm = tl.load(lmask_ptr + t * KC + k)
    w = tl.load(w_ptr + t * KC + k).to(tl.float32)
    ok = (e >= 0) & (e < NE) & (lm >= 0)
    es = tl.where(ok, e, 0)
    m = tl.load(mean_ptr + es.to(tl.int64)[:, None] * D + d[None, :], mask=ok[:, None] & dm[None, :],
                other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(ok, w, 0.0)[:, None] * m, axis=0)
    tl.store(out_ptr + t * D + d, acc, mask=dm)


def mean(out, lmask, ids, w, table, pdl: bool = None):
    """out [>= T, 3584] float32 (rows 0..T-1 written) = the late picks' weight x mean sum; table [896, 3584]."""
    T, Kc = ids.shape
    Ne, D = table.shape
    assert out.dtype == torch.float32 and out.is_contiguous() and out.shape[1] == D and Kc == K
    BD = 512
    _lmean[(T, triton.cdiv(D, BD))](out, lmask, ids.contiguous(), w.contiguous(), table, D, Ne, KC=K, BD=BD,
                                    USE_PDL=PDL if pdl is None else pdl, num_warps=4)
    return out


def mean_reference(lmask, ids, w, table):
    T = ids.shape[0]
    ok = (lmask >= 0) & (ids >= 0) & (ids < table.shape[0])
    e = ids.long().clamp(0, table.shape[0] - 1)
    out = torch.zeros(T, table.shape[1], dtype=torch.float64, device=ids.device)
    for t in range(T):
        for k in range(ids.shape[1]):
            if bool(ok[t, k]):
                out[t] += float(w[t, k]) * table[e[t, k]].double()
    return out


# ------------------------------------------------------------------------------------------------ completion
@triton.jit
def _ladd(dst_ptr, out_ptr, z_ptr, y_ptr, coef_ptr, HC, RSD, RSO, BH: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    c = tl.program_id(1)
    h = c * BH + tl.arange(0, BH)
    m = h < HC
    has = tl.load(coef_ptr + 4 * t) > 0.0
    cb = tl.load(coef_ptr + 4 * t + 1)
    o = tl.load(dst_ptr + t * RSD + h, mask=m, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + t * HC + h, mask=m, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + t * HC + h, mask=m, other=0.0).to(tl.float32)
    r = tl.where(has, o + (z + cb * y), o)            # a row without a share: its own value back (same bits)
    tl.store(out_ptr + t * RSO + h, r.to(out_ptr.dtype.element_ty), mask=m)


def add(dst, z, y, coef, out=None, pdl: bool = None):
    """out (default dst, in place) [T, Hd] = dst + z + coef[:, 1] * y on rows with coef[:, 0] > 0. dst / out may be
    row-strided views (the stored block state res[:, b, :]); z and y are contiguous [T, Hd]."""
    T, Hd = z.shape
    if out is None:
        out = dst
    assert dst.shape == (T, Hd) and out.shape == (T, Hd) and y.shape == (T, Hd)
    assert dst.stride(1) == 1 and out.stride(1) == 1 and z.is_contiguous() and y.is_contiguous()
    assert coef.dtype == torch.float32 and coef.is_contiguous() and coef.shape[0] >= T and coef.shape[1] == 4
    BH = 1024
    _ladd[(T, triton.cdiv(Hd, BH))](dst, out, z, y, coef, Hd, dst.stride(0) if T > 1 else Hd,
                                    out.stride(0) if T > 1 else Hd, BH=BH, USE_PDL=PDL if pdl is None else pdl,
                                    num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------ references
def quant_rows_reference(x):
    """The send's rows (cpustage_seg2.h's rule) with torch: x [T, 3584] (bf16 or float) -> (xq uint8 [T, 3584] in
    row-planar byte order, xe uint8 [T, 112] exponent bytes e + 128, the dequantized values float64 [T, 3584])."""
    T = x.shape[0]
    v = x.float().reshape(T, H // 32, 32)
    amax = v.abs().amax(-1)
    e = (((amax.view(torch.int32).long() + 0x1FFFF) >> 23) - 133).clamp(-100, 120)
    q = torch.round(v.double() * torch.exp2(-e.double())[..., None]).clamp(-127, 127)      # half to even
    deq = (q * torch.exp2(e.double())[..., None]).reshape(T, H)
    kk = torch.arange(32)
    planar = 8 * (kk & 3) + (kk >> 2)                                                        # byte of input k
    qb = torch.empty(T, H // 32, 32, dtype=torch.uint8)
    qb[..., planar] = q.to(torch.int8).view(torch.uint8).cpu()
    return qb.reshape(T, H), (e + 128).to(torch.uint8).cpu(), deq


def coef_reference(u, v, eps):
    """float64 (has, cb, ra, rb) per row, from the definition."""
    ud, vd = u.double(), v.double()
    rb = torch.rsqrt(ud.pow(2).mean(-1) + eps)
    ra = torch.rsqrt((ud + vd).pow(2).mean(-1) + eps)
    has = (vd != 0).any(-1)
    cb = torch.where(has, ra / rb - 1, torch.zeros_like(ra))
    return torch.stack([has.double(), cb, torch.where(has, ra, rb), rb], dim=-1)


def land_reference(u, v, g, coef):
    """xl as the landing computes it, given its own coefficients: bf16((v * ra) * g) in float32, 0 where has == 0."""
    s = torch.where(coef[:, 0] > 0, coef[:, 2], torch.zeros_like(coef[:, 2])).float()
    return ((v.float() * s[:, None]) * g.float()[None, :]).to(torch.bfloat16)


def delta_reference(u, v, g, eps, weight):
    """The definition in float64: up(norm(u + v)) - up(norm(u)), weight [Hout, 3584] (float)."""
    ud, vd, gd, W = u.double(), v.double(), g.double(), weight.double()

    def rmsn(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gd
    return (rmsn(ud + vd) - rmsn(ud)) @ W.t()


# hi2-k1 records (engine/store/build_store.py): w13 codes [6144 rows, 896 B], w2 codes [3584, 768 B], w13 scales
# [28][6144 scale rows][4], w2 scales [24][3584][4]; 2 bits per weight (code pair c -> 1, 4, -1, -4 in units of 1.5 x
# 2^(scale byte - 127), the staged nibble (c << 2) | 3); rows in MegaMoE's kernel layout (w13 row 16 j + m = gate
# neuron 8 j + m, row 16 j + 8 + m = up neuron 8 j + m; scale row of weight row r = 128 (r // 128) + 4 (r % 32) +
# (r % 128) // 32). Same decode as the band's reference (sidecar/band/reference.py decode_hi2).
REC_BYTES = 9289728
_C13, _C2 = 6144 * 896, 3584 * 768
_S13, _S2 = 28 * 6144 * 4, 24 * 3584 * 4
_FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def _scale_row(rows: int, device=None):
    r = torch.arange(rows, device=device)
    return 128 * (r // 128) + 4 * (r % 32) + (r % 128) // 32


def _dequant(codes, sf, rows: int, kq: int):
    p = [(codes >> (2 * j)) & 3 for j in range(4)]
    n = [(q << 2) | 3 for q in p]
    native = torch.stack((n[0] | (n[1] << 4), n[2] | (n[3] << 4)), dim=-1).reshape(rows, -1)
    nib = torch.stack((native & 15, native >> 4), dim=-1).flatten(-2).long()
    val = _FP4.to(codes.device)[nib]
    s = sf.permute(1, 0, 2).reshape(rows, kq * 4)[_scale_row(rows, codes.device)]
    return val * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)


def decode_hi2(rec):
    """One hi2 record uint8 [9,289,728] -> (w1 [3072, 3584], w3 [3072, 3584], w2 [3584, 3072]) float32."""
    c13 = rec[:_C13].view(6144, 896)
    c2 = rec[_C13:_C13 + _C2].view(3584, 768)
    o = _C13 + _C2
    s13 = rec[o:o + _S13].view(28, 6144, 4)
    s2 = rec[o + _S13:o + _S13 + _S2].view(24, 3584, 4)
    w13 = _dequant(c13, s13, 6144, 28)
    w2 = _dequant(c2, s2, 3584, 24)
    r = torch.arange(I, device=rec.device)
    gate_row = 16 * (r // 8) + r % 8
    return w13[gate_row], w13[gate_row + 8], w2


def _q8(a, e_min=-100):
    """The pool's activation rule, dequantized (float64): int8 with a power-of-two scale per 32."""
    v = a.float().reshape(-1, 32)
    amax = v.abs().amax(-1, keepdim=True)
    e = (((amax.view(torch.int32).long() + 0x1FFFF) >> 23) - 133).clamp(e_min, 120)
    q = torch.round(v.double() * torch.exp2(-e.double())).clamp(-127, 127)
    return (q * torch.exp2(e.double())).reshape(a.shape)


def situ(gate, up, alpha=4.0, beta=25.0):
    gate, up = gate.float(), up.float()
    a = alpha * torch.tanh(gate / alpha) * torch.sigmoid(gate)
    if beta:
        up = beta * torch.tanh(up / beta)
    return a * up


def pool_reference(experts, xd, w, flags=1, alpha=4.0, beta=25.0):
    """The pool's arithmetic in float64 (r12 cpustage.h, its test_cpustage_ref.py pipeline): experts = [(w1, w3, w2)]
    float32, xd [T, 3584] the dequantized int8 rows (quant_rows_reference), w [P, T] routing weights (0 = unused).
    -> float64 [T, 3584] (rows that use no pick are 0). flags bit 0 (CPUJOB_OUT_BF16): each record's output rounded
    to bfloat16 before the sum; bit 1 (CPUJOB_NO_BF16): gate / up not rounded to bfloat16."""
    T = xd.shape[0]
    x64 = xd.double()
    y = torch.zeros(T, H, dtype=torch.float64, device=xd.device)
    for p, (w1, w3, w2) in enumerate(experts):
        wp = w[p, :T].to(xd.device).float()
        used = wp != 0
        if not bool(used.any()):
            continue
        gate = (x64 @ w1.double().T).float()
        up = (x64 @ w3.double().T).float()
        if not flags & 2:
            gate, up = gate.bfloat16().float(), up.bfloat16().float()
        a = situ(gate, up, alpha, beta) * wp[:, None]
        out = _q8(a) @ w2.double().T
        if flags & 1:
            out = out.float().bfloat16().double()
        y += torch.where(used[:, None], out, torch.zeros_like(out))
    return y
