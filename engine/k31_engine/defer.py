"""Deferral kernels (engine/DEFERRAL.md v1): the per-call deferral plan and the late share's completion, one Triton
launch each (they replace ~20 and ~8 torch ops per layer).

plan(ids, w, row_map, bank_map, dfrom, dto, D, base, count) -> stage [16], pids [T, 16], pw [T, 16]
    A pick is a deferral candidate if its expert has no resident row, has a bank record and its rank within the token
    (by weight, ties: lower position first) is in [dfrom, dto). The candidates' distinct experts are ordered by their
    summed weight^2 over the call (ties: lower expert id); the first D take deferral slots 0.. . pids = base + slot for
    a deferred pick, else -1; pw = its weight, else 0; stage[s] = the bank index of slot s's expert, else -1; count
    (int64 scalar) += the number of deferred picks. T <= 8.

diff(u_now, u_late, g, eps) -> norm(u_now + u_late) - norm(u_now), RMSNorm with weight g, per row, in fp32.
"""
import torch
import triton
import triton.language as tl

SLOTS = 16


@triton.jit
def _dplan(ids_ptr, w_ptr, row_ptr, bank_ptr, stage_ptr, pids_ptr, pw_ptr, cnt_ptr, T, dfrom, dto, D, base,
           TB: tl.constexpr, K: tl.constexpr, SLOTSC: tl.constexpr):
    t = tl.arange(0, TB)
    k = tl.arange(0, K)
    m2 = (t[:, None] < T) & (k[None, :] < K)
    off = t[:, None] * K + k[None, :]
    ids = tl.load(ids_ptr + off, mask=m2, other=-1)
    w = tl.load(w_ptr + off, mask=m2, other=float("-inf")).to(tl.float32)
    valid = m2 & (ids >= 0)
    e = tl.where(valid, ids, 0)
    native = tl.load(row_ptr + e, mask=valid, other=-1)
    bank = tl.load(bank_ptr + e, mask=valid, other=-1)
    wi = w[:, :, None]
    wj = w[:, None, :]
    kj = k[None, None, :]
    ki = k[None, :, None]
    beats = (wj > wi) | ((wj == wi) & (kj < ki))
    rank = tl.sum(beats.to(tl.int32), axis=2)                               # [TB, K]
    cand = valid & (native < 0) & (bank >= 0) & (rank >= dfrom) & (rank < dto)
    N: tl.constexpr = TB * K
    fe = tl.reshape(e, (N,))
    fw = tl.reshape(tl.where(cand, w, 0.0), (N,))
    fc = tl.reshape(cand, (N,))
    fb = tl.reshape(bank, (N,))
    n_ar = tl.arange(0, N)
    same = fc[:, None] & fc[None, :] & (fe[:, None] == fe[None, :])          # [N, N]
    first = fc & (tl.sum((same & (n_ar[None, :] < n_ar[:, None])).to(tl.int32), axis=1) == 0)
    energy = tl.sum(tl.where(same, (fw * fw)[None, :], 0.0), axis=1)        # the expert's summed weight^2
    better = first[None, :] & ((energy[None, :] > energy[:, None]) |
                               ((energy[None, :] == energy[:, None]) & (fe[None, :] < fe[:, None])))
    slot = tl.sum(better.to(tl.int32), axis=1)
    ok = first & (slot < D)
    full = tl.sum(tl.where(same & ok[None, :], slot[None, :] + 1, 0), axis=1) - 1     # [N]: each entry's slot or -1
    s_ar = tl.arange(0, SLOTSC)
    stage = tl.sum(tl.where(ok[None, :] & (slot[None, :] == s_ar[:, None]), fb[None, :] + 1, 0), axis=1) - 1
    tl.store(stage_ptr + s_ar, stage)
    full2 = tl.reshape(full, (TB, K))
    dfr = full2 >= 0
    tl.store(pids_ptr + off, tl.where(dfr, base + full2, -1), mask=m2)
    tl.store(pw_ptr + off, tl.where(dfr, w, 0.0).to(pw_ptr.dtype.element_ty), mask=m2)
    tl.atomic_add(cnt_ptr, tl.sum(dfr.to(tl.int64)))


def plan(ids, w, row_map, bank_map, dfrom: int, dto: int, D: int, base: int, count: torch.Tensor):
    T, K = ids.shape
    assert K == 16 and T <= 8 and D <= SLOTS
    stage = torch.empty(SLOTS, dtype=torch.int32, device=ids.device)
    pids = torch.empty((T, K), dtype=torch.int32, device=ids.device)
    pw = torch.empty_like(w)
    TB = max(1, triton.next_power_of_2(T))
    _dplan[(1,)](ids.int() if ids.dtype != torch.int32 else ids, w.contiguous(), row_map, bank_map, stage, pids, pw,
                 count, T, dfrom, dto, D, base, TB=TB, K=K, SLOTSC=SLOTS, num_warps=4 if TB <= 2 else 8)
    return stage, pids, pw


@triton.jit
def _ddiff(u_ptr, l_ptr, g_ptr, o_ptr, D, eps, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    u = tl.load(u_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
    lt = tl.load(l_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + d, mask=m, other=0.0).to(tl.float32)
    a = u + lt
    ra = tl.rsqrt(tl.sum(a * a, axis=0) / D + eps)
    rb = tl.rsqrt(tl.sum(u * u, axis=0) / D + eps)
    has = tl.sum(tl.abs(lt), axis=0) > 0                       # a row without a late share adds exactly nothing
    o = tl.where(has, (a * ra - u * rb) * g, 0.0)
    tl.store(o_ptr + t * D + d, o.to(o_ptr.dtype.element_ty), mask=m)


def diff(u_now, u_late, g, eps: float):
    T, D = u_now.shape
    out = torch.empty_like(u_now)
    _ddiff[(T,)](u_now.contiguous(), u_late.contiguous(), g, out, D, float(eps), BD=triton.next_power_of_2(D),
                 num_warps=8)
    return out


@triton.jit
def _dplan_ring(ids_ptr, w_ptr, row_ptr, bank_ptr, src_ptr, ring_ptr, pids_ptr, pw_ptr, cnt_ptr, own_ptr, score_ptr,
                T, dfrom, dto, D, base, ring_base, R, LATE: tl.constexpr, TB: tl.constexpr, K: tl.constexpr,
                SLOTSC: tl.constexpr):
    """Stage 1: the lane's fetches land in the layer's ring rows and stay. Candidates and slots as _dplan;
    slot s's record goes to ring row ring[s] (the layer's ring rows in ascending keep order: rows whose owner no
    longer maps to them first, then by the owner's residency score), the old owner is unmapped (if it still maps
    there), the new one mapped (row_map[e] = ring_base + r): later picks of it are computed on time. LATE: the
    slots are also computed one layer late as pseudo rows (pids = base + slot), else fetch only (pids -1)."""
    t = tl.arange(0, TB)
    k = tl.arange(0, K)
    m2 = (t[:, None] < T) & (k[None, :] < K)
    off = t[:, None] * K + k[None, :]
    ids = tl.load(ids_ptr + off, mask=m2, other=-1)
    w = tl.load(w_ptr + off, mask=m2, other=float("-inf")).to(tl.float32)
    valid = m2 & (ids >= 0)
    e = tl.where(valid, ids, 0)
    native = tl.load(row_ptr + e, mask=valid, other=-1)
    bank = tl.load(bank_ptr + e, mask=valid, other=-1)
    wi = w[:, :, None]
    wj = w[:, None, :]
    kj = k[None, None, :]
    ki = k[None, :, None]
    beats = (wj > wi) | ((wj == wi) & (kj < ki))
    rank = tl.sum(beats.to(tl.int32), axis=2)
    cand = valid & (native < 0) & (bank >= 0) & (rank >= dfrom) & (rank < dto)
    N: tl.constexpr = TB * K
    fe = tl.reshape(e, (N,))
    fw = tl.reshape(tl.where(cand, w, 0.0), (N,))
    fc = tl.reshape(cand, (N,))
    fb = tl.reshape(bank, (N,))
    n_ar = tl.arange(0, N)
    same = fc[:, None] & fc[None, :] & (fe[:, None] == fe[None, :])
    first = fc & (tl.sum((same & (n_ar[None, :] < n_ar[:, None])).to(tl.int32), axis=1) == 0)
    energy = tl.sum(tl.where(same, (fw * fw)[None, :], 0.0), axis=1)
    better = first[None, :] & ((energy[None, :] > energy[:, None]) |
                               ((energy[None, :] == energy[:, None]) & (fe[None, :] < fe[:, None])))
    slot = tl.sum(better.to(tl.int32), axis=1)
    ok = first & (slot < D)
    s_ar = tl.arange(0, SLOTSC)
    src = tl.sum(tl.where(ok[None, :] & (slot[None, :] == s_ar[:, None]), fb[None, :] + 1, 0), axis=1) - 1
    se = tl.sum(tl.where(ok[None, :] & (slot[None, :] == s_ar[:, None]), fe[None, :] + 1, 0), axis=1) - 1
    nsel = tl.sum(ok.to(tl.int32), axis=0)
    # the ring rows, in ascending keep order
    r_ar = tl.arange(0, SLOTSC)
    rm = r_ar < R
    own = tl.load(own_ptr + r_ar, mask=rm, other=-1)
    hasown = rm & (own >= 0)
    cur = tl.load(row_ptr + tl.where(hasown, own, 0), mask=hasown, other=-1)
    live = hasown & (cur == ring_base + r_ar)
    sc = tl.load(score_ptr + tl.where(live, own, 0), mask=live, other=0.0)
    keep = tl.where(live, sc, -1.0)
    keep = tl.where(rm, keep, float("inf"))
    kb = (keep[None, :] < keep[:, None]) | ((keep[None, :] == keep[:, None]) & (r_ar[None, :] < r_ar[:, None]))
    pos = tl.sum(kb.to(tl.int32), axis=1)                                   # [R]: 0 = the first victim
    victim = tl.sum(tl.where(pos[None, :] == s_ar[:, None], r_ar[None, :] + 1, 0), axis=1) - 1   # [slot]
    use = (s_ar < nsel) & (victim >= 0) & (s_ar < R)
    ring = tl.where(use, victim, -1)
    tl.store(ring_ptr + s_ar, ring)
    tl.store(src_ptr + s_ar, tl.where(use, src, -1))
    # deferred picks of this call: their slot's pseudo row (LATE) and weight
    full = tl.sum(tl.where(same & ok[None, :], slot[None, :] + 1, 0), axis=1) - 1
    full2 = tl.reshape(full, (TB, K))
    dfr = (full2 >= 0) & (full2 < R)
    if LATE:
        tl.store(pids_ptr + off, tl.where(dfr, base + full2, -1), mask=m2)
        tl.store(pw_ptr + off, tl.where(dfr, w, 0.0).to(pw_ptr.dtype.element_ty), mask=m2)
        tl.atomic_add(cnt_ptr, tl.sum(dfr.to(tl.int64)))
    # map updates (after every load above)
    tl.debug_barrier()
    vict = rm & (pos < nsel)
    newo = tl.sum(tl.where((pos[:, None] == s_ar[None, :]) & use[None, :], se[None, :] + 1, 0), axis=1) - 1   # [R]
    tl.store(row_ptr + tl.where(live, own, 0), -1, mask=vict & live)
    tl.debug_barrier()
    tl.store(row_ptr + tl.where(newo >= 0, newo, 0), ring_base + r_ar, mask=vict & (newo >= 0))
    tl.store(own_ptr + r_ar, newo, mask=vict)


def plan_ring(ids, w, row_map, bank_map, own, score, dfrom: int, dto: int, D: int, base: int, ring_base: int, R: int,
              late: bool, count: torch.Tensor):
    """-> src [16] (bank index per slot, -1), ring [16] (ring row per slot, -1), pids [T, 16], pw [T, 16]. Updates
    row_map and own (int32 [R], the expert in each ring row) in place. T <= 8, D <= R <= 16."""
    T, K = ids.shape
    assert K == 16 and T <= 8 and D <= R <= SLOTS
    src = torch.empty(SLOTS, dtype=torch.int32, device=ids.device)
    ring = torch.empty(SLOTS, dtype=torch.int32, device=ids.device)
    pids = torch.full((T, K), -1, dtype=torch.int32, device=ids.device)
    pw = torch.zeros_like(w)
    TB = max(1, triton.next_power_of_2(T))
    _dplan_ring[(1,)](ids.int() if ids.dtype != torch.int32 else ids, w.contiguous(), row_map, bank_map, src, ring,
                      pids, pw, count, own, score, T, dfrom, dto, D, base, ring_base, R, LATE=late, TB=TB, K=K,
                      SLOTSC=SLOTS, num_warps=4 if TB <= 2 else 8)
    return src, ring, pids, pw


@triton.jit
def _cutmean(out_ptr, rows_ptr, ids_ptr, w_ptr, pids_ptr, band_ptr, mean_ptr, D, E, HAS_PIDS: tl.constexpr,
             HAS_BAND: tl.constexpr, K: tl.constexpr, BD: tl.constexpr, SERVED: tl.constexpr = False,
             STORE: tl.constexpr = False):
    """Row t, columns [c BD, (c + 1) BD) of a MegaMoE call: out[t] += sum over its cut picks (an expert id, no row,
    not deferred, not served by the band) of weight x the expert's mean output (fp32 sum, one rounding). The K picks
    are read as vectors and their mean rows gathered as one [K, BD] tile: two dependent loads per program instead of
    a chain of K (the one-program-per-row version took 7.5 us on the chain in the 11-18 fw-d4 profile)."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.arange(0, K)
    d = c * BD + tl.arange(0, BD)
    dm = d < D
    e = tl.load(ids_ptr + t * K + k).to(tl.int32)
    r = tl.load(rows_ptr + t * K + k)
    w = tl.load(w_ptr + t * K + k).to(tl.float32)
    ok = (e >= 0) & (e < E) & (r < 0)
    if HAS_PIDS:
        ok = ok & (tl.load(pids_ptr + t * K + k) < 0)
    es = tl.where(ok, e, 0)
    if HAS_BAND:
        if SERVED:                       # band_ptr = the send's per-pick flags (v3): past the cap -> the mean
            ok = ok & (tl.load(band_ptr + t * K + k) == 0)
        else:                            # band_ptr = the band map: every band expert counts as served
            ok = ok & (tl.load(band_ptr + es) < 0)
    m = tl.load(mean_ptr + es.to(tl.int64)[:, None] * D + d[None, :], mask=ok[:, None] & dm[None, :],
                other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(ok, w, 0.0)[:, None] * m, axis=0)
    if STORE:                            # cut_mean_pre: the float32 sum alone (the v4 band finish adds it)
        tl.store(out_ptr + t * D + d, acc, mask=dm)
    else:
        o = tl.load(out_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
        tl.store(out_ptr + t * D + d, (o + acc).to(out_ptr.dtype.element_ty), mask=dm)


def cut_mean_pre(msum, rows, ids, w, mean, pids=None, band=None, served=None):
    """K31_TAIL1: msum [T, D] float32 = the call's cut-pick mean sum (as cut_mean adds it), stored without reading the
    routed output: it runs beside the stager and MegaMoE, and the v4 band finish adds it to MegaMoE's output."""
    assert msum.dtype == torch.float32
    return cut_mean(msum, rows, ids, w, mean, pids, band, served, _store=True)


def cut_mean(out, rows, ids, w, mean, pids=None, band=None, served=None, _store=False):
    """In place: out [T, D] (the call's routed latent sum) += w x mean[e] for every pick with an expert id, no row
    (rows < 0), with pids not deferred (pids < 0), with band not in the band (band[e] < 0). mean [E, D]. Rows past T
    are left alone. Grid: rows x 512-wide column chunks (35 programs for 5 rows of 3,584)."""
    T, K = ids.shape
    E, D = mean.shape
    if T == 0:
        return out
    assert K & (K - 1) == 0 and out.is_contiguous() and out.shape[1] == D
    BD = 512
    _cutmean[(T, triton.cdiv(D, BD))](out, rows.contiguous(), ids.contiguous(), w.contiguous(),
                                      pids if pids is not None else rows,
                                      served if served is not None else (band if band is not None else rows), mean, D,
                                      E, HAS_PIDS=pids is not None, HAS_BAND=served is not None or band is not None,
                                      K=K, BD=BD, SERVED=served is not None, STORE=_store, num_warps=4)
    return out
    _cutmean[(T,)](out, rows.contiguous(), ids.contiguous(), w.contiguous(), pids if pids is not None else rows,
                   band if band is not None else rows, mean, D, E, HAS_PIDS=pids is not None,
                   HAS_BAND=band is not None, K=K, BD=triton.next_power_of_2(D), num_warps=8)
    return out
