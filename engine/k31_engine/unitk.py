"""unitk: mode staged's per-call plan as one Triton launch (replaces ~20 torch ops, 77-83 us in a CUDA graph).

For a call's picks (ids, weights [T, 16], T <= 16): rank each pick within its token by weight (ties: lower position
first); a pick is a candidate if its expert has no resident row, has a bank record and its rank is < strong. The
candidates' distinct experts are ordered by their largest weight (ties: lower expert id) and the first SLOTS take
slots 0, 1, ...; their picks go to row S + slot, resident picks to their rows, everything else to -1 (cut).
Outputs: the call's row ids [T, 16] (int32) and the stage list [SLOTS] (bank index per slot, or -1).
Only the candidates of rank < SP (= 4, so strong <= 4) are considered, as per-token columns [T, SP]: the pairwise
work is [T*SP, T*SP] (64 x 64 at T = 16).
"""
import torch
import triton
import triton.language as tl

SLOTS = 16
SP = 4
PDL = __import__("os").environ.get("K31_PDL", "0") == "1"


@triton.jit
def _unitk(ids_ptr, w_ptr, row_ptr, bank_ptr, out_ptr, stage_ptr, T, strong, S, score_ptr, decay, stats_ptr,
           pre_ptr, pid_ptr, pw_ptr, pt_ptr, resmax, TB: tl.constexpr, K: tl.constexpr, SPC: tl.constexpr,
           SLOTSC: tl.constexpr, NE: tl.constexpr, SCORE: tl.constexpr, PRE: tl.constexpr, ACC: tl.constexpr):
    tl.extra.cuda.gdc_wait()                 # PDL: the router's ids / weights (and earlier writes) are visible
    tl.extra.cuda.gdc_launch_dependents()
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
    # rank within the token: picks with a larger weight, or an equal weight at a lower position
    wi = w[:, :, None]
    wj = w[:, None, :]
    kj = k[None, None, :]
    ki = k[None, :, None]
    beats = (wj > wi) | ((wj == wi) & (kj < ki))
    rank = tl.sum(beats.to(tl.int32), axis=2)                       # [TB, K]
    cand = valid & (native < 0) & (bank >= 0) & (rank < strong)
    # per token, the pick of rank r (r < SP) as columns: expert, weight, candidate flag, bank index
    r_ar = tl.arange(0, SPC)
    sel = rank[:, None, :] == r_ar[None, :, None]                   # [TB, SP, K]
    ce = tl.sum(tl.where(sel, e[:, None, :], 0), axis=2)            # [TB, SP]
    cw = tl.sum(tl.where(sel, w[:, None, :], 0.0), axis=2)
    cc = tl.sum(tl.where(sel & cand[:, None, :], 1, 0), axis=2) > 0
    cb = tl.sum(tl.where(sel, bank[:, None, :], 0), axis=2)
    # flatten to N = TB * SP entries
    N: tl.constexpr = TB * SPC
    fe = tl.reshape(ce, (N,))
    fw = tl.reshape(cw, (N,))
    fc = tl.reshape(cc, (N,))
    fb = tl.reshape(cb, (N,))
    n_ar = tl.arange(0, N)
    s_ar = tl.arange(0, SLOTSC)
    if PRE:
        # the slots were staged ahead (from the pre-gate prediction): a candidate whose record is in a slot uses it,
        # any other candidate is cut; nothing new is staged
        ps = tl.load(pre_ptr + s_ar)
        match = fc[:, None] & (fb[:, None] == ps[None, :]) & (ps[None, :] >= 0)              # [N, SLOTS]
        full = tl.max(tl.where(match, s_ar[None, :], -1), axis=1)
    else:
        same = fc[:, None] & fc[None, :] & (fe[:, None] == fe[None, :])       # [N, N]
        first = fc & (tl.sum((same & (n_ar[None, :] < n_ar[:, None])).to(tl.int32), axis=1) == 0)
        score = tl.max(tl.where(same, fw[None, :], float("-inf")), axis=1)   # the expert's largest weight
        better = first[None, :] & ((score[None, :] > score[:, None]) |
                                   ((score[None, :] == score[:, None]) & (fe[None, :] < fe[:, None])))
        slot = tl.sum(better.to(tl.int32), axis=1)                            # place among distinct experts
        ok = first & (slot < SLOTSC)
        # every entry's slot = its expert's first occurrence's slot
        full = tl.sum(tl.where(same & ok[None, :], slot[None, :] + 1, 0), axis=1) - 1
        stage = tl.sum(tl.where(ok[None, :] & (slot[None, :] == s_ar[:, None]), fb[None, :] + 1, 0), axis=1) - 1
        tl.store(stage_ptr + s_ar, stage)
    # the call's rows: resident -> its row; a staged candidate -> S + slot; else -1
    full2 = tl.reshape(full, (TB, SPC))
    served = valid & (native >= 0) & (rank < resmax)          # K31_RES_MAXRANK: a token's weakest picks cut even resident
    out = tl.where(served, native, -1)
    for r in tl.static_range(SPC):
        col = tl.sum(tl.where(r_ar[None, :] == r, full2, 0), axis=1)        # [TB]: slot of the token's rank-r pick
        hit = cand & (rank == r) & (col[:, None] >= 0)
        out = tl.where(hit, S + col[:, None], out)
    tl.store(out_ptr + off, out, mask=m2)
    if SCORE:
        # residency.py's inputs: the layer's decayed scores (decay^T, then + w^2 per pick) and pick counts
        # [resident, staged, cut, all]
        if ACC:
            # accepted rows only (K31_ACCEPTED): the rows' picks wait in the layer's pending buffer [16, 16]; commit()
            # adds the accepted rows after the step's rejection sampling
            tl.store(pid_ptr + off, tl.where(valid, ids, -1), mask=m2)
            tl.store(pw_ptr + off, tl.where(valid, w * w, 0.0), mask=m2)
            tl.store(pt_ptr, T)
        else:
            eo = tl.arange(0, NE)
            em = eo < 896
            sc = tl.load(score_ptr + eo, mask=em, other=0.0)
            tl.store(score_ptr + eo, sc * decay, mask=em)
            tl.debug_barrier()
            tl.atomic_add(score_ptr + e, w * w, mask=valid)
        n_all = tl.sum(valid.to(tl.int32))
        n_res = tl.sum(served.to(tl.int32))
        n_stg = tl.sum((out >= S).to(tl.int32) * (out < S + SLOTSC).to(tl.int32) * (native < 0).to(tl.int32) * valid.to(tl.int32))
        tl.atomic_add(stats_ptr + 0, n_res)
        tl.atomic_add(stats_ptr + 1, n_stg)
        tl.atomic_add(stats_ptr + 2, n_all - n_res - n_stg)
        tl.atomic_add(stats_ptr + 3, n_all)


def plan(ids: torch.Tensor, w: torch.Tensor, row_map: torch.Tensor, bank_map: torch.Tensor, strong: int, S: int,
         out: torch.Tensor = None, stage: torch.Tensor = None, score: torch.Tensor = None, lam: float = 1.0,
         stats: torch.Tensor = None, pre: torch.Tensor = None, pend=None, resmax: int = 16):
    """S: the first row of the call's staging slots. score (float32 [896]) and stats (int32 [4]): residency inputs,
    updated in place when given (decay lam per token). pre (int32 [SLOTS], bank index per slot): the slots were
    staged ahead; candidates map onto them or are cut, and `stage` is left untouched."""
    T, K = ids.shape
    assert K == 16 and T <= 16 and strong <= SP
    if out is None:
        out = torch.empty((T, K), dtype=torch.int32, device=ids.device)
    if stage is None:
        stage = torch.empty(SLOTS, dtype=torch.int32, device=ids.device)
    TB = max(1, triton.next_power_of_2(T))
    sc = score is not None
    acc = sc and pend is not None
    pid, pw, pt = pend if acc else (out, out, out)
    _unitk[(1,)](ids.int() if ids.dtype != torch.int32 else ids, w.contiguous(), row_map, bank_map, out, stage, T,
                 strong, S, score if sc else out, float(lam ** T), stats if sc else out, pre if pre is not None else stage,
                 pid, pw, pt, int(resmax), TB=TB, K=K, SPC=SP, SLOTSC=SLOTS, NE=1024, SCORE=sc, PRE=pre is not None, ACC=acc,
                 num_warps=4 if TB <= 4 else 8, launch_pdl=PDL)
    return out, stage


@triton.jit
def _commit(score_ptr, pid_ptr, pw_ptr, pt_ptr, qsl_ptr, ns_ptr, nreq, log2lam, acc_ptr, RB: tl.constexpr,
            NE: tl.constexpr, TB: tl.constexpr, K: tl.constexpr):
    j = tl.program_id(0)
    T = tl.load(pt_ptr + j)
    if T > 0:
        r = tl.arange(0, RB)
        rm = r < nreq
        q0 = tl.load(qsl_ptr + r, mask=rm, other=0).to(tl.int32)
        q1 = tl.load(qsl_ptr + r + 1, mask=rm, other=0).to(tl.int32)
        ns = tl.load(ns_ptr + r, mask=rm, other=0).to(tl.int32)
        A = tl.sum(tl.where(rm, ns, 0), axis=0)
        t = tl.arange(0, TB)
        inq = rm[None, :] & (t[:, None] >= q0[None, :]) & (t[:, None] < q1[None, :]) & \
            (t[:, None] - q0[None, :] < ns[None, :])
        acc_row = (tl.sum(inq.to(tl.int32), axis=1) > 0) & (t < T)
        eo = tl.arange(0, NE)
        em = eo < 896
        sc = tl.load(score_ptr + j * 896 + eo, mask=em, other=0.0)
        tl.store(score_ptr + j * 896 + eo, sc * tl.exp2(A.to(tl.float32) * log2lam), mask=em)
        tl.debug_barrier()
        k = tl.arange(0, K)
        off = t[:, None] * K + k[None, :]
        m2 = acc_row[:, None] & (k[None, :] < K)
        e = tl.load(pid_ptr + j * TB * K + off, mask=m2, other=-1)
        w2 = tl.load(pw_ptr + j * TB * K + off, mask=m2, other=0.0)
        ok = m2 & (e >= 0)
        tl.atomic_add(score_ptr + j * 896 + tl.where(ok, e, 0), w2, mask=ok)
        tl.store(pt_ptr + j, 0)
        if j == 0:
            tl.atomic_add(acc_ptr, A.to(tl.int64))


def commit(score: torch.Tensor, pend_ids: torch.Tensor, pend_w2: torch.Tensor, pend_t: torch.Tensor,
           qsl: torch.Tensor, num_sampled: torch.Tensor, nreq: int, lam: float, acc: torch.Tensor) -> None:
    """After a step's sampling: per layer with pending rows (unitk with `pend`), decay the scores by lam^A (A = the
    step's accepted tokens = sum num_sampled) and add the accepted rows' squared weights (row t of request r is
    accepted if t - qsl[r] < num_sampled[r]); acc += A (promotion credit per accepted token). One launch, no host
    sync. score [L, 896] float32, pend_ids / pend_w2 [L, 16, 16], pend_t [L] int32."""
    import math
    L = score.shape[0]
    _commit[(L,)](score, pend_ids, pend_w2, pend_t, qsl, num_sampled, nreq, math.log2(lam), acc,
                  RB=max(2, triton.next_power_of_2(nreq + 1)), NE=1024, TB=16, K=16, num_warps=4)
