"""Budgeted promotion (policy.budgeted_hot) as one Triton launch per call: one program per batch row runs the
token loop with the row's scores and sets in registers (the torch loop launches ~30 kernels per token, which made
policy-heavy passes Python-bound: ~2.5 min per layer for 22 universes on 16 cuts). Same rule as the torch loop:
per token, credit += P; up to min(floor(credit), PMAX) swaps, each pairing the j-th strongest non-member with the j-th
weakest member of the state before the token's swaps (swap if it scores more and >= 0); usable = the state after
the swaps `delay` tokens earlier (the start set before that); then scores decay by lam and add w^2 of the picks."""
import triton
import triton.language as tl
import torch


@triton.jit
def _bhot(ids_ptr, w_ptr, place_ptr, out_ptr, prm_ptr, T, H, E: tl.constexpr, EB: tl.constexpr, K: tl.constexpr,
          PMAX: tl.constexpr, DELAY: tl.constexpr):
    b = tl.program_id(0)
    P = tl.load(prm_ptr)                # float64, as the torch loop's Python floats (fp32 arguments drift the credit)
    lam = tl.load(prm_ptr + 1)
    e = tl.arange(0, EB)
    em = e < E
    k = tl.arange(0, K)
    place = tl.load(place_ptr + e, mask=em, other=2147483647)
    init = ((place < H) & em).to(tl.int32)
    in_set = init
    # the last DELAY states as bits of one int32 per expert (bit 0 = the previous token's state): a rotation of
    # several loop-carried vectors miscompiled (delay 2 disagreed with the torch loop; delays 0 and 1 matched)
    hist = init * ((1 << DELAY) - 1)
    score = tl.zeros([EB], dtype=tl.float64)
    credit = tl.zeros([1], dtype=tl.float64)
    NEG: tl.constexpr = -1.0e300
    POS: tl.constexpr = 1.0e300
    for t in range(T):
        usable = init
        if PMAX > 0:
            credit = credit + P
            n = tl.minimum(tl.floor(credit), PMAX * 1.0)
            credit = credit - n
            nn = tl.sum(n.to(tl.int32), axis=0)
            cand = tl.where(em & (in_set == 0), score, NEG)
            mem = tl.where(in_set != 0, score, POS)
            new_set = in_set
            for j in tl.static_range(PMAX):
                cv = tl.max(cand, axis=0)
                ci = tl.argmax(cand, axis=0)
                mv = tl.min(mem, axis=0)
                mi = tl.argmin(mem, axis=0)
                do = (cv > mv) & (cv >= 0) & (j < nn)
                new_set = tl.where(do & (e == ci), 1, new_set)
                new_set = tl.where(do & (e == mi), 0, new_set)
                cand = tl.where(e == ci, NEG, cand)            # the next pair from the same (pre-swap) state
                mem = tl.where(e == mi, POS, mem)
            in_set = new_set
            if DELAY == 0:
                usable = in_set
            else:
                usable = (hist >> (DELAY - 1)) & 1
                hist = ((hist << 1) | in_set) & ((1 << DELAY) - 1)
        base = (b * T + t) * K
        ids_t = tl.load(ids_ptr + base + k)
        w_t = tl.load(w_ptr + base + k).to(tl.float64)
        match = e[None, :] == ids_t[:, None]                    # [K, EB]
        hit = tl.sum(tl.where(match, usable[None, :], 0), axis=1) > 0
        tl.store(out_ptr + base + k, hit.to(tl.int8))
        score = score * lam + tl.sum(tl.where(match, (w_t * w_t)[:, None], 0.0), axis=0)


def budgeted_hot_triton(ids, w, num_experts, H, P, delay, halflife, start_place):
    B, T, K = ids.shape
    assert delay <= 8 and K & (K - 1) == 0
    pmax = (max(1, int(-(-P // 1))) if P > 0 else 0)
    out = torch.empty((B, T, K), dtype=torch.int8, device=ids.device)
    prm = torch.tensor([float(P), 2.0 ** (-1.0 / halflife)], dtype=torch.float64, device=ids.device)
    _bhot[(B,)](ids.long().contiguous(), w.contiguous(), start_place.to(torch.int32).contiguous(), out, prm, T, int(H),
                E=num_experts, EB=triton.next_power_of_2(num_experts), K=K, PMAX=pmax, DELAY=int(delay), num_warps=4)
    return out.bool()
