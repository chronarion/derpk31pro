"""Helpers of the late tier's tests: unitk's row ids as torch computes them (both of its modes), random layer maps,
synthetic hi2-k1 records.

unitk_rows(ids, w, row_map, bank_map, strong, S, allrows, resmax) follows k31_engine/unitk.py: a pick's rank is its
place by weight within its token (ties: lower position); a candidate is a pick with no resident row, a bank record
and a rank < strong; the candidates' distinct experts by their largest weight (ties: lower expert id), the first 16,
hold staging slots 0..15. Row ids: a resident pick of rank < resmax -> its row; a candidate pick of a slotted expert ->
S + slot; with allrows (K31_STAGE_ALL) every non-resident pick of a slotted expert, at any rank and in any
row, -> S + slot; everything else -1."""
import torch

E, K, SLOTS = 896, 16, 16


def ranks(w):
    T, Kc = w.shape
    order = w.float().argsort(dim=1, descending=True, stable=True)
    r = torch.empty_like(order)
    r.scatter_(1, order, torch.arange(Kc, device=w.device).expand(T, Kc))
    return r


def unitk_rows(ids, w, row_map, bank_map, strong: int, S: int, allrows: bool, resmax: int = 16):
    """-> (row ids int32 [T, 16], the slotted experts {expert: slot})."""
    valid = (ids >= 0) & (ids < E)
    e = ids.long().clamp(0, E - 1)
    native, bank = row_map[e], bank_map[e]
    rk = ranks(w)
    cand = valid & (native < 0) & (bank >= 0) & (rk < strong)
    best = {}
    for t in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            if bool(cand[t, k]):
                x = int(e[t, k])
                best[x] = max(best.get(x, float("-inf")), float(w[t, k]))
    order = sorted(best, key=lambda x: (-best[x], x))[:SLOTS]
    slot = {x: i for i, x in enumerate(order)}
    out = torch.where(valid & (native >= 0) & (rk < resmax), native, torch.full_like(native, -1))
    for t in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            x = int(e[t, k])
            if x in slot and bool(valid[t, k]) and int(native[t, k]) < 0 and (allrows or bool(cand[t, k])):
                out[t, k] = S + slot[x]
    return out.to(torch.int32), slot


def rand_maps(g, nres: int, nbank: int, nband: int = 0, device="cpu"):
    """Disjoint resident rows (from row 128), bank records and band slots over 896 experts."""
    perm = torch.randperm(E, generator=g)
    row = torch.full((E,), -1, dtype=torch.int32)
    bank = torch.full((E,), -1, dtype=torch.int32)
    band = torch.full((E,), -1, dtype=torch.int32)
    row[perm[:nres]] = torch.arange(nres, dtype=torch.int32) + 128
    bank[perm[nres:nres + nbank]] = torch.arange(nbank, dtype=torch.int32)
    band[perm[nres + nbank:nres + nbank + nband]] = torch.arange(nband, dtype=torch.int32)
    return row.to(device), bank.to(device), band.to(device)


def synth_records(n: int, seed: int, rec_bytes: int = 9289728):
    """n hi2-k1 records of random codes with realistic scale bytes (117..121): uint8 [n, rec_bytes] on the CPU."""
    gen = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 256, (n, rec_bytes), generator=gen, dtype=torch.uint8)
    o = 6144 * 896 + 3584 * 768
    r[:, o:] = torch.randint(117, 122, (n, rec_bytes - o), generator=gen, dtype=torch.uint8)
    return r
