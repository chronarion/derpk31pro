"""Reference model of the prompt-chunk plan (pure Python, no torch / numpy).

What a prompt chunk of T rows asks of one MoE layer, as `prompt_path.py` computes it on the device:

  tiers of a pick (expert e, rank k = its place by routing weight within its token):
      row      row_map[e] >= 0                  -> MegaMoE on the expert's native row, round 0
      band     band_map[e] >= 0 (no row)        -> the RTX (bulk band), or cut when the profile has no band
      bank     bank_map[e] >= 0 (no row / band) -> staged from the Grace bank, in the round that holds its record
      none                                      -> cut: the mean substitute
  staging:     the layer's bank records are staged in rounds of R = 128 shared rows. Which records:
      "all"    every record with at least one pick in the chunk
      valve f  the records with the largest summed pick amplitude (amp[rank]) in the chunk, at most ceil(f x G)
      ranks r  only picks of rank < r count (r = 16: every pick); a staged record serves only those picks
  round r stages the kept records ranked [r R, (r + 1) R) (order: descending mass, ties by lower bank index: the order
  of a stable sort on the device) into shared rows 0..R-1; its MegaMoE call gets, per pick,
      the native row (round 0 only), the shared row of its record (if the record is in this round), else -1.
  Every pick is computed at most once over the rounds; picks that no round computes and the band does not serve are the
  cut picks (mean substitute).

`plan()` returns the rounds; `check()` verifies the invariants on them. Run this file: random cases, exit 0 = PASS.
The device version must reproduce `plan()` exactly (test_prompt_path.py compares them)."""
import math
import random

E, K, R = 896, 16, 128


def plan(ids, row_map, band_map, bank_map, G, amp, mode="all", valve=1.0, ranks=16, shared=R, band_on=True):
    """ids: T lists of K expert ids (-1 = no pick), sorted by weight (position = rank).
    row_map / band_map / bank_map: E ints (-1 = none); bank_map values are record slots in [0, G).
    -> dict(rounds=[(stage list [shared] of bank slots or -1, ids_r [T][K])], band [T][K] bool, cut [T][K] bool,
            kept=[bank slots in staging order])"""
    T = len(ids)
    tier = [[None] * K for _ in range(T)]
    mass = [0.0] * G
    hit = [0] * G
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            if e < 0:
                tier[t][k] = "pad"
            elif row_map[e] >= 0:
                tier[t][k] = "row"
            elif band_on and band_map[e] >= 0:
                tier[t][k] = "band"
            elif bank_map[e] >= 0 and k < ranks:
                tier[t][k] = "bank"
                mass[bank_map[e]] += amp[k]
                hit[bank_map[e]] += 1
            else:
                tier[t][k] = "none"
    cand = [j for j in range(G) if hit[j] > 0]
    if mode == "valve":
        cand.sort(key=lambda j: (-mass[j], j))
        cand = cand[:int(math.ceil(valve * G))]
    else:                                   # "all": the bank's own order (ascending slot)
        cand.sort()
    place = {j: i for i, j in enumerate(cand)}
    n_rounds = max(1, -(-len(cand) // shared))
    rounds = []
    for r in range(n_rounds):
        chunk = cand[r * shared:(r + 1) * shared]
        stage = chunk + [-1] * (shared - len(chunk))
        ids_r = [[-1] * K for _ in range(T)]
        for t in range(T):
            for k in range(K):
                e = ids[t][k]
                if tier[t][k] == "row" and r == 0:
                    ids_r[t][k] = shared + row_map[e]          # native rows follow the shared rows
                elif tier[t][k] == "bank":
                    p = place.get(bank_map[e], -1)
                    if p >= 0 and p // shared == r:
                        ids_r[t][k] = p % shared
        rounds.append((stage, ids_r))
    band = [[tier[t][k] == "band" for k in range(K)] for t in range(T)]
    cut = [[tier[t][k] == "none" or (tier[t][k] == "bank" and bank_map[ids[t][k]] not in place) for k in range(K)]
           for t in range(T)]
    return {"rounds": rounds, "band": band, "cut": cut, "kept": cand, "tier": tier}


def check(ids, row_map, band_map, bank_map, G, p, shared=R, n_rows=None):
    """Invariants of a plan; raises AssertionError with the reason."""
    T = len(ids)
    served = [[0] * K for _ in range(T)]
    seen_slots = set()
    for r, (stage, ids_r) in enumerate(p["rounds"]):
        assert len(stage) == shared
        live = [j for j in stage if j >= 0]
        assert len(set(live)) == len(live), "a record staged twice in a round"
        assert not (seen_slots & set(live)), "a record staged in two rounds"
        seen_slots |= set(live)
        for t in range(T):
            for k in range(K):
                v = ids_r[t][k]
                e = ids[t][k]
                if v < 0:
                    continue
                assert e >= 0, "a padded pick got a row"
                served[t][k] += 1
                if v >= shared:
                    assert r == 0 and row_map[e] == v - shared, "native row only in round 0, and the expert's own"
                    assert n_rows is None or v - shared < n_rows
                else:
                    assert stage[v] == bank_map[e] and bank_map[e] >= 0, "a staged pick must read its own record"
                    assert row_map[e] < 0, "a resident expert is never staged"
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            n = served[t][k] + int(p["band"][t][k]) + int(p["cut"][t][k])
            if e < 0:
                assert n == 0, "a padded pick is nobody's"
            else:
                assert n == 1, f"pick ({t},{k}) expert {e}: served {served[t][k]} band {p['band'][t][k]} cut {p['cut'][t][k]}"
    return True


def staged_bytes(p, rec_bytes=9_289_728):
    return sum(sum(1 for j in stage if j >= 0) for stage, _ in p["rounds"]) * rec_bytes


def _random_case(rng, T, H, M, G, skew=1.2):
    order = list(range(E))
    rng.shuffle(order)
    row_map, band_map, bank_map = [-1] * E, [-1] * E, [-1] * E
    for i, e in enumerate(order[:H]):
        row_map[e] = i
    for i, e in enumerate(order[H:H + M]):
        band_map[e] = i
    slots = list(range(G))
    rng.shuffle(slots)                       # a bank slot is not the expert's rank
    used = min(G, E - H - M)
    for i, e in enumerate(order[H + M:H + M + used]):
        bank_map[e] = slots[i]
    w = [1.0 / (i + 1) ** skew for i in range(E)]
    ids = []
    for _ in range(T):
        picks = set()
        while len(picks) < K:
            picks.add(order[min(E - 1, int(rng.choices(range(E), weights=w)[0]))])
        row = list(picks)
        rng.shuffle(row)
        if rng.random() < 0.05:              # a padded tail, as vLLM's padding rows have
            c = rng.randrange(K)
            row[c:] = [-1] * (K - c)
        ids.append(row)
    return ids, row_map, band_map, bank_map


def main():
    rng = random.Random(14)
    amp = [0.484, 0.119, 0.097, 0.083, 0.044, 0.04, 0.037, 0.034, 0.023, 0.022] + [0.017] * 6
    n = 0
    for T, H, M, G in ((1, 56, 50, 200), (17, 105, 118, 454), (64, 260, 216, 420), (200, 378, 216, 302),
                       (512, 56, 0, 640), (300, 105, 118, 0), (128, 0, 0, 896)):
        for mode, valve, ranks, band_on in (("all", 1.0, 16, True), ("valve", 0.5, 16, True), ("valve", 0.125, 16, True),
                                            ("all", 1.0, 4, True), ("valve", 0.25, 8, False), ("all", 1.0, 16, False)):
            ids, row_map, band_map, bank_map = _random_case(rng, T, H, M, G)
            p = plan(ids, row_map, band_map, bank_map, G, amp, mode, valve, ranks, band_on=band_on)
            check(ids, row_map, band_map, bank_map, G, p, n_rows=H)
            kept = p["kept"]
            assert len(kept) <= (G if mode == "all" else int(math.ceil(valve * G)))
            if mode == "all" and ranks == 16:        # nothing with a bank record is cut
                for t in range(T):
                    for k in range(K):
                        e = ids[t][k]
                        if e >= 0 and row_map[e] < 0 and not (band_on and band_map[e] >= 0) and bank_map[e] >= 0:
                            assert not p["cut"][t][k]
            if mode == "valve" and kept:             # the valve keeps the heaviest records
                mass = {}
                for t in range(T):
                    for k in range(min(K, ranks)):
                        e = ids[t][k]
                        if e >= 0 and p["tier"][t][k] == "bank":
                            mass[bank_map[e]] = mass.get(bank_map[e], 0.0) + amp[k]
                lo = min(mass[j] for j in kept)
                assert all(mass[j] <= lo + 1e-12 for j in mass if j not in kept), "a heavier record was left out"
            assert len(p["rounds"]) == max(1, -(-len(kept) // R))
            n += 1
    print(f"prompt_plan_model: PASS ({n} cases)")


if __name__ == "__main__":
    main()
