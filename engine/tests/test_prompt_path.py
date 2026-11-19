"""k31_engine.prompt_path.plan_rounds (torch) against tests/prompt_plan_model.plan (pure Python). CPU torch is enough
for part 1; part 2 needs the GB300 and a bound engine layer.

Run part 1 in the engine image, CPU only, one core (from engine/):
    docker run --rm --network none --cpuset-cpus 64 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_prompt_path.py
Exit 0 and "PROMPT PATH PLAN: PASS" = the device plan equals the reference on every case: the kept records and their
order, every pick's place, native row, band flag and cut flag.

Part 2 (in-server self-check, K31_PROMPT_SELFCHECK=1, first prompt chunk of each MoE layer): forward_prompt in mode t1
against today's eager branch on the same inputs: the same picks are served, so the routed latent sums agree to the
order of the float32 / bf16 additions across rounds (relative L2 < 2e-3; the bind stops above 1e-2). Sketch at the end.
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import prompt_path as pp      # noqa: E402
import prompt_plan_model as ref               # noqa: E402

E, K, R = 896, 16, 128


def case(rng, T, H, M, G):
    ids, row_map, band_map, bank_map = ref._random_case(rng, T, H, M, G)
    # weights: descending within a token, so that position = rank, as the reference assumes
    w = [sorted((rng.random() for _ in range(K)), reverse=True) for _ in range(T)]
    return ids, w, row_map, band_map, bank_map


def run(mode, valve, ranks, strong, band_on, T, H, M, G, rng):
    ids, w, row_map, band_map, bank_map = case(rng, T, H, M, G)
    # the engine's row_map holds absolute rows (shared rows first): reference rows + R
    rm = torch.tensor([r + R if r >= 0 else -1 for r in row_map], dtype=torch.int32)
    bm = torch.tensor(band_map, dtype=torch.int32) if band_on else None
    km = torch.tensor(bank_map, dtype=torch.int32)
    kept, n_keep, place, nat, is_band, cut = pp.plan_rounds(
        torch.tensor(ids, dtype=torch.int32), torch.tensor(w, dtype=torch.float32), rm, km, bm, G, strong,
        mode=mode, valve=valve, ranks=ranks)
    amp = list(pp.AMP)
    p = ref.plan(ids, row_map, band_map, bank_map, G, amp, "valve" if mode == "valve" else "all", valve,
                 strong if mode == "t1" else ranks, band_on=band_on)
    n = int(n_keep)
    assert kept[:n].tolist() == p["kept"], f"kept records differ ({mode}, T {T}, G {G})"
    assert (kept[n:] == -1).all()
    pl = {j: i for i, j in enumerate(p["kept"])}
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            tier = p["tier"][t][k]
            want_place = pl.get(bank_map[e], -1) if tier == "bank" else -1
            assert int(place[t, k]) == want_place, f"place of pick ({t},{k})"
            assert int(nat[t, k]) == (row_map[e] + R if tier == "row" else -1), f"native row of pick ({t},{k})"
            assert bool(is_band[t, k]) == p["band"][t][k] and bool(cut[t, k]) == p["cut"][t][k], f"flags ({t},{k})"
    # the rounds' ids as forward_prompt builds them
    rounds = max(1, -(-n // R))
    assert rounds == len(p["rounds"])
    rnd = torch.div(place, R, rounding_mode="floor")
    pos = place - rnd * R
    for r in range(rounds):
        got = torch.where((place >= 0) & (rnd == r), pos, nat if r == 0 else torch.full_like(nat, -1))
        assert got.tolist() == p["rounds"][r][1], f"round {r} ids"
        stage = kept[r * R:(r + 1) * R].tolist()
        stage += [-1] * (R - len(stage))
        assert stage == p["rounds"][r][0], f"round {r} stage list"


def main():
    torch.set_num_threads(1)
    # integer amplitudes: the record masses are then exact sums in float32 (device) and float64 (reference), so the
    # valve's order cannot differ by a rounding of near-ties
    pp.AMP = tuple(float(round(a * 1000)) for a in pp.AMP)
    pp._C.clear()
    rng = random.Random(14)
    n = 0
    for T, H, M, G in ((1, 56, 50, 200), (17, 105, 118, 454), (64, 260, 216, 420), (200, 378, 216, 302),
                       (512, 56, 0, 640), (300, 105, 118, 1), (128, 0, 0, 896)):
        for mode, valve, ranks, strong, band_on in (("all", 1.0, 16, 1, True), ("valve", 0.5, 16, 1, True),
                                                    ("valve", 0.125, 16, 1, True), ("t1", 1.0, 16, 4, True),
                                                    ("t1", 1.0, 16, 1, False), ("valve", 0.25, 8, 1, False),
                                                    ("all", 1.0, 16, 1, False)):
            run(mode, valve, ranks, strong, band_on, T, H, M, G, rng)
            n += 1
    print(f"PROMPT PATH PLAN: PASS ({n} cases)")


if __name__ == "__main__":
    main()

# Part 2, in hook._forward when K31_PROMPT_SELFCHECK=1 (first eager call of each layer, T > 16):
#     y_new = prompt_path.forward_prompt(..., mode t1 via env, band_eager=_band_eager, cut_mean=_dm.cut_mean)
#     y_old = <today's eager branch on clones of the inputs>
#     rel = float((y_new.float() - y_old.float()).norm() / y_old.float().norm())
#     _LOG(f"prompt path selfcheck layer {st.idx}: rel {rel:.2e}")   # expect < 2e-3; raise above 1e-2
# Both paths serve the same picks in mode t1 only when no layer has more candidates than its staged rounds hold;
# today's branch keeps the strongest experts per round by score and stages every candidate over its rounds, so the
# served sets are equal and only the order of the additions differs.
