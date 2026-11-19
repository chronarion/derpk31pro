"""unitk (Triton) against the torch reference of mode staged's plan: for every pick, the expert its row will hold
(resident row -> that expert; slot -> the staged bank record's expert; -1 -> cut) must agree, and the staged sets
must be equal; random inputs with ties. Then its time in a CUDA graph."""
import os
import sys

import torch

sys.path.insert(0, "/engine")
os.environ["K31_MODE"] = "staged"
from k31_engine import hook, unitk  # noqa: E402

dev = torch.device("cuda")
E, SL = 896, hook.SLOTS
bad = 0
for trial in range(300):
    g = torch.Generator(device="cpu").manual_seed(trial)
    perm = torch.randperm(E, generator=g)
    nres, nbank = int(torch.randint(0, 300, (1,), generator=g)), int(torch.randint(0, 500, (1,), generator=g))
    st = hook.LayerState(1, perm[:nres].tolist(), perm[nres:nres + nbank].tolist())
    st.row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
    st.bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
    st.arange = torch.arange(16, dtype=torch.long, device=dev)
    T = [1, 2, 3, 4, 5, 8, 12, 16][trial % 8]
    strong = [1, 2, 3, 4][trial % 4]
    hook.STRONG = strong
    st.strong = strong                     # the torch reference (_plan) reads the layer's own staging rank since 11-18
    pool = int(torch.randint(16, E, (1,), generator=g))          # small pools make experts repeat across tokens
    ids = torch.stack([torch.randperm(pool, generator=g)[:16] for _ in range(T)]).to(dev, torch.int32)
    w = (torch.randint(1, 6, (T, 16), generator=g).float() / 8).to(dev)   # coarse values: many ties
    out, stage = unitk.plan(ids, w, st.row_map, st.bank_map, strong, st.slot_base)
    # torch reference (hook's eager path for one round)
    picks = ids.long()
    native, bank, cand, score, valid = hook._plan(st, ids, w)
    v, e = score.topk(SL)
    ok = v > 0
    want_set = set(e[ok].tolist())
    bank_list = torch.tensor(st.bank + [0], device=dev)
    got_set = set(bank_list[stage[stage >= 0].long()].tolist())
    inv = torch.full((st.slots + st.S + 1,), -1, dtype=torch.long, device=dev)      # row -> resident expert
    for e_, r_ in enumerate(st.row):
        if r_ >= 0:
            inv[r_] = e_
    sb = st.slot_base
    is_slot = (out >= sb) & (out < sb + SL)
    holder = torch.where(out < 0, -1, torch.where(is_slot, bank_list[stage[(out - sb).clamp(0, SL - 1).long()].clamp_min(0).long()],
                                                  inv[out.clamp(0, st.slots + st.S).long()]))
    # which picks the reference stages: candidates whose expert made the set
    ref_hold = torch.where(valid & (native >= 0), picks, -1)
    in_set = torch.tensor([x in want_set for x in picks.reshape(-1).tolist()], device=dev).view_as(picks)
    ref_hold = torch.where(cand & in_set, picks, ref_hold)
    n_cand = int((score > 0).sum())
    if n_cand > SL:      # ties at the 16th place may pick different experts: compare only when nothing is left out
        continue
    if got_set != want_set or not torch.equal(holder, ref_hold):
        bad += 1
        if bad <= 3:
            print(f"trial {trial} T={T} strong={strong}: sets {sorted(got_set)} vs {sorted(want_set)}; "
                  f"{int((holder != ref_hold).sum())} picks differ")
print(f"unitk vs torch reference: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
for T in (1, 3, 4, 8, 16):
    ids = torch.stack([torch.randperm(E)[:16] for _ in range(T)]).to(dev, torch.int32)
    w = torch.rand((T, 16), device=dev)
    out = torch.empty((T, 16), dtype=torch.int32, device=dev); stage = torch.empty(SL, dtype=torch.int32, device=dev)
    for _ in range(3):
        unitk.plan(ids, w, st.row_map, st.bank_map, 3, st.slot_base, out, stage)
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        unitk.plan(ids, w, st.row_map, st.bank_map, 3, st.slot_base, out, stage)
    best = 1e9
    for _ in range(50):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gph.replay(); b.record(); b.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3)
    print(f"T={T}: unitk in a CUDA graph {best:.1f} us")
if bad:
    sys.exit(1)
# residency inputs: scores (decay then + w^2) and pick counts, against torch
st2 = hook.LayerState(1, list(range(100)), list(range(100, 400)))
st2.row_map = torch.tensor(st2.row, dtype=torch.int32, device=dev)
st2.bank_map = torch.tensor(st2.bank_idx, dtype=torch.int32, device=dev)
score = torch.rand(896, device=dev); stats = torch.zeros(4, dtype=torch.int32, device=dev)
ids = torch.stack([torch.randperm(500)[:16] for _ in range(4)]).to(dev, torch.int32)
w = torch.rand((4, 16), device=dev)
want = score.clone() * (0.9 ** 4)
want.index_add_(0, ids.reshape(-1).long(), (w * w).reshape(-1))
out, stage = unitk.plan(ids, w, st2.row_map, st2.bank_map, 3, 0, score=score, lam=0.9, stats=stats)
res = int((st2.row_map[ids.long()] >= 0).sum()); stg = int(((out >= 0) & (out < 16)).sum())
ok2 = torch.allclose(score, want, rtol=1e-5, atol=1e-6) and stats.tolist() == [res, stg, 64 - res - stg, 64]
print(f"unitk residency inputs (scores, counts {stats.tolist()}): {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    sys.exit(1)
# PRE mode: candidates map onto pre-staged slots (by bank index) or are cut; the stage list is not written
st3 = hook.LayerState(1, list(range(100)), list(range(100, 400)))
st3.row_map = torch.tensor(st3.row, dtype=torch.int32, device=dev)
st3.bank_map = torch.tensor(st3.bank_idx, dtype=torch.int32, device=dev)
ids = torch.stack([torch.randperm(500)[:16] for _ in range(3)]).to(dev, torch.int32)
w = torch.rand((3, 16), device=dev)
_, stage_full = unitk.plan(ids, w, st3.row_map, st3.bank_map, 3, 16)
pre = stage_full.clone(); pre[pre.ge(0).nonzero()[-1:]] = -1                 # drop one staged expert: it must be cut
stage_buf = torch.full((16,), 77, dtype=torch.int32, device=dev)
out_pre, st_ret = unitk.plan(ids, w, st3.row_map, st3.bank_map, 3, 16, stage=stage_buf, pre=pre)
out_full, _ = unitk.plan(ids, w, st3.row_map, st3.bank_map, 3, 16)
diff = (out_pre != out_full)
dropped = int(stage_full[pre.lt(stage_full)].numel()) if False else 1
ok3 = bool((stage_buf == 77).all()) and int(diff.sum()) >= 1 and bool((out_pre[diff] == -1).all())
print(f"unitk PRE mode (pre-staged slots; one dropped expert's picks cut, {int(diff.sum())} picks): {'PASS' if ok3 else 'FAIL'}")
sys.exit(0 if ok3 else 1)
