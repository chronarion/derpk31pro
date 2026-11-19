"""unitk PRE + FILL (K31_PRESTAGE) against a plain-Python reference: random layers, a staged-ahead list from a
perturbed prediction of the call's picks (plain planner, slot cap 12), then the real call in PRE + FILL mode. Checked
exactly: every pick's row and the late list. Also checked: when the free slots suffice, every candidate of the plain
plan is served by a slot holding its expert (no candidate lost to pre-staging). Both with and without K31_STAGE_ALL, with and without per-row caps
(K31_STAGE_ROWS).

  docker run --rm --device nvidia.com/gpu=<GB300> -v $PWD/engine:/engine:ro --entrypoint python3 \\
      vllm/vllm-openai:kimi-k31 /engine/tests/test_unitk_fill.py
"""
import os
import sys

import torch

sys.path.insert(0, "/engine")
os.environ["K31_MODE"] = "staged"
from k31_engine import hook, unitk  # noqa: E402

dev = torch.device("cuda")
E, SL, S0 = 896, unitk.SLOTS, 16


def ranks(w):
    T, K = len(w), len(w[0])
    return [[sum(1 for j in range(K) if w[t][j] > w[t][k] or (w[t][j] == w[t][k] and j < k)) for k in range(K)]
            for t in range(T)]


def ref(ids, w, row, bank, strong, pre, allrows, resmax, rcap=None):
    T, K = len(ids), len(ids[0])
    rk = ranks(w)
    cand = [[ids[t][k] >= 0 and row[ids[t][k]] < 0 and bank[ids[t][k]] >= 0 and rk[t][k] < min(strong, unitk.SP)
             and (rcap is None or rk[t][k] < rcap[t]) for k in range(K)] for t in range(T)]
    slot_of_bank = {b: s for s, b in enumerate(pre) if b >= 0}
    miss = {}                                        # expert -> its largest candidate weight
    for t in range(T):
        for k in range(K):
            if cand[t][k] and bank[ids[t][k]] not in slot_of_bank:
                e = ids[t][k]
                miss[e] = max(miss.get(e, float("-inf")), w[t][k])
    order = sorted(miss, key=lambda e: (-miss[e], e))
    free = [s for s in range(SL) if pre[s] < 0]
    late = [-1] * SL
    fill = {}
    for j, e in enumerate(order[:len(free)]):
        late[free[j]] = bank[e]
        fill[e] = free[j]
    hold = [pre[s] if pre[s] >= 0 else late[s] for s in range(SL)]
    out = [[-1] * K for _ in range(T)]
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            if e < 0:
                continue
            if row[e] >= 0 and rk[t][k] < resmax:
                out[t][k] = row[e]
            if allrows:
                if row[e] < 0 and bank[e] >= 0 and bank[e] in hold:
                    out[t][k] = S0 + hold.index(bank[e])
            elif cand[t][k]:
                s = slot_of_bank.get(bank[e], fill.get(e, -1))
                if s >= 0:
                    out[t][k] = S0 + s
    return out, late, cand


bad = checked = lost = 0
for trial in range(400):
    g = torch.Generator(device="cpu").manual_seed(1000 + trial)
    perm = torch.randperm(E, generator=g)
    nres, nbank = int(torch.randint(0, 300, (1,), generator=g)), int(torch.randint(50, 500, (1,), generator=g))
    st = hook.LayerState(1, perm[:nres].tolist(), perm[nres:nres + nbank].tolist())
    row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
    bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
    T = [1, 2, 3, 4, 5, 5, 8, 16][trial % 8]
    strong = [1, 2, 3, 4][trial % 4]
    allrows = trial % 2 == 0
    resmax = [16, 16, 8, 12][trial % 4]
    pool = int(torch.randint(16, E, (1,), generator=g))
    ids = torch.stack([torch.randperm(pool, generator=g)[:16] for _ in range(T)]).to(torch.int32)
    w = torch.randint(1, 6, (T, 16), generator=g).float() / 8
    # the prediction: the real picks with a share of them replaced and the weights jittered
    pids = ids.clone()
    swap = torch.rand((T, 16), generator=g) < 0.2
    pids[swap] = torch.randint(0, E, (int(swap.sum()),), generator=g, dtype=torch.int32)
    for t in range(T):                               # keep a token's picks distinct
        seen = set()
        for k in range(16):
            while int(pids[t, k]) in seen:
                pids[t, k] = int(torch.randint(0, E, (1,), generator=g))
            seen.add(int(pids[t, k]))
    pw = (w + torch.rand((T, 16), generator=g) * 0.2).float()
    rcap = [int(x) for x in torch.randint(0, 5, (16,), generator=g)] if trial % 4 >= 2 else None   # K31_STAGE_ROWS
    rct = torch.tensor(rcap, dtype=torch.int32, device=dev) if rcap is not None else None
    pre = torch.full((SL,), -1, dtype=torch.int32, device=dev)
    unitk.plan(pids.to(dev), pw.to(dev), row_map, bank_map, min(unitk.SP, strong + 1), S0, stage=pre, resmax=16,
               allrows=False, cap=SL - 4, rowcap=rct)
    late = torch.full((SL,), -7, dtype=torch.int32, device=dev)
    out, _ = unitk.plan(ids.to(dev), w.to(dev), row_map, bank_map, strong, S0, stage=late, pre=pre, fill=True,
                        resmax=resmax, allrows=allrows, rowcap=rct)
    torch.cuda.synchronize()
    r_out, r_late, cand = ref(ids.tolist(), w.tolist(), st.row, st.bank_idx, strong, pre.tolist(), allrows, resmax,
                              rcap)
    ok = out.cpu().tolist() == r_out and late.cpu().tolist() == r_late
    if not ok:
        bad += 1
        if bad <= 3:
            print(f"trial {trial} (T {T}, strong {strong}, allrows {allrows}): kernel / reference differ")
            print("  late", late.cpu().tolist(), r_late)
    # the plain plan's candidates are all served by a slot holding their expert when the free slots suffice
    pl_out, pl_stage = unitk.plan(ids.to(dev), w.to(dev), row_map, bank_map, strong, S0, resmax=resmax, allrows=allrows,
                                  rowcap=rct)
    hold = [p if p >= 0 else l for p, l in zip(pre.tolist(), late.tolist())]
    n_free = sum(1 for p in pre.tolist() if p < 0)
    missed = {int(ids[t, k]) for t in range(T) for k in range(16)
              if cand[t][k] and st.bank_idx[int(ids[t, k])] not in pre.tolist()}
    if len(missed) <= n_free:
        checked += 1
        o = out.cpu().tolist()
        for t in range(T):
            for k in range(16):
                if cand[t][k] and int(pl_out[t, k]) >= 0:
                    s = o[t][k] - S0
                    if not (0 <= s < SL and hold[s] == st.bank_idx[int(ids[t, k])]):
                        lost += 1
print(f"PRE + FILL vs reference: {400 - bad}/400 exact; plain-plan candidates lost in {lost} picks over {checked} "
      f"trials with enough free slots")
assert bad == 0 and lost == 0
print("PASS")
