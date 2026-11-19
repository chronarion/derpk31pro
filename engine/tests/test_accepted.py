"""K31_ACCEPTED: unitk parks a call's picks (no score change), commit() decays each layer's scores by lam^A and adds
the accepted rows' squared weights (row t of request r accepted iff t - qsl[r] < num_sampled[r]; padded rows and
layers without pending rows untouched), acc += A. Random steps against a torch reference."""
import math
import os
import sys

import torch

sys.path.insert(0, "/engine")
os.environ["K31_MODE"] = "staged"
from k31_engine import hook, unitk  # noqa: E402

dev = torch.device("cuda")
E, L = 896, 6
lam = 2.0 ** (-1.0 / 32)
bad = 0
for trial in range(400):
    g = torch.Generator(device="cpu").manual_seed(trial)
    nreq = int(torch.randint(1, 5, (1,), generator=g))
    nrows = torch.randint(1, 5, (nreq,), generator=g)                       # 1 + drafts per request
    while int(nrows.sum()) > 14:
        nrows[int(torch.argmax(nrows))] -= 1
    T = int(nrows.sum())
    Tpad = min(16, T + int(torch.randint(0, 3, (1,), generator=g)))          # graph padding rows
    nsamp = torch.stack([torch.randint(1, int(n) + 1, (1,), generator=g)[0] for n in nrows]).int()
    qsl = torch.zeros(nreq + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(nrows, 0)
    score0 = torch.rand((L, E), generator=g).to(dev)
    score = score0.clone()
    pend_ids = torch.full((L, 16, 16), -1, dtype=torch.int32, device=dev)
    pend_w2 = torch.zeros((L, 16, 16), dtype=torch.float32, device=dev)
    pend_t = torch.zeros(L, dtype=torch.int32, device=dev)
    acc = torch.zeros(1, dtype=torch.int64, device=dev)
    stats = torch.zeros((L, 4), dtype=torch.int32, device=dev)
    ran = torch.rand(L, generator=g) < 0.8                                   # layers whose unitk ran this step
    ref = score0.clone()
    A = int(nsamp.sum())
    perm = torch.randperm(E, generator=g)
    st = hook.LayerState(1, perm[:100].tolist(), perm[100:500].tolist())
    row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
    bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
    for j in range(L):
        if not ran[j]:
            continue
        ids = torch.stack([torch.randperm(E, generator=g)[:16] for _ in range(Tpad)]).int()
        ids[torch.rand((Tpad, 16), generator=g) < 0.05] = -1
        w = torch.rand((Tpad, 16), generator=g)
        unitk.plan(ids.to(dev), w.to(dev), row_map, bank_map, 3, st.slot_base, score=score[j], lam=lam,
                   stats=stats[j], pend=(pend_ids[j], pend_w2[j], pend_t[j:j + 1]))
        ref[j] *= lam ** A
        for r in range(nreq):
            for t in range(int(qsl[r]), int(qsl[r]) + int(nsamp[r])):
                for k in range(16):
                    if ids[t, k] >= 0:
                        ref[j, int(ids[t, k])] += float(w[t, k]) ** 2
    untouched = torch.equal(score, score0)
    unitk.commit(score, pend_ids, pend_w2, pend_t, qsl.to(dev), nsamp.to(dev), nreq, lam, acc)
    ok = untouched and torch.allclose(score, ref, rtol=1e-5, atol=1e-5) and int(pend_t.abs().sum()) == 0 \
        and int(acc[0]) == (A if bool(ran.any()) and bool(ran[0]) else int(acc[0]))
    # acc is added by program 0 only when layer 0 ran (layer 0 = the first MoE layer, which always runs in the engine)
    if bool(ran[0]) and int(acc[0]) != A:
        ok = False
    if not ok:
        bad += 1
        if bad <= 3:
            print(f"trial {trial}: nreq {nreq} rows {nrows.tolist()} sampled {nsamp.tolist()} Tpad {Tpad}: untouched "
                  f"{untouched}, max err {float((score - ref).abs().max()):.2e}, pend_t {pend_t.tolist()}, acc "
                  f"{int(acc[0])} vs {A}")
print(f"accepted-row commit vs torch reference: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
