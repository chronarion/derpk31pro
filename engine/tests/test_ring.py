"""Stage 1 ring (K31_RING): defer.plan_ring against a torch reference (slots as defer.plan; victims = ring rows in
ascending keep order: stale rows first, then the owner's score; old owners unmapped only if they still map there; new
owners mapped), over consecutive steps on one layer state; and stager.stage2 byte-equal to the single-destination
stager into both destinations."""
import os
import sys

import torch

sys.path.insert(0, "/engine")
os.environ["K31_MODE"] = "staged"
from k31_engine import defer  # noqa: E402

dev = torch.device("cuda")
E, R, RB = 896, 16, 300
bad = 0
for trial in range(120):
    g = torch.Generator(device="cpu").manual_seed(trial)
    perm = torch.randperm(E, generator=g)
    S = 105
    ring_base = 128 + S - R
    row_map = torch.full((E,), -1, dtype=torch.int32)
    for r, e in enumerate(perm[:S].tolist()):
        row_map[e] = 128 + r
    bank_map = torch.full((E,), -1, dtype=torch.int32)
    for j, e in enumerate(perm[:600].tolist()):                 # the bank covers the residents and more
        bank_map[e] = j
    own = perm[S - R:S].clone().int()
    if trial % 3 == 0:                                            # some ring rows stale: their owner moved natively
        row_map[int(own[2])] = 5
        own[7] = -1
    score = torch.rand(E, generator=g)
    rm, ow = row_map.to(dev), own.to(dev)
    sc = score.to(dev)
    count = torch.zeros((), dtype=torch.int64, device=dev)
    D = [1, 2, 4, 8][trial % 4]
    for step in range(4):
        T = [1, 3, 4, 8][(trial + step) % 4]
        pool = int(torch.randint(40, E, (1,), generator=g))
        ids = torch.stack([torch.randperm(pool, generator=g)[:16] for _ in range(T)]).int()
        w = torch.rand((T, 16), generator=g)
        late = (trial + step) % 2 == 0
        # torch reference
        rmh, owh = rm.cpu().clone(), ow.cpu().clone()
        order = w.argsort(dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(16).expand(T, 16))
        e = ids.long()
        cand = (rmh[e] < 0) & (bank_map[e] >= 0) & (rank >= 1) & (rank < 12)
        energy = torch.zeros(E, dtype=torch.float64)
        energy.index_add_(0, e.reshape(-1), torch.where(cand, w.double() ** 2, 0.0).reshape(-1))
        exps = [x for x in range(E) if energy[x] > 0]
        exps.sort(key=lambda x: (-float(energy[x]), x))
        chosen = exps[:D]
        keep = []
        for r in range(R):
            o = int(owh[r])
            live = o >= 0 and int(rmh[o]) == ring_base + r
            keep.append((float(score[o]) if live else -1.0, r, live, o))
        victims = sorted(keep, key=lambda x: (x[0], x[1]))[:len(chosen)]
        want_ring = [v[1] for v in victims] + [-1] * (16 - len(chosen))
        want_src = [int(bank_map[x]) for x in chosen] + [-1] * (16 - len(chosen))
        for (kv, r, live, o), x in zip(victims, chosen):
            if live:
                rmh[o] = -1
        for (kv, r, live, o), x in zip(victims, chosen):
            rmh[x] = ring_base + r
            owh[r] = x
        src, ring, pids, pw = defer.plan_ring(ids.to(dev), w.to(dev), rm, bank_map.to(dev), ow, sc, 1, 12, D, 112,
                                              ring_base, R, late, count)
        slot_of = {x: s for s, x in enumerate(chosen)}
        want_pids = torch.tensor([[112 + slot_of[int(ids[t, k])] if (late and bool(cand[t, k]) and int(ids[t, k]) in slot_of)
                                   else -1 for k in range(16)] for t in range(T)], dtype=torch.int32)
        ok = (src.cpu().tolist() == want_src and ring.cpu().tolist() == want_ring and torch.equal(rm.cpu(), rmh)
              and torch.equal(ow.cpu(), owh) and torch.equal(pids.cpu(), want_pids))
        if not ok:
            bad += 1
            if bad <= 3:
                print(f"trial {trial} step {step} T={T} D={D} late={late}: src {src.cpu().tolist()[:6]} vs {want_src[:6]}; "
                      f"ring {ring.cpu().tolist()[:6]} vs {want_ring[:6]}; row_map diff {int((rm.cpu() != rmh).sum())}, "
                      f"owner diff {int((ow.cpu() != owh).sum())}, pids diff {int((pids.cpu() != want_pids).sum())}")
            rm, ow = rmh.to(dev), owh.to(dev)
print(f"plan_ring vs torch reference: {'PASS' if bad == 0 else f'FAIL ({bad})'}")

# stage2: one read, two destinations, byte-equal to the single-destination stager
from k31_engine.stager import Stager, RingStager  # noqa: E402


class FakeMod:
    pass


rec = {"w13": 2 * 3072 * 3584 // 2, "w2": 3584 * 3072 // 2, "w13s": 2 * 3072 * 3584 // 32, "w2s": 3584 * 3072 // 32}
layout = {"native_bytes": {"w13": rec["w13"], "w2": rec["w2"], "w13s": rec["w13s"], "w2s": rec["w2s"]},
          "record_bytes": rec["w13"] // 2 + rec["w2"] // 2 + rec["w13s"] + rec["w2s"]}
rows = 40
m = FakeMod()
m.w13_weight = torch.zeros((rows, 2 * 3072, 3584 // 2), dtype=torch.uint8, device=dev)
m.w2_weight = torch.zeros((rows, 3584, 3072 // 2), dtype=torch.uint8, device=dev)
m.w13_weight_scale = torch.zeros((rows, 2 * 3072, 3584 // 32), dtype=torch.uint8, device=dev)
m.w2_weight_scale = torch.zeros((rows, 3584, 3072 // 32), dtype=torch.uint8, device=dev)
nrec = 6
bank = torch.randint(0, 256, (nrec, layout["record_bytes"]), dtype=torch.uint8).pin_memory()
ref = FakeMod()
for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
    setattr(ref, name, torch.zeros_like(getattr(m, name)))
src = torch.tensor([3, 1, 5, -1, 0] + [-1] * 11, dtype=torch.int32, device=dev)
ring = torch.tensor([7, -1, 2, 4, 9] + [-1] * 11, dtype=torch.int32, device=dev)
rs = RingStager(m, 16, 20, layout, sms=24)          # deferral rows from 16, ring rows from 20
rs(bank.data_ptr(), src, ring, True)
one = Stager(ref, 16, layout, sms=24)
one(bank.data_ptr(), src)                            # slot k -> row 16 + k
torch.cuda.synchronize()
okb = True
for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
    a, b = getattr(m, name), getattr(ref, name)
    for k in range(16):
        s_, r_ = int(src[k]), int(ring[k])
        if s_ >= 0:
            okb &= torch.equal(a[16 + k], b[16 + k])                    # the deferral row
            if r_ >= 0:
                okb &= torch.equal(a[20 + r_], b[16 + k])              # the ring row: same bytes
    untouched = [r for r in range(rows) if not (16 <= r < 32 and int(src[r - 16]) >= 0)
                 and not any(20 + int(ring[k]) == r and int(src[k]) >= 0 and int(ring[k]) >= 0 for k in range(16))]
    okb &= bool((a[untouched] == 0).all())
m2 = FakeMod()
for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
    setattr(m2, name, torch.zeros_like(getattr(m, name)))
RingStager(m2, 16, 20, layout, sms=24)(bank.data_ptr(), src, ring, False)   # fetch only: ring rows, no deferral rows
torch.cuda.synchronize()
for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
    a, b = getattr(m2, name), getattr(ref, name)
    for k in range(16):
        if int(src[k]) >= 0 and int(ring[k]) >= 0:
            okb &= torch.equal(a[20 + int(ring[k])], b[16 + k])
    okb &= bool((a[16:20] == 0).all())
print(f"stage2 dual write vs single stager: {'PASS' if okb else 'FAIL'}")
