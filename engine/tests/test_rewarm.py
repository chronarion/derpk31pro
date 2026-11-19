"""Residency.rewarm against a direct reference: per ring layer, the k strongest-scored candidates (no row, a bank
record, score > 0) replace the k weakest ring rows (stale rows first) they outscore, pairwise in order; old owners
unmapped only if they still map to the row; one fetch list per layer (bank index, ring row). CPU, torch only."""
import sys
import types

import torch

sys.path.insert(0, "/engine")
import k31_engine.residency as res  # noqa: E402

E, R = 896, 16
bad = 0
for trial in range(200):
    g = torch.Generator().manual_seed(trial)
    L = 3
    fake = types.SimpleNamespace()
    fake.score = torch.rand((L, E), generator=g) * (torch.rand((L, E), generator=g) < 0.3)
    fake.row_map = torch.full((L, E), -1, dtype=torch.int32)
    fake.bank_map = torch.full((L, E), -1, dtype=torch.int32)
    fake.st = []
    calls = []
    for j in range(L):
        perm = torch.randperm(E, generator=g)
        S, slots = 105, 128
        for r, e in enumerate(perm[:S].tolist()):
            fake.row_map[j, e] = slots + r
        for b, e in enumerate(perm[:400].tolist()):
            fake.bank_map[j, e] = b
        st = types.SimpleNamespace(ring_ok=True, ring_base=slots + S - R, bank_ptr=1000 + j,
                                   ring_owner=perm[S - R:S].clone().int())
        if trial % 2:                                        # stale rows
            fake.row_map[j, int(st.ring_owner[3])] = 7
            st.ring_owner[5] = -1
        st.ring_stager = (lambda jj: (lambda ptr, src, ring, late: calls.append((jj, ptr, src.clone(), ring.clone(), late))))(j)
        fake.st.append(st)
    # reference
    rm_ref, own_ref, want = fake.row_map.clone(), [st.ring_owner.clone() for st in fake.st], {}
    for j, st in enumerate(fake.st):
        sc, rm, bm = fake.score[j], rm_ref[j], fake.bank_map[j]
        cands = sorted([e for e in range(E) if rm[e] < 0 and bm[e] >= 0], key=lambda e: (-float(sc[e]), e))[:R]
        rows = []
        for r in range(R):
            o = int(own_ref[j][r])
            live = o >= 0 and int(rm[o]) == st.ring_base + r
            rows.append((float(sc[o]) if live else -1.0, r, live, o))
        rows.sort(key=lambda x: (x[0], x[1]))
        pairs = [(c, row) for c, row in zip(cands, rows) if float(sc[c]) > row[0] and float(sc[c]) > 0]
        for c, (kv, r, live, o) in pairs:
            if live:
                rm[o] = -1
        for c, (kv, r, live, o) in pairs:
            rm[c] = st.ring_base + r
            own_ref[j][r] = c
        want[j] = ([int(bm[c]) for c, _ in pairs], [r for _, (kv, r, live, o) in pairs])
    n = res.Residency.rewarm(fake)
    ok = torch.equal(fake.row_map, rm_ref) and all(torch.equal(st.ring_owner, own_ref[j]) for j, st in enumerate(fake.st))
    for jj, ptr, src, ring, late in calls:
        k = int((src >= 0).sum())
        ok &= (src[:k].tolist(), ring[:k].tolist()) == want[jj] and not late and ptr == 1000 + jj
    ok &= n == sum(len(w[0]) for w in want.values()) and len(calls) == sum(1 for w in want.values() if w[0])
    if not ok:
        bad += 1
        if bad <= 3:
            print(f"trial {trial}: n {n}, calls {len(calls)}, want {[len(w[0]) for w in want.values()]}")
print(f"rewarm vs reference: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
