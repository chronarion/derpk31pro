"""K31_FORCE_DRAFTS (hook._force_drafts): the forced requests' drafts in the runner's per-slot table become the
reference tokens at the right positions, up to the emulated accepted count, then (token + 1) % V; prefill rows,
rows without drafts and slots without forced tokens keep the drafter's drafts; the accepted counts follow P(i)."""
import os
import sys
import types

os.environ["K31_FORCE_DRAFTS"] = "1"
os.environ["K31_FORCE_ACCEPT"] = "0.665,0.432,0.266,0.180"
DRAFTER = os.environ.get("K31_FORCE_REJECT", "next") == "drafter"     # run once with each
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
import torch  # noqa: E402

from k31_engine import force, hook  # noqa: E402

dev = torch.device(os.environ.get("TEST_DEV", "cuda"))           # TEST_DEV=cpu while the GPUs measure
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


S, L = 4, 200
tok = torch.full((8, L), -1, dtype=torch.int32, device=dev)
ref = torch.arange(1000, 1000 + L, dtype=torch.int32, device=dev)    # the token at sequence position q is 1000 + q
for slot in (2, 5):
    tok[slot, :L - 1] = ref[1:]                  # tok[slot, p] = the token sampled at position p + 1 = 1000 + p + 1
force.INSTANCE[0] = types.SimpleNamespace(tok=tok)
# batch: row 0 = slot 2 decoding (last sampled at position 70, 4 drafts), row 1 = slot 5 decoding (position 120),
# row 2 = slot 3 not forced, row 3 = slot 5's prefill-like row with 0 drafts
idx = torch.tensor([2, 5, 3, 6], dtype=torch.int32, device=dev)
nl = torch.tensor([5, 5, 5, 1], device=dev)
F = torch.tensor([70, 120, 40, 10], device=dev)
seq = (F + nl).to(torch.int32)
cu = torch.zeros(5, dtype=torch.int32, device=dev)
cu[1:] = torch.cumsum(nl, 0)
draft = torch.full((8, S), 7, dtype=torch.int32, device=dev)          # the drafter's drafts: all 7
hook._FD["gen"] = None
counts = torch.zeros(S + 1, dtype=torch.int64)
ok = True
for step in range(4000):
    draft.fill_(7)
    hook._force_drafts(idx, seq, draft, cu)
    for b, slot in ((0, 2), (1, 5)):
        d = draft[slot].tolist()
        want = [1000 + int(F[b]) + j for j in range(1, S + 1)]           # the token at input position F + j
        n = 0
        while n < S and d[n] == want[n]:
            n += 1
        counts[n] += 1
        if DRAFTER:                              # the drafter's own draft (7) at the first rejection, later ones kept
            if n < S and (d[n] != 7 or d[n + 1:] != [7] * (S - n - 1)):
                ok = False
        elif n < S and d[n] != (want[n] + 1) % 163840:
            ok = False
    if draft[3].tolist() != [7] * S or draft[6].tolist() != [7] * S:
        ok = False
check("forced rows: reference drafts then " + ("the drafter's own" if DRAFTER else "(token + 1)") +
      "; other slots untouched", ok)
if DRAFTER:                                       # the drafter guessed the reference: the rejection takes token + 1
    hook._FD["gen"] = None
    ok2 = True
    for step in range(400):
        for b, slot in ((0, 2), (1, 5)):
            draft[slot] = torch.tensor([1000 + int(F[b]) + j for j in range(1, S + 1)], dtype=torch.int32, device=dev)
        hook._force_drafts(idx, seq, draft, cu)
        for b, slot in ((0, 2), (1, 5)):
            d = draft[slot].tolist()
            want = [1000 + int(F[b]) + j for j in range(1, S + 1)]
            n = 0
            while n < S and d[n] == want[n]:
                n += 1
            if n < S and (d[n] != want[n] + 1 or d[n + 1:] != want[n + 1:]):
                ok2 = False
    check("drafter equal to the reference at the rejection: token + 1 there, its later drafts kept", ok2)
P = torch.tensor([0.665, 0.432, 0.266, 0.180])
emp = [(counts[i:].sum() / counts.sum()).item() for i in range(1, S + 1)]
check("accepted counts follow P(i)", all(abs(e - p) < 0.03 for e, p in zip(emp, P.tolist())),
      f"(P(n >= i) measured {[round(e, 3) for e in emp]}, asked {P.tolist()})")
print("FORCE DRAFTS:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
