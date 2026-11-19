"""defer.cut_mean against torch: out[t] += sum over picks with an id, no row and (with pids) not deferred of
w x mean[e]; other rows and picks untouched; ids beyond E ignored."""
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import defer  # noqa: E402

dev = torch.device("cuda")
E, D = 896, 3584
bad = 0
for trial in range(60):
    g = torch.Generator().manual_seed(trial)
    T = [1, 2, 3, 6, 8, 16, 64][trial % 7]
    mean = (torch.randn(E, D, generator=g) * 0.05).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:16] for _ in range(T)]).int()
    ids[torch.rand(T, 16, generator=g) < 0.05] = -1
    rows = torch.where(torch.rand(T, 16, generator=g) < 0.5, torch.randint(0, 200, (T, 16), generator=g), -1).int()
    w = torch.rand(T, 16, generator=g)
    pids = torch.where(torch.rand(T, 16, generator=g) < 0.2, 5, -1).int() if trial % 2 else None
    band = torch.where(torch.rand(E, generator=g) < 0.15, torch.randint(0, 118, (E,), generator=g), -1).int() \
        if trial % 3 == 0 else None                  # the RTX band's expert -> slot map (band.py): served there
    out0 = (torch.randn(T, D, generator=g) * 0.1).to(torch.bfloat16)
    ok = (ids >= 0) & (rows < 0) & ((pids < 0) if pids is not None else True)
    served = None
    if band is not None and trial % 6 == 3:      # the send's per-pick flags instead of the map (K31_BAND_SERVED)
        served = ((band[ids.clamp_min(0).long()] >= 0) & (ids >= 0) & (torch.rand(T, 16, generator=g) < 0.7)).int()
        ok = ok & (served == 0)
    elif band is not None:
        ok = ok & (band[ids.clamp_min(0).long()] < 0)
    add = torch.zeros(T, D, dtype=torch.float32)
    for t in range(T):
        for k in range(16):
            if ok[t, k]:
                add[t] += float(w[t, k]) * mean[int(ids[t, k])].float()
    want = (out0.float() + add).to(torch.bfloat16)
    out = out0.to(dev).clone()
    defer.cut_mean(out, rows.to(dev), ids.to(dev), w.to(dev), mean.to(dev), pids.to(dev) if pids is not None else None,
                   band=band.to(dev) if band is not None else None,
                   served=served.flatten().contiguous().to(dev) if served is not None else None)
    torch.cuda.synchronize()
    err = float((out.cpu().float() - want.float()).abs().max())
    if err > 2e-2 * float(want.float().abs().max()):
        bad += 1
        print(f"trial {trial}: max err {err:.4f}")
print("cut_mean vs torch:", "PASS" if bad == 0 else f"FAIL ({bad})")
