"""Cost of mode staged's per-call planning (hook._plan + slot assignment, torch ops) and the stager launch with no
work, each captured in a CUDA graph and replayed, at T = 1, 3, 4, 8."""
import os
import sys

import torch

sys.path.insert(0, "/engine")
os.environ["K31_MODE"] = "staged"
from k31_engine import hook  # noqa: E402

dev = torch.device("cuda")
st = hook.LayerState(5, list(range(115)), list(range(115, 415)))
st.row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
st.bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
st.arange = torch.arange(16, dtype=torch.long, device=dev)
E, S, SL = 896, st.S, hook.SLOTS


def step(ids, w):
    picks = ids.long().clamp_min(0)
    native, bank, cand, score, valid = hook._plan(st, ids, w)
    nat_ids = torch.where(valid & (native >= 0), native, -1)
    v, e = score.topk(SL)
    ok = v > 0
    slot_of = torch.full((E,), -1, dtype=torch.long, device=dev)
    slot_of.scatter_(0, e, torch.where(ok, st.arange[:SL], -1))
    stage = torch.where(ok, st.bank_map[e], -1).to(torch.int32)
    slot = slot_of[picks]
    staged = cand & (slot >= 0)
    return torch.where(staged, S + slot, nat_ids).to(ids.dtype), stage


for T in (1, 3, 4, 8):
    g = torch.Generator(device="cpu").manual_seed(T)
    ids = torch.stack([torch.randperm(E, generator=g)[:16] for _ in range(T)]).to(dev, torch.int32)
    w = torch.rand((T, 16), generator=g).to(dev)
    for _ in range(3):
        step(ids, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(ids, w)
    graph.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(50):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3)
    n_staged = int((out[1] >= 0).sum())
    print(f"T={T}: plan in a CUDA graph {best:.1f} us ({n_staged} experts to stage)")
