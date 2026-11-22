"""K31_MEM_SNAPSHOT (k31_engine/memgraphs.py) on a real device: the capture and replay wrappers and the report.

  python3 tests/test_memgraphs.py        (in the vLLM image, one GPU)
Three graphs are captured: A into a pool of its own, B into A's pool, C into a pool of its own. Each allocates
inside its capture, so its pool reserves memory that no tensor holds afterwards. Checks: the captures are listed with
their caller and pools, B shares A's pool, torch's snapshot carries the fields the report reads and shows the graph
pools apart from the default pool, the replays are counted per graph, and the graphs still compute what they did.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31_engine import memgraphs  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


lines = []
memgraphs._LOG = lambda m: lines.append(m)
memgraphs.install()
memgraphs.install()                                      # twice: one wrapper
dev = torch.device("cuda:0")
x = torch.randn(1 << 20, device=dev)
outs = {}


def capture(name, n_mib, pool=None):
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    kw = {} if pool is None else {"pool": pool}
    with torch.cuda.graph(g, **kw):
        t = torch.empty(n_mib << 20, dtype=torch.uint8, device=dev)      # lives in the capture's pool
        t.fill_(1)
        outs[name] = x * 2 + t[:1 << 20].float()
        del t
    return g


ga = capture("a", 96)
gb = capture("b", 32, pool=ga.pool())
gc = capture("c", 64)
x.fill_(1.0)
ga.replay()
ga.replay()
gb.replay()
torch.cuda.synchronize()
check("replayed graphs compute what they captured", float(outs["a"][0]) == 3.0 and float(outs["b"][0]) == 3.0,
      f"({float(outs['a'][0])}, {float(outs['b'][0])})")
check("B was captured into A's pool, C into another", tuple(gb.pool()) == tuple(ga.pool()) != tuple(gc.pool()),
      f"({tuple(ga.pool())}, {tuple(gb.pool())}, {tuple(gc.pool())})")
memgraphs.report("test")
memgraphs.replays()
text = "\n".join(lines)
print(text, flush=True)
cap = [ln for ln in lines if ln.startswith("graph captures test:")]
check("the captures are listed by caller and pool (two lines for this file's two pools, one says 'a pool passed in')",
      len(cap) == 3 and all("test_memgraphs.py:capture" in ln for ln in cap)
      and sum("(a pool passed in)" in ln for ln in cap) == 1, f"({len(cap)} lines)")
grew = [r[4] for r in memgraphs._CAP]
check("each capture's growth of torch's reserved memory is recorded (A and C at least their own allocation)",
      len(grew) == 3 and grew[0] >= 96 << 20 and grew[2] >= 64 << 20, f"({[g >> 20 for g in grew]} MiB)")
snap = torch.cuda.memory_snapshot()
need = ("segment_pool_id", "stream", "segment_type", "total_size", "allocated_size")
check("torch's memory snapshot carries the fields the report reads",
      bool(snap) and all(k in snap[0] for k in need), f"({sorted(snap[0])[:12] if snap else 'empty'})")
pools = {tuple(s["segment_pool_id"]) for s in snap} if snap and "segment_pool_id" in snap[0] else set()
check("the snapshot shows the graphs' pools apart from the default pool",
      (0, 0) in pools and tuple(ga.pool()) in pools and tuple(gc.pool()) in pools, f"({sorted(pools)})")
tot = [ln for ln in lines if "graph pools reserve" in ln]
mib = float(tot[0].split("graph pools reserve ")[1].split(" MiB")[0]) if tot else -1.0
check("the graph pools' reserved memory is summed (at least the 160 MiB the two pools were grown by)",
      len(tot) == 1 and mib >= 160, f"({mib:.0f} MiB)")
rep = [ln for ln in lines if ln.startswith("graph replays so far")]
check("replays are counted per graph: 3 replays, one graph never replayed",
      len(rep) == 1 and "3 of 3 captured graphs, 1 never replayed" in rep[0], f"({rep[0][:120] if rep else ''})")
print(f"MEMGRAPHS: {'PASS' if not fails else f'FAIL ({len(fails)} checks: ' + '; '.join(fails[:3]) + ')'}")
sys.exit(0 if not fails else 1)
