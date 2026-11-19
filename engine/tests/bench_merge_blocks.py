"""Promotion merge and write-back extract (stager.merge / extract_hi2) from device buffers, as residency.py runs them,
at several grid sizes: the outputs must be byte-identical whatever the grid; time per call in a CUDA graph (only
meaningful with the GB300 otherwise idle; relative numbers are indicative next to a running engine)."""
import json
import os
import sys

import torch

sys.path.insert(0, "/engine"); sys.path.insert(0, "/engine/store")
from k31_engine import stager  # noqa: E402
from build_store import flat, kernel_rows  # noqa: E402

d = sys.argv[1]
lay = json.load(open(os.path.join(d, "layout.json")))
R = lay["record_bytes"]
dev = torch.device("cuda")
mk = kernel_rows(dev)
rows = mk(4)


class Mod:
    pass


m = Mod()
m.w13_weight, m.w2_weight, m.w13_weight_scale, m.w2_weight_scale = rows["w13"], rows["w2"], rows["w13s"], rows["w2s"]
hi = torch.empty(R, dtype=torch.uint8)
lo = torch.empty(R, dtype=torch.uint8)
for buf, ext in ((hi, "hi2"), (lo, "lo2")):
    fd = os.open(os.path.join(d, "L01." + ext), os.O_RDONLY)
    os.preadv(fd, [memoryview(buf.numpy())], 100 * R)
    os.close(fd)
d_hi, d_lo = hi.to(dev), lo.to(dev)
out = torch.empty(R, dtype=torch.uint8, device=dev)
ref_row = ref_ext = None
for blocks in (152, 304, 608, 1216):
    stager.merge(d_hi.data_ptr(), d_lo.data_ptr(), m, 1, lay, blocks=blocks)
    stager.extract_hi2(m, 1, lay, out, blocks=blocks)
    torch.cuda.synchronize()
    row = torch.cat([flat(rows[k])[1].flatten() for k in rows])     # row 1 only (the timing below writes row 2)
    if ref_row is None:
        ref_row, ref_ext = row.clone(), out.clone()
    same = torch.equal(row, ref_row) and torch.equal(out, ref_ext)
    for name, fn in (("merge", lambda: stager.merge(d_hi.data_ptr(), d_lo.data_ptr(), m, 2, lay, blocks=blocks)),
                     ("extract", lambda: stager.extract_hi2(m, 2, lay, out, blocks=blocks))):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                fn()
        g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            g.replay()
        b.record()
        b.synchronize()
        print(f"{name:7s} blocks {blocks:5d}: {a.elapsed_time(b) * 1e3 / 100:6.1f} us   outputs identical to 152 blocks: {same}",
              flush=True)
print("extract hi2 equals the hi2 record:", torch.equal(ref_ext.cpu(), hi))
