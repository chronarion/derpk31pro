"""Promotion merge (stager.merge): hi2 + lo2 records of layer 1 experts -> native rows must equal, byte for byte, the
rows vLLM's own loader writes from the official checkpoint. usage: test_merge.py [HI2_DIR LO2_DIR NATIVE_DIR] (default: /hi2 /hi2 /native, the record store holding both)"""
import json
import os
import sys

import torch

sys.path.insert(0, "/engine"); sys.path.insert(0, "/engine/store")
from k31_engine import stager  # noqa: E402
from k31_engine.ckpt import Experts  # noqa: E402
from build_store import flat, kernel_rows  # noqa: E402
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts as M  # noqa: E402

hi_dir, lo_dir, native = sys.argv[1:4] if len(sys.argv) >= 4 else ("/hi2", "/hi2", "/native")
lay = json.load(open(os.path.join(hi_dir, "layout.json")))
R = lay["record_bytes"]
dev = torch.device("cuda")
experts = [7, 100, 555, 895]
mk = kernel_rows(dev)
want, got = mk(len(experts)), mk(len(experts))
ck = Experts(native)
t = ck.read_layer(1, experts)
for r, e in enumerate(experts):
    for shard in ("w1", "w2", "w3"):
        p, s = t[e][shard]
        pre = "w2" if shard == "w2" else "w13"
        M._load_weight(M, want[pre][r], p.to(dev), shard, False)
        M._load_weight(M, want[pre + "s"][r], s.to(dev), shard, True)


class Mod:
    pass


m = Mod()
m.w13_weight, m.w2_weight, m.w13_weight_scale, m.w2_weight_scale = got["w13"], got["w2"], got["w13s"], got["w2s"]
hi = torch.empty(R, dtype=torch.uint8, pin_memory=True)
lo = torch.empty(R, dtype=torch.uint8, pin_memory=True)
for r, e in enumerate(experts):
    for buf, d in ((hi, hi_dir), (lo, lo_dir)):
        fd = os.open(os.path.join(d, "L01." + ("hi2" if d == hi_dir else "lo2")), os.O_RDONLY)
        os.preadv(fd, [memoryview(buf.numpy())], e * R); os.close(fd)
    stager.merge(hi.data_ptr(), lo.data_ptr(), m, r, lay)
    torch.cuda.synchronize()
ok = all(torch.equal(flat(want[k]), flat(got[k])) for k in want)
print(f"merge of hi2 + lo2 = vLLM's own rows for experts {experts}: {'PASS' if ok else 'FAIL'}")
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
a.record(); stager.merge(hi.data_ptr(), lo.data_ptr(), m, 0, lay); b.record(); b.synchronize()
print(f"merge time with both records in pinned host memory: {a.elapsed_time(b) * 1e3:.0f} us")
sys.exit(0 if ok else 1)
