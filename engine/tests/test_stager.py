"""The stager (k31_engine/stager.py) on real records: stage 3 records of layer 1 into slots 0, 5, 15 of
kernel-layout tensors; the slot rows must equal the records expanded on the CPU side (build_store.expand) and the
scale bytes; untouched slots stay zero. Then timing: 1, 2, 4 records at 16 / 24 / 32 SMs.
usage: test_stager.py [HI2_DIR] (default: $K31_HI2 or /hi2)"""
import json
import mmap
import os
import sys
import time

import torch

sys.path.insert(0, "/engine"); sys.path.insert(0, "/engine/store")
from k31_engine.stager import SLOTS, Stager  # noqa: E402
from build_store import expand, flat, kernel_rows  # noqa: E402

d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("K31_HI2", "/hi2")
lay = json.load(open(os.path.join(d, "layout.json")))
R = lay["record_bytes"]
dev = torch.device("cuda")
experts = [7, 100, 555]
G = len(experts)
buf = mmap.mmap(-1, G * R)
bank = torch.frombuffer(buf, dtype=torch.uint8).view(G, R)
fd = os.open(os.path.join(d, "L01.hi2"), os.O_RDONLY)
for j, e in enumerate(experts):
    os.preadv(fd, [memoryview(bank[j].numpy())], e * R)
assert int(torch.cuda.cudart().cudaHostRegister(bank.data_ptr(), bank.numel(), 0x3)) == 0
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor  # noqa: E402
bank_dev = get_accelerator_view_from_cpu_tensor(bank)


class Mod:
    pass


m = Mod()
rows = kernel_rows(dev)(SLOTS + 2)
m.w13_weight, m.w2_weight, m.w13_weight_scale, m.w2_weight_scale = rows["w13"], rows["w2"], rows["w13s"], rows["w2s"]
st = Stager(m, 2, lay)                                   # slots are rows 2..17
stage = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
slots = [0, 5, 15]
for j, k in enumerate(slots):
    stage[k] = j
st(bank_dev.data_ptr(), stage)
torch.cuda.synchronize()
fl = {k: flat(v) for k, v in rows.items()}
c13 = lay["native_bytes"]["w13"] // 2
cb = lay["codes_bytes"]
s13 = lay["native_bytes"]["w13s"]
ok = True
for j, k in enumerate(slots):
    rec = bank[j].to(dev).unsqueeze(0)
    want13 = expand(rec[:, :c13])[0]
    want2 = expand(rec[:, c13:cb])[0]
    r = 2 + k
    ok &= torch.equal(fl["w13"][r], want13) and torch.equal(fl["w2"][r], want2)
    ok &= torch.equal(fl["w13s"][r], rec[0, cb:cb + s13]) and torch.equal(fl["w2s"][r], rec[0, cb + s13:cb + lay["scale_bytes"]])
untouched = [r for r in range(SLOTS + 2) if r - 2 not in slots]
ok &= all(int(fl[k][r].abs().sum() if fl[k].dtype != torch.uint8 else fl[k][r].sum()) == 0 for k in fl for r in untouched)
print(f"stager bytes equal to the expanded records in slots {slots}, other rows untouched: {'PASS' if ok else 'FAIL'}")
for sms in (16, 24, 32):
    st.sms = sms
    for n in (1, 2, 4):
        stage.fill_(-1)
        for k in range(n):
            stage[k] = k % G
        best = 1e9
        for rep in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); st(bank_dev.data_ptr(), stage); b.record(); b.synchronize()
            best = min(best, a.elapsed_time(b) * 1e3)
        print(f"  {sms} SMs, {n} record(s): {best:.1f} us, {n * R / (best * 1e-6) / 1e9:.0f} GB/s of record bytes")
sys.exit(0 if ok else 1)
