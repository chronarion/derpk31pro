"""Kernel times of stager.py against stager2.py, without the host's share: every call is captured in a CUDA graph of
CALLS back-to-back launches and the replay is timed (best and median of REPS replays, us per call).
  stage    1 / 4 / 16 records per launch at 24 CTAs (records in device memory here; on the GB300 they are host memory)
  merge, extract at 152 / 1216 blocks
Arms: hi2-k1 records into int32 scale rows by stager.py (today) and by stager2.py, hi2-k2 records, and each with 6-bit
scale rows. usage: bench_stager2.py   (env: K31_CUDA_ARCH, CALLS=10, REPS=40, SMS=24)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_testlib as tl  # noqa: E402
from k2_testlib import LAYOUT_K1, LAYOUT_K2, recfmt  # noqa: E402
from k31_engine import sf6, stager, stager2  # noqa: E402

dev = torch.device("cuda")
CALLS, REPS, SMS = int(os.environ.get("CALLS", "10")), int(os.environ.get("REPS", "40")), int(os.environ.get("SMS", "24"))
SLOTS = stager2.SLOTS


class Mod:
    def __init__(self, rows, six):
        self.w13_weight = torch.zeros((rows, 6144, 1792), dtype=torch.int8, device=dev)
        self.w2_weight = torch.zeros((rows, 3584, 1536), dtype=torch.int8, device=dev)
        self.w13_weight_scale = torch.zeros((rows, 28, 6144), dtype=torch.int32, device=dev).transpose(1, 2)
        self.w2_weight_scale = torch.zeros((rows, 24, 3584), dtype=torch.int32, device=dev).transpose(1, 2)
        if six:
            self._k31_sf6 = (torch.zeros((rows, sf6.row_bytes(6144, 28)), dtype=torch.uint8, device=dev),
                             torch.zeros((rows, sf6.row_bytes(3584, 24)), dtype=torch.uint8, device=dev))


def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(CALLS):
            fn()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gr.replay(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / CALLS)
    out.sort()
    return out[0], out[len(out) // 2]


k1 = tl.synth_records(16, seed=5).to(dev)
k2 = recfmt.k1_to_k2(k1, LAYOUT_K2)[0].contiguous()
lo2 = tl.synth_records(1, seed=6)[0].to(dev)                     # any bytes do for the timing
arms = [("k1 -> int32 rows, stager.py (today)", LAYOUT_K1, False, k1, "old"),
        ("k1 -> int32 rows, stager2.py", LAYOUT_K1, False, k1, "new"),
        ("k2 -> int32 rows", LAYOUT_K2, False, k2, "new"),
        ("k1 -> 6-bit rows", LAYOUT_K1, True, k1, "new"),
        ("k2 -> 6-bit rows", LAYOUT_K2, True, k2, "new")]
print(f"{torch.cuda.get_device_name()}: {CALLS} calls per replay, best / median of {REPS} replays, us per call")
rounds = int(os.environ.get("ROUNDS", "3"))
res = {}
for rnd in range(rounds):                                          # interleaved: the GPU may be shared
    for name, lay, six, bank, which in arms:
        mod = Mod(SLOTS, six)
        st = stager.Stager(mod, 0, lay, sms=SMS) if which == "old" else stager2.Stager2(mod, 0, lay, sms=SMS)
        for nrec in (1, 4, 16):
            idx = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
            idx[:nrec] = torch.arange(nrec, device=dev)
            b, m = timeit(lambda: st(bank.data_ptr(), idx))
            res.setdefault((name, f"stage {nrec}"), []).append((b, m))
        hi = bank[0].contiguous()
        out = torch.empty(lay["record_bytes"], dtype=torch.uint8, device=dev)
        for blocks in (152, 1216):
            if which == "old":
                f_m = lambda: stager.merge(hi.data_ptr(), lo2.data_ptr(), mod, 0, lay, blocks=blocks)
                f_x = lambda: stager.extract_hi2(mod, 0, lay, out, blocks=blocks)
            else:
                f_m = lambda: stager2.merge(hi.data_ptr(), lo2.data_ptr(), mod, 0, lay, blocks=blocks)
                f_x = lambda: stager2.extract_hi2(mod, 0, lay, out, blocks=blocks)
            f_m(); torch.cuda.synchronize()                        # the row holds a merged record before it is extracted
            res.setdefault((name, f"merge {blocks}"), []).append(timeit(f_m))
            res.setdefault((name, f"extract {blocks}"), []).append(timeit(f_x))
        del mod, st
cols = ["stage 1", "stage 4", "stage 16", "merge 152", "merge 1216", "extract 152", "extract 1216"]
print(f"{'':38s}" + "".join(f"{c:>16s}" for c in cols))
for name, *_ in arms:
    print(f"{name:38s}" + "".join(f"{min(x[0] for x in res[(name, c)]):8.1f}/{min(x[1] for x in res[(name, c)]):7.1f}" for c in cols))
