"""Where the time of a call goes: chain times of each linear stopped after the fill, after the GEMM, and complete;
and of an empty call (launch cost of the fixed grids)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.rtxband import H, HI2_REC, MAXT, REC, Band, mxfp8_encode, pack_x
from bench.bench_band import make_calls

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True); ap.add_argument("--hi2", required=True)
ap.add_argument("--records", type=int, default=256); ap.add_argument("--experts", default="0,1,2,4,8,16")
ap.add_argument("--brief", action="store_true")
a = ap.parse_args()
dev = torch.device("cuda"); gen = torch.Generator().manual_seed(1)
mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
n = a.records
band = Band(a.lib, n, dev)
for i in range(0, n, 16):
    band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
x = torch.randn(3, H, generator=gen).to(dev); xq, xs, _ = mxfp8_encode(x, 0); xf = pack_x(xq, xs)
if a.brief:
    print(f"{'E':>2s} | {'two':>8s} | {'fused':>8s}  (us per call, chained)")
else:
    print(f"{'E':>2s} | {'l1 fill':>8s} {'l1 gemm':>8s} {'l1 all':>8s} | {'l2 fill':>8s} {'l2 gemm':>8s} {'l2 all':>8s} | {'two':>8s} | {'fused':>8s}  (us per call, chained)")
for E in [int(v) for v in a.experts.split(",")]:
    line = f"{E:2d} |"
    for which in ((3, 4) if a.brief else (1, 2, 3, 4)):
        for flags in ((16 | 256, 16 | 512, 16) if which < 3 else (16,)):
            calls = make_calls(band, n, max(E, 0), 3, max(1, min(n // max(E, 1), 64)), gen, flags=flags) if E else \
                torch.stack([band.make_call([], torch.zeros(0, MAXT), 3, flags=flags)])
            t, _ = band.bench(calls, xf, 1, which, 200)
            line += f" {t * 1e6:8.1f}"
        line += " |"
    print(line, flush=True)
