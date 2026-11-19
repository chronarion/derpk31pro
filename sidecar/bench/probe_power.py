"""Keep the band busy for a few seconds (calls queued back to back) so that power and clocks can be read from outside
(nvidia-smi --query-gpu=power.draw,clocks.sm -i <gpu> -lms 200)."""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.rtxband import H, HI2_REC, Band, mxfp8_encode, pack_x
from bench.bench_band import make_calls

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True); ap.add_argument("--hi2", required=True)
ap.add_argument("--records", type=int, default=896); ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--seconds", type=float, default=6.0); ap.add_argument("--idle-us", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda"); gen = torch.Generator().manual_seed(1)
mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
band = Band(a.lib, a.records, dev)
for i in range(0, a.records, 16):
    band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
x = torch.randn(3, H, generator=gen).to(dev); xq, xs, _ = mxfp8_encode(x, 0); xf = pack_x(xq, xs)
calls = make_calls(band, a.records, a.experts, 3, min(a.records // a.experts, 128), gen)
print("BUSY", flush=True)
t0 = time.time(); n = 0
while time.time() - t0 < a.seconds:
    if a.idle_us:
        t, _ = band.bench(calls, xf, 5, 4, 2000, pause_us=a.idle_us)
    else:
        t, _ = band.bench(calls, xf, 1, 4, 2000)
    n += 1
print(f"IDLE  {t * 1e6:.1f} us per call", flush=True)
