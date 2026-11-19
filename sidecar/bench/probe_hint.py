"""L2 prefetch hint: time of a call (launch + synchronize) by the pause between the hint and the call, and by the
number of hinted experts that the call really uses."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.rtxband import H, HI2_REC, MAXT, Band, mxfp8_encode, pack_x
from bench.bench_band import make_calls

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True); ap.add_argument("--hi2", required=True)
ap.add_argument("--records", type=int, default=896); ap.add_argument("--experts", default="1,2,4,6,8,10,12,14,16")
a = ap.parse_args()
dev = torch.device("cuda"); gen = torch.Generator().manual_seed(1)
mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
n = a.records
band = Band(a.lib, n, dev)
for i in range(0, n, 16):
    band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
x = torch.randn(3, H, generator=gen).to(dev); xq, xs, _ = mxfp8_encode(x, 0); xf = pack_x(xq, xs)
pauses = (0, 50, 200, 1000)
hi2 = torch.from_numpy(np.ascontiguousarray(mm[:32])).to(dev)
t, best = band.bench_repack(hi2, 0)
print(f"device repack of a hi2 record into a band slot: {t * 1e6:.1f} us per record (best {best * 1e6:.1f}), queued back to back")
for i in range(0, 32, 16):
    band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
del hi2
print(f"{'E':>2s} | {'no hint':>8s} | {'hint us':>8s} | " + " ".join(f"{p:>7d}" for p in pauses) + "   (us; no hint / columns: a call, launch + sync, by the pause after the hint; hint: the hint itself, launch + sync)")
for E in [int(v) for v in a.experts.split(",")]:
    calls = make_calls(band, n, E, 3, min(n // E, 128), gen)
    t0, _ = band.bench(calls, xf, 0, 4, 300)
    th, _ = band.bench_prefetch(calls, 300)
    line = f"{E:2d} | {t0 * 1e6:8.1f} | {th * 1e6:8.1f} | "
    for p in pauses:
        t, _ = band.bench(calls, xf, 4, 4, 300, pause_us=p)
        line += f"{t * 1e6:7.1f} "
    print(line, flush=True)

print("idle gaps: a call (launch + sync) after the band idled for the given time, no hint")
pauses = (0, 100, 300, 1000, 5000, 20000)
print(f"{'E':>2s} | " + " ".join(f"{p:>7d}" for p in pauses) + "   (us per call; columns = idle us before the call)")
for E in (4, 8):
    calls = make_calls(band, n, E, 3, min(n // E, 128), gen)
    line = f"{E:2d} | "
    for p in pauses:
        t, _ = band.bench(calls, xf, 5, 4, 300 if p < 5000 else 100, pause_us=p)
        line += f"{t * 1e6:7.1f} "
    print(line, flush=True)
