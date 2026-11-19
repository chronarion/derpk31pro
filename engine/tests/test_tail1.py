"""K31_TAIL1's pieces without the RTX (the band-rows path is in test_band_e2e.py):
1. w8a16.linear_norm_res with rstd (NORM 2) against the statistic-prologue kernel (NORM 1) and against the unfused
   trio (torch RMSNorm in vLLM's order, linear, add): equal up to one BF16 ulp on a few elements.
2. defer.cut_mean_pre (float32 sum, stored) against cut_mean on a zero BF16 output.
3. librtxband_fused_v4.so's finish on a call with no band rows (tag 0): out = bf16(out + msum) in one rounding and
   rstd = rsqrt(mean(out^2) + eps) of the result, against torch; and without msum / rstd, out untouched (v2 path).
"""
import ctypes
import os
import sys

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import band, defer, w8a16  # noqa: E402

dev = torch.device("cuda")
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def ulps(a, b):
    """Elements that differ, and the largest difference in BF16 ulps of the larger magnitude."""
    d = (a.float() - b.float()).abs()
    ulp = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-30) * 2.0 ** -7
    return int((d > 0).sum()), float((d / ulp).max())


g = torch.Generator(device="cpu").manual_seed(3)
eps = 1e-6
N, K = 7168, 3584
q, s = w8a16.quantize((torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16))
gw = (torch.rand(K, generator=g) + 0.5).to(dev, torch.bfloat16)
# 1. NORM 2
worst = (0, 0.0)
for M in (1, 3, 4, 5, 8):
    x = (torch.randn(M, K, generator=g) * 3).to(dev, torch.bfloat16)
    res0 = torch.randn(M, N, generator=g).to(dev, torch.bfloat16)
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1) + eps).contiguous()
    r1 = w8a16.linear_norm_res(x, q, s, gw, eps, res0.clone())
    r2 = w8a16.linear_norm_res(x, q, s, gw, eps, res0.clone(), rstd=rstd)
    xn = (xf * rstd[:, None] * gw.float()).to(torch.bfloat16)
    r3 = (res0.float() + w8a16.linear(xn, q, s).float()).to(torch.bfloat16)
    a, b = ulps(r1, r2), ulps(r2, r3)
    worst = max(worst, a, b)
check("linear_norm_res with rstd (NORM 2) = the prologue kernel and the unfused trio", worst[1] <= 1.0,
      f"(at most {worst[0]} elements differ, by <= {worst[1]:.2f} ulp)")
# 2. cut_mean_pre
E, D = 896, 3584
mean = (torch.randn(E, D, generator=g) * 0.05).to(dev, torch.bfloat16)
bad = 0
for T in (1, 3, 5, 8):
    ids = torch.stack([torch.randperm(E, generator=g)[:16] for _ in range(T)]).int().to(dev)
    rows = torch.where(torch.rand(T, 16, generator=g) < 0.5, 7, -1).int().to(dev)
    w = torch.rand(T, 16, generator=g).to(dev)
    served = (torch.rand(T * 16, generator=g) < 0.3).int().to(dev)
    out = torch.zeros(T, D, dtype=torch.bfloat16, device=dev)
    defer.cut_mean(out, rows, ids, w, mean, served=served)
    ms = torch.full((T, D), float("nan"), dtype=torch.float32, device=dev)
    defer.cut_mean_pre(ms, rows, ids, w, mean, served=served)
    if not torch.equal(ms.to(torch.bfloat16), out):
        bad += 1
check("cut_mean_pre (float32, stored) rounds to cut_mean's output", bad == 0, f"({bad} of 4 differ)")
# 3. the v4 finish without band rows
lib = os.environ.get("V4_LIB", "/engine/band/librtxband_fused_v4.so")


class _FakeGb:
    """What band.Fused.finish reads: a segment pointer (unused with tag 0), a state block with tag 0, a timeout."""
    def __init__(self):
        self.host = torch.zeros(1 << 20, dtype=torch.uint8).pin_memory()
        self.seg = ctypes.c_void_p(self.host.data_ptr())
        self.st = torch.zeros(4096, dtype=torch.uint8, device=dev)
        self.timeout_ns = 50_000


fz = band.Fused(lib, _FakeGb())
bad, worst_r = 0, 0.0
for T in (1, 3, 5, 8):
    out0 = (torch.randn(T, D, generator=g) * 0.7).to(dev, torch.bfloat16)
    msum = (torch.randn(T, D, generator=g) * 0.3).to(dev, torch.float32)
    want = (out0.float() + msum).to(torch.bfloat16)
    rs_want = torch.rsqrt(want.float().pow(2).mean(-1) + eps)
    out = out0.clone()
    rstd = torch.full((16,), float("nan"), dtype=torch.float32, device=dev)
    fz.finish(out, msum=msum, rstd=rstd, eps=eps)
    torch.cuda.synchronize()
    if not torch.equal(out, want):
        bad += 1
    worst_r = max(worst_r, float(((rstd[:T] - rs_want).abs() / rs_want).max()))
    out2 = out0.clone()
    fz.finish(out2)                                    # no msum, no rstd, tag 0: nothing to add
    torch.cuda.synchronize()
    if not torch.equal(out2, out0):
        bad += 1
check("v4 finish, tag 0: out = bf16(out + msum), rstd of the result; untouched without msum", bad == 0 and
      worst_r < 2e-6, f"({bad} mismatches, rstd rel err {worst_r:.1e})")
print("TAIL1:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
