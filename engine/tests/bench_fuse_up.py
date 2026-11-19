"""K31_FUSE_UP's fused RMSNorm + up-projection + residual add (w8a16.linear_norm_res) against the same three ops
unfused (vLLM's rms_norm, w8a16.linear, add) and the GEMM alone, for the routed up-projection 7168 x 3584 at 3-5 rows.
CUDA graphs of R calls, weights rotated past L2. The 11-18 fw-d4 profile shows the fused kernel at 28.6 us on the
chain against 11.6 us for the GEMM alone in the dense bench."""
import os
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import w8a16  # noqa: E402

dev = torch.device("cuda")
N, K = 7168, 3584
R = 24
C = 24


def timed(fn, reps=R, replays=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(replays):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (replays * reps)


def rms(x, gw, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gw.float()).to(torch.bfloat16)


ws = [w8a16.quantize(torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02) for _ in range(C)]
gw = torch.rand(K, device=dev, dtype=torch.bfloat16) + 0.5
eps = 1e-6
try:
    from vllm import _custom_ops as ops
except Exception:                                     # noqa: BLE001
    ops = None
for M in (3, 4, 5):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    res = [torch.randn(M, N, device=dev, dtype=torch.bfloat16) for _ in range(C)]
    xn = torch.empty_like(x)
    # correctness first
    r0 = res[0].clone()
    want = (r0.float() + w8a16.linear(rms(x, gw, eps), *ws[0]).float()).to(torch.bfloat16)
    got = w8a16.linear_norm_res(x, *ws[0], gw, eps, r0)
    err = ((got.float() - want.float()).norm() / want.float().norm()).item()
    t_gemm = timed(lambda i: w8a16.linear(x, *ws[i % C]))
    t_fused = timed(lambda i: w8a16.linear_norm_res(x, *ws[i % C], gw, eps, res[i % C]))
    if ops is not None:
        def unfused(i):
            ops.rms_norm(xn, x, gw, eps)
            res[i % C].add_(w8a16.linear(xn, *ws[i % C]))
    else:
        def unfused(i):
            res[i % C].add_(w8a16.linear(rms(x, gw, eps), *ws[i % C]))
    t_unf = timed(unfused)
    cfg = w8a16.config(M, N, K)
    print(f"M={M}: GEMM alone {t_gemm:6.1f} us | fused norm+GEMM+add {t_fused:6.1f} us | unfused {t_unf:6.1f} us "
          f"| config {cfg} | fused vs unfused rel {err:.1e}", flush=True)
