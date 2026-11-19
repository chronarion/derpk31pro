"""w8a16.linear_norm_res (K31_FUSE_UP) against vLLM's rms_norm, then w8a16.linear, then the residual add: the routed
latent [T, 3584] -> RMSNorm -> up-projection to 7168 -> + shared-expert output. Equal up to the variance summation order
(a handful of one-ulp differences), and graph replays equal to eager calls."""
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import w8a16  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(3)
K, N = 3584, 7168
wq, ws = w8a16.quantize((torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16))
gw = (1 + 0.1 * torch.randn(K, device=dev, generator=g)).to(torch.bfloat16)
eps = 1e-6
worst, ndiff, ntot = 0.0, 0, 0
for T in (1, 3, 5, 8, 16):
    x = (torch.randn(T, K, device=dev, generator=g) * 2).to(torch.bfloat16)
    res = (torch.randn(T, N, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    xn = torch.empty_like(x)
    ops.rms_norm(xn, x, gw, eps)
    ref = (w8a16.linear(xn, wq, ws) + res)
    out = res.clone()
    w8a16.linear_norm_res(x, wq, ws, gw, eps, out)
    d = (out.float() - ref.float()).abs()
    worst = max(worst, float(d.max() / ref.float().abs().max()))
    ndiff += int((d > 0).sum())
    ntot += d.numel()
ok = worst < 1e-2 and ndiff < ntot * 0.01
print(f"fused norm + up + residual vs rms_norm, linear, add: {ndiff} of {ntot} differ, worst rel {worst:.1e}")
T = 3
xs = torch.zeros(T, K, device=dev, dtype=torch.bfloat16)
rs = torch.zeros(T, N, device=dev, dtype=torch.bfloat16)
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    w8a16.linear_norm_res(xs, wq, ws, gw, eps, rs)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    w8a16.linear_norm_res(xs, wq, ws, gw, eps, rs)
same = True
for _ in range(4):
    x = (torch.randn(T, K, device=dev, generator=g) * 2).to(torch.bfloat16)
    r0 = (torch.randn(T, N, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    xs.copy_(x)
    rs.copy_(r0)
    gr.replay()
    e = r0.clone()
    w8a16.linear_norm_res(x, wq, ws, gw, eps, e)
    torch.cuda.synchronize()
    same = same and torch.equal(rs, e)
print(f"graph replays equal to eager calls: {same}")
ok = ok and same
print("FUSE UP:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
