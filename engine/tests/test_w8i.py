"""w8i (K31_DENSE_FMT ...:w8i, K31_W8I_DRAFT): the int8 GEMV through w8i.I8W and w8a16.linear's dispatch.
1. y = x @ dequant_ref(w)^T at 1, 3, 5, 8, 13 rows (groups of 8) and 64 / 512 rows (one GEMM on the codes, K31_I8_BIG_M),
   bf16 and fp32 out (rel < 4e-3 bf16); the GEMM against the 8-row GEMV launches on the same input.
2. The clip search: squared error of the chosen scales <= absmax scales on a heavy-tailed weight.
3. Two int8 linears in a CUDA graph, replayed: equal to eager. w8a16.linear(x, I8W, None) = I8W(x).
"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import w8a16, w8i  # noqa: E402

dev = torch.device("cuda")
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


g = torch.Generator(device="cpu").manual_seed(5)
for N, K in [(7168, 12288), (7168, 3584), (2112, 7168), (12288, 128)]:
    w = (torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16)
    iw = w8i.I8W(w)
    ref_w = w8i.dequant_ref(w)
    worst_b, worst_f = 0.0, 0.0
    for M in (1, 3, 5, 8, 13, 64, 512):
        x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
        ref = x.float() @ ref_w.T
        worst_b = max(worst_b, rel(iw(x), ref))
        yf = torch.empty(M, N, dtype=torch.float32, device=dev)
        worst_f = max(worst_f, rel(iw(x, yf), ref))
    check(f"int8 {N}x{K} vs dequant_ref at 1-512 rows", worst_b < 4e-3 and worst_f < 1e-3,
          f"(bf16 out {worst_b:.2e}, fp32 out {worst_f:.2e})")
    x = torch.randn(200, K, generator=g).to(dev, torch.bfloat16)
    big = iw(x)
    keep, w8i.BIG_M = w8i.BIG_M, 1 << 30
    gemv = iw(x)
    w8i.BIG_M = keep
    check(f"int8 {N}x{K}: one GEMM = 8-row GEMV launches at 200 rows", rel(big, gemv) < 4e-3, f"(rel {rel(big, gemv):.2e})")
# 2. clip search on a heavy-tailed weight
w = (torch.randn(2048, 4096, generator=g) * 0.02)
w[torch.rand(2048, 4096, generator=g) < 0.001] *= 12           # outliers
w = w.to(dev, torch.bfloat16)
W = w.float()
amax = W.abs().amax(1, keepdim=True) / 127
e_abs = float((torch.round(W / amax).clamp(-128, 127) * amax - W).pow(2).sum())
e_clip = float((w8i.dequant_ref(w) - W).pow(2).sum())
check("clip search: squared error <= absmax's", e_clip <= e_abs, f"({e_clip / e_abs:.3f} of absmax)")
# 3. graph + dispatch
N, K = 7168, 6144
wa, wb = [(torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16) for _ in range(2)]
ia, ib = w8i.I8W(wa), w8i.I8W(wb)
x = torch.randn(5, K, generator=g).to(dev, torch.bfloat16)
ea, eb = ia(x).clone(), ib(x).clone()
d = w8a16.linear(x, ia, None)
ya, yb = torch.empty_like(ea), torch.empty_like(eb)
for _ in range(2):
    ia(x, ya); ib(x, yb)
torch.cuda.synchronize()
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    ia(x, ya)
    ib(x, yb)
ok = True
for _ in range(10):
    ya.zero_(); yb.zero_()
    gr.replay()
    torch.cuda.synchronize()
    ok = ok and torch.equal(ya, ea) and torch.equal(yb, eb)
check("graph replay = eager; w8a16.linear dispatches to the int8 weight", ok and torch.equal(d, ea))
print("W8I:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
