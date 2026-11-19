"""K31_W8_MARLIN / K31_DENSE_FMT ...:marlin4: the Marlin paths against the engine's own kernels on the same weights.

1. w8a16.MarlinW (FP8 codes and group-128 scales from w8a16.quantize) through w8a16.linear's dispatch against the split-K
   kernel: the same products, only the summation order and BF16 output rounding differ (rel < 4e-3), at 1-5 rows.
2. Two Marlin linears in one CUDA graph, one on a side stream (their own lock workspaces), replayed: equal to eager.
3. mxfp4dg.MarlinW4 against x @ dequant(DeepGEMM's MXFP4 codes)^T (rel < 4e-3): the nibble order and the UE8M0 scale
   conversion are right; the padded N (49376 -> 49408) is sliced off.
4. The gate: marlin_preferred follows the table (faster by 5%), K31_W8_MARLIN=all takes every shape, N % 64 never.
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("K31_W8_MARLIN", "1")
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
import torch  # noqa: E402

from k31_engine import mxfp4dg, w8a16  # noqa: E402

dev = torch.device("cuda")
fails = []
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=dev)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


g = torch.Generator(device="cpu").manual_seed(1)
# 1. Marlin FP8 against the split-K kernel
for N, K in [(7168, 12288), (2112, 7168), (18432, 1536), (3584, 7168)]:
    w = (torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16)
    q, s = w8a16.quantize(w)
    mw = w8a16.MarlinW(q.clone(), s.clone())
    worst = 0.0
    for M in (1, 3, 4, 5):
        x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
        ref = w8a16.linear(x, q, s)
        y = w8a16.linear(x, mw, None)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        y2 = w8a16.linear(x, mw, None, out)
        worst = max(worst, rel(y, ref))
        if y2.data_ptr() != out.data_ptr() or not torch.equal(y2, y):
            worst = 1.0
    check(f"marlin fp8 {N}x{K} vs split-K at 1-5 rows", worst < 4e-3, f"(worst rel {worst:.2e})")
# 2. a CUDA graph with a side-stream Marlin call
N, K = 7168, 12288
wa, wb = [(torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16) for _ in range(2)]
ma, mb = [w8a16.MarlinW(*w8a16.quantize(t)) for t in (wa, wb)]
x = torch.randn(4, K, generator=g).to(dev, torch.bfloat16)
ea, eb = ma(x).clone(), mb(x).clone()
side = torch.cuda.Stream()
ya, yb = torch.empty_like(ea), torch.empty_like(eb)
for _ in range(2):                                      # warm-up outside the graph
    ya.copy_(ma(x)); yb.copy_(mb(x))
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        yb.copy_(mb(x))
    ya.copy_(ma(x))
    torch.cuda.current_stream().wait_stream(side)
ok = True
for _ in range(20):
    ya.zero_(); yb.zero_()
    graph.replay()
    torch.cuda.synchronize()
    ok = ok and torch.equal(ya, ea) and torch.equal(yb, eb)
check("two Marlin linears in a graph, one on a side stream: equal to eager", ok)
# 3. Marlin MXFP4 against DeepGEMM's codes
from deep_gemm.utils.math import per_token_cast_to_fp4  # noqa: E402
N, K = 49376, 7168
w = (torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16)
qd, sf = per_token_cast_to_fp4(w.contiguous(), use_ue8m0=True, gran_k=32)
b = qd.view(torch.uint8)
lo, hi = (b & 0xF).to(torch.int64), (b >> 4).to(torch.int64)


def dq(first, second):
    c = torch.stack((first, second), dim=-1).view(N, K)
    v = E2M1[c & 7] * torch.where(c >= 8, -1.0, 1.0)
    return (v.view(N, K // 32, 32) * sf.float().view(N, K // 32, 1)).view(N, K)


d_lo, d_hi = dq(lo, hi), dq(hi, lo)
e_lo, e_hi = rel(d_lo, w), rel(d_hi, w)
order_ok = e_lo < 0.3 < e_hi
check("DeepGEMM's MXFP4 codes: low nibble = even k", order_ok, f"(dequant error {e_lo:.3f} low-first, {e_hi:.3f} high-first)")
m4 = mxfp4dg.MarlinW4(w)
for M in (1, 4, 5):
    x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
    ref = torch.cat([x.float() @ d_lo[i:i + 8192].T for i in range(0, N, 8192)], dim=1)
    y = m4(x)
    check(f"marlin mxfp4 {N}x{K} at {M} rows vs DeepGEMM's codes", tuple(y.shape) == (M, N) and rel(y, ref) < 4e-3,
          f"(rel {rel(y, ref):.2e}, padded N {m4.pn})")
del d_lo, d_hi, lo, hi
# 4. the gate
t = {"marlin8_us": {"7168x12288": {"3": 10.0}, "2112x7168": {"3": 9.4}}}
f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(t, f)
f.close()
w8a16._MTABLE.clear()
os.environ["K31_MARLIN_TABLE"] = f.name
w8a16._TABLE["t"] = {"w8a16_best_us": {"7168x12288": {"3": 29.1}, "2112x7168": {"3": 9.5}}}
ok = w8a16.marlin_preferred(7168, 12288) and not w8a16.marlin_preferred(2112, 7168) and \
    not w8a16.marlin_preferred(3584, 3584) and not w8a16.marlin_preferred(49376, 7168)
w8a16.MARLIN = "all"
ok = ok and w8a16.marlin_preferred(3584, 3584) and not w8a16.marlin_preferred(49376, 7168)
check("gate: table (5% margin), all, N % 64", ok)
print("MARLIN:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
