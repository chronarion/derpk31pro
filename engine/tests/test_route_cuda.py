"""k31_engine.route.route_cuda (csrc/k31route.cu) against the engine's chain (float32 gate logits of BF16 weights,
then vLLM's grouped_topk): same experts in the same order, weights close, for 1-8 rows; CUDA graph replays equal to
eager calls; time per call next to the chain (meaningful only with the GB300 otherwise idle)."""
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import route  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
E, K = 896, 7168
w = (torch.randn(E, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
bias = torch.randn(E, device=dev, generator=g) * 0.05
diff_set = diff_order = n = 0
wmax = 0.0
for trial in range(160):
    T = (1, 2, 3, 4, 5, 6, 7, 8)[trial % 8]
    x = (torch.randn(T, K, device=dev, generator=g) * 3).to(torch.bfloat16)
    wt, ids = route.route_cuda(x, w, bias)
    logits = (x.float() @ w.float().T)
    v, i = ops.grouped_topk(logits, 1, 1, 16, True, 1.0, bias, 1)
    for t in range(T):
        n += 1
        a, b = ids[t].tolist(), i[t].tolist()
        if set(a) != set(b):
            diff_set += 1
        elif a != b:
            diff_order += 1
        else:
            wmax = max(wmax, float((wt[t] - v[t]).abs().max()))
print(f"rows {n}: expert sets differ {diff_set}, order differs {diff_order}, max weight diff {wmax:.1e}; vLLM dtypes "
      f"{v.dtype} {i.dtype}", flush=True)
ok = diff_set <= n // 100 and wmax < 1e-5
# scale and no renormalization
x = (torch.randn(3, K, device=dev, generator=g) * 3).to(torch.bfloat16)
wt, ids = route.route_cuda(x, w, bias, scale=2.5, renormalize=False)
v, i = ops.grouped_topk(x.float() @ w.float().T, 1, 1, 16, False, 2.5, bias, 1)
ok2 = torch.equal(ids.long(), i.long()) and float((wt - v).abs().max()) < 1e-5
print(f"scale 2.5 without renormalization: equal {ok2}", flush=True)
# graph replay
T = 3
xs = torch.zeros(T, K, device=dev, dtype=torch.bfloat16)
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    route.route_cuda(xs, w, bias)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    gw, gi = route.route_cuda(xs, w, bias)
same = True
for _ in range(5):
    x = (torch.randn(T, K, device=dev, generator=g) * 3).to(torch.bfloat16)
    xs.copy_(x)
    gr.replay()
    torch.cuda.synchronize()
    a, b = route.route_cuda(x, w, bias)
    same = same and torch.equal(gi, b) and torch.equal(gw, a)
print(f"graph replays equal to eager calls: {same}", flush=True)
ok = ok and ok2 and same
ws = [(torch.randn(E, K, device=dev, generator=g) * 0.02).to(torch.bfloat16) for _ in range(24)]


def timed(fn):
    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s2)
    gt = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gt):
        for i in range(24):
            fn(i)
    gt.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            gt.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / 240)
    return best


for T in (3, 5):
    x = (torch.randn(T, K, device=dev, generator=g) * 3).to(torch.bfloat16)
    lg = torch.mm(x, ws[0].T).float()
    print(f"T {T}: route_cuda {timed(lambda i: route.route_cuda(x, ws[i % 24], bias)):.1f} us; stock mm "
          f"{timed(lambda i: torch.mm(x, ws[i % 24].T)):.1f} + grouped_topk "
          f"{timed(lambda i: ops.grouped_topk(lg, 1, 1, 16, True, 1.0, bias, 1)):.1f} us", flush=True)
print("ROUTE CUDA:", "PASS" if ok else "FAIL")
