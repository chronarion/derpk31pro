"""k31_engine.route against the engine's chain (float32 gate logits of BF16 weights, then vLLM's grouped_topk):
same experts in the same order, weights close; a CUDA graph replay; time per call next to the chain (only meaningful
with the GB300 otherwise idle)."""
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
for trial in range(120):
    T = (1, 2, 3, 3, 5, 8, 12, 16)[trial % 8]
    x = (torch.randn(T, K, device=dev, generator=g) * 3).to(torch.bfloat16)
    wt, ids = route.route(x, w, bias)
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
print(f"rows {n}: expert sets differ {diff_set}, order differs {diff_order}, max weight diff {wmax:.1e} (float32 logits "
      f"summed in another order: near-ties may flip)", flush=True)
ok = diff_set <= n // 100 and wmax < 1e-5
# graph replay
T = 3
xs = torch.zeros(T, K, device=dev, dtype=torch.bfloat16)
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    route.route(xs, w, bias)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    gw, gi = route.route(xs, w, bias)
same = True
for _ in range(5):
    x = (torch.randn(T, K, device=dev, generator=g) * 3).to(torch.bfloat16)
    xs.copy_(x)
    gr.replay()
    torch.cuda.synchronize()
    a, b = route.route(x, w, bias)
    same = same and torch.equal(gi, b) and torch.equal(gw, a)
print(f"graph replays equal to eager calls: {same}", flush=True)
ok = ok and same
# time per call: route vs the chain (gate GEMV in torch + grouped_topk), CUDA graphs, rotating weights
ws = [(torch.randn(E, K, device=dev, generator=g) * 0.02).to(torch.bfloat16) for _ in range(24)]
x = (torch.randn(T, K, device=dev, generator=g) * 3).to(torch.bfloat16)
for name, fn in (("route (one launch)", lambda i: route.route(x, ws[i % 24], bias)),
                 ("gate mm + grouped_topk", lambda i: ops.grouped_topk(torch.mm(x, ws[i % 24].T).float(), 1, 1, 16,
                                                                       True, 1.0, bias, 1))):
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
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        gt.replay()
    e1.record()
    torch.cuda.synchronize()
    print(f"time {name}: {e0.elapsed_time(e1) * 1e3 / 240:.1f} us per call", flush=True)
print("ROUTE:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
