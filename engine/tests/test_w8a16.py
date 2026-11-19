"""k31_engine.w8a16: accuracy against a float64 reference of the same dequantized weights, bit-equal reruns, a CUDA
graph replay, and the time per call next to DeepGEMM's FP8 GEMM (only meaningful with the GB300 otherwise idle)."""
import os
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import w8a16  # noqa: E402

dev = torch.device("cuda")
SHAPES = [(12288, 7168), (7168, 12288), (7168, 3584), (7168, 6144), (3584, 7168), (2112, 7168), (18432, 1536),
          (49376, 7168)]
fails = []
g = torch.Generator(device=dev).manual_seed(0)
for (N, K) in SHAPES:
    w = torch.randn(N, K, device=dev, generator=g) * 0.02
    q, s = w8a16.quantize(w)
    wd = (q.float().view(N, K // 128, 128) * s.float()[..., None]).view(N, K)
    for M in (1, 3, 6, 12, 16, 17, 64, 200):
        x = (torch.randn(M, K, device=dev, generator=g) * 2).to(torch.bfloat16)
        y = w8a16.linear(x, q, s)
        ref = (x.double() @ wd.double().T)
        err = float((y.double() - ref).abs().max() / ref.abs().max())
        rerun = w8a16.linear(x, q, s)
        same = bool(torch.equal(y, rerun))
        ok = err < 8e-3 and same
        if not ok:
            fails.append(f"{N}x{K} M{M}")
        if M in (3, 16, 200) or not ok:
            print(f"{'ok  ' if ok else 'FAIL'} {N:6d} x {K:6d} M {M:2d}: max rel err {err:.1e} (bf16 output), rerun "
                  f"bit-equal {same}, config {w8a16.config(M, N, K)}", flush=True)
# graph replay
N, K, M = 7168, 3584, 3
w = torch.randn(N, K, device=dev, generator=g) * 0.02
q, s = w8a16.quantize(w)
xs = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
ys = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    w8a16.linear(xs, q, s, ys)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    w8a16.linear(xs, q, s, ys)
worst = 0.0
for _ in range(5):
    x = (torch.randn(M, K, device=dev, generator=g) * 2).to(torch.bfloat16)
    xs.copy_(x)
    gr.replay()
    torch.cuda.synchronize()
    worst = max(worst, float((ys - w8a16.linear(x, q, s)).abs().max()))
print(f"{'ok  ' if worst == 0 else 'FAIL'} graph replays equal to eager calls (max diff {worst})", flush=True)
if worst != 0:
    fails.append("graph")
# time per call (rotating weights past L2)
if os.environ.get("TIME", "1") == "1":
    import triton
    for (N, K) in SHAPES:
        C = max(2, -(-(512 << 20) // (N * K)))
        ws = [w8a16.quantize(torch.randn(N, K, device=dev, generator=g) * 0.02) for _ in range(C)]
        for M in (3, 12):
            x = (torch.randn(M, K, device=dev, generator=g)).to(torch.bfloat16)
            out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            R = 24
            s2 = torch.cuda.Stream()
            s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for i in range(3):
                    w8a16.linear(x, *ws[i % C], out)
            torch.cuda.current_stream().wait_stream(s2)
            gt = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gt):
                for i in range(R):
                    w8a16.linear(x, *ws[i % C], out)
            gt.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                gt.replay()
            e1.record()
            torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1e3 / (10 * R)
            nb = N * K + N * (K // 128) * 2
            print(f"time {N:6d} x {K:6d} M {M:2d}: {us:7.1f} us  {nb / us / 1e6:5.2f} TB/s  config "
                  f"{w8a16.config(M, N, K)}", flush=True)
# row-strided inputs (KDA's f_b_proj reads the 128-wide f_a slice of the in_proj output): equal to the contiguous copy
g2 = torch.Generator(device="cuda").manual_seed(7)
wide = torch.randn(3, 49376, device="cuda", generator=g2).to(torch.bfloat16)
sl = wide[:, 49152:49280]                                   # [3, 128], row stride 49376
wq2, ws2 = w8a16.quantize((torch.randn(12288, 128, device="cuda", generator=g2) * 0.05).to(torch.bfloat16))
a2 = w8a16.linear(sl, wq2, ws2)
b2 = w8a16.linear(sl.contiguous(), wq2, ws2)
one = w8a16.linear(wide[:1, 49152:49280], wq2, ws2)
ok_s = bool(torch.equal(a2, b2)) and bool(torch.equal(one, b2[:1])) and sl.stride(0) == 49376
print(f"{'ok' if ok_s else 'FAIL'}   row-strided input (stride {sl.stride(0)}): equal to the contiguous copy {ok_s}")
if not ok_s:
    fails.append("strided")
print("W8A16:", "PASS" if not fails else f"FAIL {fails}", flush=True)
sys.exit(1 if fails else 0)

