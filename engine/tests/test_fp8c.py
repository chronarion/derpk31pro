"""fp6.py's `fp8c` format (FP8 128 x 128 block weights, vLLM's per-token-group activation quant with float scales,
CUTLASS blockwise GEMM): the same math as the format's apply at the wide decode shapes, against BF16 (rel err of a
W8A8 GEMM) and the dequantized reference (layout check); time per call, quant included, next to W8A16 (meaningful only
on an otherwise idle GB300); a CUDA graph replay."""
import os
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import w8a16  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8  # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(5)
TIME = os.environ.get("TIME", "1") == "1"


def quant_block(w):
    N, K = w.shape
    wb = w.float().view(N // 128, 128, K // 128, 128)
    sw = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)
    wq = (wb / sw[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(N, K).contiguous()
    return wq, sw.contiguous()


def fp8c(x, wq, sw):
    xq, sx = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=False)
    return ops.cutlass_scaled_mm(xq, wq.t(), sx, sw.t(), torch.bfloat16)


def timed(fn, C):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(24):
            fn(i)
    gr.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / 240)
    return best


ok = True
for (N, K) in ((12288, 7168), (7168, 12288), (7168, 6144)):
    w = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    wq, sw = quant_block(w)
    for M in (3, 5):
        x = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        y = fp8c(x, wq, sw).float()
        ref = x.float() @ w.float().T
        rel = float((y - ref).norm() / ref.norm())
        xq, sx = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=False)
        xd = (xq.float().view(M, K // 128, 128) * sx.float()[..., None]).view(M, K)
        wd = (wq.float().view(N // 128, 128, K // 128, 128) * sw[:, None, :, None]).view(N, K)
        rd = xd @ wd.T
        lay = float((y - rd).norm() / rd.norm())
        good = rel < 5e-2 and lay < 5e-3        # W8A8 FP8 ~3.7% vs BF16; BF16 output rounding ~1.7e-3
        ok = ok and good
        print(f"{'ok' if good else 'FAIL'}  {N} x {K} M {M}: rel err vs BF16 {rel:.2e} (W8A8), vs dequantized {lay:.1e}",
              flush=True)
    if TIME:
        C = max(2, -(-512 * 2**20 // (N * K)))
        ws8 = [quant_block((torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)) for _ in range(C)]
        ww = [w8a16.quantize((torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)) for _ in range(C)]
        x = torch.randn(3, K, device=dev, generator=g).to(torch.bfloat16)
        t_c = timed(lambda i: fp8c(x, *ws8[i % C]), C)
        t_w = timed(lambda i: w8a16.linear(x, *ww[i % C]), C)
        print(f"      time at 3 rows: fp8c (quant + CUTLASS) {t_c:.1f} us, W8A16 {t_w:.1f} us", flush=True)
# graph replay with new inputs
N, K = 12288, 7168
w = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
wq, sw = quant_block(w)
xs = torch.zeros(3, K, device=dev, dtype=torch.bfloat16)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fp8c(xs, wq, sw)
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    yg = fp8c(xs, wq, sw)
same = True
for _ in range(3):
    x = torch.randn(3, K, device=dev, generator=g).to(torch.bfloat16)
    xs.copy_(x)
    gr.replay()
    torch.cuda.synchronize()
    same = same and torch.equal(yg, fp8c(x, wq, sw))
print(f"graph replays equal to eager calls: {same}")
ok = ok and same
print("FP8C:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
