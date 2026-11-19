"""Dense GEMMs of K3.1's decode step at M = 1/3/6 rows: the image's FP8 block path (DeepGEMM fp8_fp4_gemm_nt with
per-token 128-group activation quant, as vLLM's fp8_per_block) vs Humming weight-only formats with BF16 inputs
(FP8 group-32, MXFP6 E2M3, NVFP4, MXFP4) vs BF16 cuBLAS. Per shape: time (CUDA graph replay of 50 calls), GB/s of weight
bytes, and the relative error of the output against the BF16 GEMM of the same weights.
Run in the engine image on the GB300 with nothing else on the GPU."""
import sys
import time

import torch

dev = torch.device("cuda")
torch.manual_seed(0)
SHAPES = {                                           # (N, K), per-step count
    "kda_in": ((49376, 7168), 69), "kda_out": ((7168, 12288), 69), "shared_gate_up": ((12288, 7168), 92),
    "shared_down": ((7168, 6144), 92), "latent_down": ((3584, 7168), 92), "latent_up": ((7168, 3584), 92),
    "mla_q_b": ((18432, 1536), 24), "mla_o": ((7168, 12288), 24),
}
MS = [int(m) for m in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "3", "6"])]


def timeit(fn, n=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


def fp8_block(w):
    """vLLM fp8_per_block weights: 128x128 blocks, float32 scales; returns a callable x -> y via DeepGEMM."""
    from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    wq, ws = per_block_cast_to_fp8(w, use_ue8m0=True)
    N = w.shape[0]

    def run(x, out):
        xq, xs = per_token_group_quant_fp8(x, 128, use_ue8m0=True, column_major_scales=True)
        fp8_gemm_nt((xq, xs), (wq, ws), out)
    return run, wq.numel() + ws.numel() * 4


HFMT = {
    "fp6": {"dtype": "float6e2m3", "scale_dtype": "float8e8m0", "group_size": 32, "weight_scale_type": "group"},
    "nvfp4": {"dtype": "float4e2m1", "scale_dtype": "float8e4m3", "group_size": 16, "weight_scale_type": "group",
              "weight_scale_2_type": "tensor"},
    "mxfp4": {"dtype": "float4e2m1", "scale_dtype": "float8e8m0", "group_size": 32, "weight_scale_type": "group"},
    "fp8": {"dtype": "float8e4m3", "scale_dtype": "float8e8m0", "group_size": 32, "weight_scale_type": "group"},
}


def humming_fp6(w, a8=False, fmt="fp6"):
    from humming.layer import HummingLayer
    N, K = w.shape
    wc = dict(HFMT[fmt])
    ic = {"dtype": "float8e4m3", "input_quant_mode": "dynamic_token"} if a8 else None
    lay = HummingLayer(shape_n=N, shape_k=K, weight_config=wc, input_config=ic, torch_dtype=torch.bfloat16,
                       pad_n_to_multiple=256, pad_k_to_multiple=128).to(dev)
    lay.load_from_unquantized(w)
    lay.transform()
    nbytes = sum(p.numel() * p.element_size() for n, p in lay.named_parameters())
    return (lambda x, out: out.copy_(lay(x)[:, :N])), nbytes


rows = []
for name, ((N, K), count) in SHAPES.items():
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = (x.float() @ w.float().T)
        res = {}
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        res["bf16"] = (timeit(lambda: torch.matmul(x, w.T, out=out)), N * K * 2, 0.0)
        for label, make in (("fp8b128", lambda: fp8_block(w)), ("h-fp8", lambda: humming_fp6(w, fmt="fp8")),
                            ("h-fp6", lambda: humming_fp6(w)), ("h-nvfp4", lambda: humming_fp6(w, fmt="nvfp4")),
                            ("h-mxfp4", lambda: humming_fp6(w, fmt="mxfp4"))):
            try:
                run, nb = make()
                run(x, out)
                torch.cuda.synchronize()
                err = float(((out.float() - ref).norm() / ref.norm()))
                res[label] = (timeit(lambda: run(x, out)), nb, err)
            except Exception as e:                        # report and keep going
                res[label] = (float("nan"), 0, float("nan"))
                print(f"{name} M={M} {label}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        line = f"{name:15s} N {N:6d} K {K:6d} M {M}: " + " | ".join(
            f"{k} {t:7.1f} us {nb / t / 1e3 if t == t and t > 0 else 0:6.0f} GB/s err {e:.4f}" for k, (t, nb, e) in res.items())
        print(line, flush=True)
        rows.append((name, M, count, res))
for M in MS:
    tot = {k: sum(c * r[k][0] for n, m, c, r in rows if m == M) / 1e3
           for k in ("bf16", "fp8b128", "h-fp8", "h-fp6", "h-nvfp4", "h-mxfp4")}
    print(f"per decode step at M={M} (these shapes x their counts): " + ", ".join(f"{k} {v:.2f} ms" for k, v in tot.items()))
