"""vLLM's Marlin kernels on the engine's decode-size dense shapes: FP8 weights x BF16 activations (the W8A16 format:
FP8 E4M3 [N, K] with one BF16 scale per row and 128-wide K group, w8a16.quantize) and MXFP4 x BF16 (E2M1 with E8M0
scales per 32), against what the engine's kernels measured (profiles/w8a16-bench.json).

CHECK=1: correctness only (each shape at 4 rows against x @ dequant(W)^T in float32), safe while the GB300 is busy.
Otherwise timing: each case is a CUDA graph of R calls over enough weight copies to defeat L2 (as bench_dense_dg.py);
run with the GB300 otherwise idle. ROWS=3,4,5 picks the row counts; ONLY=NxK limits the shapes.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine import w8a16  # noqa: E402
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mf8  # noqa: E402
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mf4  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

dev = torch.device("cuda")
SHAPES = [(49376, 7168), (12288, 7168), (7168, 12288), (7168, 3584), (7168, 6144), (3584, 7168), (2112, 7168),
          (18432, 1536), (7168, 35840), (28672, 7168), (7168, 14336), (163840, 7168)]
ONLY = os.environ.get("ONLY", "")
ROWS = [int(r) for r in os.environ.get("ROWS", "3,4,5").split(",")]
CHECK = os.environ.get("CHECK", "0") == "1"
L2_BYTES = 256 << 20
R = 24
WS = marlin_make_workspace_new(dev, 4)


class _L(torch.nn.Module):
    pass


def marlin8_prepare(q: torch.Tensor, s: torch.Tensor):
    """w8a16.quantize's (q [N, K] e4m3, s [N, K / 128] bf16) -> Marlin's (weight, scales)."""
    N, K = q.shape
    L = _L()
    L.weight = torch.nn.Parameter(q, requires_grad=False)
    L.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    L.weight_block_size = [1, 128]
    L.output_size_per_partition, L.input_size_per_partition = N, K
    L.orig_dtype = torch.bfloat16
    mf8.prepare_fp8_layer_for_marlin(L, size_k_first=False)
    return L.weight.data, L.weight_scale.data


def marlin_call(x, w, sc, gs, N, K, tid, atomic, fp32red, out=None):
    """One Marlin GEMM as vLLM's apply functions call it: N (and K) padded to Marlin's tiles, the output sliced back
    (a view)."""
    bits = 8 if tid == scalar_types.float8_e4m3fn.id else 4
    pn, pk = mf8.marlin_repacked_nk(w, num_bits=bits)
    assert pk == K, f"K {K} would need padding to {pk}"
    y = torch.ops.vllm.marlin_gemm(a=x, b_q_weight=w, b_bias=None, b_scales=sc, a_scales=None, global_scale=gs,
                                   b_zeros=None, workspace=WS, b_q_type_id=tid, size_m=x.shape[0], size_n=pn,
                                   size_k=pk, use_atomic_add=atomic, use_fp32_reduce=fp32red)
    return y[:, :N]


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def mxfp4_quantize(w: torch.Tensor):
    """BF16 [N, K] -> (packed uint8 [N, K / 2] E2M1, two per byte, low nibble first; E8M0 scales uint8 [N, K / 32];
    the float32 dequantized weights). Nearest E2M1 magnitude by bucketize, in row chunks."""
    N, K = w.shape
    grid = E2M1.to(w.device)
    bounds = ((grid[1:] + grid[:-1]) / 2).contiguous()
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(N, K // 32, dtype=torch.uint8, device=w.device)
    deq = torch.empty(N, K, dtype=torch.float32, device=w.device)
    for i in range(0, N, 4096):
        wf = w[i:i + 4096].float().view(-1, K // 32, 32)
        e = torch.ceil(torch.log2(wf.abs().amax(-1).clamp_min(1e-30) / 6.0)).clamp(-127, 127)
        sc = torch.exp2(e)
        v = wf / sc[..., None]
        idx = torch.bucketize(v.abs(), bounds)
        deq[i:i + 4096] = (grid[idx] * torch.sign(v) * sc[..., None]).view(-1, K)
        code = (idx | ((v < 0).to(torch.int64) << 3)).view(-1, K).to(torch.uint8)
        packed[i:i + 4096] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[i:i + 4096] = (e + 127).to(torch.uint8)
    return packed, scales, deq


def marlin4_prepare(packed: torch.Tensor, scales: torch.Tensor, N: int, K: int):
    L = _L()
    L.weight = torch.nn.Parameter(packed, requires_grad=False)
    L.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    L.output_size_per_partition, L.input_size_per_partition = N, K
    L.params_dtype = torch.bfloat16
    mf4.prepare_fp4_layer_for_marlin(L)
    return L.weight.data, L.weight_scale.data


def timed(fn, reps=R, replays=10):
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


def copies(nbytes):
    return max(2, -(-2 * L2_BYTES // nbytes))


def check():
    fails = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for N, K in SHAPES:
        if ONLY and f"{N}x{K}" not in ONLY.split(","):
            continue
        w = (torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16)
        x = torch.randn(4, K, generator=g).to(dev, torch.bfloat16)
        q, s = w8a16.quantize(w)
        ref = torch.cat([x.float() @ (q[i:i + 8192].float().view(-1, K // 128, 128) * s[i:i + 8192].float()[..., None])
                         .view(-1, K).T for i in range(0, N, 8192)], dim=1)
        mw, ms = marlin8_prepare(q, s)
        for atomic in (False, True):
            y = marlin_call(x, mw, ms, None, N, K, scalar_types.float8_e4m3fn.id, atomic, True).float()
            rel = ((y - ref).norm() / ref.norm()).item()
            ok = rel < 5e-3
            print(f"{'PASS' if ok else 'FAIL'} marlin fp8 {N}x{K} atomic={atomic}: rel {rel:.2e}")
            if not ok:
                fails.append(f"fp8 {N}x{K}")
        if (N, K) == (49376, 7168) or ONLY:
            pk, sc, deq = mxfp4_quantize(w)
            ref4 = x.float() @ deq.T
            m4w, m4s = marlin4_prepare(pk, sc, N, K)
            y = marlin_call(x, m4w, m4s, None, N, K, scalar_types.float4_e2m1f.id, False, True).float()
            rel = ((y - ref4).norm() / ref4.norm()).item()
            ok = rel < 5e-3
            print(f"{'PASS' if ok else 'FAIL'} marlin mxfp4 {N}x{K}: rel {rel:.2e}")
            if not ok:
                fails.append(f"mxfp4 {N}x{K}")
        del w, q, s, mw, ms
        torch.cuda.empty_cache()
    print("MARLIN CHECK:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
    return not fails


def bench():
    tab = json.load(open("/engine/profiles/w8a16-bench.json"))
    old = tab.get("w8a16_best_us", {})
    f8d, m4d, bfu = tab.get("fp8_default", {}), tab.get("mxfp4dg_us", {}), tab.get("bf16_us", {})
    out = {"marlin8_us": {}, "marlin4_us": {}}
    for N, K in SHAPES:
        key = f"{N}x{K}"
        if ONLY and key not in ONLY.split(","):
            continue
        C = copies(N * K)
        ws = []
        for _ in range(C):
            q, s = w8a16.quantize(torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
            ws.append(marlin8_prepare(q, s))
            del q, s
        torch.cuda.empty_cache()
        for M in ROWS:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            best = None
            for atomic in (False, True):
                for fp32red in (True, False):
                    try:
                        us = timed(lambda i: marlin_call(x, ws[i % C][0], ws[i % C][1], None, N, K,
                                                         scalar_types.float8_e4m3fn.id, atomic, fp32red))
                    except Exception as e:                       # noqa: BLE001
                        print(f"  {key} M={M} atomic={atomic} fp32red={fp32red}: {str(e)[:100]}")
                        continue
                    if best is None or us < best[0]:
                        best = (us, atomic, fp32red)
            if best is None:
                continue
            us = best[0]
            out["marlin8_us"].setdefault(key, {})[str(M)] = us
            o = (old.get(key) or {}).get(str(M)) or (old.get(key) or {}).get("3")
            print(f"{key:>12} M={M}: marlin fp8 {us:6.1f} us {N * K / us / 1e6:5.2f} TB/s (atomic={best[1]}, "
                  f"fp32red={best[2]}) | w8a16 {o if o is None else round(o, 1)} | fp8 cutlass/dg "
                  f"{(f8d.get(key) or {}).get('3')} | bf16 {round((bfu.get(key) or {}).get('3') or 0, 1)}", flush=True)
        del ws
        torch.cuda.empty_cache()
        if (N, K) == (49376, 7168):
            C4 = copies(N * K // 2)
            w4 = []
            for _ in range(C4):
                pk, sc, _d = mxfp4_quantize(torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
                w4.append(marlin4_prepare(pk, sc, N, K))
                del pk, sc, _d
            torch.cuda.empty_cache()
            for M in ROWS:
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                best = min(timed(lambda i: marlin_call(x, w4[i % C4][0], w4[i % C4][1], None, N, K,
                                                       scalar_types.float4_e2m1f.id, a, True)) for a in (False, True))
                out["marlin4_us"].setdefault(key, {})[str(M)] = best
                print(f"{key:>12} M={M}: marlin mxfp4 {best:6.1f} us {N * K * 0.53 / best / 1e6:5.2f} TB/s | mxfp4dg "
                      f"{(m4d.get(key) or {}).get('3')}", flush=True)
            del w4
            torch.cuda.empty_cache()
    json.dump(out, open(os.environ.get("OUT", "/kl/bench-marlin.json"), "w"), indent=1)


if __name__ == "__main__":
    if CHECK:
        sys.exit(0 if check() else 1)
    bench()
