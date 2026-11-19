"""Decode-size dense GEMMs on the GB300: what DeepGEMM's FP8 GEMM reaches against HBM bandwidth per engine shape, and
whether its global knobs (set_tc_util, set_block_size_multiple_of) or FP4 weights do better.

Shapes are the engine's DeepGEMM FP8 linears from the 3-row decode profile (N x K, weight bytes = N * K): the big ones run
at 0.9-4.5 TB/s there (N / 128 tiles under the 152 SMs). Weights rotate over enough copies to defeat L2; each case is a
CUDA graph of R calls replayed, time per call = graph time / R (graph node gaps included, as in the engine).
Run with the GB300 otherwise idle. DG_PRINT_CONFIGS=1 prints DeepGEMM's chosen configs.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/third_party")
import deep_gemm as dg  # noqa: E402
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8  # noqa: E402

dev = torch.device("cuda")
SHAPES = [(49376, 7168), (12288, 7168), (7168, 12288), (7168, 3584), (7168, 6144), (3584, 7168), (2112, 7168), (18432, 1536)]
# the DSpark drafter's own BF16 linears (context_proj, MLP gate_up / down, the shared LM head; its o_proj, q_b and
# fused q/kv-a shapes are above): only W8A16 against BF16 cuBLAS, which they run on today (K31_DRAFT_W8 uses W8A16
# only where it measured faster)
DRAFT_SHAPES = [(7168, 35840), (28672, 7168), (7168, 14336), (163840, 7168)]
SHAPES = SHAPES + DRAFT_SHAPES
L2_BYTES = 256 << 20
R = 24
ONLY = os.environ.get("ONLY", "")
BEST: dict = {}


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
    return e0.elapsed_time(e1) * 1e3 / (replays * reps)       # us per call


def copies(nbytes):
    return max(2, -(-2 * L2_BYTES // nbytes))


def run_fp8(M, N, K, setup=None):
    C = copies(N * K)
    ws = [per_block_cast_to_fp8(torch.randn(N, K, device=dev, dtype=torch.bfloat16), use_ue8m0=True) for _ in range(C)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    xa = per_token_cast_to_fp8(x, use_ue8m0=True)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    if setup:
        setup()
    return timed(lambda i: dg.fp8_gemm_nt(xa, ws[i % C], out))


def run_fp4(M, N, K):
    C = copies(N * K // 2)
    ws = []
    for _ in range(C):
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        ws.append(per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32))
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    xa = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    return timed(lambda i: dg.fp8_fp4_gemm_nt(xa, ws[i % C], out, recipe_a=(1, 32), recipe_b=(1, 32)))


def run_scaled_mm(M, N, K):
    """torch._scaled_mm with 1x128 activation and 128x128 weight block scales (cuBLASLt block scaling on Blackwell)."""
    C = copies(N * K)
    ws = []
    for _ in range(C):
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        wb = w.view(N // 128, 128, K // 128, 128).float()
        sw = wb.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0                  # [N/128, K/128]
        wq = (wb / sw[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
        ws.append((wq, sw.contiguous()))
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    xb = x.view(M, K // 128, 128).float()
    sx = xb.abs().amax(-1).clamp_min(1e-12) / 448.0                               # [M, K/128]
    xq = (xb / sx[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)
    last = None
    for sa, sb in ((sx.t().contiguous().t(), ws[0][1]), (sx, ws[0][1]), (sx.t().contiguous().t(), ws[0][1].t().contiguous().t())):
        try:
            torch._scaled_mm(xq, ws[0][0].t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            break
        except Exception as e:
            last = e
            sa = None
    if sa is None:
        raise last
    lay_a, lay_b = sa, sb
    return timed(lambda i: torch._scaled_mm(xq, ws[i % C][0].t(), scale_a=lay_a,
                                            scale_b=ws[i % C][1] if lay_b.is_contiguous() else ws[i % C][1].t().contiguous().t(),
                                            out_dtype=torch.bfloat16))


def run_cutlass_fp8(M, N, K):
    """vLLM's CUTLASS blockwise FP8 GEMM (the engine's path for the KDA in_proj: 1x128 activation, 128x128 weight scales),
    checked against the dequantized reference before timing."""
    from vllm import _custom_ops as ops
    C = copies(N * K)
    Np = (N + 127) // 128 * 128
    ws = []
    for _ in range(C):
        w = torch.randn(Np, K, device=dev) * 0.02
        wb = w.view(Np // 128, 128, K // 128, 128)
        sw = (wb.abs().amax(dim=(1, 3)) / 448).float().clamp_min(1e-12)
        ws.append(((wb / sw[:, None, :, None]).to(torch.float8_e4m3fn).view(Np, K), sw))
    x = torch.randn(M, K, device=dev)
    xb = x.view(M, K // 128, 128)
    sx = (xb.abs().amax(-1) / 448).float().clamp_min(1e-12)
    xq = (xb / sx[..., None]).to(torch.float8_e4m3fn).view(M, K)
    sxc = sx.t().contiguous().t()
    wq, sw = ws[0]
    y = ops.cutlass_scaled_mm(xq, wq.t(), sxc, sw.t(), torch.bfloat16)
    ref = (xq.float().view(M, K // 128, 128) * sx[..., None]).view(M, K) @ \
        (wq.float().view(Np // 128, 128, K // 128, 128) * sw[:, None, :, None]).view(Np, K).T
    err = float((y.float() - ref).norm() / ref.norm())
    assert err < 1e-2, f"cutlass blockwise layout check failed: rel err {err}"
    return timed(lambda i: ops.cutlass_scaled_mm(xq, ws[i % C][0].t(), sxc, ws[i % C][1].t(), torch.bfloat16))


def run_cublaslt_bf16(M, N, K):
    C = copies(2 * N * K)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(C)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    return timed(lambda i: dg.cublaslt_gemm_nt(x, ws[i % C], out))


def run_w8a16(M, N, K, cfg=None):
    sys.path.insert(0, "/engine")
    from k31_engine import w8a16
    C = copies(N * K)
    ws = [w8a16.quantize(torch.randn(N, K, device=dev, dtype=torch.bfloat16)) for _ in range(C)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    return timed(lambda i: w8a16.linear(x, *ws[i % C], out, cfg))


def run_mxfp4dg(M, N, K):
    """k31_engine.mxfp4dg: the fused MXFP8 activation quant + DeepGEMM's FP8 x MXFP4 GEMM, as the engine calls it."""
    sys.path.insert(0, "/engine")
    from k31_engine import mxfp4dg
    C = copies(N * K // 2)
    ws = [mxfp4dg.prepare(torch.randn(N, K, device=dev, dtype=torch.bfloat16)) for _ in range(C)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    return timed(lambda i: mxfp4dg.linear(x, ws[i % C][0], ws[i % C][1], N, out))


def w8a16_cfgs(N, K):
    """The default config and a small sweep around it: rows per program, K slice (programs per N tile), warps, stages."""
    sys.path.insert(0, "/engine")
    from k31_engine import w8a16
    import triton
    groups = K // 128
    out = [None]
    for BN in (32, 64, 128):
        nt = triton.cdiv(N, BN)
        for per_sm in (2, 4, 8):
            split = min(groups, max(1, (per_sm * 152 + nt - 1) // nt))
            per = triton.cdiv(groups, split)
            split = triton.cdiv(groups, per)
            for nw, ns in ((4, 3), (4, 4), (8, 3)):
                out.append((BN, per * 128, split, nw, ns))
    return out


def run_bf16(M, N, K):
    C = copies(2 * N * K)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(C)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    return timed(lambda i: torch.mm(x, ws[i % C].T))


def main():
    print(f"{torch.cuda.get_device_name()}, DeepGEMM {dg.__version__}, num_sms {dg.get_num_sms()}, tc_util "
          f"{dg.get_tc_util()}", flush=True)
    tc0 = dg.get_tc_util()
    print(f"torch {torch.__version__}; dg.cublaslt_gemm_nt schema: "
          f"{getattr(getattr(dg.cublaslt_gemm_nt, 'default', dg.cublaslt_gemm_nt), '_schema', '?')}", flush=True)
    variants = [("default", None)]
    if os.environ.get("KNOBS", "0") == "1":      # DeepGEMM's global knobs (no effect in the 11-18 smoke runs)
        for v in sorted({max(1, int(tc0 * f)) for f in (0.25, 0.5, 0.75)} - {tc0}):
            variants.append((f"tc_util {v}", lambda v=v: dg.set_tc_util(v)))
        for b in (16, 32, 64):
            variants.append((f"block multiple {b}", lambda b=b: dg.set_block_size_multiple_of(b)))
    rows = []
    for (N, K) in SHAPES:
        if rows:
            dump(rows)
        if ONLY and f"{N}x{K}" not in ONLY:
            continue
        nb = N * K
        draft = (N, K) in DRAFT_SHAPES
        for M in ((3,) if draft else tuple(int(v) for v in os.environ.get("MS", "3,6,12").split(","))):
            for name, setup in ([] if draft else variants if M == 3 else variants[:1]):
                dg.set_tc_util(tc0)
                try:
                    dg.set_block_size_multiple_of(1)
                except Exception:
                    pass
                try:
                    t = run_fp8(M, N, K, setup)
                    rows.append((N, K, M, f"fp8 {name}", t, nb / t / 1e6))
                except Exception as e:
                    rows.append((N, K, M, f"fp8 {name}", float("nan"), 0.0))
                    print(f"  {N}x{K} M {M} fp8 {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                print(f"{N:6d} x {K:6d}  M {M:2d}  {rows[-1][3]:24s} {rows[-1][4]:8.1f} us  {rows[-1][5]:5.2f} TB/s",
                      flush=True)
            dg.set_tc_util(tc0)
            if M == 3 or M == 12:
                best = None
                for cfg in (w8a16_cfgs(N, K) if M == 3 else [None]):
                    try:
                        t = run_w8a16(M, N, K, cfg)
                    except Exception as e:
                        print(f"  {N}x{K} M {M} w8a16 {cfg}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                        continue
                    by = N * K + N * (K // 128) * 2
                    if cfg is None or best is None or t < best[0]:
                        if cfg is None:
                            rows.append((N, K, M, "w8a16 default", t, by / t / 1e6))
                            print(f"{N:6d} x {K:6d}  M {M:2d}  {'w8a16 default':24s} {t:8.1f} us  {by / t / 1e6:5.2f} TB/s",
                                  flush=True)
                        if cfg is not None:
                            best = (t, cfg)
                if best is not None:
                    by = N * K + N * (K // 128) * 2
                    BEST.setdefault(f"{N}x{K}", {})[str(M)] = list(best[1])
                    rows.append((N, K, M, f"w8a16 best {best[1]}", best[0], by / best[0] / 1e6))
                    print(f"{N:6d} x {K:6d}  M {M:2d}  w8a16 best {best[1]}  {best[0]:8.1f} us  "
                          f"{by / best[0] / 1e6:5.2f} TB/s", flush=True)
            if M == 3 and not draft:
                try:
                    t = run_mxfp4dg(M, N, K)
                    rows.append((N, K, M, "mxfp4dg (quant + gemm)", t, (nb // 2) / t / 1e6))
                    print(f"{N:6d} x {K:6d}  M {M:2d}  {'mxfp4dg (quant + gemm)':24s} {t:8.1f} us  {(nb // 2) / t / 1e6:5.2f} "
                          f"TB/s", flush=True)
                except Exception as e:
                    print(f"  {N}x{K} M {M} mxfp4dg: {type(e).__name__}: {str(e)[:160]}", flush=True)
            for name, fn, by in (("fp8 cutlass blockwise", run_cutlass_fp8, nb),
                                 ("mxfp4 weights (fp8 x fp4)", run_fp4, nb // 2),
                                 ("fp8 _scaled_mm blockwise", run_scaled_mm, nb),
                                 ("bf16 dg.cublaslt_gemm_nt", run_cublaslt_bf16, 2 * nb),
                                 ("bf16 cuBLAS", run_bf16, 2 * nb)):
                if M != 3 or (draft and name != "bf16 cuBLAS"):
                    continue
                try:
                    t = fn(M, N, K)
                    rows.append((N, K, M, name, t, by / t / 1e6))
                except Exception as e:
                    rows.append((N, K, M, name, float("nan"), 0.0))
                    print(f"  {N}x{K} M {M} {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                print(f"{N:6d} x {K:6d}  M {M:2d}  {rows[-1][3]:24s} {rows[-1][4]:8.1f} us  {rows[-1][5]:5.2f} TB/s",
                      flush=True)
    # the router's gate (BF16 weights [896, 7168], float32 out): w8a16.linear16 against cuBLAS (torch mm)
    try:
        sys.path.insert(0, "/engine")
        from k31_engine import w8a16
        E, K = 896, 7168
        C = copies(2 * E * K)
        gws = [torch.randn(E, K, device=dev, dtype=torch.bfloat16).contiguous() for _ in range(C)]
        x = torch.randn(3, K, device=dev, dtype=torch.bfloat16)
        for name, fn in (("gate linear16 (split-K)", lambda i: w8a16.linear16(x, gws[i % C])),
                         ("gate torch mm (cuBLAS)", lambda i: torch.mm(x, gws[i % C].T).float())):
            t = timed(fn)
            rows.append((E, K, 3, name, t, 2 * E * K / t / 1e6))
            print(f"{E:6d} x {K:6d}  M  3  {name:24s} {t:8.1f} us  {2 * E * K / t / 1e6:5.2f} TB/s", flush=True)
    except Exception as e:
        print(f"  gate: {type(e).__name__}: {str(e)[:160]}", flush=True)
    dump(rows)
    print("BENCH DONE", flush=True)


def dump(rows):
    """The table the engine reads (w8a16.config / preferred, the chain's MXFP4 decision); written after every shape."""
    import json
    table = {"rows": rows, "best": BEST, "fp8_default": {}, "w8a16_best_us": {}}
    for (N, K, M, name, t, _) in rows:
        # the FP8 kernel the engine runs at this shape (11-17 trace): CUTLASS blockwise for the KDA in_proj (N = 49376,
        # not a multiple of 128), DeepGEMM for every other dense shape
        engine_fp8 = "fp8 cutlass blockwise" if N == 49376 else "fp8 default"
        if name == engine_fp8 and t == t:
            table["fp8_default"].setdefault(f"{N}x{K}", {})[str(M)] = t
        if name.startswith("w8a16 best"):
            table["w8a16_best_us"].setdefault(f"{N}x{K}", {})[str(M)] = t
        if name == "mxfp4dg (quant + gemm)":
            table.setdefault("mxfp4dg_us", {}).setdefault(f"{N}x{K}", {})[str(M)] = t
        if name == "bf16 cuBLAS" and t == t:
            table.setdefault("bf16_us", {}).setdefault(f"{N}x{K}", {})[str(M)] = t
    with open(os.environ.get("BENCH_JSON", "/cache/bench_dense_dg.json"), "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f} s", flush=True)
