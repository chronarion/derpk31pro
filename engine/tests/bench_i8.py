"""Hecate's int8 decode GEMV (engine/csrc/k31i8: dense_i8.cu unchanged) on the engine's dense shapes, against the
same weights' float32 reference (CHECK=1, safe on a busy GPU) or timed (quiet GPU: CUDA graphs of R calls over
enough weight copies to defeat L2, as bench_marlin.py). Weights: int8 code - 128 with one float32 scale per output
row (absmax / 127, Hecate's to_i8_copy), packed in the kernel's mma fragment order. The k31i8 build takes BF16 x
(converted to f16 pairs on load) and writes BF16 (or fp32 / fp16) y: no conversion kernels around it.
ROWS=3,4,5; ONLY=NxK,..; K31I8=path of k31i8.so (default /build/k31i8.so).
"""
import importlib.util
import json
import os
import sys

import torch

dev = torch.device("cuda")
spec = importlib.util.spec_from_file_location("k31i8", os.environ.get("K31I8", "/build/k31i8.so"))
k31i8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k31i8)
SHAPES = [(12288, 7168), (7168, 12288), (7168, 3584), (7168, 6144), (3584, 7168), (2112, 7168), (18432, 1536),
          (7168, 35840), (28672, 7168), (7168, 14336), (163840, 7168)]
ONLY = os.environ.get("ONLY", "")
ROWS = [int(r) for r in os.environ.get("ROWS", "3,4,5").split(",")]
CHECK = os.environ.get("CHECK", "0") == "1"
L2_BYTES = 256 << 20
R = 24


def pack(w: torch.Tensor):
    """BF16 [N, K] -> (packed uint8 codes, float32 scales [N], spw): Hecate's to_i8_copy."""
    W = w.float()
    N, K = W.shape
    assert N % 16 == 0 and K % 32 == 0
    spw = 4 if (N // 16) % 4 == 0 else 2 if (N // 16) % 2 == 0 else 1
    s = (W.abs().amax(1) / 127.0).clamp_min(1e-12)
    q = (torch.round(W / s[:, None]) + 128).clamp_(0, 255).to(torch.uint8)
    deq = (q.float() - 128) * s[:, None]
    q = q.view(N // (16 * spw), spw, 2, 8, K // 32, 2, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 8, 2).contiguous().view(-1)
    return q, s.contiguous(), spw, deq


class I8:
    def __init__(self, w):
        self.q, self.s, self.spw, deq = pack(w)
        self.N, self.K = w.shape
        self.y32 = torch.zeros(8 * self.N, dtype=torch.float32, device=dev)
        self.ctr = torch.zeros(max(1, self.N // 16), dtype=torch.int32, device=dev)
        self.deq = deq if CHECK else None

    def __call__(self, x, y):
        k31i8.dense_i8(x, self.q, self.s, y, self.y32, self.ctr, self.spw)
        return y


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


def check():
    fails = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for N, K in SHAPES:
        if ONLY and f"{N}x{K}" not in ONLY.split(","):
            continue
        w = (torch.randn(N, K, generator=g) * 0.02).to(dev, torch.bfloat16)
        lin = I8(w)
        worst = 0.0
        for M in (1, 3, 5, 8):
            x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
            ref = torch.cat([x.float() @ lin.deq[i:i + 8192].T for i in range(0, N, 8192)], dim=1)
            for odt in (torch.bfloat16, torch.float32):
                y = torch.empty(M, N, dtype=odt, device=dev)
                lin(x, y)
                worst = max(worst, ((y.float() - ref).norm() / ref.norm()).item())
        ok = worst < 3e-3
        print(f"{'PASS' if ok else 'FAIL'} int8 GEMV {N}x{K} at 1-8 rows (bf16 x; bf16 / fp32 out): worst rel {worst:.2e}",
              flush=True)
        if not ok:
            fails.append(f"{N}x{K}")
        del w, lin
        torch.cuda.empty_cache()
    print("I8 CHECK:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
    return not fails


def bench():
    tab = json.load(open("/engine/profiles/w8a16-bench.json"))
    mtab = json.load(open("/kl/bench-marlin.json")) if os.path.exists("/kl/bench-marlin.json") else {}
    old = tab.get("w8a16_best_us", {})
    out = {"i8_us": {}, "i8_conv_us": {}}
    for N, K in SHAPES:
        key = f"{N}x{K}"
        if ONLY and key not in ONLY.split(","):
            continue
        C = max(2, -(-2 * L2_BYTES // (N * K)))
        ws = []
        for _ in range(C):
            ws.append(I8(torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02))
        torch.cuda.empty_cache()
        for M in ROWS:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            yb = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
            t_k = timed(lambda i: ws[i % C](x, yb))
            t_c = t_k
            out["i8_us"].setdefault(key, {})[str(M)] = t_k
            out["i8_conv_us"].setdefault(key, {})[str(M)] = t_c
            o = (old.get(key) or {}).get("3")
            mr = ((mtab.get("marlin8_us") or {}).get(key) or {}).get(str(M))
            print(f"{key:>12} M={M}: int8 GEMV bf16 {t_k:6.1f} us {N * K / t_k / 1e6:5.2f} TB/s | w8a16 "
                  f"{o if o is None else round(o, 1)} | marlin {mr if mr is None else round(mr, 1)}",
                  flush=True)
        del ws
        torch.cuda.empty_cache()
    json.dump(out, open(os.environ.get("OUT", "/kl/bench-i8.json"), "w"), indent=1)


if __name__ == "__main__":
    if CHECK:
        sys.exit(0 if check() else 1)
    bench()
