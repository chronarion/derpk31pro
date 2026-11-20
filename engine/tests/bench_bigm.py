"""Timing of the prompt-size dense GEMMs (k31_engine.bigm) against today's paths, on the engine's shapes, and the
config table K31_BIGM_CFG reads.

Run from engine/ with the GB300 otherwise idle (~5-8 min: Triton compiles of ~30 configs, then ~2 min of GPU):
    GB=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "GB300") {print $1; exit}'); mkdir -p /tmp/bigm-out
    docker run --rm --device nvidia.com/gpu=$GB --ulimit core=0 --network none --cpuset-cpus 60-61 \
        --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/engine:ro -v /tmp/bigm-out:/out \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/bench_bigm.py
Writes /out/bigm-bench.json (BENCH_JSON): per kind ("i8" int8 I8W, "w8" W8A16 split-K format, "m8" W8A16 Marlin
format), shape "NxK" and row count, the fastest implementation and tile config, its time and today's. Copy it to
engine/profiles/ and set K31_BIGM_CFG=/engine/profiles/bigm-bench.json with K31_I8_BIG_M_KERNEL=1 and
K31_W8A16_BIG_M=1 to serve the table's choices. Every timed output is checked against today's path first (rel < 2e-3;
a config that fails is reported and left out).
Each case: C weight copies (>= 256 MB together, so a call's weights are not in L2 from the previous one) and two x
copies, called eagerly as the engine does (prompt chunks run eager), 3 warm-up calls, then CUDA events over R calls.
The last lines: per shape at 4,096 rows today vs the best, and the sum over one 4,096-row prompt chunk with the
default profile's layer counts.
Env: MS=512,1024,2048,4096,8192 (row counts); KINDS=i8,w8,m8; ONLY=NxK,..; QUICK=1 (default configs only); R=10.
"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import bigm, flags, w8a16, w8i  # noqa: E402

dev = torch.device("cuda")
flags._over.update({"K31_I8_BIG_M": "16", "K31_I8_BIG_M_KERNEL": "0", "K31_W8A16_BIG_M": "0", "K31_BIGM_CFG": ""})
MS = [int(m) for m in os.environ.get("MS", "512,1024,2048,4096,8192").split(",")]
KINDS = os.environ.get("KINDS", "i8,w8,m8").split(",")
ONLY = [v for v in os.environ.get("ONLY", "").split(",") if v]
QUICK = os.environ.get("QUICK", "0") == "1"
R = int(os.environ.get("R", "10"))
OUT = os.environ.get("BENCH_JSON", "/out/bigm-bench.json")
# kind -> [(N, K, layers per forward in the default profile)]
SHAPES = {
    "i8": [(7168, 12288, 93), (7168, 3584, 92), (18432, 1536, 24), (2112, 7168, 24)],   # o_proj, routed up, q_b, qkv-a
    "w8": [(3584, 7168, 92), (7168, 6144, 0)],          # routed latent down; shared down (on Marlin in the profile)
    "m8": [(7168, 6144, 92)],                           # shared down (K31_W8_MARLIN)
}
# (BM, BN, BK, GROUP_M, warps, stages)
CFGS = {
    "i8": [bigm.DEFAULT["i8"], (128, 128, 128, 8, 4, 3), (128, 256, 64, 8, 8, 3), (128, 64, 64, 8, 4, 4),
           (128, 128, 64, 8, 8, 4), (64, 128, 64, 8, 4, 3), (256, 128, 64, 8, 8, 3), (128, 128, 64, 16, 4, 3)],
    "w8": [bigm.DEFAULT["w8"], (128, 128, 128, 8, 8, 2), (128, 64, 128, 8, 4, 3), (64, 128, 128, 8, 4, 3),
           (128, 256, 128, 8, 8, 2)],
    "hilo": [bigm.DEFAULT["hilo"], (128, 128, 128, 8, 8, 3), (128, 64, 64, 8, 4, 4), (128, 256, 64, 8, 8, 3)],
    "m8": [bigm.DEFAULT["m8"], (128, 128, 128, 8, 4, 3), (128, 256, 64, 8, 8, 3), (64, 128, 64, 8, 4, 3),
           (256, 128, 64, 8, 8, 3), (128, 128, 64, 8, 8, 4)],
}
if QUICK:
    CFGS = {k: v[:1] for k, v in CFGS.items()}
g = torch.Generator(device=dev).manual_seed(0)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def timed(fn, C):
    """Mean us per call of fn(i), i cycling over the copies, eager launches."""
    for i in range(3):
        fn(i % C)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(R):
        fn(i % C)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / R


def copies(nbytes):
    return max(2, min(4, math.ceil((256 << 20) / max(1, nbytes))))


table, rows_out = {}, []


def record(kind, N, K, M, today, cands):
    """cands: [(impl, cfg, us)]; the fastest goes into the table."""
    flop = 2.0 * M * N * K
    best = min(cands, key=lambda c: c[2]) if cands else None
    for impl, cfg, us in cands:
        rows_out.append([kind, N, K, M, impl, list(cfg) if cfg else None, round(us, 2), round(flop / us / 1e6, 1)])
    rows_out.append([kind, N, K, M, "today", None, round(today, 2), round(flop / today / 1e6, 1)])
    if best:
        table.setdefault(kind, {}).setdefault(f"{N}x{K}", {})[str(M)] = {
            "impl": best[0], "cfg": list(best[1]) if best[1] else None, "us": round(best[2], 2),
            "today_us": round(today, 2)}
        print(f"{kind} {N:6d} x {K:6d} M {M:5d}: today {today:8.1f} us ({flop / today / 1e6:6.1f} TFLOP/s) | best "
              f"{best[0]} {best[1]} {best[2]:8.1f} us ({flop / best[2] / 1e6:6.1f} TFLOP/s) x{today / best[2]:.2f}",
              flush=True)


def keep(cands, kind, N, K, M, impl, cfg, us, err):
    if us is not None:
        cands.append((impl, cfg, us))
    else:
        print(f"  {kind} {N}x{K} M {M} {impl} {cfg}: left out ({err})", flush=True)


def try_impl(name, fn, ref, C):
    """(us, None) or (None, reason): checked against today's output first."""
    try:
        y = fn(0)
        r = rel(y, ref)
        if not r < 2e-3:
            return None, f"rel {r:.1e} vs today"
        return timed(fn, C), None
    except Exception as e:                                       # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def bench_i8():
    for N, K, _ in SHAPES["i8"]:
        if ONLY and f"{N}x{K}" not in ONLY:
            continue
        C = copies(N * K)
        ws = [w8i.I8W((torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)) for _ in range(C)]
        for M in MS:
            xs = [torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16) for _ in range(2)]
            today = timed(lambda i: w8i.big_m(xs[i % 2], ws[i]).to(torch.bfloat16), C)
            ref = w8i.big_m(xs[0], ws[0]).to(torch.bfloat16)
            cands = []
            for cfg in CFGS["i8"]:
                us, err = try_impl("triton", lambda i, cfg=cfg: bigm.i8_linear(xs[i % 2], ws[i], None, impl="triton",
                                                                               cfg=cfg), ref, C)
                keep(cands, "i8", N, K, M, "triton", cfg, us, err)
            us, err = try_impl("unpack", lambda i: bigm.i8_linear(xs[i % 2], ws[i], None, impl="unpack"), ref, C)
            keep(cands, "i8", N, K, M, "unpack", None, us, err)
            record("i8", N, K, M, today, cands)
            del xs, ref
        del ws
        torch.cuda.empty_cache()


def bench_w8():
    for N, K, _ in SHAPES["w8"]:
        if ONLY and f"{N}x{K}" not in ONLY:
            continue
        C = copies(N * K)
        ws = [w8a16.quantize(torch.randn(N, K, device=dev, generator=g) * 0.02) for _ in range(C)]
        for M in MS:
            xs = [torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16) for _ in range(2)]
            today = timed(lambda i: w8a16.linear(xs[i % 2], *ws[i]), C)
            ref = w8a16.linear(xs[0], *ws[0])
            cands = []
            for impl, key in (("triton", "w8"), ("hilo", "hilo")):
                for cfg in CFGS[key]:
                    us, err = try_impl(impl, lambda i, cfg=cfg, impl=impl: bigm.w8_linear(xs[i % 2], *ws[i], None,
                                                                                         impl=impl, cfg=cfg), ref, C)
                    keep(cands, "w8", N, K, M, impl, cfg, us, err)
            us, err = try_impl("cublas", lambda i: bigm.w8_linear(xs[i % 2], *ws[i], None, impl="cublas"), ref, C)
            keep(cands, "w8", N, K, M, "cublas", None, us, err)
            record("w8", N, K, M, today, cands)
            del xs, ref
        del ws
        torch.cuda.empty_cache()


def bench_m8():
    for N, K, _ in SHAPES["m8"]:
        if ONLY and f"{N}x{K}" not in ONLY:
            continue
        C = copies(N * K)
        ws = [w8a16.MarlinW(*w8a16.quantize(torch.randn(N, K, device=dev, generator=g) * 0.02)) for _ in range(C)]
        for M in MS:
            xs = [torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16) for _ in range(2)]
            today = timed(lambda i: ws[i](xs[i % 2]), C)
            ref = ws[0](xs[0])
            cands = []
            for cfg in CFGS["m8"]:
                us, err = try_impl("triton", lambda i, cfg=cfg: bigm.marlin_linear(xs[i % 2], ws[i], None,
                                                                                   impl="triton", cfg=cfg), ref, C)
                keep(cands, "m8", N, K, M, "triton", cfg, us, err)
            us, err = try_impl("cublas", lambda i: bigm.marlin_linear(xs[i % 2], ws[i], None, impl="cublas"), ref, C)
            keep(cands, "m8", N, K, M, "cublas", None, us, err)
            record("m8", N, K, M, today, cands)
            del xs, ref
        del ws
        torch.cuda.empty_cache()


t0 = time.time()
for kind in KINDS:
    {"i8": bench_i8, "w8": bench_w8, "m8": bench_m8}[kind]()
# one 4,096-row chunk with the default profile's layer counts
M = 4096 if 4096 in MS else max(MS)
tot_today = tot_best = 0.0
for kind in KINDS:
    for N, K, layers in SHAPES[kind]:
        e = table.get(kind, {}).get(f"{N}x{K}", {}).get(str(M))
        if e and layers:
            tot_today += layers * e["today_us"]
            tot_best += layers * e["us"]
            print(f"chunk {M}: {kind} {N}x{K} x {layers} layers: today {layers * e['today_us'] / 1e3:7.1f} ms, "
                  f"best {layers * e['us'] / 1e3:7.1f} ms ({e['impl']} {e['cfg']})")
print(f"chunk {M}: these dense GEMMs today {tot_today / 1e3:.1f} ms, with the table {tot_best / 1e3:.1f} ms, "
      f"saved {(tot_today - tot_best) / 1e3:.1f} ms per {M}-row chunk ({time.time() - t0:.0f} s)", flush=True)
table["rows"] = rows_out
table["rows_cols"] = ["kind", "N", "K", "M", "impl", "cfg", "us", "TFLOP/s"]
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(table, f, indent=1)
print("wrote", OUT)
