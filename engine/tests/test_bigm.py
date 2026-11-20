"""k31_engine.bigm on the GB300: the prompt-size GEMMs against the engine's existing paths for the same weights.

Run from engine/ (~2-4 min, most of it Triton compiles; safe beside a running server if
~3 GB of HBM are free, timings are not taken here):
    GB=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "GB300") {print $1; exit}')
    docker run --rm --device nvidia.com/gpu=$GB --ulimit core=0 --network none --cpuset-cpus 60-61 \
        --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /engine/tests/test_bigm.py
Exit 0 and "BIGM: PASS" = every check below held. "rel" is ||a - b|| / ||b|| over the whole output.
1. int8 (w8i.I8W) on the engine's shapes at 17 / 300 / 1,024 / 4,096 rows: the Triton GEMM on packed codes and the
   unpack + cuBLAS form against today's big_m (rel < 2e-3 for BF16 outputs; float32 outputs rel < 1e-5: the same
   exact products, float32 sums in another order), against a float64 reference on 256 sampled rows, bit-equal
   reruns, float32 and strided outputs; the unpack kernel bit-equal to w8i.unpack; I8W.__call__ follows the runtime
   flag K31_I8_BIG_M_KERNEL both ways.
2. W8A16, split-K format (q, s) at 300 / 1,024 / 4,096 rows: per-group promotion, hi/lo and the cuBLAS form against
   today's 64-row tiles (rel < 2e-3; the promotion's bit-equal share is reported), against float64 on sampled rows,
   row-strided x, the hi/lo dequant exact (hi + lo = q * s, hi = BF16(q * s)); w8a16.linear follows
   K31_W8A16_BIG_M / _ROWS / _MIN_K (below the row count and for K < 1,024 it stays on today's tiles, bit-equal).
3. W8A16, Marlin format (w8a16.MarlinW, vLLM's gptq_marlin_repack on the GPU): bigm's dequant equal to BF16(q * s)
   bit for bit (checks the byte map and the scale map against the real repack), the Triton GEMM and the cuBLAS form
   against Marlin's own GEMM (rel < 2e-3), w8a16.linear's dispatch.
4. K31_BIGM_CFG: a table entry is followed ("1" takes its implementation and config).
5. Every implementation launched with PDL (K31_PDL=1 servers) equals its plain launch bit for bit.
The printed "max|d|/max|ref|" values are for the record; the verdicts use rel.
"""
import contextlib
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import bigm, flags, w8a16, w8i  # noqa: E402

dev = torch.device("cuda")
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def same_share(a, b):
    return float((a == b).float().mean())


@contextlib.contextmanager
def runtime_flags(**kv):
    """Set runtime flags the way the control file does (flags.py's overrides), restored on exit."""
    old = dict(flags._over)
    flags._over.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        flags._over.clear()
        flags._over.update(old)


# today's paths unless a check sets a flag (whatever the container's environment says)
flags._over.update({"K31_I8_BIG_M": "16", "K31_I8_BIG_M_KERNEL": "0", "K31_W8A16_BIG_M": "0", "K31_BIGM_CFG": ""})
g = torch.Generator(device=dev).manual_seed(3)
SAMPLE = torch.randperm(4096, generator=torch.Generator().manual_seed(4))[:256].sort().values.to(dev)


def rows(M):
    return SAMPLE[SAMPLE < M] if M > 256 else torch.arange(M, device=dev)


# ------------------------------------------------------------------------------------------------ 1. int8 (I8W)
I8_SHAPES = [(7168, 3584), (7168, 12288), (18432, 1536), (2112, 7168)]   # routed up, o_proj, q_b, fused qkv-a
for N, K in I8_SHAPES:
    w = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    iw = w8i.I8W(w)
    del w
    codes = w8i.unpack(iw.q, N, K, iw.spw)
    cu = bigm.i8_unpack(iw)
    check(f"int8 {N}x{K}: unpack kernel = w8i.unpack - 128", torch.equal(cu.float(), codes.float() - 128))
    deq = (codes.double() - 128) * iw.s.double()[:, None]
    del codes, cu
    for M in (17, 300, 1024, 4096):
        x = (torch.randn(M, K, device=dev, generator=g)).to(torch.bfloat16)
        old32 = w8i.big_m(x, iw)                                 # today's prompt path, float32 before rounding
        old = old32.to(torch.bfloat16)
        r = rows(M)
        ref = x[r].double() @ deq.t()
        for impl in ("triton", "unpack"):
            y = bigm.i8_linear(x, iw, None, impl=impl)
            y2 = bigm.i8_linear(x, iw, None, impl=impl)
            y32 = torch.empty(M, N, device=dev, dtype=torch.float32)
            bigm.i8_linear(x, iw, y32, impl=impl)
            ok = rel(y, old) < 2e-3 and rel(y32, old32) < 1e-5 and torch.equal(y, y2) and rel(y[r], ref) < 4e-3
            check(f"int8 {impl} {N}x{K} M {M}", ok,
                  f"(vs big_m {rel(y, old):.1e}, bit-equal {same_share(y, old):.3f}; fp32 {rel(y32, old32):.1e}; "
                  f"vs fp64 {rel(y[r], ref):.1e}; rerun equal {torch.equal(y, y2)})")
        if M == 300:
            wide = torch.zeros(M, N + 64, device=dev, dtype=torch.bfloat16)
            bigm.i8_linear(x, iw, wide[:, 32:32 + N], impl="triton")
            y = bigm.i8_linear(x, iw, None, impl="triton")
            check(f"int8 {N}x{K}: strided out", torch.equal(wide[:, 32:32 + N], y) and
                  float(wide[:, :32].abs().max()) == 0 and float(wide[:, 32 + N:].abs().max()) == 0)
            with runtime_flags(K31_I8_BIG_M_KERNEL="1"):
                via_flag = iw(x)
            with runtime_flags(K31_I8_BIG_M_KERNEL="0"):
                via_old = iw(x)
            check(f"int8 {N}x{K}: I8W follows K31_I8_BIG_M_KERNEL", torch.equal(via_flag, y) and torch.equal(via_old, old))
        del x, old32, old
    del iw, deq
    torch.cuda.empty_cache()

# -------------------------------------------------------------------------------------- 2. W8A16, split-K format
W8_SHAPES = [(3584, 7168), (7168, 6144), (7168, 3584), (2112, 7168), (12288, 128)]   # latent down first
for N, K in W8_SHAPES:
    wf = torch.randn(N, K, device=dev, generator=g) * 0.02
    wf[torch.rand(N, K, device=dev, generator=g) < 0.002] *= 8              # some heavy-tailed groups
    q, s = w8a16.quantize(wf)
    del wf
    wd = (q.double().view(N, K // 128, 128) * s.double()[..., None]).view(N, K)
    w2 = bigm.w8_dequant(q, s)
    hi, lo = w2[:, :K], w2[:, K:]
    check(f"W8A16 {N}x{K}: hi + lo = q * s exactly, hi = BF16(q * s)",
          torch.equal(hi.double() + lo.double(), wd) and torch.equal(hi, wd.float().to(torch.bfloat16)))
    del w2, hi, lo
    for M in (300, 1024, 4096):
        base = torch.randn(M, K + 128, device=dev, generator=g).to(torch.bfloat16)
        for strided in (False, True):
            x = base[:, 64:64 + K] if strided else base[:, :K].contiguous()
            old = w8a16.linear(x, q, s)                          # today's 64-row tiles (M > 16)
            r = rows(M)
            ref = x[r].double() @ wd.t()
            for impl in ("triton", "hilo", "cublas"):
                y = bigm.w8_linear(x, q, s, None, impl=impl)
                y2 = bigm.w8_linear(x, q, s, None, impl=impl)
                ok = rel(y, old) < 2e-3 and torch.equal(y, y2) and rel(y[r], ref) < 8e-3
                check(f"W8A16 {impl} {N}x{K} M {M}{' strided x' if strided else ''}", ok,
                      f"(vs 64-row tiles {rel(y, old):.1e}, bit-equal {same_share(y, old):.3f}; vs fp64 "
                      f"{rel(y[r], ref):.1e}; rerun equal {torch.equal(y, y2)})")
            if M == 1024 and not strided:
                with runtime_flags(K31_W8A16_BIG_M="1"):
                    a = w8a16.linear(x, q, s)
                    b = w8a16.linear(x[:1000], q, s)             # below K31_W8A16_BIG_M_ROWS: today's tiles
                with runtime_flags(K31_W8A16_BIG_M="1", K31_W8A16_BIG_M_ROWS="512"):
                    c = w8a16.linear(x[:600], q, s)
                big_ok = torch.equal(a, bigm.w8_linear(x, q, s, None, impl="triton")) if K >= 1024 else \
                    torch.equal(a, old)
                check(f"W8A16 {N}x{K}: w8a16.linear follows K31_W8A16_BIG_M / _ROWS / _MIN_K",
                      big_ok and torch.equal(b, w8a16.linear(x[:1000], q, s)) and
                      (torch.equal(c, bigm.w8_linear(x[:600], q, s, None, impl="triton")) if K >= 1024 else True))
            del x, old
        del base
    del q, s, wd
    torch.cuda.empty_cache()

# -------------------------------------------------------------------------------------- 3. W8A16, Marlin format
M8_SHAPES = [(7168, 6144), (3584, 7168), (12288, 7168)]                 # the shared down first
for N, K in M8_SHAPES:
    wf = torch.randn(N, K, device=dev, generator=g) * 0.02
    q, s = w8a16.quantize(wf)
    del wf
    weff = (q.float().view(N, K // 128, 128) * s.float()[..., None]).view(N, K).to(torch.bfloat16)
    mw = w8a16.MarlinW(q.clone(), s.clone())
    del q, s
    check(f"Marlin {N}x{K}: bigm dequant = BF16(q * s) of the real repack", torch.equal(bigm.marlin_dequant(mw), weff))
    for M in (300, 1024, 4096):
        x = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        old = mw(x)                                              # Marlin's own GEMM
        r = rows(M)
        ref = x[r].double() @ weff.double().t()
        for impl in ("triton", "cublas"):
            y = bigm.marlin_linear(x, mw, None, impl=impl)
            y2 = bigm.marlin_linear(x, mw, None, impl=impl)
            ok = rel(y, old) < 2e-3 and torch.equal(y, y2) and rel(y[r], ref) < 4e-3
            check(f"Marlin {impl} {N}x{K} M {M}", ok,
                  f"(vs Marlin {rel(y, old):.1e}, bit-equal {same_share(y, old):.3f}; vs fp64 {rel(y[r], ref):.1e}; "
                  f"rerun equal {torch.equal(y, y2)})")
        if M == 1024:
            with runtime_flags(K31_W8A16_BIG_M="1"):
                a = w8a16.linear(x, mw, None)
                b = w8a16.linear(x[:16], mw, None)
            check(f"Marlin {N}x{K}: w8a16.linear follows K31_W8A16_BIG_M",
                  torch.equal(a, bigm.marlin_linear(x, mw, None, impl="triton")) and torch.equal(b, mw(x[:16])))
        del x, old
    del mw, weff
    torch.cuda.empty_cache()

# -------------------------------------------------------------------------------------------- 4. the config table
N, K, M = 3584, 7168, 1024
q, s = w8a16.quantize(torch.randn(N, K, device=dev, generator=g) * 0.02)
x = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"w8": {f"{N}x{K}": {"512": {"impl": "hilo", "cfg": [128, 64, 64, 8, 4, 4]}}}}, f)
with runtime_flags(K31_BIGM_CFG=f.name):
    a = bigm.w8_linear(x, q, s, None, impl="1")
    picked = bigm.choice("w8", N, K, M, "1")
b = bigm.w8_linear(x, q, s, None, impl="hilo", cfg=(128, 64, 64, 8, 4, 4))
os.unlink(f.name)
check("K31_BIGM_CFG: '1' follows the table's implementation and config",
      picked == ("hilo", (128, 64, 64, 8, 4, 4)) and torch.equal(a, b), f"(picked {picked})")

# --------------------------------------------------------------------------- 5. PDL launches (K31_PDL=1 servers)
pdl = []
iw = w8i.I8W((torch.randn(7168, 3584, device=dev, generator=g) * 0.02).to(torch.bfloat16))
mw = w8a16.MarlinW(*w8a16.quantize(torch.randn(7168, 6144, device=dev, generator=g) * 0.02))
xs = [torch.randn(1024, k, device=dev, generator=g).to(torch.bfloat16) for k in (3584, 7168, 6144)]
calls = [lambda: bigm.i8_linear(xs[0], iw, None, impl="triton"), lambda: bigm.i8_linear(xs[0], iw, None, impl="unpack"),
         lambda: bigm.w8_linear(xs[1], q, s, None, impl="triton"), lambda: bigm.w8_linear(xs[1], q, s, None, impl="hilo"),
         lambda: bigm.w8_linear(xs[1], q, s, None, impl="cublas"),
         lambda: bigm.marlin_linear(xs[2], mw, None, impl="triton"),
         lambda: bigm.marlin_linear(xs[2], mw, None, impl="cublas")]
plain = [c() for c in calls]
bigm.PDL = True
try:
    with_pdl = [c() for c in calls]
    torch.cuda.synchronize()
finally:
    bigm.PDL = False
check("PDL launches (griddepcontrol at entry) = plain launches, every implementation",
      all(torch.equal(a, b) for a, b in zip(plain, with_pdl)))

print("BIGM:", "PASS" if not fails else f"FAIL ({len(fails)}: {', '.join(fails[:8])})", flush=True)
sys.exit(1 if fails else 0)
