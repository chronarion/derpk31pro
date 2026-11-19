"""k31_engine.bigm without a GPU: the tile maps and the kernels' arithmetic in Triton's interpreter, then every kernel
compiled for the GB300 (sm_103a) at the configs the engine uses.

Run in the engine image, CPU only (from engine/):
    docker run --rm --network none --cpuset-cpus 60-61 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_bigm_cpu.py
Exit 0 and "BIGM CPU: PASS" =
1. Marlin's 8-bit byte map (bigm.marlin_byte_index: the bit map _m8_codes reads) equals a direct implementation of
   vLLM's permutation (marlin_permute_weights with get_weight_perm(8), packed 4 bytes per int32), and the scale map
   equals marlin_permute_scales' column order, on several shapes.
2. In Triton's interpreter (TRITON_INTERPRET=1; its BF16 dot is unusable, so the kernels' INTERP switch multiplies in
   float32: the products of BF16 values are exact either way), float32 outputs against float64 references:
   int8 GEMM on packed codes (spw 4 / 2 / 1, even and masked N, M not a multiple of the tile, strided out), the
   int8 unpack (bit-equal to w8i.unpack), the column scale; W8A16 split-K format by promotion and by hi/lo (strided
   x, masked N), the hi/lo dequant (hi + lo = q * s exactly) and its cuBLAS form (torch on the CPU); Marlin format
   GEMM and dequant (bit-equal to BF16(q * s)).
3. Each kernel compiled for sm_103a with the JIT's divisibility hints at its default config and every candidate the
   timing script sweeps (no GPU needed): shared memory within the 227 KiB limit, no local memory (spills), registers
   reported.
"""
import os
import subprocess
import sys
import types

if len(sys.argv) < 2 or sys.argv[1] != "aot":
    os.environ["TRITON_INTERPRET"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def bf16_rtz(t: torch.Tensor) -> torch.Tensor:
    """float32 -> BF16 by truncation: the interpreter's conversion (it ignores fp_downcast_rounding; compiled Triton
    rounds to nearest even, which the GPU test checks)."""
    return t.float().contiguous().view(torch.int32).bitwise_and(-65536).view(torch.float32).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------- 1. Marlin byte map
def vllm_perm8():
    """get_weight_perm(num_bits=8, is_a_8bit=False), written out."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    return np.array(perm).reshape(-1, 4)[:, [0, 2, 1, 3]].ravel()


def marlin_bytes_ref(qw: np.ndarray) -> np.ndarray:
    """q_w [K, N] (one byte per weight, k-major as GPTQ) -> Marlin's bytes [K / 16, 16 N] in memory order
    (marlin_permute_weights with 16 x 16 tiles, then the 1,024-entry permutation; marlin_weights packs bytes 4j..4j+3
    of a row into int32 j, little-endian, so the byte order is the element order)."""
    K, N = qw.shape
    t = qw.reshape(K // 16, 16, N // 16, 16).transpose(0, 2, 1, 3).reshape(K // 16, N * 16)
    return t.reshape(-1, 1024)[:, vllm_perm8()].reshape(K // 16, N * 16)


def marlin_scales_ref(s_kn: np.ndarray) -> np.ndarray:
    """marlin_permute_scales for group 128 < K: rows of 64 columns permuted by scale_perm."""
    perm = [i + 8 * j for i in range(8) for j in range(8)]
    G, N = s_kn.shape
    return s_kn.reshape(-1, 64)[:, perm].reshape(G, N)


def part1():
    from k31_engine import bigm
    for K, N in [(128, 64), (256, 192), (384, 512), (1536, 7168 // 4)]:
        idx = np.arange(K * N, dtype=np.int64).reshape(K, N)
        lay = marlin_bytes_ref(idx)                              # lay[p] = flat (k, n) index stored at byte p
        pos = np.empty(K * N, dtype=np.int64)
        pos[lay.ravel()] = np.arange(K * N)
        mine = bigm.marlin_byte_index(K, N).numpy()
        check(f"Marlin byte map {K}x{N}", np.array_equal(pos.reshape(K, N), mine))
        G = K // 128
        sidx = np.arange(G * N).reshape(G, N)
        sp = marlin_scales_ref(sidx)
        col = bigm.marlin_scale_index(N).numpy()
        check(f"Marlin scale map {K}x{N}", all(np.array_equal(sp[g, col], sidx[g]) for g in range(G)))


# ------------------------------------------------------------------------------------- 2. interpreter arithmetic
def marlin_weight(q: torch.Tensor, s: torch.Tensor):
    """A w8a16.MarlinW stand-in built on the CPU from (q FP8 [N, K], s BF16 [N, K / 128]): Marlin's bytes (as int32
    words) and its permuted scales with 2^120 folded in (exact in BF16)."""
    N, K = q.shape
    qw = q.view(torch.uint8).t().contiguous().numpy()
    b = np.ascontiguousarray(marlin_bytes_ref(qw))
    w = torch.from_numpy(b).view(torch.int32)
    sm = torch.from_numpy(marlin_scales_ref(s.float().t().contiguous().numpy()))
    sm = (sm * 2.0 ** 120).to(torch.bfloat16)
    assert torch.equal(sm.float() * 2.0 ** -120, torch.from_numpy(marlin_scales_ref(s.float().t().contiguous().numpy())))
    return types.SimpleNamespace(w=w.contiguous(), s=sm.contiguous(), N=N, K=K)


def part2():
    from k31_engine import bigm, w8a16, w8i
    g = torch.Generator().manual_seed(11)

    # int8 GEMM on packed codes
    for N, K, cfg in [(128, 128, (32, 64, 64, 2, 4, 3)), (96, 96, (16, 32, 32, 2, 4, 3)),
                      (80, 64, (32, 16, 32, 1, 4, 3)), (192, 128, (64, 128, 32, 2, 4, 3))]:
        w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
        q, s, spw = w8i.quantize(w)
        iw = types.SimpleNamespace(N=N, K=K, q=q, s=s, spw=spw)
        codes = w8i.unpack(q, N, K, spw).float() - 128
        for M in (1, 37, 70):
            x = torch.randn(M, K, generator=g).to(torch.bfloat16)
            ref = (x.double() @ codes.double().t()) * s.double()
            y32 = torch.empty(M, N)
            bigm.i8_linear(x, iw, y32, impl="triton", cfg=cfg)
            yb = bigm.i8_linear(x, iw, None, impl="triton", cfg=cfg)
            ok = rel(y32, ref) < 1e-6 and rel(yb, ref) < 8e-3 and torch.equal(yb, bf16_rtz(y32))
            check(f"int8 GEMM {N}x{K} spw {spw} M {M} cfg {cfg[:3]}", ok,
                  f"(fp32 out {rel(y32, ref):.1e}, bf16 out {rel(yb, ref):.1e})")
        # strided output (a column slice of a wider buffer)
        x = torch.randn(9, K, generator=g).to(torch.bfloat16)
        wide = torch.zeros(9, N + 32)
        bigm.i8_linear(x, iw, wide[:, 16:16 + N], impl="triton", cfg=cfg)
        ref = (x.double() @ codes.double().t()) * s.double()
        check(f"int8 GEMM {N}x{K}: strided out", rel(wide[:, 16:16 + N], ref) < 1e-6 and
              float(wide[:, :16].abs().max()) == 0 and float(wide[:, 16 + N:].abs().max()) == 0)
        # the unpack kernel and the column scale
        cu = bigm.i8_unpack(iw)
        check(f"int8 unpack {N}x{K} spw {spw} = w8i.unpack - 128", torch.equal(cu.float(), codes))
        y32 = torch.randn(5, N, generator=g)
        out = torch.empty(5, N, dtype=torch.bfloat16)
        bigm._scale_cols[(1, triton_cdiv(N, 64))](y32, s, out, 5, N, N, BM=8, BN=64)
        check(f"column scale {N}", torch.equal(out, bf16_rtz(y32 * s)))
    # masked N tiles (the wrapper picks even tiles; the masked path is kept correct for other callers)
    N, K = 160, 64
    w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
    q, s, spw = w8i.quantize(w)
    codes = w8i.unpack(q, N, K, spw).float() - 128
    x = torch.randn(21, K, generator=g).to(torch.bfloat16)
    y = torch.zeros(21, N)
    BM, BN, BK = 16, 64 if spw == 4 else 32, 32
    bigm._i8_mm[(triton_cdiv(21, BM) * triton_cdiv(N, BN),)](x, q, s, y, 21, N, K, K, N, SPW=spw, BM=BM, BN=BN, BK=BK,
                                                             GROUP_M=2, EVEN_N=False, PDL=False, INTERP=True)
    check(f"int8 GEMM masked N ({N} rows, BN {BN}, spw {spw})",
          rel(y, (x.double() @ codes.double().t()) * s.double()) < 1e-6)

    # W8A16 split-K format: promotion, hi/lo, dequant + mm
    for N, K in [(64, 256), (96, 384), (48, 128)]:
        w = torch.randn(N, K, generator=g) * 0.02
        w[torch.rand(N, K, generator=g) < 0.01] *= 9
        q, s = w8a16.quantize(w)
        wd = (q.float().view(N, K // 128, 128) * s.float()[..., None]).view(N, K)
        for M in (3, 45):
            base = torch.randn(M, K + 64, generator=g).to(torch.bfloat16)
            x = base[:, 32:32 + K]                                       # rows strided (stride K + 64)
            ref = x.double() @ wd.double().t()
            for impl, cfg in [("triton", (16, 32, 128, 2, 8, 3)), ("hilo", (16, 32, 64, 2, 4, 3)),
                              ("hilo", (32, 16, 128, 1, 4, 3))]:
                y32 = torch.empty(M, N)
                bigm.w8_linear(x, q, s, y32, impl=impl, cfg=cfg)
                yb = bigm.w8_linear(x, q, s, None, impl=impl, cfg=cfg)
                ok = rel(y32, ref) < 1e-6 and torch.equal(yb, bf16_rtz(y32))
                check(f"W8A16 {impl} {N}x{K} M {M} cfg {cfg[:3]} (strided x)", ok, f"(fp32 out {rel(y32, ref):.1e})")
        w2 = bigm.w8_dequant(q, s)
        exact = torch.equal(w2[:, :K].float() + w2[:, K:].float(), wd) and torch.equal(w2[:, :K], bf16_rtz(wd))
        check(f"W8A16 hi/lo dequant {N}x{K}: hi + lo = q * s exactly, hi = bf16(q * s)", exact)
        x = torch.randn(7, K, generator=g).to(torch.bfloat16)
        yc = bigm.w8_linear(x, q, s, None, impl="cublas")
        check(f"W8A16 cublas form {N}x{K} (CPU torch.mm)", rel(yc, x.double() @ wd.double().t()) < 4e-3,
              f"({rel(yc, x.double() @ wd.double().t()):.1e})")
    # masked N for the promotion and hi/lo kernels
    N, K, M = 40, 256, 19
    q, s = w8a16.quantize(torch.randn(N, K, generator=g) * 0.02)
    wd = (q.float().view(N, K // 128, 128) * s.float()[..., None]).view(N, K)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    ref = x.double() @ wd.double().t()
    y1, y2 = torch.zeros(M, N), torch.zeros(M, N)
    bigm._w8_mm[(triton_cdiv(M, 16) * triton_cdiv(N, 32),)](x, q, s.t().float().contiguous(), y1, M, N, K, K // 128, K, N, BM=16, BN=32,
                                                            GROUP_M=2, EVEN_N=False, PDL=False, INTERP=True)
    bigm._w8_hilo_mm[(triton_cdiv(M, 16) * triton_cdiv(N, 32),)](x, q, s, y2, M, N, K, K // 128, K, N, BM=16, BN=32,
                                                                 BK=64, GROUP_M=2, EVEN_N=False, PDL=False,
                                                                 INTERP=True)
    check("W8A16 promotion / hi/lo, masked N (40 rows, BN 32)", rel(y1, ref) < 1e-6 and rel(y2, ref) < 1e-6,
          f"({rel(y1, ref):.1e}, {rel(y2, ref):.1e})")

    # Marlin format
    for N, K in [(64, 128), (128, 256), (192, 384)]:
        w = torch.randn(N, K, generator=g) * 0.02
        q, s = w8a16.quantize(w)
        mw = marlin_weight(q, s)
        weff = bf16_rtz((q.float().view(N, K // 128, 128) * s.float()[..., None]).view(N, K))
        check(f"Marlin dequant {N}x{K} = BF16(q * s)", torch.equal(bigm.marlin_dequant(mw), weff))
        for M, cfg in [(5, (16, 64, 64, 2, 4, 3)), (33, (32, 64, 128, 1, 4, 3)), (40, (16, 128, 32, 2, 4, 3))]:
            if N % cfg[1]:
                continue
            x = torch.randn(M, K, generator=g).to(torch.bfloat16)
            ref = x.double() @ weff.double().t()
            y32 = torch.empty(M, N)
            bigm.marlin_linear(x, mw, y32, impl="triton", cfg=cfg)
            check(f"Marlin GEMM {N}x{K} M {M} cfg {cfg[:3]}", rel(y32, ref) < 1e-6, f"({rel(y32, ref):.1e})")


def triton_cdiv(a, b):
    return -(-a // b)


# ------------------------------------------------------------------------------------------- 3. AOT for sm_103a
# kernel -> list of (constexprs, num_warps, num_stages); the defaults plus the timing script's sweep candidates
def aot_cases():
    from k31_engine import bigm
    c = []
    for (BM, BN, BK, GM, nw, ns) in [bigm.DEFAULT["i8"], (128, 128, 128, 8, 4, 3), (128, 256, 64, 8, 8, 3),
                                     (128, 64, 64, 8, 4, 4), (64, 128, 64, 8, 4, 3), (128, 128, 64, 8, 8, 4),
                                     (256, 128, 64, 8, 8, 3)]:
        c.append(("_i8_mm", dict(SPW=4, BM=BM, BN=BN, BK=BK, GROUP_M=GM, EVEN_N=True, PDL=True, INTERP=False), nw, ns,
                  {"x_ptr": "*bf16", "q_ptr": "*u8", "s_ptr": "*fp32", "y_ptr": "*bf16"}))
    c.append(("_i8_mm", dict(SPW=1, BM=128, BN=128, BK=64, GROUP_M=8, EVEN_N=True, PDL=True, INTERP=False), 4, 3,
              {"x_ptr": "*bf16", "q_ptr": "*u8", "s_ptr": "*fp32", "y_ptr": "*fp32"}))
    for (BM, BN, BK, GM, nw, ns) in [bigm.DEFAULT["w8"], (128, 128, 128, 8, 8, 2), (128, 64, 128, 8, 4, 3),
                                     (64, 128, 128, 8, 4, 3), (128, 256, 128, 8, 8, 2)]:
        c.append(("_w8_mm", dict(BM=BM, BN=BN, GROUP_M=GM, EVEN_N=True, PDL=True, INTERP=False), nw, ns,
                  {"x_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "y_ptr": "*bf16"}))
    for (BM, BN, BK, GM, nw, ns) in [bigm.DEFAULT["hilo"], (128, 128, 128, 8, 8, 3), (128, 64, 64, 8, 4, 4),
                                     (128, 256, 64, 8, 8, 3)]:
        c.append(("_w8_hilo_mm", dict(BM=BM, BN=BN, BK=BK, GROUP_M=GM, EVEN_N=True, PDL=True, INTERP=False), nw, ns,
                  {"x_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*bf16", "y_ptr": "*bf16"}))
    for (BM, BN, BK, GM, nw, ns) in [bigm.DEFAULT["m8"], (128, 128, 128, 8, 4, 3), (128, 256, 64, 8, 8, 3),
                                     (64, 128, 64, 8, 4, 3), (256, 128, 64, 8, 8, 3), (128, 128, 64, 8, 8, 4)]:
        c.append(("_m8_mm", dict(BM=BM, BN=BN, BK=BK, GROUP_M=GM, PDL=True, INTERP=False), nw, ns,
                  {"x_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*bf16", "y_ptr": "*bf16"}))
    c.append(("_i8_unpack", dict(SPW=4, BN=128, BK=64, EVEN_N=True, INTERP=False), 4, 1,
              {"q_ptr": "*u8", "out_ptr": "*bf16"}))
    c.append(("_scale_cols", dict(BM=32, BN=128), 4, 1, {"y_ptr": "*fp32", "s_ptr": "*fp32", "out_ptr": "*bf16"}))
    c.append(("_w8_dequant", dict(BN=64, BK=128), 4, 1, {"q_ptr": "*fp8e4nv", "s_ptr": "*bf16", "w2_ptr": "*bf16"}))
    c.append(("_m8_dequant", dict(BN=64, BK=64, INTERP=False), 4, 1,
              {"q_ptr": "*fp8e4nv", "s_ptr": "*bf16", "w_ptr": "*bf16"}))
    return c


def part3():
    import inspect
    import tempfile
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from k31_engine import bigm
    cuobjdump = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/cuobjdump")
    for name, cx, nw, ns, ptrs in aot_cases():
        fn = getattr(bigm, name)
        params = list(inspect.signature(fn.fn).parameters)
        sig, attrs = {}, {}
        for i, p in enumerate(params):
            if p in cx:
                sig[p] = "constexpr"
            elif p in ptrs:
                sig[p] = ptrs[p]
                attrs[(i,)] = [["tt.divisibility", 16]]
            else:
                sig[p] = "i32"
                if p not in ("M",):
                    attrs[(i,)] = [["tt.divisibility", 16]]
        tag = f"{name} {','.join(f'{k}={v}' for k, v in cx.items() if k not in ('PDL', 'INTERP'))} w{nw} s{ns}"
        try:
            c = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cx, attrs=attrs),
                               target=GPUTarget("cuda", 103, 32), options={"num_warps": nw, "num_stages": ns})
        except Exception as e:                                   # noqa: BLE001
            check(f"AOT {tag}", False, str(e).splitlines()[-1][:300] if str(e) else repr(e))
            continue
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(c.asm["cubin"])
            f.flush()
            res = subprocess.run([cuobjdump, "--dump-resource-usage", f.name], capture_output=True, text=True).stdout
        reg = res.split("REG:")[1].split()[0] if "REG:" in res else "?"
        local = res.split("LOCAL:")[1].split()[0] if "LOCAL:" in res else "?"
        smem = c.metadata.shared
        ok = smem <= 232448 and local in ("0", "?")
        check(f"AOT {tag}", ok, f"(smem {smem}, tmem {c.metadata.tmem_size}, regs {reg}, local {local})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aot":
        part3()
        print("AOT:", "PASS" if not fails else f"FAIL ({len(fails)})", flush=True)
        sys.exit(1 if fails else 0)
    part1()
    part2()
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "aot"], env=env)
    if r.returncode != 0:
        fails.append("AOT compile (see above)")
    print("BIGM CPU:", "PASS" if not fails else f"FAIL ({', '.join(fails)})", flush=True)
    sys.exit(1 if fails else 0)
