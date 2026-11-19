"""K31_MEGAMOE_NOPREP (k31_engine/noprep.py, csrc/noprep_plan.cu): CPU checks, no GPU.

Run in the engine image, CPU only (from the repository root):
    docker run --rm --network none --cpuset-cpus 62-63 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD"/engine:/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_noprep_cpu.py
Exit 0 and "NOPREP CPU: PASS" = every part passed (NOPREP_CPU_PARTS=1,3 runs some). NOPREP_CPU_BUILD=1 also builds
the plan kernel's torch extension (nvcc, ~1-2 min) and compares its scoring SASS with vLLM's grouped top-k kernel.

1. The plan kernel's unitk part, transliterated phase by phase from csrc/noprep_plan.cu (model_plan), against unitk's
   own Triton kernel run by Triton's CPU interpreter (TRITON_INTERPRET=1, the PDL calls stubbed): row ids, stage list,
   pick counts and pending rows equal, residency scores equal to float32 rounding of the adds' order, on random calls
   (ties, experts repeated across rows, invalid picks, strong 1-4, resmax caps, T 1-16, three residency modes).
2. The plan kernel's top-16 selection (the 16th largest lane maximum as a threshold, the candidates ranked against
   each other; past 64 candidates 16 rounds of the warp maximum), modelled the same way, against a sort of vLLM's keys
   (twiddled selection score << 32 | 65535 - id): random scores, all equal, the top in two lanes, many ties, -inf
   scores, almost nothing finite.
3. The x half: noprep.linear_mx (the latent down with the MXFP8 epilogue) against w8a16.linear + vLLM's
   prepare_megamoe_inputs, all in the interpreter, at 3584 x 7168 with the engine's latent down config (BN 32, split 4)
   and two others (BN 64 / split 1, BN 128 / split 2): bf16 rows, FP8 codes and scale words byte-equal; and the
   epilogue alone (noprep._mx_store) on rows built for the scale's edges (zeros, group maxima of exactly 448 x 2^k and
   one bf16 step either side, tiny and huge rows). The interpreter's float arithmetic is not the GPU's
   (tests/test_noprep.py checks that); this checks the layout, the grouping and the byte order.
4. Triton compiles noprep._w8a16_mx for sm_103 (no GPU needed); its PTX quantizes with the prep kernel's instructions
   (div.full.f32, cvt.rn.satfinite.e4m3x2.f32).
"""
import os
import random
import struct
import sys
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ENGINE = os.environ.get("ENGINE", "/engine")
sys.path.insert(0, ENGINE)
from k31_engine import noprep, w8a16  # noqa: E402
from k31_engine.noprep import _mx_store  # noqa: E402

fails = []
E, K, SP, SLOTS = 896, 16, 4, 16
PREP = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/prepare_megamoe.py"


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def real_prep_module():
    """vLLM's prepare_megamoe.py on the real Triton (vLLM swaps in placeholders when no GPU is visible), loaded from a
    copy in /tmp (Triton's JIT reads a kernel's source from its file)."""
    import importlib.util
    import tempfile
    src = open(PREP).read().replace("from vllm.triton_utils import tl, triton",
                                    "import triton\nimport triton.language as tl")
    path = os.path.join(tempfile.mkdtemp(prefix="k31_noprep_"), "prepare_megamoe_real.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("prepare_megamoe_real", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cpu_stubs():
    """The interpreter has no griddepcontrol and no CUDA stream: no-ops / a stream id of 0."""
    tl.extra.cuda.gdc_wait = lambda *a, **k: None
    tl.extra.cuda.gdc_launch_dependents = lambda *a, **k: None
    torch.cuda.current_stream = lambda *a, **k: types.SimpleNamespace(cuda_stream=0)


# ------------------------------------------------------------------------------------- 1. unitk, the kernel's model
def model_plan(ids, w, row_map, bank_map, strong, S, resmax, score=None, decay=1.0, stats=None, pend=None,
               allrows=False):
    """csrc/noprep_plan.cu phases 2a-3 in Python (float32 where the kernel computes in float)."""
    T = len(ids)
    f32 = np.float32
    pk = []
    for t in range(T):
        for k in range(K):
            i, wv = ids[t][k], f32(w[t][k])
            valid = 0 <= i < E
            e = i if valid else 0
            native = row_map[e] if valid else -1
            bank = bank_map[e] if valid else -1
            rank = sum(1 for j in range(K) if f32(w[t][j]) > wv or (f32(w[t][j]) == wv and j < k))
            pk.append(dict(valid=valid, e=e, native=native, bank=bank, rank=rank, w=wv,
                           cand=valid and native < 0 and bank >= 0 and rank < strong))
    ent = []
    for t in range(T):
        for r in range(SP):
            ce, cw, cc, cb = 0, f32(0), False, 0
            for k in range(K):
                p = pk[t * K + k]
                if p["rank"] == r:
                    ce, cw, cc, cb = p["e"], p["w"], p["cand"], p["bank"]
            ent.append(dict(e=ce, w=cw, c=cc, b=cb))
    NE = len(ent)
    for n in range(NE):
        first, sc = False, f32(-np.inf)
        if ent[n]["c"]:
            first = True
            for m in range(NE):
                if ent[m]["c"] and ent[m]["e"] == ent[n]["e"]:
                    if m < n:
                        first = False
                    sc = max(sc, ent[m]["w"])
        ent[n]["first"], ent[n]["score"] = first, sc
    for n in range(NE):
        slot = sum(1 for m in range(NE) if ent[m]["first"] and (ent[m]["score"] > ent[n]["score"] or (
            ent[m]["score"] == ent[n]["score"] and ent[m]["e"] < ent[n]["e"])))
        ent[n]["slot"], ent[n]["ok"] = slot, ent[n]["first"] and slot < SLOTS
    stage, hold = [-1] * SLOTS, [-1] * SLOTS
    for n in range(NE):
        full = -1
        if ent[n]["c"]:
            for m in range(NE):
                if ent[m]["ok"] and ent[m]["c"] and ent[m]["e"] == ent[n]["e"]:
                    full = ent[m]["slot"]
        ent[n]["full"] = full
        if ent[n]["ok"]:
            stage[ent[n]["slot"]] = ent[n]["b"]
            hold[ent[n]["slot"]] = ent[n]["e"]
    rows = [[0] * K for _ in range(T)]
    n_res = n_stg = n_all = 0
    if score is not None and pend is None:
        for e in range(E):
            score[e] = f32(score[e]) * f32(decay)
    for t in range(T):
        for k in range(K):
            p = pk[t * K + k]
            served = p["valid"] and p["native"] >= 0 and p["rank"] < resmax
            out = p["native"] if served else -1
            if allrows:
                sidx = -1
                for s_ in range(SLOTS):
                    if hold[s_] >= 0 and hold[s_] == p["e"]:
                        sidx = s_
                if p["valid"] and p["native"] < 0 and sidx >= 0:
                    out = S + sidx
            else:
                for r in range(SP):
                    col = ent[t * SP + r]["full"]
                    if p["cand"] and p["rank"] == r and col >= 0:
                        out = S + col
            rows[t][k] = out
            if score is not None:
                if pend is not None:
                    pend[0][t * K + k] = ids[t][k] if p["valid"] else -1
                    pend[1][t * K + k] = p["w"] * p["w"] if p["valid"] else f32(0)
                elif p["valid"]:
                    score[p["e"]] = f32(score[p["e"]] + p["w"] * p["w"])
                n_all += p["valid"]
                n_res += served
                n_stg += S <= out < S + SLOTS and p["native"] < 0 and p["valid"]
    if score is not None:
        stats[0] += n_res
        stats[1] += n_stg
        stats[2] += n_all - n_res - n_stg
        stats[3] += n_all
        if pend is not None:
            pend[2][0] = T
    return rows, stage


def part1():
    from k31_engine import unitk
    rng = random.Random(1)
    bad = n = bad_all = changed = 0
    for trial in range(300):
        T = [1, 2, 3, 4, 5, 6, 8, 11, 13, 16][trial % 10]
        strong = 1 + trial % 4
        resmax = [16, 16, 12, 9][(trial // 4) % 4]
        perm = list(range(E))
        rng.shuffle(perm)
        nres, nbank = rng.randrange(0, 400), rng.randrange(0, 450)
        row_map, bank_map = [-1] * E, [-1] * E
        for j, e in enumerate(perm[:nres]):
            row_map[e] = 128 + j
        for j, e in enumerate(perm[nres:nres + nbank]):
            bank_map[e] = j
        pool = rng.choice([20, 40, 120, E])                     # small pools repeat experts across rows
        ids = [rng.sample(range(pool), K) for _ in range(T)]
        if trial % 7 == 3:
            ids[0][5] = -1                                       # an invalid pick
        w = [[rng.choice([0.125, 0.25, 0.375, 0.5, rng.random()]) for _ in range(K)] for _ in range(T)]   # ties
        S = 16 * (trial % 8)
        mode = trial % 3                                         # 0 plain, 1 scores + counts, 2 pending rows
        score0 = [rng.random() for _ in range(E)]
        lam = 0.97
        kw = {}
        if mode:
            kw.update(score=torch.tensor(score0, dtype=torch.float32), stats=torch.zeros(4, dtype=torch.int32),
                      lam=lam)
        if mode == 2:
            kw["pend"] = (torch.full((16, 16), -7, dtype=torch.int32), torch.full((16, 16), -7.0),
                          torch.zeros(1, dtype=torch.int32))
        out, stage = unitk.plan(torch.tensor(ids, dtype=torch.int32), torch.tensor(w, dtype=torch.float32),
                                torch.tensor(row_map, dtype=torch.int32), torch.tensor(bank_map, dtype=torch.int32),
                                strong, S, resmax=resmax, **kw)
        score = [np.float32(v) for v in score0] if mode else None
        stats = [0, 0, 0, 0] if mode else None
        pend = ([-7] * 256, [np.float32(-7.0)] * 256, [0]) if mode == 2 else None
        rows, stg = model_plan(ids, w, row_map, bank_map, strong, S, resmax, score, np.float32(float(lam ** T)), stats,
                               pend)
        ok = out.tolist() == rows and stage.tolist() == stg
        if mode:
            ok = ok and kw["stats"].tolist() == stats and \
                np.allclose(kw["score"].numpy(), np.array(score, dtype=np.float32), rtol=1e-6, atol=0)
        if mode == 2:
            ok = ok and kw["pend"][0].reshape(-1)[:T * K].tolist() == pend[0][:T * K] and \
                np.array_equal(kw["pend"][1].reshape(-1)[:T * K].numpy(), np.array(pend[1][:T * K], np.float32)) \
                and int(kw["pend"][2][0]) == T
        n += 1
        if not ok:
            bad += 1
            if bad <= 3:
                print(f"  trial {trial} T={T} strong={strong} resmax={resmax} mode={mode}: rows equal "
                      f"{out.tolist() == rows}, stage {stage.tolist()} vs {stg}")
        # K31_STAGE_ALL (unitk's ALLROWS): the reference from its definition on unitk's own output: every valid
        # non-resident pick of an expert that holds a slot -> that slot's row; everything else as without it
        inv_bank = {j: e for e, j in enumerate(bank_map) if j >= 0}
        held = {inv_bank[b]: s_ for s_, b in enumerate(stage.tolist()) if b >= 0}
        want = [[S + held[ids[t][k]] if (0 <= ids[t][k] < E and row_map[ids[t][k]] < 0 and ids[t][k] in held)
                 else out[t][k].item() for k in range(K)] for t in range(T)]
        rows_a, stg_a = model_plan(ids, w, row_map, bank_map, strong, S, resmax, allrows=True)
        if rows_a != want or stg_a != stage.tolist():
            bad_all += 1
            if bad_all <= 3:
                nd = sum(a != b for ra, rb in zip(rows_a, want) for a, b in zip(ra, rb))
                print(f"  ALLROWS trial {trial} T={T}: {nd} picks differ")
        changed += sum(a != b for ra, rb in zip(rows_a, out.tolist()) for a, b in zip(ra, rb))
    check("1. the plan kernel's unitk part (model) = unitk's Triton kernel (interpreter)", bad == 0,
          f"({n} calls, {bad} differ)")
    check("1b. ALLROWS (K31_STAGE_ALL) in the plan kernel's model = the rule applied to unitk's output", bad_all == 0
          and changed > 0, f"({n} calls, {bad_all} differ; {changed} picks newly on a slot)")


# ------------------------------------------------------------------------------------- 2. the top-16 selection
def twiddle(f):
    b = struct.unpack("<I", struct.pack("<f", float(f)))[0]
    return b ^ (0xFFFFFFFF if b & 0x80000000 else 0x80000000)


def model_select(sel):
    """The kernel's phase 1b for one row: sel = the 896 twiddled selection scores -> the top 16 keys, descending."""
    keys = [(sel[e] << 32) | (65535 - e) for e in range(E)]
    lanes = [[keys[lane + 32 * i] for i in range(E // 32)] for lane in range(32)]
    m = [max(L) for L in lanes]
    r = [sum(1 for mj in m if mj > ml) for ml in m]
    tau = m[r.index(K - 1)]
    cand = [kk for L in lanes for kk in L if kk >= tau]          # lane-major, as the kernel's scan places them
    if len(cand) <= 64:
        top = [None] * K
        for kj in cand:
            rank = sum(1 for kq in cand if kq > kj)
            if rank < K:
                top[rank] = kj
        return top, len(cand)
    taken = [set() for _ in range(32)]
    top = []
    for _ in range(K):
        lm, li = [], []
        for lane, L in enumerate(lanes):
            best, bi = 0, -1
            for i, kk in enumerate(L):
                if i not in taken[lane] and kk > best:
                    best, bi = kk, i
            lm.append(best)
            li.append(bi)
        gm = max(lm)
        for lane in range(32):
            if li[lane] >= 0 and lm[lane] == gm:
                taken[lane].add(li[lane])
        top.append(gm)
    return top, len(cand)


def part2():
    rng = random.Random(2)
    bad = n = big = 0
    for trial in range(600):
        kind = trial % 6
        if kind == 0:
            s = [rng.gauss(0, 1) for _ in range(E)]
        elif kind == 1:
            s = [0.5] * E                                         # every score equal: the ids decide
        elif kind == 2:
            s = [rng.gauss(0, 1) - (0 if e % 32 < 2 else 5) for e in range(E)]     # the top in two lanes
        elif kind == 3:
            s = [round(rng.gauss(0, 1), 1) for _ in range(E)]    # many ties
        elif kind == 4:
            s = [rng.gauss(0, 1) if rng.random() < 0.9 else float("-inf") for _ in range(E)]
            for e in rng.sample(range(E), 3):
                s[e] = 0.0
        else:
            s = [float("-inf")] * E                               # almost nothing finite
            for e in rng.sample(range(E), 5):
                s[e] = rng.gauss(0, 1)
        sel = [twiddle(np.float32(v)) for v in s]
        top, C = model_select(sel)
        want = sorted(((sel[e] << 32) | (65535 - e) for e in range(E)), reverse=True)[:K]
        big += C > 64
        n += 1
        bad += top != want
    check("2. the plan kernel's top-16 selection (model) = a sort of vLLM's keys", bad == 0 and big > 0,
          f"({n} rows, {big} past 64 candidates, {bad} differ)")


# ------------------------------------------------------------------------------------- 3. the x half
@triton.jit
def _epilogue_only(y_ptr, xq_ptr, xs_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    """noprep._mx_store on given bf16 rows, one program per BN columns."""
    pn = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    out = tl.load(y_ptr + rm[:, None] * N + rn[None, :], mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    _mx_store(out, rm, rn, pn, M, N, xq_ptr, N, xs_ptr, N // 128, 1, BM, BN)


def _prep_x(pm, y):
    M, N = y.shape
    xr = torch.zeros(16, N, dtype=torch.float8_e4m3fn)
    xsr = torch.zeros(16, N // 128, dtype=torch.int32)
    pm.prepare_megamoe_inputs(y, torch.zeros(M, K), torch.zeros(M, K, dtype=torch.int32), xr[:M], xsr[:M],
                              torch.zeros(M, K, dtype=torch.int64), torch.zeros(M, K))
    return xr[:M], xsr[:M]


def _eq8(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def part3():
    pm = real_prep_module()
    noprep._quantize_fp8_groups = pm._quantize_fp8_groups        # the real Triton function (placeholder without GPU)
    g = torch.Generator().manual_seed(3)
    N, Kd = 3584, 7168
    q, s = w8a16.quantize(torch.randn(N, Kd, generator=g) * 0.02)
    bad = n = 0
    key = f"{N}x{Kd}"
    keep = w8a16._OVR.get(key)
    for cfg in ((32, 1792, 4, 4, 3), (64, 7168, 1, 4, 3), (128, 3584, 2, 4, 3)):
        w8a16._OVR[key] = cfg
        for M in (1, 3, 5, 16):
            x = (torch.randn(M, Kd, generator=g) * 2).to(torch.bfloat16)
            if M >= 3:
                x[1] = 0                                          # every group's amax below 1e-4: the clamp
                x[2] *= 1e-6
            xq = torch.zeros(16, N, dtype=torch.float8_e4m3fn)
            xs = torch.zeros(16, N // 128, dtype=torch.int32)
            y = noprep.linear_mx(x, q, s, xq, xs)
            y_ref = w8a16.linear(x, q, s)
            xr, xsr = _prep_x(pm, y_ref)
            same = _eq8(y, y_ref) and _eq8(xq[:M], xr) and torch.equal(xs[:M], xsr)
            n += 1
            if not same:
                bad += 1
                print(f"  cfg {cfg} M={M}: y equal {_eq8(y, y_ref)}, codes differ "
                      f"{int((xq[:M].view(torch.uint8) != xr.view(torch.uint8)).sum())}, scale words differ "
                      f"{int((xs[:M] != xsr).sum())}")
    if keep is None:
        w8a16._OVR.pop(key, None)
    else:
        w8a16._OVR[key] = keep
    check("3a. x half: latent down + MXFP8 epilogue = w8a16 + vLLM's prep (interpreter, byte-equal)", bad == 0,
          f"({n} calls over 3 configs)")
    # the scale's edges through the epilogue alone
    rows = torch.randn(8, N, generator=g).float()
    rows[0] = 0
    for gi in range(N // 32):
        k = gi % 40 - 20                                          # the other values stay below 448 x 2^k: |x| < 2^(k+5)
        for r in (1, 3, 4):
            rows[r, 32 * gi:32 * gi + 32] *= 2.0 ** (k + 2)
        rows[1, 32 * gi + gi % 32] = 448.0 * 2.0 ** k             # group maximum exactly 448 x 2^k
        rows[2, 32 * gi:32 * gi + 32] = -1.75 * 2.0 ** k          # all equal, 448 x 2^(k - 8), negative
        rows[3, 32 * gi + 3] = 448.0 * 2.0 ** k + 2.0 ** (k + 1)  # one bf16 step above 448 x 2^k
        rows[4, 32 * gi + 7] = 448.0 * 2.0 ** k - 2.0 ** (k + 1)  # one bf16 step below
    rows[5] *= 1e-30
    rows[6] *= 1e37
    rows = rows.to(torch.bfloat16)
    for BM, BN in ((8, 32), (8, 64)):
        xq = torch.zeros(8, N, dtype=torch.float8_e4m3fn)
        xs = torch.zeros(8, N // 128, dtype=torch.int32)
        _epilogue_only[(N // BN,)](rows, xq, xs, 8, N, BM=BM, BN=BN)
        xr, xsr = _prep_x(pm, rows)
        ok = _eq8(xq, xr) and torch.equal(xs, xsr)
        check(f"3b. epilogue on the scale's edges (BN {BN}) = vLLM's prep (interpreter, byte-equal)", ok,
              f"(codes differ {int((xq.view(torch.uint8) != xr.view(torch.uint8)).sum())}, "
              f"scale words differ {int((xs != xsr).sum())})")


# ------------------------------------------------------------------------------------- 4. compile for sm_103
def part4():
    import subprocess
    code = r'''
import os, re, sys
sys.path.insert(0, %r)
sys.path.insert(0, %r)
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import test_noprep_cpu as T
from k31_engine import noprep
noprep._quantize_fp8_groups = T.real_prep_module()._quantize_fp8_groups
fn = noprep._w8a16_mx
sig = {"x_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*bf16", "ws_ptr": "*fp32", "cnt_ptr": "*i32", "y_ptr": "*bf16",
       "M": "i32", "N": "i32", "K": "i32", "NG": "i32", "sxm": "i32", "xq_ptr": "*fp8e4nv", "xq_sm": "i32",
       "xs_ptr": "*i32", "xs_sm": "i32", "xs_sk": "i32", "BM": "constexpr", "BN": "constexpr", "KCH": "constexpr",
       "SPLIT": "constexpr"}
names = fn.arg_names
cx = {(names.index("BM"),): 16, (names.index("BN"),): 32, (names.index("KCH"),): 1792, (names.index("SPLIT"),): 4}
k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cx), target=GPUTarget("cuda", 103, 32),
                   options={"num_warps": 4, "num_stages": 3})
ptx = k.asm["ptx"]
print("DIV", sorted(set(re.findall(r"div\.[a-z0-9.]+", ptx))))
print("CVT8", sorted(set(re.findall(r"cvt\.[a-z.]*e4m3[a-z0-9.]*", ptx))))
print("BYTE_STORES", len(re.findall(r"st\.global[a-z0-9.]*\.(?:b8|u8)", ptx)))
'''
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, "-c", code % (ENGINE, here)], capture_output=True, text=True,
                       env=dict(os.environ, TRITON_INTERPRET="0"))
    print("  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode:
        print(r.stderr[-3000:])
    ok = r.returncode == 0 and "DIV ['div.full.f32']" in r.stdout and "cvt.rn.satfinite.e4m3x2.f32" in r.stdout
    check("4. _w8a16_mx compiles for sm_103; quantization with the prep kernel's instructions", ok)


# ------------------------------------------------------------------------------------- 5. the plan kernel's build
def part5():
    """NOPREP_CPU_BUILD=1: the plan kernel's extension builds (nvcc) and its scoring SASS is vLLM's."""
    import glob
    import re
    import subprocess
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
    bdir = "/tmp/k31_noprep_build"
    os.makedirs(bdir, exist_ok=True)
    src = os.path.join(ENGINE, "k31_engine", "csrc", "noprep_plan.cu")
    try:
        load(name="k31_noprep_plan", sources=[src], build_directory=bdir, verbose=False,
             extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_103a,code=sm_103a"])
        built = True
    except Exception as ex:                      # noqa: BLE001
        print(f"  build: {ex}")
        built = False
    check("5a. the plan kernel's extension builds", built)
    if not built:
        return
    env = dict(os.environ, PATH="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin:" +
               os.environ.get("PATH", ""))
    so = glob.glob(os.path.join(bdir, "*.so"))[0]
    mine = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True, env=env).stdout
    lib = "/usr/local/lib/python3.12/dist-packages/vllm/_moe_C_stable_libtorch.abi3.so"
    fn = ("_ZN4vllm3moe17single_group_topk6detail30single_group_topk_block_kernelIffiLNS0_11ScoringFuncE1ELi896ELi16E"
          "EEvPKT_PfPT1_PKT0_llbfb")
    xdir = "/tmp/k31_noprep_vllm"
    os.makedirs(xdir, exist_ok=True)
    subprocess.run(["cuobjdump", "-xelf", "all", lib], cwd=xdir, capture_output=True, env=env)
    ref = ""
    for cub in sorted(glob.glob(os.path.join(xdir, "*sm_100*.cubin"))):
        r = subprocess.run(["cuobjdump", "-sass", "-fun", fn, cub], capture_output=True, text=True, env=env).stdout
        if "MUFU.EX2" in r:
            ref = r
            break

    def consts(sass):
        """Float immediates of FFMA / FMUL / FADD in order of first use (the sigmoid's tanhf, 0.5, 1e-20, ...)."""
        seen = []
        for c in re.findall(r"(?:FFMA|FMUL|FADD)\s[^;]*?(-?\d+\.\d+(?:e[-+]\d+)?)\s*;", sass):
            if c not in seen:
                seen.append(c)
        return seen

    a, b = consts(mine), consts(ref)
    print(f"  float constants, plan kernel: {a}")
    print(f"  float constants, vLLM top-k:  {b}")
    bfly = lambda s: re.findall(r"SHFL\.BFLY PT, R\d+, R\d+, (0x[0-9a-f]+), 0x1f", s)   # noqa: E731
    ok = bool(ref) and set(b) <= set(a) and bfly(mine)[:5] == bfly(ref)[:5] == ["0x10", "0x8", "0x4", "0x2", "0x1"]
    check("5b. the plan kernel's scoring / weight arithmetic = vLLM's top-k (SASS constants, warp-sum order)", ok)


if __name__ == "__main__":
    _cpu_stubs()
    parts = os.environ.get("NOPREP_CPU_PARTS", "1,2,3,4" + (",5" if os.environ.get("NOPREP_CPU_BUILD") == "1" else ""))
    for p in parts.split(","):
        {"1": part1, "2": part2, "3": part3, "4": part4, "5": part5}[p.strip()]()
    print(f"NOPREP CPU: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
