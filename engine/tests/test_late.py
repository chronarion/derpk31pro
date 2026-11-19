"""The late tier's kernels on the GB300, without a pool (K31_LATE; latetier.py, late.py, csrc/late.cu). After ./build.sh
(liblate.so and k31i8.so in engine/band/), from the repo root:

    docker run --rm --network none --device nvidia.com/gpu=$GB --ipc host --cpuset-cpus 40-47 --ulimit core=0 \
      --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/repo -w /repo --entrypoint python3 vllm/vllm-openai:kimi-k31 \
      engine/tests/test_late.py

(K31_LATE_LIB / K31_I8_LIB override the libraries' paths; K31_PDL=1 runs the launches with programmatic dependent
launch as the engine's profiles do.) Checks, exit 0 = PASS:
  1 late.plan on the GPU == plan_reference, 300 calls (T 1..8, D 0..8, rank windows, band maps, ties), the rows built
    as unitk builds them in both of its modes; with ALLROWS (K31_STAGE_ALL) no slotted expert is ever late
  2 late.mean == mean_reference (rel < 1e-6)
  3 late.add: shares added within one bf16 rounding of the float32 formula, rows without a share bit-identical, in place,
    out of place and on a row-strided block state
  4 late_land on device rows: xl == bf16((v ra) g) bit for bit given its coefficients; coefficients against float64
    (rel < 2e-6); mfix; bf16 rows; rows without a share -> has 0, xl 0; two launches bit-equal
  5 the completion through the int8 GEMV (w8i.I8W, float32 out): z + cb y against the float64 definition
    up(norm(u + v)) - up(norm(u)) on the int8 weights' values, shares from 1e-3 to 5x |u| (rel < 1e-2)
  6 plan + mean + landing + GEMV + add captured in a CUDA graph on a side stream (fork / join as the engine does),
    3 replays bit-equal to eager
  7 timing (information only): each kernel alone, 200 launches
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("K31_I8_LIB", os.path.abspath("engine/band/k31i8.so"))
from k31_engine import late, latetier, w8i  # noqa: E402
import late_testlib as tl_  # noqa: E402

dev = torch.device("cuda")
lib = latetier.Lib(os.environ.get("K31_LATE_LIB", "engine/band/liblate.so"))
PDL = os.environ.get("K31_PDL", "0") == "1"
EPS = 1e-6
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def maps(g, nres, nbank, nband):
    return tl_.rand_maps(g, nres, nbank, nband, device=dev)


# 1. plan
bad, bad_allrows = [], []
for trial in range(300):
    g = torch.Generator().manual_seed(trial)
    T = 1 + trial % 8
    nband = int(torch.randint(0, 100, (1,), generator=g)) if trial % 3 == 0 else 0
    row, bank, band = maps(g, int(torch.randint(0, 300, (1,), generator=g)), int(torch.randint(0, 500, (1,), generator=g)),
                           nband)
    pool = int(torch.randint(16, late.E, (1,), generator=g))
    ids = torch.stack([torch.randperm(pool, generator=g)[:16] for _ in range(T)]).to(torch.int32)
    if trial % 5 == 0:
        ids[0, 3] = -1
    ids = ids.to(dev)
    w = (torch.randint(1, 7, (T, 16), generator=g).float() / 8).to(dev)
    allrows = trial % 2 == 1                                          # unitk's two modes (K31_STAGE_ALL)
    rows, slotted = tl_.unitk_rows(ids, w, row, bank, trial % 5, 0, allrows)
    lfrom = int(torch.randint(0, 5, (1,), generator=g)) if trial % 2 else 0
    lto = int(torch.randint(lfrom, 17, (1,), generator=g))
    D = trial % 9
    bm = band if nband else None
    lslot = torch.full((8,), 7, dtype=torch.int32, device=dev)
    lw = torch.full((8, 8), 9.0, device=dev)
    lmask = torch.full((8, 16), 7, dtype=torch.int32, device=dev)
    cnt = torch.zeros(1, dtype=torch.int64, device=dev)
    late.plan(ids, w, row, bank, rows, lfrom, lto, D, lslot, lw, lmask, cnt, band_map=bm, pdl=PDL)
    r_slot, r_w, r_mask, r_n = late.plan_reference(ids, w, row, bank, rows, lfrom, lto, D, band_map=bm)
    if not (torch.equal(lslot, r_slot) and torch.equal(lw, r_w) and torch.equal(lmask[:T], r_mask) and int(cnt) == r_n):
        bad.append(trial)
    owner = {int(b): x for x, b in enumerate(bank.tolist()) if b >= 0}
    if allrows and {owner[b] for b in lslot.tolist() if b >= 0} & set(slotted):
        bad_allrows.append(trial)
check("1 plan == plan_reference, 300 calls (unitk rows, strict and ALLROWS)", not bad and not bad_allrows,
      f"(bad {bad[:8]}, ALLROWS violations {bad_allrows[:8]})")

# 2. mean
g = torch.Generator().manual_seed(11)
table = (torch.randn(late.E, late.H, generator=g) * 0.05).bfloat16().to(dev)
worst = 0.0
for T in (1, 3, 5, 8):
    ids = torch.randint(0, late.E, (T, 16), generator=g).to(dev, torch.int32)
    w = torch.rand(T, 16, generator=g).to(dev)
    lmask = torch.where(torch.rand(T, 16, generator=g) < 0.3, torch.randint(0, 8, (T, 16), generator=g),
                        torch.tensor(-1)).to(dev, torch.int32)
    out = torch.full((8, late.H), 7.0, device=dev)
    late.mean(out, lmask, ids, w, table, pdl=PDL)
    ref = late.mean_reference(lmask, ids, w, table)
    worst = max(worst, float((out[:T].double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)))
check("2 mean == mean_reference", worst < 1e-6, f"({worst:.1e})")

# 3. add
g = torch.Generator().manual_seed(12)
ok3, worst = True, 0.0
for T in (1, 4, 8):
    dst0 = (torch.randn(T, 7168, generator=g) * 0.5).bfloat16().to(dev)
    dst0[0, :7] = -0.0
    z = (torch.randn(T, 7168, generator=g) * 0.01).to(dev)
    y = (torch.randn(T, 7168, generator=g) * 0.3).bfloat16().to(dev)
    coef = torch.zeros(8, 4, device=dev)
    coef[:T, 0] = (torch.arange(T, device=dev) % 2).float()
    coef[:T, 1] = (torch.randn(T, generator=g) * 0.01).to(dev)
    want = torch.where(coef[:T, :1] > 0, dst0.float() + (z + coef[:T, 1:2] * y.float()), dst0.float())
    d = dst0.clone()
    late.add(d, z, y, coef, pdl=PDL)
    o = torch.empty_like(dst0)
    late.add(dst0, z, y, coef, out=o, pdl=PDL)
    blk = torch.zeros(T, 3, 7168, dtype=torch.bfloat16, device=dev)
    blk[:, 1] = dst0
    late.add(blk[:, 1, :], z, y, coef, pdl=PDL)
    nos = coef[:T, 0] == 0
    for t_ in (d, o, blk[:, 1]):
        ok3 &= torch.equal(t_[nos].view(torch.int16), dst0[nos].view(torch.int16))
        ok3 &= bool(((t_.float() - want).abs() <= want.abs() * 2 ** -7 + 1e-30).all())
    ok3 &= torch.equal(blk[:, 0], torch.zeros_like(blk[:, 0])) and torch.equal(blk[:, 2], torch.zeros_like(blk[:, 2]))
check("3 add: within one bf16 ulp of the float32 formula, rows without a share bit-identical (in place / out / "
      "strided)", ok3)

# 4. landing on device rows
g = torch.Generator().manual_seed(13)
gw = (torch.rand(late.H, generator=g) * 1.5 + 0.25).bfloat16().to(dev)
xl = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev)
coef = torch.zeros(8, 4, device=dev)
ok4, worst = True, 0.0
for T in (1, 2, 3, 5, 8):
    for scale in (1e-3, 0.3, 5.0):
        u = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16().to(dev)
        v = (torch.randn(T, late.H, generator=g) * 0.7 * scale).to(dev)
        if T >= 2:
            v[1] = 0.0
        mfix = (torch.randn(T, late.H, generator=g) * 0.05).to(dev) if T == 5 else None
        if mfix is not None:
            mfix[1] = 0.0
        for vin in (v, v.bfloat16()):
            lib.land(None, None, u, vin, mfix, gw, xl, coef, T, EPS, 1, 0.0, pdl=PDL)
            torch.cuda.synchronize()
            vv = vin.float() - (mfix if mfix is not None else 0.0)
            cr = late.coef_reference(u, vv, EPS)
            ok4 &= torch.equal(coef[:T, 0], cr[:, 0].float())
            worst = max(worst, float(((coef[:T, 1:4].double() - cr[:, 1:4]).abs() / cr[:, 2:3].abs()).max()))
            ok4 &= torch.equal(xl[:T].view(torch.int16), late.land_reference(u, vv, gw, coef[:T]).view(torch.int16))
            ok4 &= bool((xl[:T][cr[:, 0] == 0].float() == 0).all())
            x2, c2 = xl.clone(), coef.clone()
            lib.land(None, None, u, vin, mfix, gw, xl, coef, T, EPS, 1, 0.0, pdl=PDL)
            torch.cuda.synchronize()
            ok4 &= torch.equal(x2, xl) and torch.equal(c2, coef)
check("4 landing on device rows: xl bit-exact, coefficients vs float64, mfix, bf16 rows, repeatable",
      ok4 and worst < 2e-6, f"(coefficients worst rel {worst:.1e})")

# 5. the completion through the int8 GEMV
g = torch.Generator().manual_seed(14)
W = (torch.randn(7168, late.H, generator=g) * 0.02).bfloat16().to(dev)
iw = w8i.I8W(W)
W8 = w8i.dequant_ref(W)
z = torch.zeros(8, 7168, device=dev)
worst5 = 0.0
for T in (1, 5, 8):
    for scale in (1e-3, 0.03, 0.3, 1.0, 5.0):
        u = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16().to(dev)
        v = (torch.randn(T, late.H, generator=g) * 0.7 * scale).to(dev)
        un = u.float()
        xn = ((un * torch.rsqrt(un.pow(2).mean(-1, keepdim=True) + EPS)) * gw.float()).bfloat16()
        y = iw(xn)                                                 # the layer's on-time routed output (bf16)
        lib.land(None, None, u, v, None, gw, xl, coef, T, EPS, 1, 0.0, pdl=PDL)
        iw(xl[:T], z[:T])
        torch.cuda.synchronize()
        d = z[:T] + coef[:T, 1:2] * y.float()
        ref = late.delta_reference(u, v, gw, EPS, W8)
        worst5 = max(worst5, rel(d, ref))
check("5 completion z + cb y through the int8 GEMV == up(norm(u + v)) - up(norm(u))", worst5 < 1e-2,
      f"(worst rel {worst5:.1e}: bf16 xl and y)")

# 6. a CUDA graph with the engine's fork / join
g = torch.Generator().manual_seed(15)
T = 5
row, bank, band = maps(g, 200, 400, 0)
ids = torch.stack([torch.randperm(600, generator=g)[:16] for _ in range(T)]).to(dev, torch.int32)
w = torch.rand(T, 16, generator=g).to(dev)
e = ids.long()
rows = torch.where(row[e] >= 0, row[e], -1).to(torch.int32)
u = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16().to(dev)
v = (torch.randn(T, late.H, generator=g) * 0.2).to(dev)
yb = (torch.randn(T, 7168, generator=g) * 0.3).bfloat16().to(dev)
dst = (torch.randn(T, 7168, generator=g) * 0.5).bfloat16().to(dev)
lslot = torch.full((8,), -1, dtype=torch.int32, device=dev)
lw = torch.zeros(8, 8, device=dev)
lmask = torch.full((8, 16), -1, dtype=torch.int32, device=dev)
cnt = torch.zeros(1, dtype=torch.int64, device=dev)
mfix = torch.zeros(8, late.H, device=dev)
side = torch.cuda.Stream()


def chain(out):
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        late.plan(ids, w, row, bank, rows, 0, 16, 5, lslot, lw, lmask, cnt, pdl=PDL)
        late.mean(mfix, lmask, ids, w, table, pdl=PDL)
        lib.land(None, None, u, v, mfix, gw, xl, coef, T, EPS, 1, 0.0, stream=side, pdl=PDL)
        iw(xl[:T], z[:T])
        ev = torch.cuda.Event()
        ev.record(side)
    main.wait_event(ev)
    late.add(dst, z[:T], yb, coef, out=out, pdl=PDL)


eager = torch.empty_like(dst)
chain(eager)
torch.cuda.synchronize()
cap = torch.empty_like(dst)
gr = torch.cuda.CUDAGraph()
s0 = torch.cuda.Stream()
s0.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s0):
    chain(cap)                                                     # warm-up on the capture stream
    torch.cuda.synchronize()
    with torch.cuda.graph(gr, stream=s0):
        chain(cap)
ok6 = True
for _ in range(3):
    cap.zero_()
    gr.replay()
    torch.cuda.synchronize()
    ok6 &= torch.equal(cap.view(torch.int16), eager.view(torch.int16))
check("6 plan + mean + landing + GEMV + add in a CUDA graph (side-stream fork / join): replays == eager", ok6)

# 7. timing
def timeit(fn, n=200):
    for _ in range(10):
        fn()
    s, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e_.record()
    torch.cuda.synchronize()
    return 1e3 * s.elapsed_time(e_) / n


T = 5
t_plan = timeit(lambda: late.plan(ids, w, row, bank, rows, 0, 16, 5, lslot, lw, lmask, cnt, pdl=PDL))
t_mean = timeit(lambda: late.mean(mfix, lmask, ids, w, table, pdl=PDL))
t_land = timeit(lambda: lib.land(None, None, u, v, mfix, gw, xl, coef, T, EPS, 1, 0.0, pdl=PDL))
t_gemv = timeit(lambda: iw(xl[:T], z[:T]))
t_add = timeit(lambda: late.add(dst, z[:T], yb, coef, out=cap, pdl=PDL))
print(f"TIMING (us per launch, T = 5, back to back): plan {t_plan:.1f}, mean {t_mean:.1f}, landing from device rows "
      f"{t_land:.1f}, int8 up GEMV {t_gemv:.1f}, add {t_add:.1f}", flush=True)

print("LATE TEST " + ("PASS" if not fails else f"FAIL ({len(fails)}: {fails})"), flush=True)
sys.exit(0 if not fails else 1)
