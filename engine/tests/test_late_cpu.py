"""The late tier (K31_LATE) on the CPU: no GPU anywhere. Run in the engine image on one core (from the repo root):

    docker run --rm --network none --cpuset-cpus 70 --ulimit core=0 --user $(id -u):$(id -g) -e HOME=/tmp \
      -v "$PWD":/repo -w /repo --entrypoint bash vllm/vllm-openai:kimi-k31 -c \
      'engine/k31_engine/csrc/build_late.sh emu && engine/pool/build.sh && python3 engine/tests/test_late_cpu.py'

What runs where: the Triton kernels (late.plan / mean / add) in Triton's interpreter (TRITON_INTERPRET=1); late.cu's
two kernels compiled for the host (build_late.sh emu: late_emu.h runs one CTA as 512 host threads, the same source as
the GB300 build); the pool is tests/late_fakepool.py (a Python thread, float64 arithmetic), and in E the real one
(engine/pool's libcpustage.so through cpupool.CpuPool, one thread on this core). Checks, exit 0 = PASS:
  A1 plan == plan_reference on 400 random calls (T 1..8, D 0..8, rank windows, band maps, ties), the rows built as
     unitk builds them, in both of its modes; with ALLROWS (K31_STAGE_ALL) no slotted expert is late
  A2 mean == mean_reference (float32 sums, rel < 1e-6)
  A3 add: rows with a share = dst + z + cb y (float32, one rounding); rows without one bit-identical (signed zeros
     too), in place, out of place, row-strided dst
  B1 the send's int8 rows (host twin of quant_block) == quant_rows_reference byte for byte, edge cases included
  B2 the job header bytes == CpuJob2 packed by hand (cpustage_seg2.h layout)
  B3 the landing's coefficients (host twin of row_coef) against float64 (rel < 1e-6), small shares, v = -u
  B4 late_layout == the header's constants; the completion identity in float64 with the prescaled input
  C1 late_send (emulated): the segment's header / rows / exponents == the host twins, door tag, state words; a call
     with no record publishes nothing
  C2 late_land on device rows (emulated): xl == bf16((v ra) g) bit for bit given its coefficients; coefficients
     against float64; mfix; rows without a share give has 0 and xl 0; deterministic
  C3 late_send -> stand-in pool -> late_land: the pool's rows (== pool_reference) land bit-equal to the device-row
     landing of the same rows; counters
  C4 no job, a foreign layer's job, the pool down, the pool silent (timeout), a refused job: has 0, counted
  C5 seven jobs in a row (the slot ring wraps): every landing gets its own job's rows
  D  latetier.Tier end to end (post -> rot -> join) with stand-in layer modules: the landed delta against the float64
     definition up(norm(u + v)) - up(norm(u)), v = pool rows - mfix; the AttnRes block-state path; a job the pool
     does not answer adds nothing (bit-identical attention output); the on-time output out = residual + y unchanged
  E  the real pool (cpupool.CpuPool, libcpustage.so) between the emulated send and landing: the rows of a job the
     "GB300" posted == the pool's single-thread kernel on the same records and rows (cpupool run_single), bit for bit;
     the landing that reads them == the device-row landing; six jobs through the ring; a job the pool holds back
     times out and adds nothing, and the ring goes on
"""
import contextlib
import os
import struct
import sys
import time

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31_engine import late, latetier  # noqa: E402
import late_fakepool as fp  # noqa: E402
import late_testlib as tl_  # noqa: E402

EMU = os.environ.get("LATE_EMU_LIB", "build/late/liblate_emu.so")
torch.set_num_threads(1)
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


class _St:                                   # torch.cuda stand-ins: everything here runs in order on the CPU
    cuda_stream = 0

    def wait_stream(self, s):
        pass

    def wait_event(self, e):
        pass

    def synchronize(self):
        pass


class _Ev:
    def record(self, stream=None):
        pass

    def query(self):
        return True


torch.cuda.Stream = lambda *a, **k: _St()
torch.cuda.Event = lambda *a, **k: _Ev()
torch.cuda.stream = lambda s: contextlib.nullcontext()
torch.cuda.current_stream = lambda *a, **k: _St()
torch.Tensor.pin_memory = lambda self, *a, **k: self
NS = _St()


rand_maps = tl_.rand_maps


# ---------------------------------------------------------------------------------------------------------- A
bad, bad_allrows, n_late = [], [], [0, 0]
for trial in range(400):
    g = torch.Generator().manual_seed(trial)
    T = 1 + trial % 8
    nres = int(torch.randint(0, 300, (1,), generator=g))
    nbank = int(torch.randint(0, 500, (1,), generator=g))
    nband = int(torch.randint(0, 100, (1,), generator=g)) if trial % 3 == 0 else 0
    row, bank, band = rand_maps(g, nres, nbank, nband)
    pool = int(torch.randint(16, late.E, (1,), generator=g))          # small pools: experts repeat across rows
    ids = torch.stack([torch.randperm(pool, generator=g)[:16] for _ in range(T)]).to(torch.int32)
    if trial % 5 == 0:
        ids[0, 3] = -1                                                # a padding pick
    w = (torch.randint(1, 7, (T, 16), generator=g).float() / 8)        # coarse: ties in weight and in energy
    allrows = trial % 2 == 1                                          # unitk's two modes (K31_STAGE_ALL)
    rows, slotted = tl_.unitk_rows(ids, w, row, bank, trial % 5, 0, allrows)
    lfrom = int(torch.randint(0, 5, (1,), generator=g)) if trial % 2 else 0
    lto = int(torch.randint(lfrom, 17, (1,), generator=g))
    D = trial % 9
    bm = band if nband else None
    lslot = torch.full((8,), 7, dtype=torch.int32)
    lw = torch.full((8, 8), 9.0)
    lmask = torch.full((8, 16), 7, dtype=torch.int32)
    cnt = torch.zeros(1, dtype=torch.int64)
    late.plan(ids, w, row, bank, rows, lfrom, lto, D, lslot, lw, lmask, cnt, band_map=bm, pdl=False)
    r_slot, r_w, r_mask, r_n = late.plan_reference(ids, w, row, bank, rows, lfrom, lto, D, band_map=bm)
    if not (torch.equal(lslot, r_slot) and torch.equal(lw, r_w) and torch.equal(lmask[:T], r_mask) and
            int(cnt) == r_n):
        bad.append(trial)
    owner = {int(bank[x]): x for x in range(late.E) if int(bank[x]) >= 0}
    late_experts = {owner[int(b)] for b in lslot.tolist() if b >= 0}
    if allrows and late_experts & set(slotted):
        bad_allrows.append(trial)
    n_late[allrows] += r_n
check("A1 plan == plan_reference on 400 calls (unitk rows, strict and ALLROWS); ALLROWS: no slotted expert late",
      not bad and not bad_allrows, f"(bad trials {bad[:8]}, ALLROWS violations {bad_allrows[:8]}; late picks in the "
      f"strict trials {n_late[0]}, in the ALLROWS trials {n_late[1]}: different calls)")

g = torch.Generator().manual_seed(11)
table = (torch.randn(late.E, late.H, generator=g) * 0.05).bfloat16()
worst = 0.0
for T in (1, 3, 5, 8):
    ids = torch.randint(0, late.E, (T, 16), generator=g).to(torch.int32)
    w = torch.rand(T, 16, generator=g)
    lmask = torch.where(torch.rand(T, 16, generator=g) < 0.3, torch.randint(0, 8, (T, 16), generator=g),
                        torch.tensor(-1)).to(torch.int32)
    out = torch.full((8, late.H), 7.0)
    late.mean(out, lmask, ids, w, table, pdl=False)
    ref = late.mean_reference(lmask, ids, w, table)
    worst = max(worst, float((out[:T].double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)))
check("A2 mean == mean_reference", worst < 1e-6, f"(max err / max {worst:.1e})")

g = torch.Generator().manual_seed(12)
okA3, worst = True, 0.0
for T in (1, 4, 8):
    dst0 = (torch.randn(T, 7168, generator=g) * 0.5).bfloat16()
    dst0[0, :7] = -0.0
    z = torch.randn(T, 7168, generator=g) * 0.01
    y = (torch.randn(T, 7168, generator=g) * 0.3).bfloat16()
    coef = torch.zeros(8, 4)
    coef[:T, 0] = (torch.arange(T) % 2).float()                         # rows 1, 3, ... have a share
    coef[:T, 1] = torch.randn(T, generator=g) * 0.01
    want = torch.where(coef[:T, :1] > 0, dst0.float() + (z + coef[:T, 1:2] * y.float()), dst0.float()).bfloat16()
    d = dst0.clone()
    late.add(d, z, y, coef, pdl=False)
    o = torch.empty_like(dst0)
    late.add(dst0, z, y, coef, out=o, pdl=False)
    blk = torch.zeros(T, 3, 7168, dtype=torch.bfloat16)
    blk[:, 1] = dst0
    late.add(blk[:, 1, :], z, y, coef, pdl=False)
    nos = coef[:T, 0] == 0
    same_noshare = all(torch.equal(t[nos].view(torch.int16), dst0[nos].view(torch.int16)) for t in (d, o, blk[:, 1]))
    worst = max(worst, max(float((t.float() - want.float()).abs().max()) for t in (d, o, blk[:, 1])))
    okA3 &= same_noshare and torch.equal(blk[:, 0], torch.zeros_like(blk[:, 0])) and \
        torch.equal(blk[:, 2], torch.zeros_like(blk[:, 2]))
check("A3 add: shares added, other rows bit-identical (in place, out of place, row-strided)",
      okA3 and worst <= 2 ** -6, f"(max |diff| vs float32 formula {worst:.1e}: bf16 rounding of one fused add)")

# ---------------------------------------------------------------------------------------------------------- B
lib = latetier.Lib(EMU)
lo = lib.layout
g = torch.Generator().manual_seed(21)
x = (torch.randn(6, late.H, generator=g) * torch.tensor([1e-3, 0.5, 3.0, 1e2, 1e-30, 7.0])[:, None]).bfloat16()
x[5, :32] = 0.0                                                            # an all-zero block
x[5, 32:64] = torch.arange(32).float().bfloat16() - 16                   # integers: exact steps
x[5, 64:96] = 127.0                                                         # amax exactly 127
x[5, 96:128] = torch.tensor([0.5, 1.5, 2.5, -0.5] * 8).bfloat16() * 2 ** -6   # rounding ties
xq, xe = lib.host_quant_rows(x)
rq, re_, _ = late.quant_rows_reference(x)
check("B1 the send's int8 rows == the rule's reference, byte for byte", np.array_equal(xq, rq.numpy()) and
      np.array_equal(xe, re_.numpy()), f"(rows {xq.shape[0]}, exponents {sorted(set(xe.reshape(-1).tolist()))[:3]}..)")

lsl = np.array([12, 7, 300, -1, 5, -1, -1, -1], dtype=np.int32)          # the plan fills from 0: n = 3
lwv = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
hb = lib.host_header(41, 17, 5, lsl, lwv)
wexp = np.zeros((8, 8), dtype=np.float32)
wexp[:3, :5] = lwv[:3, :5]
want = fp.pack_header((41 << 16) | (17 << 8) | 5, 3, 5, 17, 1, 4.0, 25.0, -100, [12, 7, 300], wexp)
check("B2 job header bytes == CpuJob2 packed by hand", hb == want, f"({len(hb)} B, sizeof {lo['job_size']})")

g = torch.Generator().manual_seed(22)
worst, okB3 = 0.0, True
for scale in (1e-4, 1e-2, 0.3, 1.0, 5.0):
    u = torch.randn(late.H, generator=g).float()
    v = torch.randn(late.H, generator=g).float() * scale
    ud, vd = u.double(), v.double()
    sums = [float((ud * ud).sum()), float((ud * vd).sum()), float((vd * vd).sum()), float(vd.abs().sum())]
    c = lib.host_coef(sums, True, 1e-6)
    cr = late.coef_reference(u[None], v[None], 1e-6)[0]
    worst = max(worst, abs(c[1] - cr[1]) / max(abs(cr[1]), 1e-12), abs(c[2] - cr[2]) / cr[2], abs(c[3] - cr[3]) / cr[3])
c0 = lib.host_coef([3.0, 0.0, 0.0, 0.0], True, 1e-6)
okB3 &= c0[0] == 0 and c0[1] == 0 and c0[2] == c0[3]
cneg = lib.host_coef([3584.0, -3584.0, 3584.0, 3584.0], True, 1e-6)       # v = -u: the sum is zero
okB3 &= bool(np.isfinite(cneg).all()) and cneg[0] == 1 and 500 < cneg[2] < 2000     # true ra = rsqrt(eps) = 1e3
check("B3 landing coefficients against float64", okB3 and worst < 1e-6, f"(worst rel {worst:.1e})")

seg_consts = dict(bytes=655360, door=128, state=256, done=512, job=4096, job_bytes=512, x=8192, x_stride=32768,
                  res=139264, res_bytes=114688, nslot=4, pmax=8, maxt=8, h=3584, kb=112, magic=0x3253555043313343,
                  job_size=328, off_slot=40, off_w=72, state_bytes=128, threads=512)
g = torch.Generator().manual_seed(23)
U = torch.randn(64, late.H, generator=g, dtype=torch.float64) * 0.02
gw = torch.rand(late.H, generator=g, dtype=torch.float64) + 0.25
u = torch.randn(3, late.H, generator=g, dtype=torch.float64)
v = torch.randn(3, late.H, generator=g, dtype=torch.float64) * torch.tensor([[1e-3], [0.3], [4.0]], dtype=torch.float64)
cf = late.coef_reference(u, v, 1e-6)
y = (u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6) * gw) @ U.t()
delta = ((v * cf[:, 2:3]) * gw) @ U.t() + cf[:, 1:2] * y
ident = rel(delta, late.delta_reference(u, v, gw, 1e-6, U))
check("B4 layout == cpustage_seg2.h; identity up(ra g v) + cb y == up(norm(u+v)) - up(norm(u))",
      all(lo[k] == v_ for k, v_ in seg_consts.items()) and ident < 1e-12, f"(identity rel {ident:.1e})")

# ---------------------------------------------------------------------------------------------------------- C
seg = latetier.Segment(lib)
st = torch.zeros(16, dtype=torch.int64)
g = torch.Generator().manual_seed(31)
REC = late.REC_BYTES


def synth_records(n, gen):
    r = torch.randint(0, 256, (n, REC), generator=gen, dtype=torch.uint8)
    o = late._C13 + late._C2
    r[:, o:] = torch.randint(117, 122, (n, REC - o), generator=gen, dtype=torch.uint8)    # scale bytes
    return r


bank = synth_records(4, g)
LAYER = 7
T = 5
x = (torch.randn(T, late.H, generator=g) * 0.5).bfloat16()
lslot = torch.tensor([2, 0, 3, -1, -1, -1, -1, -1], dtype=torch.int32)
lw = torch.zeros(8, 8)
lw[0, 0], lw[0, 3], lw[1, 1], lw[1, 4], lw[2, 2], lw[2, 0] = 0.31, 0.12, 0.22, 0.40, 0.05, 0.02
lib.send(seg.host, st, lslot, lw, x, T, LAYER, stream=NS)
tag = (1 << 16) | (LAYER << 8) | T
xo = seg.x_off(1)
xq_t, xe_t = lib.host_quant_rows(x)
okC1 = bytes(seg.u8[seg.job_off(1):seg.job_off(1) + 512]) == lib.host_header(1, LAYER, T, lslot.numpy(), lw.numpy())
okC1 &= np.array_equal(seg.u8[xo:xo + T * late.H].reshape(T, late.H), xq_t)
okC1 &= np.array_equal(seg.u8[xo + 8 * late.H:xo + 8 * late.H + T * 112].reshape(T, 112), xe_t)
okC1 &= seg.get(lo["door"]) == tag and int(st[0]) == 1 and int(st[1]) == tag and int(st[2]) == 1 and int(st[3]) == 3
st_before = st.clone()
lib.send(seg.host, st, torch.full((8,), -1, dtype=torch.int32), lw, x, T, LAYER, stream=NS)
okC1 &= seg.get(lo["door"]) == tag and int(st[1]) == 0 and int(st[0]) == 1 and int(st[2]) == int(st_before[2])
check("C1 late_send (emulated): header, rows, exponents, door, state; no record -> no job", okC1)

gwt = (torch.rand(late.H, generator=g) * 1.5 + 0.25).bfloat16()
EPS = 1e-6
okC2, worst = True, 0.0
for Tt in (1, 2, 5, 8):
    u = (torch.randn(Tt, late.H, generator=g) * 0.7).bfloat16()
    v = torch.randn(Tt, late.H, generator=g) * 0.2
    if Tt >= 2:
        v[1] = 0.0                                                 # a row without a share
    mfix = torch.randn(Tt, late.H, generator=g) * 0.05 if Tt == 5 else None
    if mfix is not None:
        mfix[1] = v[1]                                             # rows - mfix = 0 exactly
    xl = torch.full((8, late.H), 3.0).bfloat16()
    coef = torch.full((8, 4), 9.0)
    lib.land(None, None, u, v, mfix, gwt, xl, coef, Tt, EPS, LAYER, 0.0, stream=NS)
    vv = v - mfix if mfix is not None else v
    cr = late.coef_reference(u, vv, EPS)
    hasr = cr[:, 0] > 0
    okC2 &= torch.equal(coef[:Tt, 0], cr[:, 0].float())
    worst = max(worst, float(((coef[:Tt, 1:4].double() - cr[:, 1:4]).abs() / cr[:, 2:3].abs()).max()))
    okC2 &= torch.equal(xl[:Tt].view(torch.int16), late.land_reference(u, vv, gwt, coef[:Tt]).view(torch.int16))
    okC2 &= bool((xl[:Tt][~hasr].float() == 0).all())
    xl2, coef2 = xl.clone(), coef.clone()
    lib.land(None, None, u, v, mfix, gwt, xl2, coef2, Tt, EPS, LAYER, 0.0, stream=NS)
    okC2 &= torch.equal(xl2, xl) and torch.equal(coef2, coef)
    vb = v.bfloat16()                                              # the bf16 device-row variant
    lib.land(None, None, u, vb, None, gwt, xl2, coef2, Tt, EPS, LAYER, 0.0, stream=NS)
    crb = late.coef_reference(u, vb.float(), EPS)
    okC2 &= torch.equal(xl2[:Tt].view(torch.int16), late.land_reference(u, vb.float(), gwt, coef2[:Tt]).view(torch.int16))
    worst = max(worst, float(((coef2[:Tt, 1:4].double() - crb[:, 1:4]).abs() / crb[:, 2:3].abs()).max()))
check("C2 late_land on device rows (emulated): xl bit-exact, coefficients vs float64, mfix, empty rows, repeatable",
      okC2 and worst < 2e-6, f"(coefficients worst rel {worst:.1e})")

pool = fp.FakePool(seg, {LAYER: bank})                           # a pool serves jobs published after its start
pool.start()
lib.send(seg.host, st, lslot, lw, x, T, LAYER, stream=NS)        # seq 2, slot 2
okC3 = pool.wait_served(1) and int(st[0]) == 2
rows = torch.from_numpy(seg.rows(2, T).copy())
experts = [late.decode_hi2(bank[p]) for p in (2, 0, 3)]
_, _, xd = late.quant_rows_reference(x)
want = late.pool_reference(experts, xd, lw[:3], 1, 4.0, 25.0)
okC3 &= rel(rows, want) < 1e-6 and bool((rows[:, :] != 0).any(1).all())
u = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16()
mfix = torch.randn(T, late.H, generator=g) * 0.01
xl_p, cf_p = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
c_before = st.clone()
lib.land(seg.host, st, u, None, mfix, gwt, xl_p, cf_p, T, EPS, LAYER, 30e6, stream=NS)
xl_d, cf_d = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
lib.land(None, None, u, rows, mfix, gwt, xl_d, cf_d, T, EPS, LAYER, 0.0, stream=NS)
okC3 &= torch.equal(xl_p, xl_d) and torch.equal(cf_p, cf_d) and bool((cf_p[:T, 0] == 1).all())
okC3 &= int(st[4]) - int(c_before[4]) == 1 and int(st[1]) == 0 and int(st[12]) - int(c_before[12]) == T
check("C3 send -> stand-in pool -> landing: rows == pool_reference, landed bit-equal to device rows, counters", okC3,
      f"(pool rows vs float64 reference {rel(rows, want):.1e})")


def land_once(timeout_us, layer=LAYER, Tt=T):
    xl1, cf1 = torch.full((8, late.H), 2.0).bfloat16(), torch.full((8, 4), 5.0)
    t0 = time.time()
    lib.land(seg.host, st, u[:Tt], None, None, gwt, xl1, cf1, Tt, EPS, layer, timeout_us, stream=NS)
    return xl1, cf1, time.time() - t0


def zero_landing(xl1, cf1, Tt=T):
    return bool((cf1[:Tt, 0] == 0).all() and (cf1[:Tt, 1] == 0).all() and (xl1[:Tt].float() == 0).all())


okC4 = True
s0 = st.clone()
xl1, cf1, _ = land_once(1e6)                                          # no send since the last landing
okC4 &= zero_landing(xl1, cf1) and int(st[7]) - int(s0[7]) == 1
lib.send(seg.host, st, lslot, lw, x, T, LAYER, stream=NS)
pool.wait_served(2)
xl1, cf1, _ = land_once(1e6, layer=LAYER + 1)                        # another layer's landing: never its rows
okC4 &= zero_landing(xl1, cf1) and int(st[7]) - int(s0[7]) == 2 and int(st[1]) == 0
pool.stop()
lib.send(seg.host, st, lslot, lw, x, T, LAYER, stream=NS)            # the pool is stopped (running = 0)
xl1, cf1, dt_down = land_once(5e6)
okC4 &= zero_landing(xl1, cf1) and int(st[8]) - int(s0[8]) == 1 and dt_down < 2.0
seg.set(lo["state"], 1)                                               # "running" but nobody serves: timeout
lib.send(seg.host, st, lslot, lw, x, T, LAYER, stream=NS)
xl1, cf1, dt_to = land_once(20e3)
okC4 &= zero_landing(xl1, cf1) and int(st[5]) - int(s0[5]) == 1 and dt_to >= 0.02
pool = fp.FakePool(seg, {LAYER: bank})                               # serves from the next job on
pool.start()
nseq = int(st[0]) + 1
pool.refuse.add(nseq)
lib.send(seg.host, st, lslot, lw, x, T, LAYER, stream=NS)
pool.wait_served(1)
xl1, cf1, _ = land_once(30e6)
okC4 &= zero_landing(xl1, cf1) and int(st[6]) - int(s0[6]) == 1
check("C4 no job / foreign layer / pool down / timeout / refused: nothing lands, each counted", okC4,
      f"(down {1e3 * dt_down:.0f} ms, timeout {1e3 * dt_to:.0f} ms for a 20 ms bound)")

okC5 = True
for j in range(7):                                                    # the ring wraps: slot = seq % 4
    Tt = 1 + (j * 3) % 8
    xj = (torch.randn(Tt, late.H, generator=g) * 0.5).bfloat16()
    lsj = torch.tensor([j % 4, (j + 1) % 4] + [-1] * 6, dtype=torch.int32)
    lwj = torch.zeros(8, 8)
    lwj[0, :Tt] = 0.1 + 0.01 * j
    lwj[1, 0] = 0.05
    nserved = len(pool.served)
    lib.send(seg.host, st, lsj, lwj, xj, Tt, LAYER, stream=NS)
    okC5 &= pool.wait_served(nserved + 1)
    slot = int(st[0]) % 4
    rj = torch.from_numpy(seg.rows(slot, Tt).copy())
    _, _, xdj = late.quant_rows_reference(xj)
    wantj = late.pool_reference([pool._experts(LAYER, j % 4), pool._experts(LAYER, (j + 1) % 4)], xdj, lwj[:2], 1)
    uj = (torch.randn(Tt, late.H, generator=g) * 0.7).bfloat16()
    xa, ca = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
    lib.land(seg.host, st, uj, None, None, gwt, xa, ca, Tt, EPS, LAYER, 30e6, stream=NS)
    xb, cbb = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
    lib.land(None, None, uj, wantj.float(), None, gwt, xb, cbb, Tt, EPS, LAYER, 0.0, stream=NS)
    okC5 &= rel(rj, wantj) < 1e-6 and torch.equal(ca, cbb) and torch.equal(xa, xb)
pool.stop()
check("C5 seven jobs through the ring: each landing gets its own rows", okC5, f"(stale {pool.stale})")

# ---------------------------------------------------------------------------------------------------------- D


class _Norm:                                  # vLLM's RMSNorm: (x * rsqrt(mean(x^2) + eps)) * weight in float32
    def __init__(self, w, eps):
        self.weight, self.variance_epsilon = w, eps

    def __call__(self, x):
        xf = x.float()
        return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)) *
                self.weight.float()).to(x.dtype)


class _I8:                                    # stand-in for w8i.I8W: y = x @ W^T (W the int8 weights' values)
    def __init__(self, W):
        self.W = W

    def __call__(self, x, out=None):
        y = x.float() @ self.W.t()
        if out is None:
            return y.to(torch.bfloat16)
        out.copy_(y)
        return out


class _Up:
    def __init__(self, W):
        self._k31_i8 = (_I8(W), W.shape[0])

    def __call__(self, x):
        return self._k31_i8[0](x), None


class _Rot:
    def __init__(self, W, gw, eps):
        self.norm, self.up_proj = _Norm(gw, eps), _Up(W)


class _Layer:
    pass


g = torch.Generator().manual_seed(41)
Wup = (torch.randn(7168, late.H, generator=g) * 0.01)
rot = _Rot(Wup, gwt, EPS)
seg2 = latetier.Segment(lib)
tier = latetier.Tier(torch.device("cpu"), lib, seg2, None)
L = 11                                       # (L + 1) % 12 == 0: L ends an AttnRes block
row, bmap, _ = rand_maps(g, 200, 400)
lay = _Layer()
lay.idx, lay.row_map, lay.bank_map, lay.band_map = L, row, bmap, None
lay.cut_mean = (torch.randn(late.E, late.H, generator=g) * 0.02).bfloat16()
tier.cfg = {L: (4, 0, 16, 1e6), L + 1: (4, 0, 16, 1e6), 4: (4, 0, 16, 1e6), 5: (0, 0, 16, 50.0)}
okD, detail, why = True, [], []


def need(cond, what):
    global okD
    if not cond:
        why.append(what)
        okD = False

for case in ("attn", "block", "silent"):
    Lc = 4 if case == "attn" else L
    lay.idx = Lc
    big = {}
    bnk = torch.nonzero(bmap >= 0).flatten()
    T = 5
    ids = torch.stack([torch.cat([bnk[torch.randperm(len(bnk), generator=g)[:6]],
                                  torch.nonzero(row >= 0).flatten()[torch.randperm(200, generator=g)[:10]]])
                       for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, 16, generator=g) * 0.3 + 0.01
    rows_u = torch.where(row[ids.long()] >= 0, row[ids.long()], -1).to(torch.int32)
    # the bank records the plan can name: a synthetic bank big enough for the layer's indices, records generated lazily
    lslot_ref, lw_ref, lmask_ref, _ = late.plan_reference(ids, w, row, bmap, rows_u, 0, 16, 4)
    nrec = int(bmap.max()) + 1

    class _Bank(dict):
        shape = (nrec, REC)

        def __missing__(self, k):
            v = synth_records(1, torch.Generator().manual_seed(1000 + k))[0]
            self[k] = v
            return v
    bank_d = _Bank()
    pool = fp.FakePool(seg2, {Lc: bank_d})
    if case == "silent":
        pool.drop.add(int(tier.st[0]) + 1)
    pool.start()
    x = (torch.randn(T, late.H, generator=g) * 0.5).bfloat16()
    tier.post(lay, x, ids, w, rows_u, NS)
    need(tier.sent is not None, f"{case}: no send recorded")
    need(torch.equal(tier.lslot, lslot_ref) and torch.equal(tier.lw, lw_ref), f"{case}: plan != reference")
    if case != "silent":
        need(pool.wait_served(1), f"{case}: pool did not serve")
    else:
        time.sleep(0.5)
    u = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16()
    shared = (torch.randn(T, 7168, generator=g) * 0.2).bfloat16()
    sh0 = shared.clone()
    out = tier.rot(rot, u, shared, lambda: None)
    y_ref = rot.up_proj(rot.norm(u))[0]
    need(out is not None and tier.pend is not None, f"{case}: rot did not take the call")
    need(out is not None and torch.equal(out, (sh0.float() + y_ref.float()).bfloat16()), f"{case}: on-time output")
    pool.stop()
    attn = (torch.randn(T, 7168, generator=g) * 0.4).bfloat16()
    res = torch.zeros(T, 8, 7168, dtype=torch.bfloat16)
    res[:, (Lc + 1) // 12] = (torch.randn(T, 7168, generator=g) * 0.4).bfloat16()
    res0 = res.clone()
    got = tier.join(Lc + 1, attn, res)
    need(tier.pend is None, f"{case}: pend left")
    rows_p = torch.from_numpy(seg2.rows(int(tier.st[0]) % 4, T).copy()).double()
    mf = late.mean_reference(lmask_ref, ids, w, lay.cut_mean)
    dref = late.delta_reference(u.float(), (rows_p - mf).float(), gwt.float(), EPS, Wup)
    zt, cft = tier.z[:T], tier.coef[:T]
    d32 = zt + cft[:, 1:2] * y_ref.float()                         # the engine's delta in float32 (late.add's sum)
    b = (Lc + 1) // 12
    base = attn if case == "attn" else res0[:, b]
    want_out = torch.where(cft[:, :1] > 0, base.float() + d32, base.float())
    landed = got if case == "attn" else res[:, b]
    one_ulp = bool(((landed.float() - want_out).abs() <= want_out.abs() * 2 ** -7 + 1e-30).all())   # interpreter: truncating cast
    uses = (lw_ref[:, :T] > 0).any(0)                             # rows holding one of the late experts
    if case == "attn":
        r = rel(torch.where(cft[:, :1] > 0, d32, torch.zeros_like(d32)), dref)
        need(r < 2e-2, f"attn: rel {r:.1e}")
        need(one_ulp, "attn: final add not within one bf16 rounding")
        need(torch.equal(cft[:, 0] > 0, uses), f"attn: has {cft[:, 0].tolist()} for rows {uses.tolist()}")
        need(torch.equal(res, res0), "attn: block state touched")
        detail.append(f"attention path: delta vs float64 rel {r:.1e}")
    elif case == "block":
        r = rel(torch.where(cft[:, :1] > 0, d32, torch.zeros_like(d32)), dref)
        need(r < 2e-2, f"block: rel {r:.1e}")
        need(one_ulp, "block: final add not within one bf16 rounding")
        need(got is attn, "block: attention output replaced")
        need(torch.equal(res[:, :b], res0[:, :b]) and torch.equal(res[:, b + 1:], res0[:, b + 1:]), "block: others")
        detail.append(f"block-state path rel {r:.1e}")
    else:
        need(torch.equal(got.view(torch.int16), attn.view(torch.int16)) and torch.equal(res, res0), "silent: changed")
        need(bool((cft[:, 0] == 0).all()), f"silent: has {cft[:, 0].tolist()}")
        detail.append(f"silent pool: unchanged")
c = tier.counters()
need(c["n_added"] == 2 and c["n_timeout"] == 1 and c["n_jobs"] == 3, f"counters {c}")
detail.append(f"counters jobs {c['n_jobs']}, added {c['n_added']}, timeouts {c['n_timeout']}")
check("D  Tier post -> rot -> join against up(norm(u + v)) - up(norm(u))", okD,
      "(" + "; ".join(detail) + ")" + (f" FAILED: {why}" if why else ""))

# ---------------------------------------------------------------------------------------------------------- E
from k31_engine import cpupool  # noqa: E402

core = min(os.sched_getaffinity(0))
cp = cpupool.CpuPool(os.environ.get("K31_POOL_LIB", "engine/band/libcpustage.so"), threads=1, core0=core)
bankE = tl_.synth_records(4, 51).numpy()
LE = 13
cp.set_layer(LE, bankE)
cp.start(None)
segE = latetier.Segment.attach(lib, cp, dev=cp.addr)                # the emulated kernels use host addresses
stE = torch.zeros(16, dtype=torch.int64)
stE[0] = cp.next_seq() - 1
okE, notes = True, []
g = torch.Generator().manual_seed(52)
for j in range(6):
    Tj = (5, 1, 8, 3, 2, 7)[j]
    xj = (torch.randn(Tj, late.H, generator=g) * 0.5).bfloat16()
    slots = [(j + q) % 4 for q in range(1 + j % 3)]
    lsj = torch.tensor(slots + [-1] * (8 - len(slots)), dtype=torch.int32)
    lwj = torch.zeros(8, 8)
    for q in range(len(slots)):
        lwj[q, torch.randperm(Tj, generator=g)[:max(1, Tj // 2)]] = 0.05 + 0.1 * torch.rand(1, generator=g)
    lib.send(segE.dev, stE, lsj, lwj, xj, Tj, LE, stream=NS)
    tagj = int(stE[1])
    sl = (tagj >> 16) % 4
    if not segE.wait(segE.done_off(sl), tagj, 60.0):
        okE = False
        notes.append(f"job {j} not served")
        break
    got = segE.rows(sl, Tj).copy()
    want = cp.run_single([bankE[s_] for s_ in slots], xj.float().numpy(), lwj[:len(slots), :Tj].numpy())
    okE &= bool(np.array_equal(got.view(np.uint32), want.view(np.uint32)))
    uj = (torch.randn(Tj, late.H, generator=g) * 0.7).bfloat16()
    xa, ca = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
    xb, cbb = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
    lib.land(segE.dev, stE, uj, None, None, gwt, xa, ca, Tj, EPS, LE, 30e6, stream=NS)
    lib.land(None, None, uj, torch.from_numpy(got), None, gwt, xb, cbb, Tj, EPS, LE, 0.0, stream=NS)
    okE &= torch.equal(xa, xb) and torch.equal(ca, cbb) and int(stE[1]) == 0
cp.hold(True)                                                      # the pool takes no job: the landing gives up
lib.send(segE.dev, stE, lsj, lwj, xj, Tj, LE, stream=NS)
xa, ca = torch.full((8, late.H), 2.0).bfloat16(), torch.full((8, 4), 5.0)
s0 = stE.clone()
lib.land(segE.dev, stE, uj, None, None, gwt, xa, ca, Tj, EPS, LE, 20e3, stream=NS)
okE &= int(stE[5] - s0[5]) == 1 and bool((ca[:Tj, :2] == 0).all() and (xa[:Tj].float() == 0).all())
cp.hold(False)
lib.send(segE.dev, stE, lsj, lwj, xj, Tj, LE, stream=NS)          # the next job is served as usual
tagj = int(stE[1])
okE &= segE.wait(segE.done_off((tagj >> 16) % 4), tagj, 60.0)
xa, ca = torch.zeros(8, late.H, dtype=torch.bfloat16), torch.zeros(8, 4)
lib.land(segE.dev, stE, uj, None, None, gwt, xa, ca, Tj, EPS, LE, 30e6, stream=NS)
okE &= bool((ca[:Tj, 0] >= 0).all()) and int(stE[4]) >= 7
st_pool = cp.stats()
cp.stop()
check("E  the real pool between the emulated kernels: rows bit-equal to its single-thread kernel, landings equal, a "
      "held job times out", okE, f"(pool: done {st_pool['done']}, stale {st_pool['stale']}, abandoned "
      f"{st_pool['abandoned']}, mean job {st_pool['mean_job_us']:.0f} us on one thread; {'; '.join(notes)})")

print("LATE CPU TEST " + ("PASS" if not fails else f"FAIL ({len(fails)}: {fails})"), flush=True)
sys.exit(0 if not fails else 1)
