"""The late tier end to end on the GB300 (latetier.Tier: post after unitk, rot = the routed output transform, join in the
next layer's _post_attn_norm), with a stand-in pool (tests/late_fakepool.py) and stand-in layer modules (an RMSNorm in
float32 and the int8 up GEMV, w8i.I8W, as the engine's routed_expert_up_proj:w8i). After ./build.sh, from the repo root
(LATE_HI2 = a record file to take the bank from; without it, synthetic records; LATE_POOL=real: the real pool,
engine/pool's libcpustage.so through cpupool.CpuPool with 4 threads on the container's last 4 cores, instead of the
stand-in):

    docker run --rm --network none --device nvidia.com/gpu=$GB --ipc host --cpuset-cpus 40-47 --ulimit core=0 \
      --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/repo -w /repo -v /models/k31-hi2:/hi2:ro \
      -e LATE_HI2=/hi2/L01.hi2 --entrypoint python3 vllm/vllm-openai:kimi-k31 engine/tests/test_late_e2e.py
    (and once more with -e LATE_POOL=real)

Checks, exit 0 = PASS:
  1 eager, attention path (layer 4 -> 5): the plan == plan_reference; the on-time output residual + y unchanged by the
    tier (bit-equal to the plain three steps); the landed delta (z + cb y) against the float64 definition
    up(norm(u + v)) - up(norm(u)) with v = the pool's rows - the late picks' on-time mean (rel < 2e-2); the attention
    output + delta within one bf16 rounding
  2 eager, block-state path (layer 11 ends an AttnRes block): the delta goes into res[:, 1] only
  3 eager, the pool does not answer (dropped job): the attention output passes bit-identical, a timeout counted
  4 a layer without mean tables (mfix none): v = the pool's rows
  5 a sent job whose call never reaches its landing: the next call joins and drops it (no open fork)
  6 post + rot + join captured in one CUDA graph (two layers), 3 replays with new inputs: each served, each against
    the float64 definition, and one replay bit-equal to an eager run on the same inputs
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("K31_I8_LIB", os.path.abspath("engine/band/k31i8.so"))
from k31_engine import late, latetier, w8i  # noqa: E402
import late_fakepool as fp  # noqa: E402

dev = torch.device("cuda")
lib = latetier.Lib(os.environ.get("K31_LATE_LIB", "engine/band/liblate.so"))
EPS = 1e-6
NREC = 48
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def wait_gpu(timeout_s=120.0):
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.time()
    while not ev.query():
        assert time.time() - t0 < timeout_s, "GPU work did not finish"
        time.sleep(0.0005)


def records():
    path = os.environ.get("LATE_HI2", "")
    if path:
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = np.frombuffer(os.pread(fd, NREC * late.REC_BYTES, 0), dtype=np.uint8).reshape(NREC, late.REC_BYTES)
        finally:
            os.close(fd)
        return torch.from_numpy(raw.copy()), f"{path} records 0-{NREC - 1}"
    gen = torch.Generator().manual_seed(1)
    r = torch.randint(0, 256, (NREC, late.REC_BYTES), generator=gen, dtype=torch.uint8)
    o = late._C13 + late._C2
    r[:, o:] = torch.randint(117, 122, (NREC, late.REC_BYTES - o), generator=gen, dtype=torch.uint8)
    return r, "synthetic records"


class Norm:                     # vLLM's RMSNorm arithmetic: (x * rsqrt(mean(x^2) + eps)) * weight, float32, one rounding
    def __init__(self, w):
        self.weight, self.variance_epsilon = w, EPS

    def __call__(self, x):
        xf = x.float()
        return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)) * self.weight.float()).to(x.dtype)


class Up:                       # a ReplicatedLinear on the int8 GEMV (fp6.py's apply for routed_expert_up_proj:w8i)
    def __init__(self, iw):
        self._k31_i8 = (iw, iw.N)

    def __call__(self, x):
        return self._k31_i8[0](x), None


class Rot:
    def __init__(self, iw, gw):
        self.norm, self.up_proj = Norm(gw), Up(iw)


class Layer:
    pass


g = torch.Generator().manual_seed(5)
bank_cpu, src = records()
bank_cpu = bank_cpu.pin_memory()
gw = (torch.rand(late.H, generator=g) * 1.5 + 0.25).bfloat16().to(dev)
W = (torch.randn(7168, late.H, generator=g) * 0.01).bfloat16().to(dev)
iw = w8i.I8W(W)
W8 = w8i.dequant_ref(W)
rot = Rot(iw, gw)
perm = torch.randperm(late.E, generator=g)
row_map = torch.full((late.E,), -1, dtype=torch.int32)
bank_map = torch.full((late.E,), -1, dtype=torch.int32)
row_map[perm[:300]] = torch.arange(300, dtype=torch.int32) + 128
bank_map[perm[300:300 + NREC]] = torch.arange(NREC, dtype=torch.int32)
table = (torch.randn(late.E, late.H, generator=g) * 0.02).bfloat16().to(dev)
layers = {}
for idx in (4, 5, 11, 12, 20, 21):
    st = Layer()
    st.idx, st.row_map, st.bank_map, st.band_map = idx, row_map.to(dev), bank_map.to(dev), None
    st.cut_mean = None if idx == 20 else table
    st.bank, st.bank_cpu = list(range(NREC)), bank_cpu
    layers[idx] = st
REAL = os.environ.get("LATE_POOL", "") == "real"


class RealPool:
    """cpupool.CpuPool behind the stand-in's interface (drop -> hold the pool for that job, wait_served -> nothing:
    the landing's own bound waits)."""

    def __init__(self, banks):
        from k31_engine import cpupool
        cores = sorted(os.sched_getaffinity(0))
        self.cp = cpupool.CpuPool(os.environ.get("K31_POOL_LIB", "engine/band/libcpustage.so"), threads=4,
                                  core0=cores[-4])
        for i, b in banks.items():
            self.cp.set_layer(i, b.numpy())
        cpupool.pin_threads(set(cores[:-4]))
        self.cp.start(None)
        self.drop = set()
        self.served, self.stale = [], 0

    def wait_served(self, n, timeout_s=0):
        return True

    def stop(self):
        self.stale = self.cp.stats()["stale"]
        self.cp.stop()


if REAL:
    pool = RealPool({i: bank_cpu for i in layers})
    seg = latetier.Segment.attach(lib, pool.cp)
else:
    seg = latetier.Segment(lib)
tier = latetier.Tier(dev, lib, seg, pool.cp if REAL else None)
tier.cfg = {i: (4, 0, 16, 5e6) for i in layers}          # 5 s bounds: the stand-in pool starts slowly
if not REAL:
    pool = fp.FakePool(seg, {i: bank_cpu for i in layers}, device="cuda")
    pool.start()
T = 5
res_idx = None


def inputs(gen):
    bnk = perm[300:300 + NREC]
    resd = perm[:300]
    ids = torch.stack([torch.cat([bnk[torch.randperm(NREC, generator=gen)[:6]], resd[torch.randperm(300, generator=gen)[:10]]])
                       for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, 16, generator=gen) * 0.3 + 0.01
    rows = torch.where(row_map[ids.long()] >= 0, row_map[ids.long()], -1).to(torch.int32)
    rows[0, 0] = 3                                                # one bank pick staged on time: never late
    x = (torch.randn(T, late.H, generator=gen) * 0.5).bfloat16()
    u = (torch.randn(T, late.H, generator=gen) * 0.7).bfloat16()
    sh = (torch.randn(T, 7168, generator=gen) * 0.2).bfloat16()
    attn = (torch.randn(T, 7168, generator=gen) * 0.4).bfloat16()
    return [t.to(dev) for t in (ids, w, rows, x, u, sh, attn)]


def reference(L, ids, w, rows, x, u, slot):
    """float64 delta of layer L's late share from the pool rows the job left in `slot`."""
    st = layers[L]
    lslot, lw, lmask, _ = late.plan_reference(ids, w, st.row_map, st.bank_map, rows, 0, 16, 4)
    rws = torch.from_numpy(seg.rows(slot, T).copy()).to(dev).double()
    mf = late.mean_reference(lmask, ids, w, st.cut_mean) if st.cut_mean is not None else 0.0
    return late.delta_reference(u, rws - mf, gw, EPS, W8), lslot, lw


def run_layer(L, ids, w, rows, x, u, sh, attn, res, serve=True):
    st = layers[L]
    main = torch.cuda.current_stream()
    tier.post(st, x, ids, w, rows, main)
    out = tier.rot(rot, u, sh, lambda: None)
    got = tier.join(L + 1, attn, res)
    return out, got


# 1-4 eager
for case, L in (("attention", 4), ("block state", 11), ("silent pool", 4), ("no mean tables", 20)):
    ids, w, rows, x, u, sh, attn = inputs(g)
    res = torch.zeros(T, 8, 7168, dtype=torch.bfloat16, device=dev)
    res[:, (L + 1) // 12] = (torch.randn(T, 7168, generator=g) * 0.4).bfloat16().to(dev)
    res0, sh0 = res.clone(), sh.clone()
    if case == "silent pool":
        if REAL:
            pool.cp.hold(True)
        else:
            pool.drop.add(int(tier.st[0]) + 1)
    c0 = tier.counters()
    out, got = run_layer(L, ids, w, rows, x, u, sh, attn, res)
    wait_gpu()
    if case == "silent pool" and REAL:
        pool.cp.hold(False)
    c1 = tier.counters()
    slot = int(tier.st[0]) % 4
    y = iw(rot.norm(u))
    ok = torch.equal(out, (sh0.float() + y.float()).bfloat16())                  # the on-time output
    dref, lslot_r, lw_r = reference(L, ids, w, rows, x, u, slot)
    ok &= torch.equal(tier.lslot, lslot_r) and torch.equal(tier.lw, lw_r)
    d32 = tier.z[:T] + tier.coef[:T, 1:2] * y.float()
    b = (L + 1) // 12
    base = attn if (L + 1) % 12 else res0[:, b]
    landed = got if (L + 1) % 12 else res[:, b]
    if case == "silent pool":
        ok &= torch.equal(got.view(torch.int16), attn.view(torch.int16)) and torch.equal(res, res0)
        ok &= c1["n_timeout"] - c0["n_timeout"] == 1 and bool((tier.coef[:T, 0] == 0).all())
        detail = f"(timeout after {(c1['wait_ns'] - c0['wait_ns']) / 1e6:.0f} ms)"
    else:
        r = rel(d32, dref)
        want_out = base.float() + d32
        ok &= r < 2e-2 and bool(((landed.float() - want_out).abs() <= want_out.abs() * 2 ** -7 + 1e-30).all())
        uses = (lw_r[:, :T] > 0).any(0)                              # rows holding one of the call's late experts
        ok &= c1["n_added"] - c0["n_added"] == 1 and torch.equal(tier.coef[:T, 0] > 0, uses)
        if (L + 1) % 12 == 0:
            ok &= got is attn and torch.equal(res[:, :b], res0[:, :b]) and torch.equal(res[:, b + 1:], res0[:, b + 1:])
        else:
            ok &= torch.equal(res, res0)
        detail = f"(delta vs float64 rel {r:.1e}; {src})"
    check(f"{['1', '2', '3', '4'][['attention', 'block state', 'silent pool', 'no mean tables'].index(case)]} eager, "
          f"{case}", ok, detail)

# 5. a send without its landing
ids, w, rows, x, u, sh, attn = inputs(g)
tier.post(layers[4], x, ids, w, rows, torch.cuda.current_stream())
n_drop = tier.n_drop
tier.post(layers[5], x, ids, w, rows, torch.cuda.current_stream())      # the next call: flushes layer 4's send
out = tier.rot(rot, u, sh, lambda: None)
got = tier.join(6, attn, None)
wait_gpu()
check("5 a send that never reaches its landing is joined and dropped by the next call",
      tier.n_drop == n_drop + 1 and tier.pend is None and tier.sent is None)

# 6. a CUDA graph over two layers
bufs = inputs(g)
ids, w, rows, x, u, sh, attn = bufs
sh2, attn2 = sh.clone(), attn.clone()
cs = torch.cuda.Stream()
cs.wait_stream(torch.cuda.current_stream())
outs = {}


def forward():
    outs["a"] = run_layer(4, ids, w, rows, x, u, sh, attn, None)
    outs["b"] = run_layer(5, ids, w, rows, x, u, sh2, outs["a"][1], None)


with torch.cuda.stream(cs):
    forward()
    wait_gpu()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=cs):
        forward()
ok6, worst = True, 0.0
for r_ in range(3):
    nb = inputs(g)
    for dst, s_ in zip(bufs, nb):
        dst.copy_(s_)
    sh2.copy_(sh)
    c0 = tier.counters()
    gr.replay()
    wait_gpu()
    c1 = tier.counters()
    ok6 &= c1["n_added"] - c0["n_added"] == 2 and c1["n_jobs"] - c0["n_jobs"] == 2
    seqb = int(tier.st[0])
    dref_b, _, _ = reference(5, ids, w, rows, x, u, seqb % 4)
    y = iw(rot.norm(u))
    d32 = tier.z[:T] + tier.coef[:T, 1:2] * y.float()
    worst = max(worst, rel(d32, dref_b))
    if r_ == 2:                                                    # the same inputs eagerly
        cap = (outs["b"][1].clone(), outs["a"][1].clone())
        sh.copy_(nb[5])
        sh2.copy_(nb[5])
        e_a = run_layer(4, ids, w, rows, x, u, sh, attn, None)
        e_b = run_layer(5, ids, w, rows, x, u, sh2, e_a[1], None)
        wait_gpu()
        ok6 &= torch.equal(cap[1], e_a[1]) and torch.equal(cap[0], e_b[1])
check("6 two layers in a CUDA graph: every replay served and landed; replay == eager bit for bit", ok6 and worst < 2e-2,
      f"(worst delta vs float64 {worst:.1e})")
pool.stop()
print(f"counters: {tier.counters()}; {'real pool' if REAL else 'stand-in pool'}, stale {pool.stale}", flush=True)
print("LATE E2E TEST " + ("PASS" if not fails else f"FAIL ({len(fails)}: {fails})"), flush=True)
sys.exit(0 if not fails else 1)
