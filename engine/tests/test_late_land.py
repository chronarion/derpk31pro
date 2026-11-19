"""The late tier's job protocol on the GB300 with a stand-in pool (tests/late_fakepool.py: a Python thread serving the
pinned segment with the pool's arithmetic in float64; the real pool is engine/pool's libcpustage.so). After ./build.sh,
from the repo root:

    docker run --rm --network none --device nvidia.com/gpu=$GB --ipc host --cpuset-cpus 40-47 --ulimit core=0 \
      --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/repo -w /repo --entrypoint python3 vllm/vllm-openai:kimi-k31 \
      engine/tests/test_late_land.py

Every landing has a bound (at most a few seconds): nothing here can hang the GPU. The test waits for the GPU by polling
events (time.sleep between polls), so the stand-in pool's thread runs while a landing spins. Checks, exit 0 = PASS:
  1 the segment pinned and mapped (late_map), its layout as cpustage_seg2.h says
  2 late_send on the GPU: header, int8 rows and exponents in host memory == the host twins byte for byte; the door tag
    (release) and the state words; a call with no record publishes nothing
  3 send -> pool -> landing: the pool's rows (== pool_reference) read over C2C land bit-equal to the device-row
    landing of the same rows; counters (added, rows)
  4 no job, another layer's job, the pool stopped (no wait), the pool silent (timeout at the bound), a refused job:
    has 0 and xl 0 each time, each counted; the measured wait of the timeout within [bound, bound + 1 ms]
  5 seven jobs back to back (the ring wraps): each landing gets its own job's rows
  6 send + landing captured in a CUDA graph (late stream forked and joined), 3 replays with new rows each time: the pool
    serves every replay's job and each landing equals the device-row landing of that job's rows
  7 the real pool (engine/pool's libcpustage.so through cpupool.CpuPool, 4 threads on the container's last 4 cores,
    its segment pinned by map_device): jobs the GPU posts are served, their rows bit-equal to the pool's single-thread
    kernel on the same records and rows (cpupool run_single), the landing == the device-row landing; a held pool
    (cpupool hold) times the landing out; the ring goes on after it
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31_engine import late, latetier  # noqa: E402
import late_fakepool as fp  # noqa: E402

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


def wait_gpu(timeout_s=60.0):
    """Wait for the current stream without holding the GIL in a blocking call (the stand-in pool is a Python thread)."""
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.time()
    while not ev.query():
        assert time.time() - t0 < timeout_s, "GPU work did not finish"
        time.sleep(0.0005)


def synth_records(n, seed):
    gen = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 256, (n, late.REC_BYTES), generator=gen, dtype=torch.uint8)
    o = late._C13 + late._C2
    r[:, o:] = torch.randint(117, 122, (n, late.REC_BYTES - o), generator=gen, dtype=torch.uint8)
    return r


seg = latetier.Segment(lib)
lo = seg.lo
check("1 segment pinned and mapped, cpustage_seg2.h layout", seg.dev is not None and lo["bytes"] == 655360 and
      lo["nslot"] == 4 and lo["job_size"] == 328, f"(host {seg.host:#x}, device {seg.dev:#x})")
st = torch.zeros(16, dtype=torch.int64, device=dev)
LAYER, T = 9, 5
bank = synth_records(4, 1)
g = torch.Generator().manual_seed(2)
x = (torch.randn(T, late.H, generator=g) * 0.5).bfloat16().to(dev)
lslot = torch.tensor([2, 0, 3, -1, -1, -1, -1, -1], dtype=torch.int32, device=dev)
lw = torch.zeros(8, 8)
lw[0, 0], lw[0, 3], lw[1, 1], lw[1, 4], lw[2, 2], lw[2, 0] = 0.31, 0.12, 0.22, 0.40, 0.05, 0.02
lw = lw.to(dev)
gw = (torch.rand(late.H, generator=g) * 1.5 + 0.25).bfloat16().to(dev)

# 2. the send's bytes
lib.send(seg.dev, st, lslot, lw, x, T, LAYER, pdl=PDL)
wait_gpu()
tag = (1 << 16) | (LAYER << 8) | T
xo = seg.x_off(1)
xq_t, xe_t = lib.host_quant_rows(x.cpu())
ok2 = bytes(seg.u8[seg.job_off(1):seg.job_off(1) + 512]) == lib.host_header(1, LAYER, T, lslot.cpu().numpy(),
                                                                             lw.cpu().numpy())
ok2 &= np.array_equal(seg.u8[xo:xo + T * late.H].reshape(T, late.H), xq_t)
ok2 &= np.array_equal(seg.u8[xo + 8 * late.H:xo + 8 * late.H + T * 112].reshape(T, 112), xe_t)
s_ = st.tolist()
ok2 &= seg.get(lo["door"]) == tag and s_[0] == 1 and s_[1] == tag and s_[2] == 1 and s_[3] == 3 and s_[11] > 0
lib.send(seg.dev, st, torch.full((8,), -1, dtype=torch.int32, device=dev), lw, x, T, LAYER, pdl=PDL)
wait_gpu()
s_ = st.tolist()
ok2 &= seg.get(lo["door"]) == tag and s_[1] == 0 and s_[0] == 1 and s_[2] == 1
check("2 late_send: header / rows / exponents in host memory == host twins, door, state; empty call sends nothing", ok2)

# 3. through the pool (a pool serves the jobs published after its start: send again)
pool = fp.FakePool(seg, {LAYER: bank}, device="cuda")
pool.start()
lib.send(seg.dev, st, lslot, lw, x, T, LAYER, pdl=PDL)            # seq 2, slot 2
wait_gpu()
ok3 = pool.wait_served(1) and int(st[0]) == 2
rows = torch.from_numpy(seg.rows(2, T).copy()).to(dev)
_, _, xd = late.quant_rows_reference(x)
want = late.pool_reference([pool._experts(LAYER, p) for p in (2, 0, 3)], xd.to(dev), lw[:3], 1)
ok3 &= rel(rows, want) < 1e-6
u = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16().to(dev)
mfix = (torch.randn(T, late.H, generator=g) * 0.01).to(dev)
xl_p, cf_p = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
xl_d, cf_d = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
s0 = st.clone()
lib.land(seg.dev, st, u, None, mfix, gw, xl_p, cf_p, T, EPS, LAYER, 5e6, pdl=PDL)
lib.land(None, None, u, rows, mfix, gw, xl_d, cf_d, T, EPS, LAYER, 0.0, pdl=PDL)
wait_gpu()
ok3 &= torch.equal(xl_p, xl_d) and torch.equal(cf_p, cf_d) and bool((cf_p[:T, 0] == 1).all())
d = (st - s0).tolist()
ok3 &= d[4] == 1 and d[12] == T and int(st[1]) == 0
check("3 send -> pool -> landing over C2C == the device-row landing of the same rows; counters", ok3,
      f"(pool rows vs float64 {rel(rows, want):.1e}, wait {d[9] / 1e3:.1f} us)")


def land(timeout_us, layer=LAYER):
    xl1 = torch.full((8, late.H), 2.0, dtype=torch.bfloat16, device=dev)
    cf1 = torch.full((8, 4), 5.0, device=dev)
    lib.land(seg.dev, st, u, None, None, gw, xl1, cf1, T, EPS, layer, timeout_us, pdl=PDL)
    wait_gpu()
    return bool((cf1[:T, :2] == 0).all() and (xl1[:T].float() == 0).all())


ok4, s0 = True, st.clone()
ok4 &= land(5e6)                                                  # no job (the last landing cleared the tag)
lib.send(seg.dev, st, lslot, lw, x, T, LAYER, pdl=PDL)
wait_gpu()
pool.wait_served(2)
ok4 &= land(5e6, layer=LAYER + 1)                                 # another layer's landing never reads these rows
pool.stop()                                                       # running = 0
lib.send(seg.dev, st, lslot, lw, x, T, LAYER, pdl=PDL)
s1 = st.clone()
ok4 &= land(5e6)
dw_down = int(st[9] - s1[9])
seg.set(lo["state"], 1)                                           # "running", nobody serving: the bound
lib.send(seg.dev, st, lslot, lw, x, T, LAYER, pdl=PDL)
s1 = st.clone()
ok4 &= land(200.0)
dw_to = int(st[9] - s1[9])
pool = fp.FakePool(seg, {LAYER: bank}, device="cuda")
pool.start()
pool.refuse.add(int(st[0]) + 1)
lib.send(seg.dev, st, lslot, lw, x, T, LAYER, pdl=PDL)
wait_gpu()
pool.wait_served(1)
ok4 &= land(5e6)
d = (st - s0).tolist()
ok4 &= d[7] == 2 and d[8] == 1 and d[5] == 1 and d[6] == 1 and dw_down < 50_000 and 200_000 <= dw_to <= 1_200_000
check("4 no job / other layer / pool stopped / pool silent / refused: nothing lands, each counted", ok4,
      f"(empty {d[7]}, down {d[8]} in {dw_down / 1e3:.1f} us, timeout {d[5]} after {dw_to / 1e3:.1f} us for a 200 us "
      f"bound, refused {d[6]})")

ok5 = True
for j in range(7):
    Tj = 1 + (j * 3) % 8
    xj = (torch.randn(Tj, late.H, generator=g) * 0.5).bfloat16().to(dev)
    lsj = torch.tensor([j % 4, (j + 1) % 4] + [-1] * 6, dtype=torch.int32, device=dev)
    lwj = torch.zeros(8, 8, device=dev)
    lwj[0, :Tj] = 0.1 + 0.01 * j
    lwj[1, 0] = 0.05
    n0 = len(pool.served)
    lib.send(seg.dev, st, lsj, lwj, xj, Tj, LAYER, pdl=PDL)
    wait_gpu()
    ok5 &= pool.wait_served(n0 + 1)
    slot = int(st[0]) % 4
    rj = torch.from_numpy(seg.rows(slot, Tj).copy()).to(dev)
    _, _, xdj = late.quant_rows_reference(xj)
    wj = late.pool_reference([pool._experts(LAYER, j % 4), pool._experts(LAYER, (j + 1) % 4)], xdj.to(dev), lwj[:2], 1)
    uj = (torch.randn(Tj, late.H, generator=g) * 0.7).bfloat16().to(dev)
    xa, ca = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
    xb, cb = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
    lib.land(seg.dev, st, uj, None, None, gw, xa, ca, Tj, EPS, LAYER, 5e6, pdl=PDL)
    lib.land(None, None, uj, wj.float(), None, gw, xb, cb, Tj, EPS, LAYER, 0.0, pdl=PDL)
    wait_gpu()
    ok5 &= rel(rj, wj) < 1e-6 and torch.equal(ca, cb) and torch.equal(xa, xb)
check("5 seven jobs through the ring: each landing gets its own rows", ok5)

# 6. send + landing in a CUDA graph, served by the pool at every replay
xg = (torch.randn(T, late.H, generator=g) * 0.5).bfloat16().to(dev)
ug = (torch.randn(T, late.H, generator=g) * 0.7).bfloat16().to(dev)
xl_g, cf_g = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
ls = torch.cuda.Stream()


def step():
    main = torch.cuda.current_stream()
    ls.wait_stream(main)
    with torch.cuda.stream(ls):
        lib.send(seg.dev, st, lslot, lw, xg, T, LAYER, stream=ls, pdl=PDL)
        lib.land(seg.dev, st, ug, None, None, gw, xl_g, cf_g, T, EPS, LAYER, 5e6, stream=ls, pdl=PDL)
        ev = torch.cuda.Event()
        ev.record(ls)
    main.wait_event(ev)


cs = torch.cuda.Stream()
cs.wait_stream(torch.cuda.current_stream())
gr = torch.cuda.CUDAGraph()
with torch.cuda.stream(cs):
    step()                                                         # warm-up (served like any job)
    wait_gpu()
    with torch.cuda.graph(gr, stream=cs):
        step()
ok6 = True
for r in range(3):
    xg.copy_((torch.randn(T, late.H, generator=g) * 0.5).bfloat16())
    n0 = len(pool.served)
    gr.replay()
    wait_gpu()
    ok6 &= pool.wait_served(n0 + 1) and len(pool.served) == n0 + 1 and pool.served[-1][1] == 0
    slot = int(st[0]) % 4
    rr = torch.from_numpy(seg.rows(slot, T).copy()).to(dev)
    xb, cb = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
    lib.land(None, None, ug, rr, None, gw, xb, cb, T, EPS, LAYER, 0.0, pdl=PDL)
    wait_gpu()
    _, _, xdr = late.quant_rows_reference(xg)
    wr = late.pool_reference([pool._experts(LAYER, p) for p in (2, 0, 3)], xdr.to(dev), lw[:3], 1)
    ok6 &= torch.equal(xl_g, xb) and torch.equal(cf_g, cb) and bool((cf_g[:T, 0] == 1).all()) and rel(rr, wr) < 1e-6
pool.stop()
check("6 send + landing in a CUDA graph: every replay's job served and landed", ok6, f"(served {len(pool.served)})")

# 7. the real pool
from k31_engine import cpupool  # noqa: E402

cores = sorted(os.sched_getaffinity(0))
cp = cpupool.CpuPool(os.environ.get("K31_POOL_LIB", "engine/band/libcpustage.so"), threads=4, core0=cores[-4])
bank7 = bank.numpy()
cp.set_layer(LAYER, bank7)
cpupool.pin_threads(set(cores[:-4]))                               # this process off the pool's cores
cp.start(None)
seg7 = latetier.Segment.attach(lib, cp)                            # pinned and mapped by map_device()
st7 = torch.zeros(16, dtype=torch.int64, device=dev)
st7[0] = cp.next_seq() - 1
ok7 = True
for j in range(8):
    Tj = 1 + (j * 5) % 8
    xj = (torch.randn(Tj, late.H, generator=g) * 0.5).bfloat16().to(dev)
    slots = [(j + q) % 4 for q in range(1 + j % 4)]
    lsj = torch.tensor(slots + [-1] * (8 - len(slots)), dtype=torch.int32, device=dev)
    lwj = torch.zeros(8, 8)
    for q in range(len(slots)):
        lwj[q, torch.randperm(Tj, generator=g)[:max(1, Tj // 2)]] = 0.05 + 0.1 * float(torch.rand(1, generator=g))
    uj = (torch.randn(Tj, late.H, generator=g) * 0.7).bfloat16().to(dev)
    xa, ca = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
    lib.send(seg7.dev, st7, lsj, lwj.to(dev), xj, Tj, LAYER, pdl=PDL)
    lib.land(seg7.dev, st7, uj, None, None, gw, xa, ca, Tj, EPS, LAYER, 5e6, pdl=PDL)
    wait_gpu()
    sl = int(st7[0]) % 4
    got = seg7.rows(sl, Tj).copy()
    want = cp.run_single([bank7[s_] for s_ in slots], xj.float().cpu().numpy(), lwj[:len(slots), :Tj].numpy())
    ok7 &= bool(np.array_equal(got.view(np.uint32), want.view(np.uint32)))
    xb, cb = torch.zeros(8, late.H, dtype=torch.bfloat16, device=dev), torch.zeros(8, 4, device=dev)
    lib.land(None, None, uj, torch.from_numpy(got).to(dev), None, gw, xb, cb, Tj, EPS, LAYER, 0.0, pdl=PDL)
    wait_gpu()
    ok7 &= torch.equal(xa, xb) and torch.equal(ca, cb)
s7 = st7.clone()
cp.hold(True)
lib.send(seg7.dev, st7, lsj, lwj.to(dev), xj, Tj, LAYER, pdl=PDL)
lib.land(seg7.dev, st7, uj, None, None, gw, xa, ca, Tj, EPS, LAYER, 300.0, pdl=PDL)
wait_gpu()
ok7 &= int(st7[5] - s7[5]) == 1 and bool((ca[:Tj, :2] == 0).all())
cp.hold(False)
lib.send(seg7.dev, st7, lsj, lwj.to(dev), xj, Tj, LAYER, pdl=PDL)
lib.land(seg7.dev, st7, uj, None, None, gw, xa, ca, Tj, EPS, LAYER, 5e6, pdl=PDL)
wait_gpu()
ok7 &= int(st7[4] - s7[4]) == 1
ps7 = cp.stats()
c7 = st7.tolist()
cp.stop()
check("7 the real pool: GPU-posted jobs bit-equal to its single-thread kernel, landings equal, a held pool times out",
      ok7, f"(mean wait {c7[9] / max(1, c7[4] + c7[5]) / 1e3:.1f} us, pool mean job {ps7['mean_job_us']:.0f} us on 4 "
      f"threads, stale {ps7['stale']})")

print("LATE LAND TEST " + ("PASS" if not fails else f"FAIL ({len(fails)}: {fails})"), flush=True)
sys.exit(0 if not fails else 1)
