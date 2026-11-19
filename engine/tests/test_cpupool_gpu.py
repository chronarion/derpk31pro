"""The Grace pool's segment seen from the GB300 (K31_POOL): map_device(), jobs written by the GPU into the ring and
served by the real pool, their rows read back by the GPU. No kernel of the pool's own runs on the GPU (the late tier's
send and landing kernels are tested with it); this checks the mapping and the visibility in both directions.

Run in the engine image from engine/ (after ./build.sh), on the GB300 with 4 cores (3 pool threads + this process):
    docker run --rm --network none --device nvidia.com/gpu=$GB --cpuset-cpus 64-67 --user $(id -u):$(id -g) \
        -e HOME=/tmp -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /engine/tests/test_cpupool_gpu.py
Exit 0 and "CPUPOOL GPU: PASS" =
  1. map_device() pins the segment and returns a device pointer; through the device view the GPU reads the magic
     the library wrote on the host;
  2. a job written by the GPU: its header (cpupool.job_header) and its rows quantized on the host (run_single's
     quantizer, put in row-planar order) are staged in HBM and written into slot seq % 4 through the device view by
     elementwise kernels; after a synchronize, the door is written by the GPU the same way: the pool runs the job
     (done tag, status 0) and its rows equal run_single's bit for bit;
  3. the GPU reads the result rows through the device view (an elementwise kernel into HBM): bit-equal to the host;
  4. 2-3 for eight jobs in a row (every slot twice, 1 to 8 records, 1 to 8 rows), timing printed for information.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import cpupool as cp  # noqa: E402

dev = torch.device("cuda")
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def make_bank(n, seed):
    rng = np.random.default_rng(seed)
    bank = np.empty((n, cp.REC_BYTES), dtype=np.uint8)
    codes = 3584 * 768 + 6144 * 896
    for e in range(n):
        bank[e, :codes] = rng.integers(0, 256, size=codes, dtype=np.uint8)
        bank[e, codes:] = rng.integers(110, 122, size=cp.REC_BYTES - codes, dtype=np.uint8)
    return bank


def gpu_write(seg, off, data: np.ndarray):
    """Bytes into the segment from the GPU: staged in HBM, written by an elementwise kernel through the device view."""
    src = torch.from_numpy(np.ascontiguousarray(data).view(np.uint8).reshape(-1)).to(dev)
    dst = seg[off:off + src.numel()]
    torch.add(src, 0, out=dst)


def gpu_read(seg, off, n) -> np.ndarray:
    """Bytes of the segment read by the GPU (an elementwise kernel into HBM), then copied to the host."""
    return torch.add(seg[off:off + n], 0).cpu().numpy()


cpus = sorted(os.sched_getaffinity(0))
nt = min(3, len(cpus) - 1)
core0 = cpus[-nt]
if nt < 1 or cpus[-nt:] != list(range(core0, core0 + nt)):
    nt, core0 = 1, cpus[-1]
pool = cp.CpuPool(threads=nt, core0=core0)
seg_dev = pool.map_device()
seg = pool.seg_t
check("map_device", isinstance(seg_dev, int) and seg_dev != 0 and seg.is_cuda and seg.numel() == cp.SEG_BYTES,
      f"device pointer {seg_dev:#x}, host {pool.addr:#x}")
magic = int(gpu_read(seg, 0, 8).view(np.uint64)[0])
check("the GPU reads the host's magic", magic == cp.MAGIC, f"{magic:#x}")

bank = make_bank(8, 3)
pool.set_layer(5, bank)
pool.start(set(cpus) - set(range(core0, core0 + nt)) or None)
rng = np.random.default_rng(11)
planar = cp.planar_index()
seq = pool.next_seq()
bad, times = 0, []
for k, (P, T) in enumerate([(1, 1), (2, 3), (3, 5), (8, 5), (1, 8), (4, 2), (5, 5), (8, 8)]):
    slots = [int(s) for s in rng.integers(0, 8, size=P)]
    x = (rng.standard_normal((T, cp.H)) * 0.5).astype(np.float32)
    w = np.zeros((P, T), dtype=np.float32)
    for p in range(P):
        w[p, rng.integers(0, T, size=2)] = rng.uniform(0.02, 0.3, size=2)
    want, xq, xe = pool.run_single([bank[s] for s in slots], x, w, want_q=True)
    xp = np.empty_like(xq)
    xp[:, planar] = xq
    tg = cp.tag(seq, 5, T)
    s = seq % cp.NSLOT
    gpu_write(seg, cp.OFF_JOB + s * cp.JOB_BYTES, cp.job_header(tg, slots, w, T, 5))
    gpu_write(seg, cp.OFF_X + s * cp.X_STRIDE, xp)
    gpu_write(seg, cp.OFF_X + s * cp.X_STRIDE + cp.X_OFF_E, xe)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_write(seg, cp.OFF_DOOR, np.array([tg], dtype=np.uint64))
    torch.cuda.synchronize()
    st = pool.wait(tg, 30.0)
    times.append((time.perf_counter() - t0) * 1e6)
    host_ok = st == 0 and np.array_equal(pool.result(tg, T).view(np.uint32), want.view(np.uint32))
    res = gpu_read(seg, cp.OFF_RES + s * cp.RES_BYTES, T * cp.H * 4).view(np.float32).reshape(T, cp.H)
    gpu_ok = np.array_equal(res.view(np.uint32), want.view(np.uint32))
    check(f"job {k}: {P} records x {T} rows written by the GPU", host_ok and gpu_ok,
          f"status {st}, rows on the host {'equal' if host_ok else 'DIFFERENT'}, read by the GPU "
          f"{'equal' if gpu_ok else 'DIFFERENT'}, door to done {times[-1]:.0f} us (incl. a synchronize)")
    seq += 1
st = pool.stats()
check("pool counters", st["done"] == 8 and st["refused"] == 0 and st["stale"] == 0, f"{st}")
pool.close()
print("CPUPOOL GPU: PASS" if not fails else f"CPUPOOL GPU: FAIL ({', '.join(fails)})")
sys.exit(0 if not fails else 1)
