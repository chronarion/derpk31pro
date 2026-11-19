"""k31_engine.cpupool (K31_POOL, the Grace pool's engine-side wrapper) on the CPU: no GPU. Needs
engine/band/libcpustage.so (./build.sh or engine/pool/build.sh) and at least 2 cores (with 3 or more the pool leaves
one for this process).

Run in the engine image, from engine/ (the CPU set picks the pool's cores: the highest ones, up to 4 threads):
    docker run --rm --network none --cpuset-cpus 67-69 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_cpupool.py
Exit 0 and "CPUPOOL: PASS" =
  1. from_env without K31_POOL=1 returns None and starts nothing (the default path);
  2. the library's segment layout equals the module's (cpustage_layout), the magic is written;
  3. start(): the process's other threads move to the host cores, each pool thread is pinned to its own core;
  4. jobs posted through the ring (1 / 2 / 3 / 8 records, 1-5 rows) give rows bit-equal to run_single on the same
     records; a refused job (bad slot) reports status 1; jobs held and released run in order (FIFO); land() and
     result() address the right slot; next_seq() is above every tag seen; stats() count what happened;
  5. selfcheck() passes; a pool on a core outside the CPU set does not start;
  6. from_env with K31_POOL=1: banks registered from (idx, bank, nrec), started, self-check line, cpupool.POOL set.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
os.environ.pop("K31_POOL", None)
from k31_engine import cpupool as cp  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def make_bank(n, seed):
    """n synthetic hi2-k1 records: random codes, scale bytes 110-121 (a few 0: empty blocks)."""
    rng = np.random.default_rng(seed)
    bank = np.empty((n, cp.REC_BYTES), dtype=np.uint8)
    codes = 3584 * 768 + 6144 * 896
    for e in range(n):
        bank[e, :codes] = rng.integers(0, 256, size=codes, dtype=np.uint8)
        sc = rng.integers(110, 122, size=cp.REC_BYTES - codes, dtype=np.uint8)
        sc[rng.integers(0, sc.size, size=40)] = 0
        bank[e, codes:] = sc
    return bank


def rows(rng, T):
    x = (rng.standard_normal((T, cp.H)) * 0.5).astype(np.float32)
    x[:, 64:96] = 0.0
    return x


def weights(rng, n, T):
    w = np.zeros((n, T), dtype=np.float32)
    for p in range(n):
        w[p, rng.integers(0, T, size=1 + int(rng.integers(0, min(T, 3))))] = rng.uniform(0.02, 0.3)
    return w


def same(a, b):
    return np.array_equal(np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32))


# 1. the default path
check("from_env without K31_POOL", cp.from_env([(7, None, 0)]) is None and cp.POOL is None)

cpus = sorted(os.sched_getaffinity(0))
if len(cpus) < 2:
    print("CPUPOOL: need at least 2 cores")
    sys.exit(1)
nt = min(4, len(cpus) - 1)
core0 = cpus[-nt]
if cpus[-nt:] != list(range(core0, core0 + nt)):
    nt, core0 = 1, cpus[-1]
host = set(cpus) - set(range(core0, core0 + nt))

# 2. layout
pool = cp.CpuPool(threads=nt, core0=core0)
check("segment magic", pool.word(0) == cp.MAGIC and pool.word(8) == cp.SEG_BYTES, f"{pool.word(0):#x}")
bank = make_bank(6, 1)
pool.set_layer(7, bank)
pool.set_layer(9, bank, 3)

# 3. start and pinning
pool.start(host)
masks = {}
for tid in os.listdir("/proc/self/task"):
    try:
        masks[int(tid)] = os.sched_getaffinity(int(tid))
    except ProcessLookupError:
        pass
pinned = sorted(next(iter(m)) for m in masks.values() if len(m) == 1 and next(iter(m)) >= core0)
others = [m for m in masks.values() if not (len(m) == 1 and next(iter(m)) >= core0)]
check("pool threads pinned one per core", pinned == list(range(core0, core0 + nt)), f"{pinned}")
check("other threads on the host cores", all(m == host for m in others), f"{len(others)} threads")
check("running word", pool.stats()["running"] == 1)

# 4. jobs through the ring against run_single
rng = np.random.default_rng(5)
seq, bad, njobs = pool.next_seq(), 0, 0
for P in (1, 2, 3, 8):
    for T in (1, 3, 5):
        slots = [int(s) for s in rng.integers(0, 6, size=P)]
        x, w = rows(rng, T), weights(rng, P, T)
        want = pool.run_single([bank[s] for s in slots], x, w)
        tg = pool.post(seq, 7, slots, x, w)
        st = pool.wait(tg, 30.0)
        ok = st == 0 and same(pool.result(tg, T), want)
        bad += not ok
        njobs += 1
        seq += 1
check("ring jobs bit-equal to run_single", bad == 0, f"{njobs - bad} of {njobs}")
# what post() wrote (the library's own header and quantizer) against job_header / planar_index / run_single's rows
slots, x, w = [3, 1, 4], rows(rng, 5), weights(rng, 3, 5)
pool.hold(True)
time.sleep(0.002)
tg = pool.post(seq, 7, slots, x, w)
s = seq % cp.NSLOT
hdr = pool.u8[cp.OFF_JOB + s * cp.JOB_BYTES:cp.OFF_JOB + s * cp.JOB_BYTES + cp.JOB2_BYTES]
_, xq, xe = pool.run_single([bank[i] for i in slots], x, w, want_q=True)
xo = cp.OFF_X + s * cp.X_STRIDE
xp = pool.u8[xo:xo + 5 * cp.H].view(np.int8).reshape(5, cp.H)
check("job_header equals the library's header", np.array_equal(hdr, cp.job_header(tg, slots, w, 5, 7)))
check("slot rows: row-planar int8 + exponent bytes", np.array_equal(xp[:, cp.planar_index()], xq)
      and np.array_equal(pool.u8[xo + cp.X_OFF_E:xo + cp.X_OFF_E + 5 * cp.KB].reshape(5, cp.KB), xe))
pool.hold(False)
check("that job runs", pool.wait(tg, 30.0) == 0)
seq += 1
njobs += 1
x, w = rows(rng, 2), weights(rng, 1, 2)
tg = pool.post(seq, 7, [6], x, w)
seq += 1
check("a bad slot is refused", pool.wait(tg, 30.0) == 1)
tg = pool.post(seq, 9, [4], x, w)                       # layer 9 holds 3 records
seq += 1
check("a slot beyond the layer's records is refused", pool.wait(tg, 30.0) == 1)
s0 = pool.stats()
pool.hold(True)
time.sleep(0.002)
jobs = []
for k in range(3):
    slots = [int(s) for s in rng.integers(0, 6, size=2)]
    x, w = rows(rng, 4), weights(rng, 2, 4)
    jobs.append((pool.post(seq, 7, slots, x, w), pool.run_single([bank[s] for s in slots], x, w)))
    seq += 1
pool.hold(False)
okq = all(pool.wait(tg, 30.0) == 0 for tg, _ in jobs) and all(same(pool.result(tg, 4), want) for tg, want in jobs)
s1 = pool.stats()
check("held jobs run in order", okq and s1["done"] - s0["done"] == 3 and s1["queue_max"] >= 2,
      f"queue_max {s1['queue_max']}")
fake = rng.standard_normal((3, cp.H)).astype(np.float32)
ft = cp.tag(seq + 5, 11, 3)
pool.land(ft, fake, 0)
check("land / result / done address the tag's slot",
      same(pool.result(ft, 3), fake) and pool.done((seq + 5) % 4) == (ft, 0))
check("next_seq above every tag", pool.next_seq() == seq + 6, f"{pool.next_seq()} vs {seq + 6}")
st = pool.stats()
check("stats", st["refused"] == 2 and st["done"] == njobs + 3 and st["picks"] > 0 and st["mean_job_us"] > 0,
      f"{st}")

# 5. self-check; a pool that cannot pin
r0 = pool.stats()["resets"]
us = pool.selfcheck(7)
check("selfcheck", us > 0, f"{us:.0f} us")
fresh = pool.door() == 0 and all(pool.done(s) == (0, 0) for s in range(cp.NSLOT)) and pool.next_seq() == 1
check("selfcheck leaves a fresh ring the pool has followed", fresh and pool.stats()["resets"] == r0 + 1)
slots, x, w = [2, 5], rows(rng, 5), weights(rng, 2, 5)
tg = pool.post(1, 7, slots, x, w)
check("a sender starting at 1 after the self-check is served",
      pool.wait(tg, 30.0) == 0 and same(pool.result(tg, 5), pool.run_single([bank[i] for i in slots], x, w)))
pool.stop()
check("stopped", pool.stats()["running"] == 0)
outside = next((c for c in range(1024) if c not in set(cpus)), None)
if outside is not None:
    p2 = cp.CpuPool(threads=1, core0=outside)
    try:
        p2.start()
        check("a core outside the CPU set refuses to start", False)
        p2.stop()
    except RuntimeError as e:
        check("a core outside the CPU set refuses to start", "not all in" in str(e))

# 6. from_env
os.environ.update({"K31_POOL": "1", "K31_POOL_THREADS": str(nt), "K31_POOL_CORE0": str(core0)})
lines = []
p3 = cp.from_env([(7, bank, 6), (8, None, 0), (9, bank, 2)], log=lines.append)
check("from_env started", p3 is not None and cp.POOL is p3 and p3.started and sorted(p3.layers) == [7, 9],
      lines[0] if lines else "")
check("from_env self-check", bool(lines) and "bit-equal" in lines[0])
p3.stop()

print("CPUPOOL: PASS" if not fails else f"CPUPOOL: FAIL ({', '.join(fails)})")
sys.exit(0 if not fails else 1)
