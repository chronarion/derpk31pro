"""The bulk band request (K31_BAND_BULK, prompt chunks) end to end on real records, against the decode calls.

Runs like test_band_e2e.py: a container with both GPUs (CDI) and --ipc host; a band sidecar subprocess on the RTX with
K31_BAND_BULK=1 serves layers 1-2 (118 records each, ranks 105..222 of rank-pass1.json); this process works on the
GB300 (CUDA_VISIBLE_DEVICES = its UUID).
  1. T = 1,000 rows of random routings (resident, band and other experts mixed), layer 1 and layer 2: the band picks
     through decode calls of 8 rows (mode 0, the service kernel) against one bulk request (mode 1, the launched
     kernels in units of <= 8 picks of one expert): float32 sums of the same products in another order (rel < 1e-5).
  2. The finish's add: y (bf16) + result against bf16(y + result) computed here (bit-equal).
  3. Mode switches both ways; a decode call after the switch back still serves (the service kernel restarted).
  4. A record that enters the band while the service kernel is parked (launched repack): completed, status 1.
  5. Timing of one request at 1,000 and 4,096 rows (the sidecar's own split: copy in, plan + pack, launches, out).
  6. The plan / hint kernel (K31_BAND_HINT, librtxband_fused_v5): flags, stats and rows equal with and without it
     (reported as "HINT:", apart from the bulk verdict).
Env: RTX, GB300 (UUIDs). Exit code 0 = PASS.
"""
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, "/engine")
sys.path.insert(0, "/work/code")
from k31_engine import band  # noqa: E402

SEG = "/dev/shm/k31band-bulktest"
LAYERS, HBM, N = (1, 2), 105, 118
RANK = "/rank/rank-pass1.json"
BUILD = os.environ.get("BULK_BUILD", "/work/build")       # the sidecar's libraries (librtxband.so with the bulk kernels)
GBLIB = os.environ.get("BULK_GBLIB", "/engine/band")      # this side's: librtxband_gb300.so, libbandbulk.so
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["RTX"], CUDA_MODULE_LOADING="EAGER", K31_BAND_BULK="1",
           K31_BAND_BULK_ROWS="4096")
side = subprocess.Popen([sys.executable, "-m", "band.sidecar", "--build", BUILD, "--hi2", "/hi2", "--rank", RANK,
                         "--hbm", str(HBM), "--n", str(N), "--seg", SEG, "--layers", f"{LAYERS[0]}-{LAYERS[-1]}",
                         "--status-s", "30"], cwd="/work/code", env=env)
try:
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    g = torch.Generator().manual_seed(1)
    rank = json.load(open(RANK))["layers"]
    t0 = time.time()
    while not os.path.exists(SEG + ".ready"):
        assert side.poll() is None, f"the sidecar left: {side.returncode}"
        assert time.time() - t0 < 600, "no ready file after 600 s"
        time.sleep(1)
    m = np.load(SEG + ".map.npy")
    gb = band.Gb(os.path.join(GBLIB, "librtxband_gb300.so"), SEG, dev, timeout_us=20000)
    bulk = band.Bulk(os.path.join(GBLIB, "libbandbulk.so"), os.path.join(BUILD, "libbandhost.so"), SEG + ".bulk", dev)
    check("bulk segment mapped", bulk.max_rows == 4096, f"(max rows {bulk.max_rows}, mode {bulk.mode})")

    def layer_state(L):
        ex = [int(e) for e in rank[str(L)]]
        row = torch.full((896,), -1, dtype=torch.int32)
        for r, e in enumerate(sorted(ex[:HBM])):
            row[e] = r
        return types.SimpleNamespace(idx=L, row_map=row.to(dev), band_map=torch.tensor(m[L], device=dev)), ex

    def routing(ex, T, gen):
        pool = torch.tensor(ex[:HBM + N + 200])
        ids = torch.stack([pool[torch.randperm(len(pool), generator=gen)[:16]] for _ in range(T)]).to(torch.int32)
        w = torch.rand((T, 16), generator=gen) + 0.01
        return ids, (w / w.sum(1, keepdim=True))

    def by_calls(st, x, ids, w):
        """The band picks through decode calls of 8 rows, as hook._band_eager makes them: float32 sums."""
        call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
        a = gb.args(call, 20000)
        y = torch.zeros(x.shape, dtype=torch.float32, device=dev)
        T = x.shape[0]
        for r0 in range(0, T, band.MAXT):
            r1 = min(T, r0 + band.MAXT)
            band.quant(x[r0:r1].contiguous(), gb.xq, gb.xs)
            for base in range(0, 16 * band.MAXT, band.MAXE):
                tmp = torch.zeros(3, dtype=torch.int32, device=dev)
                band.plan(ids[r0:r1].contiguous(), w[r0:r1].contiguous(), st.row_map, st.band_map, call, tmp, base)
                if int(call[0]) == 0:
                    break
                gb.dst[:r1 - r0].zero_()
                gb.send(a)
                gb.finish(a)
                y[r0:r1] += gb.dst[:r1 - r0]
        torch.cuda.synchronize()
        return y

    def result_rows(T):
        res = torch.frombuffer(bulk.buf, dtype=torch.uint8)[bulk.off_res:bulk.off_res + T * 3584 * 4]
        return res.view(torch.float32).view(T, 3584).clone()

    # 1-2. decode calls against one bulk request, two layers
    for L in LAYERS:
        st, ex = layer_state(L)
        T = 1000
        ids, w = routing(ex, T, g)
        ids, w = ids.to(dev), w.to(dev)
        x = (torch.randn(T, 3584, generator=g) * torch.exp(0.5 * torch.randn(T, 1, generator=g))).to(torch.bfloat16)
        x = x.to(dev)
        bulk.set_mode(False)
        ya = by_calls(st, x, ids, w)
        bulk.set_mode(True)
        y0 = (torch.randn(T, 3584, generator=g) * 0.1).to(torch.bfloat16).to(dev)
        y = y0.clone()
        bulk.send(st, x, ids, w)
        bulk.finish(st, y)
        torch.cuda.synchronize()
        yb = result_rows(T)
        rel = float((yb.to(dev) - ya).norm() / ya.norm())
        n_band = int(((st.row_map[ids.long()] < 0) & (st.band_map[ids.long()] >= 0)).sum())
        check(f"layer {L}: one bulk request = decode calls of 8 rows ({T} rows, {n_band} band picks)", rel < 1e-5,
              f"(rel {rel:.2e}; errors {bulk.err.tolist()})")
        want = (y0.float() + yb.to(dev)).to(torch.bfloat16)
        check(f"layer {L}: finish adds the result into bf16 rows", torch.equal(y, want),
              f"({int((y != want).sum())} differ)")

    # 3. back to decode mode: a decode call is served by the restarted service kernel
    st, ex = layer_state(1)
    ids, w = routing(ex, 8, g)
    x = torch.randn(8, 3584, generator=g).to(torch.bfloat16).to(dev)
    bulk.set_mode(False)
    ya = by_calls(st, x, ids.to(dev), w.to(dev))
    check("decode calls after the switch back", gb.running() and bool(torch.isfinite(ya).all()) and float(ya.abs().sum()) > 0,
          f"(service running {gb.running()})")

    # 4. an entry while the service kernel is parked (launched repack)
    bulk.set_mode(True)
    cmd = band.BandCmd(SEG + ".cmd", os.path.join(BUILD, "libbandhost.so"))
    L = 2
    ex = [int(e) for e in rank[str(L)]]
    outsider = ex[HBM + N + 5]                                  # not in the band: takes the slot of band expert 0
    slot = int(m[L, ex[HBM]])
    seq = cmd.post(L, outsider, slot)
    t1, got = time.time(), []
    while time.time() - t1 < 30 and not got:
        got = [s for s in cmd.poll() if s[0] == seq]
        time.sleep(0.01)
    check("entry while parked: completed by the launched repack", bool(got) and got[0][1] == 1, f"({got})")
    bulk.set_mode(False)

    # 5. timing
    for T in (1000, 4096):
        st, ex = layer_state(1)
        ids, w = routing(ex, T, g)
        x = torch.randn(T, 3584, generator=g).to(torch.bfloat16).to(dev)
        y = torch.zeros(T, 3584, dtype=torch.bfloat16, device=dev)
        bulk.set_mode(True)
        ts = []
        for _ in range(3):
            torch.cuda.synchronize()
            a0 = time.perf_counter()
            bulk.send(st, x, ids.to(dev), w.to(dev))
            bulk.finish(st, y)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - a0)
        parts = [bulk.get(bulk.OFF_TIMES + 8 * i) / 1e6 for i in range(4)]
        print(f"INFO bulk request of {T} rows: {1e3 * min(ts):.1f} ms end to end; sidecar ms: copy in {parts[0]:.2f}, "
              f"plan + pack {parts[1]:.2f}, launches {parts[2]:.2f}, gather + out {parts[3]:.2f}", flush=True)
    bulk.set_mode(False)

    # 6. the plan / hint kernel (K31_BAND_HINT, librtxband_fused_v5): served flags and stats equal to the send's own,
    #    rows bit-identical with the hint (reported apart: it does not decide the bulk verdict)
    try:
        fz = band.Fused(os.path.join(GBLIB, "librtxband_fused_v5.so"), gb)
        ok6, n6 = fz.has_hint, 0
        for trial in range(16):
            L = LAYERS[trial % 2]
            st, ex = layer_state(L)
            T = 1 + trial % 8
            ids, w = routing(ex, T, g)
            ids32, wf = ids.to(dev).int().contiguous(), w.to(dev).float().contiguous()
            x = torch.randn(T, 3584, generator=g).to(torch.bfloat16).to(dev)
            res = []
            for hinted in (False, True):
                stats = torch.zeros(3, dtype=torch.int32, device=dev)
                served = torch.zeros(16 * 16, dtype=torch.int32, device=dev)
                out = torch.zeros(T, 3584, dtype=torch.bfloat16, device=dev)
                if hinted:
                    fz.hint(ids32, wf, st.row_map, st.band_map, stats, L, served=served)
                    fz.qsend(ids32, wf, st.row_map, st.band_map, x, None, L, served=None)
                else:
                    fz.qsend(ids32, wf, st.row_map, st.band_map, x, stats, L, served=served)
                fz.finish(out, timeout_us=20000)
                torch.cuda.synchronize()
                res.append((stats.clone(), served.clone(), out.clone()))
            same = all(torch.equal(a_, b_) for a_, b_ in zip(res[0], res[1]))
            ok6 &= same
            n6 += int(res[0][0][0])
        print(f"{'PASS' if ok6 else 'FAIL'} hint: served flags, stats and rows equal with and without the plan / hint "
              f"kernel (16 calls, {n6} band picks served)", flush=True)
        print("HINT:", "PASS" if ok6 else "FAIL", flush=True)
    except Exception as ex:                      # noqa: BLE001
        print(f"FAIL hint: {ex}", flush=True)
        print("HINT: FAIL", flush=True)
finally:
    side.terminate()
    try:
        side.wait(60)
    except subprocess.TimeoutExpired:
        side.kill()
print("BAND BULK:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
