"""The persistent service kernel on the RTX, driven from a CPU thread through the mailbox (no GB300).

    python tests/test_service.py --build /work/build --hi2 /hi2/L01.hi2 [--records 256] [--seg /dev/shm/rtxband_test]

Every result row must equal the launched kernels' output for the same call bit for bit. Covers: calls back to back
and paced, 1..16 experts, 1..8 rows, an empty call, hints (results unchanged), work lists that name slots outside the
store or too many experts (refused), a record entering the band while the kernel runs (copy-engine transfer +
repack item), a kernel that parks itself when idle and is relaunched, and the counters and error word after every
phase.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.rtxband import H, HI2_REC, MAXT, Band                               # noqa: E402
from band.service import Mailbox, Service, make_trace                         # noqa: E402


def stats(ns, col=0):
    v = np.sort(ns[:, col]) / 1000.0
    return f"p50 {v[len(v) // 2]:6.1f}  p90 {v[int(len(v) * 0.9)]:6.1f}  p99 {v[min(len(v) - 1, int(len(v) * 0.99))]:6.1f}  max {v[-1]:6.1f} us"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/work/build")
    ap.add_argument("--hi2", required=True)
    ap.add_argument("--records", type=int, default=256)
    ap.add_argument("--seg", default="/dev/shm/rtxband_test")
    ap.add_argument("--grid", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
    n = a.records
    band = Band(os.path.join(a.build, "librtxband.so"), n, dev)
    for i in range(0, n, 16):
        band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
    mb = Mailbox(os.path.join(a.build, "libbandhost.so"), a.seg, create=True)
    svc = Service(band, os.path.join(a.build, "librtxband_svc.so"), mb)
    print(f"service kernel: {svc.regs} registers per thread, {svc.per_sm} CTAs per SM, {svc.sms} SMs -> grid "
          f"{a.grid or svc.per_sm * svc.sms}; mailbox {mb.bytes} B at {a.seg}", flush=True)
    ok = True

    # references first: nothing else may run on the device once the service kernel is there
    # (the last slot is kept out of the traces: the entry test rewrites it)
    tr = make_trace(band, n - 1, 128, T=3, e_lo=7, e_hi=10, seed=1)
    odd = [make_trace(band, n - 1, 12, T=T, e_lo=lo, e_hi=hi, seed=10 + j)
           for j, (T, lo, hi) in enumerate(((1, 1, 1), (8, 16, 16), (3, 1, 3), (8, 4, 12), (2, 0, 2)))]
    # the entry test: slot `x` holds a wrong record until the service repacks the right one into it
    x = n - 1
    tr_x = make_trace(band, n - 1, 8, T=3, e_lo=4, e_hi=6, seed=77)
    tr_x["calls"] = tr_x["calls"].copy()
    calls32 = tr_x["calls"].view(np.int32)
    calls32[:, 8] = x                                    # every call's first expert becomes slot x
    for i in range(8):
        call = torch.from_numpy(calls32[i].copy()).to(dev)
        xq, xs = torch.from_numpy(tr_x["xq"][i]).to(dev), torch.from_numpy(tr_x["xs"][i]).to(dev)
        from band.rtxband import pack_x
        tr_x["ref"][i] = band.run(call, pack_x(xq, xs), fused=False)[:3].cpu().numpy()
    right, right_base = band.store[x].clone(), int(band.sbase[x])
    band.store[x] = band.store[x - 1]
    band.sbase[x] = band.sbase[x - 1]
    hi2_x = np.ascontiguousarray(mm[x]).copy()
    assert svc.lib.rtxband_svc_pin(hi2_x.ctypes.data, HI2_REC) == 0
    torch.cuda.synchronize()

    def check(name, t, **kw):
        nonlocal ok, seq
        ns, rows, late = mb.trace(t["calls"], t["rows"], kw.pop("hints", None), t["rows"].shape[1], seq, **kw)
        seq += t["calls"].shape[0]
        same = bool(np.array_equal(rows, t["ref"]))
        st = mb.state()
        good = same and late == 0 and st["error"] == 0 and int(ns[:, 2].max()) <= 2
        ok &= good
        print(f"{name:46s} {t['calls'].shape[0]:4d} calls  bit-equal {same}  timeouts {late} | round trip {stats(ns)}"
              f" | on the RTX: tag seen to done p50 {np.median(ns[:, 3]) / 1000:5.1f}, of it staging "
              f"{np.median(ns[:, 4]) / 1000:4.1f} us" + ("" if good else "   <-- FAIL"), flush=True)
        return ns

    seq = 1
    svc.start(grid=a.grid)
    time.sleep(0.01)
    assert svc.running() and mb.state()["running"] == 1
    check("7-10 experts, T = 3, back to back", tr)
    check("7-10 experts, T = 3, one per 234 us", tr, period_us=234.0)
    check("the same with hints (recall 0.8, 140 us ahead)", tr, period_us=234.0, hints=tr["hints"], hint_lead_us=140.0)
    for t, name in zip(odd, ("1 expert, T = 1", "16 experts, T = 8", "1-3 experts, T = 3", "4-12 experts, T = 8",
                             "0-2 experts (empty calls), T = 2")):
        check(name, t, period_us=234.0)
    # work lists that are not work lists: refused with the error bit, nothing computed, the next call is served
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
    b32 = bad["calls"].view(np.int32)
    b32[0, 8] = n + 5                                    # a slot outside the store
    b32[1, 8 + 3] = -1
    b32[2, 0] = 17                                       # more experts than a call takes
    ns, rows, late = mb.trace(bad["calls"][:8], bad["rows"][:8], None, 3, seq, period_us=234.0)
    seq += 8
    refused = [int(x) for x in ns[:, 2]] == [3, 3, 3, 0, 0, 0, 0, 0] and not rows[:3].any() and \
        bool(np.array_equal(rows[3:], tr["ref"][3:8])) and mb.state()["error"] == 0
    print(f"three calls with slots outside the store or 17 experts: refused, the following five served: {refused}",
          flush=True)
    ok &= refused
    # a record enters the band while the kernel runs
    wrong = mb.trace(tr_x["calls"], tr_x["rows"], None, 3, seq)[1]
    seq += 8
    fits, base, t_copy, t_pack = svc.enter(hi2_x, x)
    ns = check("after a record entered the last slot", tr_x, period_us=234.0)
    print(f"  entry: fits {fits}, scale base {base} (launched repack: {right_base}), copy {t_copy * 1e6:.0f} us, repack "
          f"item {t_pack * 1e6:.0f} us; before it the calls differed from the reference: "
          f"{not np.array_equal(wrong, tr_x['ref'])}", flush=True)
    ok &= fits and base == right_base and not np.array_equal(wrong, tr_x["ref"])
    svc.stop()
    st = mb.state()
    same_rec = bool((band.store[x] == right).all()) and int(band.sbase[x]) == right_base
    ctl = svc.ctl.cpu().numpy().view(np.int32)           # ints 160..165: fin, fin_gen, phase, err, range; 192..255: sync
    clean = int(np.abs(ctl[192:256]).sum()) == 0 and ctl[160] == 0 and ctl[162] == 0 and ctl[163] == 0 and \
        ctl[164] == 0 and ctl[165] == 0
    print(f"stopped: running {st['running']}, error {st['error']}, calls served {st['calls']}, hints {st['hints']}; "
          f"the entered slot equals the launched repack: {same_rec}; counters zero: {clean}", flush=True)
    ok &= st["running"] == 0 and st["error"] == 0 and same_rec and clean

    # parking: the kernel leaves after its idle limit, a relaunch continues with the next seq
    svc.start(grid=a.grid, idle_limit_s=0.02)
    check("with a 20 ms idle limit", tr)
    time.sleep(0.2)
    parked = not svc.running() and mb.state()["running"] == 0
    svc.start(grid=a.grid)
    check("relaunched after parking", tr, period_us=234.0)
    svc.stop()
    print(f"parked by itself after the idle limit: {parked}")
    ok &= parked
    print("SERVICE TESTS", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
