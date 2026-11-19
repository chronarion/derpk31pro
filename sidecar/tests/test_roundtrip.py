"""The GB300 side of the mailbox against the service kernel on the RTX: send, finish, late, hint.

    CUDA_VISIBLE_DEVICES=<RTX uuid> python tests/test_roundtrip.py --build /work/build --hi2 /hi2/L01.hi2
        --gb300 <GB300 uuid> [--records 256]

Two processes, as bench/roundtrip.py. Checked:
  - every call added on time equals the launched kernels' output bit for bit, paced and back to back, with hints;
  - a finish that gives up adds nothing and marks the call late; band_late one call later adds exactly the call's
    rows to the late buffer; with a bound in the middle of the distribution every call is in exactly one of the two;
  - a call that is still not there when band_late looks is lost and counted, and the sequence goes on, also when
    the sends come faster than the RTX serves them;
  - a work list without experts is not sent at all;
  - with the service kernel gone the finish does not spin (status "down"); after a relaunch the calls sent in
    between are skipped and the next ones are served;
  - nothing is ever added twice or to the wrong buffer; the service's error word stays zero.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.gbproc import GbProcess                                              # noqa: E402
from band.rtxband import HI2_REC, Band, LIB_SUFFIX  # noqa: E402
from band.service import Mailbox, Service, make_trace                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/work/build"); ap.add_argument("--hi2", required=True)
    ap.add_argument("--records", type=int, default=256); ap.add_argument("--gb300", required=True)
    ap.add_argument("--seg", default="/dev/shm/rtxband_test2"); ap.add_argument("--calls", type=int, default=400)
    a = ap.parse_args()
    mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
    n = a.records
    band = Band(os.path.join(a.build, f"librtxband{LIB_SUFFIX}.so"), n, torch.device("cuda"))
    for i in range(0, n, 16):
        band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
    tr = make_trace(band, n, a.calls, T=3, e_lo=7, e_hi=10, seed=21)
    trace_file = a.seg + ".trace.npz"
    np.savez(trace_file, calls=tr["calls"], xq=tr["xq"], xs=tr["xs"], hints=tr["hints"], ref=tr["ref"])
    tr0 = make_trace(band, n, a.calls, T=2, e_lo=0, e_hi=2, seed=22)            # a third of the calls have no expert
    trace0 = a.seg + ".trace0.npz"
    np.savez(trace0, calls=tr0["calls"], xq=tr0["xq"], xs=tr0["xs"], hints=tr0["hints"], ref=tr0["ref"])
    n_empty = int((tr0["n_exp"] == 0).sum())
    mb = Mailbox(os.path.join(a.build, "libbandhost.so"), a.seg, create=True)
    svc = Service(band, os.path.join(a.build, f"librtxband_svc{LIB_SUFFIX}.so"), mb)
    torch.cuda.synchronize()
    svc.start()
    gb = GbProcess(a.build, a.seg, trace_file, a.gb300)
    ok = True
    N = a.calls

    def check(name, cond, r, alive=True):
        nonlocal ok
        st = mb.state()
        clean = r["stray_dst"] == 0 and r["stray_late"] == 0 and r["errors"] == 0 and st["error"] == 0 and \
            r["added_equal"] == r["added"] and r["late_equal"] == r["late_added"]
        good = bool(cond) and clean
        ok &= good
        print(f"{name:58s} on time {r['added']:4d}  late {r['timeouts']:4d} -> collected {r['late_added']:4d}, lost "
              f"{r['lost']:4d}  down {r['down']:4d}  all rows bit-equal and in the right buffer: {clean}"
              + ("" if good else "   <-- FAIL"), flush=True)

    try:
        run = lambda **kw: gb.run(alive=svc.running, **kw)                         # noqa: E731
        run(name="warm", period_us=0, timeout_us=2000)
        r = run(name="warm", period_us=0, timeout_us=2000)
        # Another process on the GB300 (time slices of milliseconds) makes the paced launches arrive in bursts: a
        # late call may then be looked for before the RTX has it. With a shared GB300 "collected" becomes
        # "collected or lost"; everything else is checked as it is.
        shared = r["host_us_p50"] > 500.0
        print(f"GB300 {'shared with another process' if shared else 'to ourselves'}: a call takes "
              f"{r['host_us_p50']:.0f} us from launch to synchronized on the host", flush=True)
        back = (lambda r: r["late_added"] + r["lost"]) if shared else (lambda r: r["late_added"] if r["lost"] == 0 else -1)
        r = run(name="paced", period_us=234, timeout_us=2000, with_late=1)
        check("paced, 2 ms bound", r["added"] == N and r["timeouts"] == 0, r)
        r = run(name="back to back", period_us=0, timeout_us=2000, with_late=1)
        check("back to back, 2 ms bound", r["added"] == N and r["timeouts"] == 0, r)
        r = run(name="hinted", period_us=234, hint_lead_us=140, timeout_us=2000, with_late=1)
        check("hints 140 us ahead", r["added"] == N and r["timeouts"] == 0, r)
        r = run(name="all late", period_us=234, timeout_us=20, with_late=1)
        check("20 us bound right after the send: every call late", r["added"] == 0 and back(r) == N, r)
        r = run(name="half late", period_us=234, timeout_us=66, with_late=1)
        check("66 us bound: each call on time or collected late", 0 < r["added"] < N and r["added"] + back(r) == N, r)
        r = run(name="race", period_us=234, timeout_us=20, work_us=50, with_late=1)
        check("finish 50 us after the send, 20 us bound", r["added"] + back(r) == N and r["added"] > 0, r)
        r = run(name="no late kernel", period_us=234, timeout_us=20, with_late=0)
        check("20 us bound, nobody collects: nothing added", r["added"] == 0 and r["late_added"] == 0, r)
        r = run(name="lost", period_us=234, timeout_us=5, with_late=2)
        check("5 us bound, collected right behind the finish: all lost", r["added"] == 0 and r["late_added"] == 0
              and r["lost"] == N, r)
        r = run(name="flood", period_us=0, timeout_us=5, with_late=1)
        check("back to back with a 5 us bound: collected or lost", r["added"] + r["late_added"] + r["lost"] == N, r)
        time.sleep(0.05)
        r = run(name="after the losses", period_us=234, timeout_us=2000, with_late=1)
        check("paced again after the losses", r["added"] == N, r)
        # layers without a band expert: nothing is sent, the finish returns at once
        gb.load(trace0)
        served = mb.state()["calls"]
        r = run(name="empty", period_us=234, timeout_us=2000, with_late=1)
        sent = mb.state()["calls"] - served
        check(f"0-2 experts, 2 rows: {n_empty} calls without an expert are not sent", r["empty"] == n_empty
              and r["added"] == N - n_empty and sent == N - n_empty, r)
        gb.load(trace_file)
        # the service kernel leaves; the GB300 goes on sending
        svc.stop()
        r = gb.run(name="down", period_us=234, timeout_us=2000, with_late=1)
        check(f"service kernel gone: no spin (finish p99 {r['finish_us_p99']:.1f} us of a 2000 us bound)",
              r["down"] == N and r["added"] == 0 and r["finish_us_p99"] < 200.0, r)
        svc.start()
        time.sleep(0.02)
        r = run(name="relaunched", period_us=234, timeout_us=2000, with_late=1)
        check("relaunched: the calls sent meanwhile are skipped", r["added"] == N, r)
    finally:
        gb.close()
        svc.stop()
    st = mb.state()
    print(f"service stopped: error word {st['error']}, calls served {st['calls']}")
    ok &= st["error"] == 0
    print("ROUND-TRIP TESTS", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
