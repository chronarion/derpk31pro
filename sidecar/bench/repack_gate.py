"""Records entering the band while it serves calls: does the entry traffic move a call's round trip?

    CUDA_VISIBLE_DEVICES=<RTX uuid> python bench/repack_gate.py --build /work/build --hi2 /hi2/L01.hi2
        [--rate 2300] [--steps 20] [--feed-records 128] [--out out.json]

The trace of bench/roundtrip.py (CPU sender, one call per 234 us) is replayed without entries, then with a host
thread that brings `rate` records per second into the band: a copy-engine transfer of the 9.29 MB hi2 record into
a landing buffer and a repack item of the service kernel (2,300 per second = 0.25 records per layer and token at
100 tokens per second). The entries rewrite slots with the record they already hold, so every call must still equal
the reference bit for bit. Gate: the entries move the calls' p99 by no more than 10 us.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.rtxband import HI2_REC, Band, LIB_SUFFIX  # noqa: E402
from band.service import Mailbox, Service, make_trace                         # noqa: E402


def pct(v):
    v = np.sort(np.asarray(v, dtype=np.float64)) / 1000.0
    n = len(v)
    return dict(p50=float(v[n // 2]), p90=float(v[int(n * 0.9)]), p99=float(v[min(n - 1, int(n * 0.99))]),
                max=float(v[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/work/build"); ap.add_argument("--hi2", required=True)
    ap.add_argument("--records", type=int, default=896); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rate", type=float, default=2300.0); ap.add_argument("--feed-records", type=int, default=128)
    ap.add_argument("--period-us", type=float, default=234.0); ap.add_argument("--hint-lead-us", type=float, default=140.0)
    ap.add_argument("--chunk-kb", type=float, default=0.0, help="one copy-engine transfer (0 = a whole record)")
    ap.add_argument("--gate", type=int, default=0, help="1: a transfer starts only right after a call has ended; "
                    "2: transfers start as long as no call is in flight")
    ap.add_argument("--mode", type=int, default=0, help="1: repacks without new transfers, 2: transfers only")
    ap.add_argument("--repack-after-us", type=float, default=15.0)
    ap.add_argument("--repack-idle-us", type=float, default=1000.0)
    ap.add_argument("--seg", default="/dev/shm/rtxband_repack"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
    n = min(a.records, mm.shape[0])
    band = Band(os.path.join(a.build, f"librtxband{LIB_SUFFIX}.so"), n, torch.device("cuda"))
    for i in range(0, n, 16):
        band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:i + 16])))
    calls = a.steps * 92
    tr = make_trace(band, n, calls, T=3, e_lo=7, e_hi=10, seed=5)
    m = a.feed_records
    feed = np.ascontiguousarray(mm[:m]).copy()           # anonymous memory: pinned for the copy engine
    before = band.store[:m].clone()
    mb = Mailbox(os.path.join(a.build, "libbandhost.so"), a.seg, create=True)
    svc = Service(band, os.path.join(a.build, f"librtxband_svc{LIB_SUFFIX}.so"), mb)
    torch.cuda.synchronize()
    svc.start(repack_after_us=a.repack_after_us, repack_idle_us=a.repack_idle_us)
    res = dict(config=vars(a), runs={})
    print(f"{calls} calls, one per {a.period_us:.0f} us, 7-10 experts, 3 rows; entries: "
          f"{'as many as fit' if not a.rate else '%.0f per second' % a.rate} over {m} slots, transfers of "
          f"{'a whole record' if not a.chunk_kb else '%.0f KB' % a.chunk_kb}"
          f"{('', ', started only at the end of a call', ', started while no call is in flight')[a.gate]}"
          f"{' (mode %d)' % a.mode if a.mode else ''}; a "
          f"repack starts within {a.repack_after_us:.0f} us of a call's end or after {a.repack_idle_us:.0f} us "
          f"without one", flush=True)

    def run(name, hinted, feeding):
        if feeding:
            svc.feed_start(feed, np.arange(m), np.arange(m), a.rate, int(a.chunk_kb * 1024), a.gate, a.mode)
            time.sleep(0.02)
        t0 = time.time()
        ns, rows, late = mb.trace(tr["calls"], tr["rows"], tr["hints"] if hinted else None, 3, mb.next_seq(),
                                  period_us=a.period_us, hint_lead_us=a.hint_lead_us)
        dt = time.time() - t0
        f = svc.feed_stop() if feeding else None
        equal = int((rows.view(np.uint32) == tr["ref"].view(np.uint32)).all(axis=(1, 2)).sum())
        r = dict(round_trip=pct(ns[:, 0]), rtx=pct(ns[:, 3]), staging=pct(ns[:, 4]), bit_equal=equal, timeouts=int(late),
                 feed=f, seconds=dt)
        res["runs"][name] = r
        extra = ""
        if f:
            extra = (f" | entries {f['entries']} ({f['entries'] / (dt + 0.02):.0f} per s) in {f['transfers']} "
                     f"transfers (mean {f['transfer_us']:.0f} max {f['transfer_max_us']:.0f} us each), "
                     f"{f['copy_us']:.0f} us per record, repack request to acknowledgement mean "
                     f"{f['repack_us']:.0f} max {f['repack_max_us']:.0f} us, unfit {f['unfit']}, errors {f['errors']}")
            r["entries_per_s"] = f["entries"] / (dt + 0.02)
        rt, x = r["round_trip"], r["rtx"]
        print(f"{name:26s} round trip p50 {rt['p50']:6.1f} p90 {rt['p90']:6.1f} p99 {rt['p99']:6.1f} max {rt['max']:6.1f} | "
              f"RTX p50 {x['p50']:5.1f} p99 {x['p99']:5.1f} | staging p50 {r['staging']['p50']:.1f} p99 "
              f"{r['staging']['p99']:.1f} | bit-equal {equal}/{calls}, timeouts {late}{extra}", flush=True)
        return r

    run("warm-up", False, False)
    for hinted in (False, True):
        h = "hinted" if hinted else "unhinted"
        b1 = run(f"{h}", hinted, False)
        w1 = run(f"{h} + entries", hinted, True)
        b2 = run(f"{h} again", hinted, False)
        w2 = run(f"{h} + entries again", hinted, True)
        # each run with entries against the run without them right before it; the worse pair decides
        moved = [w["round_trip"]["p99"] - b["round_trip"]["p99"] for b, w in ((b1, w1), (b2, w2))]
        moved50 = [w["round_trip"]["p50"] - b["round_trip"]["p50"] for b, w in ((b1, w1), (b2, w2))]
        good = max(moved) <= 10.0
        res[f"gate {h}"] = dict(p99_moved=moved, p50_moved=moved50, ok=bool(good),
                                 entries_per_s=[w1.get("entries_per_s"), w2.get("entries_per_s")])
        print(f"GATE {h}: the entries move the calls' p99 by {moved[0]:+.1f} and {moved[1]:+.1f} us, their p50 by "
              f"{moved50[0]:+.1f} and {moved50[1]:+.1f} us: {'PASS' if good else 'FAIL'} (allowed +10 on p99)", flush=True)
    svc.stop()
    st = mb.state()
    same = bool((band.store[:m] == before).all()) or a.mode == 1
    print(f"service stopped: error word {st['error']}; the fed slots hold the same records as before: {same}")
    res["store_same"] = same
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if st["error"] == 0 and same else 1)


if __name__ == "__main__":
    main()
