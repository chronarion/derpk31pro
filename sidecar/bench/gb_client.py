"""The GB300's half of the round-trip benchmark and tests: a process with the GB300 as its only device that
replays the trace through the send / finish / late kernels on request. Started by bench/roundtrip.py and
tests/test_roundtrip.py; commands arrive as JSON lines on stdin, every answer is one "RESULT {json}" line.

    {"cmd": "run", "name": ..., "period_us": 234, "hint_lead_us": null | 140, "timeout_us": 2000, "work_us": 0,
     "with_late": false, "n": null, "load": null | {"gib": 2, "grid": 128}, "out": "/path/run.npz"}
    {"cmd": "load", "trace": "/path/trace.npz"}
    {"cmd": "quit"}

A run's answer has the counts (statuses, bit-equality of the rows added on time and of the rows collected late
against the reference, rows that must be zero); its log goes to the .npz file.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.client import LATE_ADDED, LATE_LOST, ST_ADDED, ST_DOWN, ST_EMPTY, ST_ERROR, ST_TIMEOUT, Gb300   # noqa: E402
from band.service import Mailbox                                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--seg", required=True)
ap.add_argument("--trace", required=True)
a = ap.parse_args()


def load_file(path):
    with np.load(path) as z:                             # every array once (an .npz member is re-read per access)
        t = {k: z[k] for k in ("calls", "xq", "xs", "hints", "ref")}
    gb.load_trace(t["calls"], t["xq"], t["xs"], t["hints"])
    return t, t["ref"].view(np.uint32)


mb = Mailbox(os.path.join(a.build, "libbandhost.so"), a.seg, create=False)
gb = Gb300(os.path.join(a.build, "librtxband_gb300.so"), mb)
tr, ref = load_file(a.trace)
print("RESULT " + json.dumps(dict(device=torch.cuda.get_device_name(0), calls=int(tr["calls"].shape[0]),
                                  device_mib=torch.cuda.memory_allocated() / 2**20)), flush=True)
for line in sys.stdin:
    c = json.loads(line)
    if c["cmd"] == "quit":
        break
    if c["cmd"] == "load":                               # another trace file
        tr, ref = load_file(c["trace"])
        print("RESULT " + json.dumps(dict(calls=int(tr["calls"].shape[0]))), flush=True)
        continue
    load = c.get("load")
    n = c.get("n") or tr["calls"].shape[0]
    period = float(c.get("period_us", 234.0))
    if load:
        gb.load_start(load.get("gib", 2.0), load.get("grid", 128), seconds=n * max(period, 150.0) * 1e-6 + 1.0)
    r = gb.run(period, c.get("hint_lead_us"), float(c.get("timeout_us", 2000.0)), float(c.get("work_us", 0.0)),
               int(c.get("with_late", 0)), n)
    rate = gb.load_stop() if load else 0.0
    log, dst, late = r["log"], r["dst"].view(np.uint32), r["late"].view(np.uint32)
    status, lstat = log[:, 6], log[:, 8]
    on = status == ST_ADDED
    got = lstat == LATE_ADDED
    eq_dst = (dst == ref[:n]).all(axis=(1, 2))
    eq_late = (late == ref[:n]).all(axis=(1, 2))
    zero_dst = (dst == 0).all(axis=(1, 2))
    zero_late = (late == 0).all(axis=(1, 2))
    if c.get("out"):
        np.savez(c["out"], log=log, host=r["host"])
    fin = np.sort(log[:, 4] - log[:, 2]) / 1000.0        # the finish kernel, first poll to end
    host = np.sort(r["host"][:, 0]) / 1000.0             # graph launch -> synchronized, on the host
    ans = dict(name=c.get("name", ""), calls=int(n), added=int(on.sum()), added_equal=int((on & eq_dst).sum()),
               timeouts=int((status == ST_TIMEOUT).sum()), errors=int((status == ST_ERROR).sum()),
               empty=int((status == ST_EMPTY).sum()), down=int((status == ST_DOWN).sum()),
               late_added=int(got.sum()), late_equal=int((got & eq_late).sum()), lost=int((lstat == LATE_LOST).sum()),
               stray_dst=int((~on & ~zero_dst).sum()), stray_late=int((~got & ~zero_late).sum()),
               finish_us_p50=float(fin[n // 2]), finish_us_p99=float(fin[min(n - 1, int(n * 0.99))]),
               host_us_p50=float(host[n // 2]),
               state=r["state"], load_gb_s=rate, out=c.get("out"))
    print("RESULT " + json.dumps(ans), flush=True)
