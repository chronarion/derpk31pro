"""Round trip of a layer's band call on one DGX Station: GB300 send -> RTX service kernel -> GB300 finish.

    CUDA_VISIBLE_DEVICES=<RTX uuid> python bench/roundtrip.py --build /work/build --hi2 /hi2/L01.hi2
        --gb300 <GB300 uuid> [--steps 20] [--period-us 234] [--hint-lead-us 140] [--recall 0.8] [--out out.json]

Two processes share the mailbox (a /dev/shm segment, registered with CUDA in each): this one owns the RTX (store,
service kernel) and starts bench/gb_client.py with the GB300 as its only device, as an engine process and a band
sidecar would be. The trace: `steps` decode steps of 92 layer calls, T rows, real layer-1 records as stand-ins.
  --dist uniform   e_lo..e_hi distinct band experts per call (default 7-10)
  --dist rows3     the simulated band's counts at three rows per step (mean 9.6, p90 14, p99 19), capped at --cap
  --store-layers 92 --per-layer 120   a full-size store (92 layers of 120 slots, the 896 records repeated); a call
                   takes its experts from its own layer's slots. Without it the calls walk a shuffled list of
                   --records records, so nothing is in L2 unless a hint put it there (the same holds per layer).
Every call is checked bit for bit against the launched kernels' output for the same work list.

Times per call (us):
  round trip   GB300 timer, start of the send kernel -> end of the finish kernel (the rows are added)
  visible      start of the send kernel -> the finish sees the done tag
  send, add    the send kernel up to the published tag; the finish from the tag to its end
  RTX          RTX timer, tag seen by the poller -> done tag published; of it staging (tag seen -> call released)
  cpu          the same call sent and awaited by a CPU thread instead of the GB300 (no GB300 involved)
Without --gb300 only the cpu rows are measured.
Hinted runs: the call's hint list (its experts, 1 - recall of them replaced by records it does not use) goes out
--hint-lead-us before the send. From the GB300 the hint kernel is in the call's graph and the send waits on the
device for the lead, so another process on the GB300 cannot shorten it; the CPU sender paces it on the host.
  --race W:S[,W:S...]   the engine's placement: the finish starts W us after the send (the layer's own kernels) and
                   spins at most S us; a call that is not there is collected by band_late before the next send.
  --load           repeat the GB300 runs while a second GB300 kernel streams host memory (the stager's stand-in).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.gbproc import GbProcess                                              # noqa: E402
from band.rtxband import HI2_REC, Band, LIB_SUFFIX  # noqa: E402
from band.service import Mailbox, Service, make_trace                         # noqa: E402

LINES = (60, 70, 80, 95, 105, 115)


def pct(v):
    v = np.sort(np.asarray(v, dtype=np.float64)) / 1000.0
    n = len(v)
    if n == 0:
        return dict(p50=0.0, p90=0.0, p99=0.0, max=0.0, mean=0.0)
    return dict(p50=float(v[n // 2]), p90=float(v[int(n * 0.9)]), p99=float(v[min(n - 1, int(n * 0.99))]),
                max=float(v[-1]), mean=float(v.mean()))


def fmt(d):
    return f"p50 {d['p50']:6.1f}  p90 {d['p90']:6.1f}  p99 {d['p99']:6.1f}  max {d['max']:7.1f}"


def fill_store(band, mm, n, layers, per_layer):
    """Slots < 896 hold their own record; the others repeat them so that a layer's slots are distinct records."""
    base = min(n, mm.shape[0])
    for i in range(0, base, 16):
        band.load(i, torch.from_numpy(np.ascontiguousarray(mm[i:min(i + 16, base)])))
    for k in range(base, n):
        src = ((k // per_layer) * 131 + k % per_layer) % base if layers else k % base
        band.store[k] = band.store[src]
        band.sbase[k] = band.sbase[src]
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/work/build"); ap.add_argument("--hi2", required=True)
    ap.add_argument("--records", type=int, default=896); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--e-lo", type=int, default=7); ap.add_argument("--e-hi", type=int, default=10)
    ap.add_argument("--dist", default="uniform"); ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--store-layers", type=int, default=0); ap.add_argument("--per-layer", type=int, default=120)
    ap.add_argument("--period-us", type=float, default=234.0)
    ap.add_argument("--hint-lead-us", type=float, default=140.0)
    ap.add_argument("--recall", type=float, default=0.8)
    ap.add_argument("--gb300", default="", help="UUID of the GB300 for the client process")
    ap.add_argument("--seg", default="/dev/shm/rtxband_mailbox")
    ap.add_argument("--timeout-us", type=float, default=2000.0)
    ap.add_argument("--race", default="", help="W:S pairs (us): finish W after the send, spin bound S")
    ap.add_argument("--load", action="store_true"); ap.add_argument("--load-grid", type=int, default=128)
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--out", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
    n = a.store_layers * a.per_layer if a.store_layers else min(a.records, mm.shape[0])
    band = Band(os.path.join(a.build, f"librtxband{LIB_SUFFIX}.so"), n, dev)
    fill_store(band, mm, n, a.store_layers, a.per_layer)
    calls = a.steps * 92
    tr = make_trace(band, n, calls, T=a.rows, e_lo=a.e_lo, e_hi=a.e_hi, seed=5, recall=a.recall, dist=a.dist,
                    cap=a.cap, layers=a.store_layers)
    trace_file = a.seg + ".trace.npz"
    np.savez(trace_file, calls=tr["calls"], xq=tr["xq"], xs=tr["xs"], hints=tr["hints"], ref=tr["ref"])
    mb = Mailbox(os.path.join(a.build, "libbandhost.so"), a.seg, create=True)
    svc = Service(band, os.path.join(a.build, f"librtxband_svc{LIB_SUFFIX}.so"), mb)
    ne = tr["n_exp"]
    what = f"{a.e_lo}-{a.e_hi} experts" if a.dist == "uniform" else f"experts as {a.dist} capped at {a.cap}"
    store = f"{a.store_layers} layers x {a.per_layer} slots" if a.store_layers else f"{n} records"
    print(f"{a.tag} trace: {calls} calls ({a.steps} steps x 92 layers), {a.rows} rows, {what} (mean {ne.mean():.2f}, "
          f"max {ne.max()}), store {store} ({n * band.store.shape[1] / 2**30:.1f} GiB, device "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated); service grid {a.grid or svc.per_sm * svc.sms} "
          f"CTAs ({svc.regs} registers, {svc.per_sm} per SM); period {a.period_us:.0f} us, hints "
          f"{a.hint_lead_us:.0f} us ahead at recall {a.recall}", flush=True)
    torch.cuda.synchronize()
    svc.start(grid=a.grid)
    res = dict(config=vars(a), mean_experts=float(ne.mean()), runs={})

    def by_experts(v, mask=None):
        m = np.ones(len(ne), bool) if mask is None else mask
        return {int(e): float(v[ne[m] == e].mean() / 1000) for e in np.unique(ne[m])}

    def cpu_run(name, **kw):
        hints = kw.pop("hints", None)
        s0 = mb.state()
        ns, rows, late = mb.trace(tr["calls"], tr["rows"], hints, a.rows, mb.next_seq(), timeout_us=a.timeout_us, **kw)
        s1 = mb.state()
        equal = int((rows.view(np.uint32) == tr["ref"].view(np.uint32)).all(axis=(1, 2)).sum())
        r = dict(round_trip=pct(ns[:, 0]), rtx=pct(ns[:, 3]), staging=pct(ns[:, 4]), timeouts=int(late),
                 bit_equal=equal, calls=calls, by_experts=by_experts(ns[:, 0]),
                 rtx_marks_p50=[float(np.median(ns[:, k]) / 1000) for k in (4, 5, 6, 7, 3)],
                 tag_store=pct(ns[1:, 8]),
                 poll_turn_us=(s1["poll_ns"] - s0["poll_ns"]) / max(s1["poll_turns"] - s0["poll_turns"], 1) / 1000)
        res["runs"]["cpu " + name] = r
        m = r["rtx_marks_p50"]
        print(f"cpu   {name:22s} round trip {fmt(r['round_trip'])} | RTX {fmt(r['rtx'])} | marks p50: released "
              f"{m[0]:.1f}, first worker {m[1]:.1f}, sum stored {m[2]:.1f}, fenced {m[3]:.1f}, done {m[4]:.1f}; tag "
              f"store {r['tag_store']['p50']:.1f}; poll turn {r['poll_turn_us']:.2f} | bit-equal {equal}/{calls}, "
              f"timeouts {late}", flush=True)

    cpu_run("warm-up")
    cpu_run("unhinted", period_us=a.period_us)
    cpu_run("hinted", period_us=a.period_us, hints=tr["hints"], hint_lead_us=a.hint_lead_us)
    cpu_run("unhinted, no gaps")
    gb = None
    try:
        if a.gb300:
            gb = GbProcess(a.build, a.seg, trace_file, a.gb300)
            print(f"GB300 client on {gb.info['device']}: {gb.info['calls']} calls loaded, "
                  f"{gb.info['device_mib']:.0f} MiB of device memory", flush=True)

            def gb_run(name, **kw):
                out = f"{a.seg}.{len(res['runs'])}.npz"
                c = gb.run(alive=svc.running, name=name, out=out, **kw)
                with np.load(out) as z:
                    log, host = z["log"], z["host"]
                os.remove(out)
                ok = log[:, 6] == 0
                lg = log[ok]
                rt, vis = lg[:, 4] - lg[:, 0], lg[:, 3] - lg[:, 0]
                r = dict(round_trip=pct(rt), visible=pct(vis), send=pct(lg[:, 1] - lg[:, 0]),
                         add=pct(lg[:, 4] - lg[:, 3]), wait=pct(log[:, 3] - log[:, 2]),
                         finish=pct(log[:, 4] - log[:, 2]), rtx=pct(lg[:, 7] & 0xFFFFFFFF),
                         staging=pct(lg[:, 7] >> 32), host=pct(host[:, 0]),
                         gaps_over_20us=int((log[:, 5] > 20000).sum()),
                         visible_within={str(x): float((vis <= x * 1000).mean()) if len(vis) else 0.0 for x in LINES},
                         by_experts=by_experts(rt, ok) if ok.any() else {}, **{k: c[k] for k in (
                             "calls", "added", "added_equal", "timeouts", "errors", "down", "late_added", "late_equal",
                             "lost", "stray_dst", "stray_late", "load_gb_s")})
                res["runs"]["gb300 " + name] = r
                return r

            def show(name, r):
                extra = f" | stream load {r['load_gb_s']:.0f} GB/s" if r["load_gb_s"] else ""
                print(f"gb300 {name:22s} round trip {fmt(r['round_trip'])} | visible p50 {r['visible']['p50']:.1f} "
                      f"p99 {r['visible']['p99']:.1f} | send p50 {r['send']['p50']:.1f} p99 {r['send']['p99']:.1f} | "
                      f"add p50 {r['add']['p50']:.1f} p99 {r['add']['p99']:.1f} | RTX {fmt(r['rtx'])} | host p50 "
                      f"{r['host']['p50']:.1f} | bit-equal {r['added_equal']}/{r['added']}, timeouts "
                      f"{r['timeouts']}, poll gaps > 20 us: {r['gaps_over_20us']}{extra}", flush=True)

            t = dict(timeout_us=a.timeout_us)
            gb_run("warm", period_us=0, **t); gb_run("warm", period_us=0, **t)
            for rep in ("", "-2"):
                show("unhinted" + rep, gb_run("unhinted" + rep, period_us=a.period_us, **t))
                show("hinted" + rep, gb_run("hinted" + rep, period_us=a.period_us, hint_lead_us=a.hint_lead_us, **t))
            show("unhinted-nogaps", gb_run("unhinted-nogaps", period_us=0, **t))
            for name in ("unhinted", "hinted"):
                r = res["runs"]["gb300 " + name]
                print(f"      {name}: share of calls visible within " + ", ".join(
                    f"{x} us: {r['visible_within'][str(x)] * 100:.1f}%" for x in LINES), flush=True)
            for pair in [p for p in a.race.split(",") if p]:
                W, S = (float(x) for x in pair.split(":"))
                for name, lead in (("unhinted", None), ("hinted", a.hint_lead_us)):
                    r = gb_run(f"race {W:.0f}+{S:.0f} {name}", period_us=a.period_us, hint_lead_us=lead, timeout_us=S,
                               work_us=W, with_late=1)
                    print(f"gb300 race: finish {W:.0f} us after the send, spin bound {S:.0f} us, {name:8s}: on time "
                          f"{r['added']}/{r['calls']} ({r['added'] / r['calls'] * 100:.1f}%, bit-equal "
                          f"{r['added_equal']}), late and collected one call later {r['late_added']} (bit-equal "
                          f"{r['late_equal']}), lost {r['lost']}, service down {r['down']}; the finish spun "
                          f"p50 {r['wait']['p50']:.1f} mean {r['wait']['mean']:.1f} us and took mean "
                          f"{r['finish']['mean']:.1f} us; stray rows {r['stray_dst'] + r['stray_late']}", flush=True)
            if a.load:
                ld = dict(gib=2.0, grid=a.load_grid)
                show("unhinted+load", gb_run("unhinted+load", period_us=a.period_us, load=ld, **t))
                show("hinted+load", gb_run("hinted+load", period_us=a.period_us, hint_lead_us=a.hint_lead_us, load=ld, **t))
                show("unhinted after load", gb_run("unhinted after load", period_us=a.period_us, **t))
    finally:
        if gb is not None:
            gb.close()
        svc.stop()
    st = mb.state()
    res["state"] = st
    print(f"service stopped: error word {st['error']}, calls served {st['calls']}, hints taken {st['hints']}", flush=True)
    src = "gb300" if a.gb300 else "cpu"
    res["gates"] = []
    for name, p50, p99 in (("unhinted", 85.0, 120.0), ("hinted", 70.0, 95.0)):
        r = res["runs"].get(f"{src} {name}")
        if r is None:
            continue
        good = r["round_trip"]["p50"] <= p50 and r["round_trip"]["p99"] <= p99
        equal = r.get("added_equal", r.get("bit_equal")) == r["calls"] and r["timeouts"] == 0
        res["gates"].append(dict(name=name, time=bool(good), bits=bool(equal)))
        print(f"GATE {name:9s} ({src}): p50 {r['round_trip']['p50']:.1f} <= {p50:.0f} and p99 {r['round_trip']['p99']:.1f} "
              f"<= {p99:.0f}: {'PASS' if good else 'FAIL'}; bit-equal and no timeouts: {'PASS' if equal else 'FAIL'}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if st["error"] == 0 else 1)


if __name__ == "__main__":
    main()
