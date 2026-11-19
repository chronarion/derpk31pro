"""Timing of the band kernel on real records.

    python bench/bench_band.py --lib /work/build/librtxband.so --hi2 /work/data/L01.hi2 [--records 896]
                               [--experts 1,4,8,16] [--rows 1,3,8] [--json out.json]

Every call uses different experts and the work lists cycle through the whole store, so no record is re-read while
it could still be in the 128 MB L2 cache (unless --same-experts). Times per call:
  sync   one call launched and synchronized from the host: median, p90
  chain  calls queued back to back on one stream, one synchronize: mean per call (what a longer graph pays per call)
  graph  a CUDA graph holding all the work lists' calls, replayed: mean per call
  graph1 a one-call CUDA graph launched and synchronized: median, p90
  hint   as sync, after an L2 prefetch of the call's experts has completed (the pre-gated case): median, p90
  two    chain time of the two-launch form (linear 1, linear 2); l1, l2: chain times of each linear alone
The fused single-launch kernel unless --two. GB/s = experts x 9,289,728 B / time: hi2 record bytes served per second
(the band record read is 8,773,632 B per expert).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band.rtxband import H, HI2_REC, MAXT, REC, Band, mxfp8_encode, pack_x             # noqa: E402


def make_calls(band, n_records, E, T, n_calls, gen, same=False, flags=16):
    calls = []
    for i in range(n_calls):
        base = 0 if same else i * E
        slots = [(base + j) % n_records for j in range(E)]
        w = torch.zeros(E, MAXT)
        for e in range(E):
            use = torch.rand(T, generator=gen) < 0.5
            use[int(torch.randint(0, T, (1,), generator=gen))] = True
            w[e, :T] = torch.where(use, torch.rand(T, generator=gen) * 0.2 + 0.01, torch.zeros(T))
        calls.append(band.make_call(slots, w, T, flags=flags))
    return torch.stack(calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--hi2", required=True)
    ap.add_argument("--records", type=int, default=896)
    ap.add_argument("--experts", default="1,2,4,8,16")
    ap.add_argument("--rows", default="1,3,8")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--same-experts", action="store_true")
    ap.add_argument("--modes", default="sync,chain,graph,graph1,hint,two,l1,l2")
    ap.add_argument("--two", action="store_true", help="time the two-launch form in sync / chain / graph modes")
    ap.add_argument("--json", default=None)
    ap.add_argument("--flags", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(1)
    mm = np.memmap(args.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
    n = min(args.records, mm.shape[0])
    band = Band(args.lib, n, dev)
    for a in range(0, n, 16):
        band.load(a, torch.from_numpy(np.ascontiguousarray(mm[a:min(n, a + 16)])))
    torch.cuda.synchronize()
    print(f"{n} records of {args.hi2} in the band store ({n * REC / 1e9:.2f} GB), device "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    modes = args.modes.split(",")
    res = []
    hdr = f"{'E':>2s} {'T':>1s} |"
    for m in modes:
        hdr += f" {m + ' us':>10s} {'GB/s':>6s}" + (f" {'p90':>7s}" if m in ("sync", "graph1", "hint") else "") + " |"
    print(hdr, flush=True)
    for E in [int(v) for v in args.experts.split(",")]:
        for T in [int(v) for v in args.rows.split(",")]:
            x = torch.randn(T, H, generator=gen).to(dev)
            xq, xs, _ = mxfp8_encode(x, 0)
            xf = pack_x(xq, xs)
            n_calls = max(1, min(n // E, 128))
            calls = make_calls(band, n, E, T, n_calls, gen, same=args.same_experts, flags=args.flags)
            row = dict(E=E, T=T, n_calls=n_calls)
            line = f"{E:2d} {T:1d} |"
            for m in modes:
                wh = 3 if args.two else 4
                if m == "sync":
                    t, p90 = band.bench(calls, xf, 0, wh, args.iters)
                elif m == "chain":
                    t, p90 = band.bench(calls, xf, 1, wh, args.iters)
                elif m == "graph":
                    t, p90 = band.bench(calls, xf, 2, wh, 20)
                elif m == "graph1":
                    t, p90 = band.bench(calls, xf, 3, wh, args.iters)
                elif m == "hint":
                    t, p90 = band.bench(calls, xf, 4, wh, args.iters, pause_us=20 + 12 * E)
                elif m == "two":
                    t, p90 = band.bench(calls, xf, 1, 3, args.iters)
                elif m == "l1":
                    t, p90 = band.bench(calls, xf, 1, 1, args.iters)
                elif m == "l2":
                    t, p90 = band.bench(calls, xf, 1, 2, args.iters)
                else:
                    raise ValueError(m)
                nbytes = E * HI2_REC * (2 / 3 if m == "l1" else 1 / 3 if m == "l2" else 1)
                row[m] = dict(us=t * 1e6, p90_or_best_us=p90 * 1e6, gbps=nbytes / t / 1e9)
                line += f" {t * 1e6:10.1f} {nbytes / t / 1e9:6.0f}" + (f" {p90 * 1e6:7.1f}" if m in ("sync", "graph1", "hint") else "") + " |"
            print(line, flush=True)
            res.append(row)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
