"""Correctness of the band kernel against the torch reference (band/reference.py).

flags column: 1 weight after linear 2, 2 no bf16 rounding of gate / up, 4 activation rounded to bf16, 8 precise SiTU
(expf / tanhf; without it the kernel uses MegaMoE's fast-math forms and the reference torch's), 16 each expert's
output rounded to bf16. 16 = the MegaMoE preset (24 with precise SiTU), 29 = the runner preset.

    python tests/test_band.py --lib /work/build/librtxband.so [--hi2 /work/data/L01.hi2] [--quick]

Records: real layer-1 experts from the hi2 store when given, and synthetic ones (random codes, scale codes like
K3.1's). For each case: relative error of y (norm and worst element against the row's largest value), the number
of intermediate FP8 codes and block scales that differ from the reference's, and a determinism check.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band import reference as ref                                             # noqa: E402
from band.rtxband import (BREC, CODES13, CODES2, FLAG_ACT_BF16, FLAG_NO_BF16, FLAG_OUT_BF16, FLAG_PRECISE,   # noqa: E402
                          FLAG_WEIGHT_AFTER, FLAGS_MEGAMOE, FLAGS_RUNNER, H, HI2_REC, I, KQ2, MAXE, MAXT, SF13, SF2, W2C,
                          W13C, W13S, Band, aq_index, aq_scale_index, mxfp8_encode, pack_x, repack)


def synthetic_records(n, gen, device):
    rec = torch.randint(0, 256, (n, HI2_REC), generator=gen, dtype=torch.uint8).to(device)
    o = CODES13 + CODES2
    # scale codes like K3.1's (mostly 120 and 119) with a tail over the whole 16-code span 107..122
    u = torch.rand((n, SF13 + SF2), generator=gen)
    sf = torch.where(u < 0.76, 120, torch.where(u < 0.97, 119, 107 + torch.randint(0, 16, u.shape, generator=gen)))
    sf[:, 0], sf[:, 1] = 107, 122
    rec[:, o:] = sf.to(torch.uint8).to(device)
    return rec


def case(band, recs, slots, T, w, x_scale, flags, sf_min, gen, dev, experts_cache, fused=True):
    x = torch.randn(T, H, generator=gen).to(dev) * x_scale
    xq, xs, xv = mxfp8_encode(x, 0)
    xf = band.pack_x(xq.contiguous(), xs.contiguous()).clone()                 # the device packer
    assert bool((xf == pack_x(xq, xs)).all()), "device packer and torch packer disagree"
    call = band.make_call(slots, w, T, flags=flags, sf_min=sf_min)
    y = band.run(call, xf, fused=fused).clone()
    aq = band.aq.clone()
    y2 = band.run(call, xf, fused=not fused).clone()                          # the other launch form: same bits
    experts = []
    for s in slots:
        if s not in experts_cache:
            experts_cache[s] = ref.decode_hi2(recs[s])
        experts.append(experts_cache[s])
    wt = torch.as_tensor(w, dtype=torch.float32).reshape(len(slots), MAXT)
    yr, inter = ref.forward(xv, experts, wt, flags=flags & ~FLAG_PRECISE, sf_min=sf_min)
    err = (y[:T].double() - yr)
    rel = float(err.norm() / yr.norm().clamp_min(1e-300))
    worst = float((err.abs().amax(1) / yr.abs().amax(1).clamp_min(1e-300)).max())
    idx, sidx = aq_index(dev), aq_scale_index(dev)
    code_diff = sf_diff = total = 0
    for e, (codes, sf) in enumerate(inter):
        rec_e = aq[e * KQ2 * BREC:(e + 1) * KQ2 * BREC]
        kc = rec_e[idx.view(-1)].view(MAXT, I)[:T]
        ks = rec_e[sidx.view(-1)].view(MAXT, I // 32)[:T]
        used = (wt[e, :T] != 0).to(dev)
        code_diff += int((kc != codes)[used].sum())
        sf_diff += int((ks != sf)[used].sum())
        total += int(used.sum()) * I
        assert int((kc[~used] & 0x7F).sum()) == 0, "rows that do not use an expert must give (signed) zero activations"
    tail = float(y[T:].abs().max()) if T < MAXT else 0.0
    gain = float((err * yr).sum() / (yr * yr).sum())          # systematic part of the error: a gain on y ...
    bias = float(err.mean() / yr.pow(2).mean().sqrt())        # ... or an offset, in units of y's rms
    return dict(rel=rel, worst=worst, code_diff=code_diff, sf_diff=sf_diff, total=total, same=bool((y == y2).all()),
                tail=tail, ynorm=float(yr.norm()), gain=gain, bias=bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--hi2", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(args.seed)
    n_syn, n_real = 6, (20 if args.hi2 else 0)
    recs = synthetic_records(n_syn, gen, dev)
    if args.hi2:
        mm = np.memmap(args.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
        pick = torch.randperm(mm.shape[0], generator=gen)[:n_real].tolist()
        real = torch.from_numpy(np.stack([mm[i] for i in pick])).to(dev)
        recs = torch.cat((recs, real))
        print(f"real records: layer file {args.hi2}, experts {pick}")
    band = Band(args.lib, recs.shape[0], dev)
    band.load(0, recs)

    # 1. the stager's expansion keeps the record's bits and sets the low two (as build_store.py --check)
    nat = ref.expand(recs[:2, :CODES13])
    pairs_back = ((nat & 0x0C) >> 2) | ((nat & 0xC0) >> 4)
    b = pairs_back.view(2, -1, 2)
    assert bool(((b[..., 0] | (b[..., 1] << 4)) == recs[:2, :CODES13]).all()) and bool(((nat & 0x33) == 0x33).all())
    print("1. stager expansion round-trips, low bits set: ok")

    # 2. the repack is lossless: same multiset of 2-bit pairs; scale codes = 4-bit code + base; and a record whose
    #    scale codes do not fit (a zero code, or a span over 16) is refused
    rp, base0 = band.store[0], int(band.sbase[0])
    def pair_counts(t):
        return torch.stack([(((t >> (2 * j)) & 3).view(-1, 1) == torch.arange(4, device=dev)).sum(0) for j in range(4)]).sum(0)
    src_pairs = pair_counts(recs[0, :CODES13 + CODES2])
    dst_pairs = pair_counts(rp[:W13C]) + pair_counts(rp[W13C + W13S:W13C + W13S + W2C])
    sc = torch.cat((rp[W13C:W13C + W13S], rp[W13C + W13S + W2C:]))
    dst_sf = torch.bincount(torch.cat((sc & 15, sc >> 4)).long() + base0, minlength=256)
    src_sf = torch.bincount(recs[0, CODES13 + CODES2:].long(), minlength=256)
    assert bool((src_pairs == dst_pairs).all()) and bool((src_sf == dst_sf).all())
    bad = recs[:2].clone()
    bad[0, CODES13 + CODES2 + 5] = 0
    bad[1, CODES13 + CODES2 + 7] = bad[1, CODES13 + CODES2:].amin() + 16
    assert not bool(repack(bad)[2].any())
    # the device repack kernels write the same bytes as the torch repack
    spare = Band(args.lib, 1, dev)
    for j in (0, recs.shape[0] - 1):
        assert spare.load_device(0, recs[j].contiguous())
        assert bool((spare.store[0] == band.store[j]).all()) and int(spare.sbase[0]) == int(band.sbase[j])
    assert not spare.load_device(0, bad[0].contiguous()) and not spare.load_device(0, bad[1].contiguous())
    assert int(spare._range.abs().sum()) == 0
    del spare
    print("2. repack keeps the multiset of pairs and the scale codes; out-of-range records are refused; the device "
          "repack equals the torch repack: ok")

    cache = {}
    rows = []
    syn = list(range(n_syn))
    real_slots = list(range(n_syn, n_syn + n_real))
    def rand_w(n, T):
        w = torch.zeros(n, MAXT)
        for e in range(n):
            k = int(torch.randint(1, T + 1, (1,), generator=gen))
            sel = torch.randperm(T, generator=gen)[:k]
            w[e, sel] = torch.rand(k, generator=gen) * 0.29 + 0.01
        return w
    cases = []
    for name, pool in (("synthetic", syn), ("real L01", real_slots)):
        if not pool:
            continue
        for (n, T) in ((1, 1), (1, 8), (4, 3), (min(8, len(pool)), 8), (min(16, len(pool)), 8), (min(5, len(pool)), 1)):
            for flags, sf_min in ((FLAGS_MEGAMOE, 105), (FLAGS_MEGAMOE | FLAG_PRECISE, 105), (FLAGS_RUNNER, 0),
                                  (FLAGS_RUNNER & ~FLAG_PRECISE, 0), (0, 105), (FLAG_PRECISE, 105)):
                cases.append((name, pool[:n], T, flags, sf_min, 1.0))
        cases.append((name, pool[:3], 3, FLAG_NO_BF16 | FLAG_PRECISE, 105, 1.0))
        cases.append((name, pool[:3], 3, FLAG_WEIGHT_AFTER | FLAG_ACT_BF16 | FLAG_PRECISE, 0, 1.0))
        cases.append((name, pool[:3], 3, FLAGS_MEGAMOE, 0, 0.2))
        cases.append((name, pool[:3], 3, FLAGS_MEGAMOE, 105, 4.0))
        if args.quick:
            cases = cases[:6]
    print(f"{'records':10s} {'E':>2s} {'T':>1s} {'flags':>5s} {'sfmin':>5s} {'xscale':>6s} | {'rel err':>9s} {'worst':>9s} | "
          f"{'FP8 codes differing':>22s} {'scales':>6s} | same bits (fused / two launches), tail | gain, offset of the error")
    ok = True
    for name, slots, T, flags, sf_min, xsc in cases:
        w = rand_w(len(slots), T)
        r = case(band, recs, slots, T, w, xsc, flags, sf_min, gen, dev, cache, fused=bool(len(rows) % 2 == 0))
        rows.append(r)
        good = r["rel"] < 1e-2 and r["same"] and r["tail"] == 0.0
        ok &= good
        print(f"{name:10s} {len(slots):2d} {T:1d} {flags:5d} {sf_min:5d} {xsc:6.1f} | {r['rel']:9.2e} {r['worst']:9.2e} | "
              f"{r['code_diff']:8d} of {r['total']:9d} {r['sf_diff']:6d} | {str(r['same']):5s}  {r['tail']:.1e} | "
              f"{r['gain']:+.1e} {r['bias']:+.1e}" + ("" if good else "   <-- FAIL"))
    # 3. an empty call gives zeros
    call = band.make_call([], torch.zeros(0, MAXT), 3)
    band.y.fill_(1.0)
    z = band.run(call, torch.zeros(band.sizes[2], dtype=torch.uint8, device=dev), fused=True).clone()
    band.y.fill_(1.0)
    z2 = band.run(call, torch.zeros(band.sizes[2], dtype=torch.uint8, device=dev), fused=False)
    zero = float(z.abs().max()) == 0.0 and float(z2.abs().max()) == 0.0
    ok &= zero and int(band.sync.abs().sum()) == 0 and not band.error()
    print("3. empty call:", "zeros" if zero else "NOT ZERO", "; counters back at zero:", int(band.sync.abs().sum()) == 0)
    # 4. calls queued back to back without a synchronize in between (the counters must hand over cleanly): 96
    #    random calls, both launch forms mixed, against the same calls run one at a time
    pool = list(range(recs.shape[0]))
    queued, lists = [], []
    for i in range(96):
        n = int(torch.randint(0, min(MAXE, len(pool)) + 1, (1,), generator=gen))
        T = int(torch.randint(1, MAXT + 1, (1,), generator=gen))
        slots = [pool[j] for j in torch.randperm(len(pool), generator=gen)[:n].tolist()]
        xq, xs, _ = mxfp8_encode(torch.randn(T, H, generator=gen).to(dev), 0)
        lists.append((band.make_call(slots, rand_w(n, T) if n else torch.zeros(0, MAXT), T), pack_x(xq, xs)))
    torch.cuda.synchronize()
    for i, (call, xf) in enumerate(lists):
        queued.append(band.run(call, xf, fused=bool(i % 3), sync=False).clone())
    torch.cuda.synchronize()
    chained = all(bool((q == band.run(call, xf, fused=False).clone()).all()) for q, (call, xf) in zip(queued, lists))
    clean = int(band.sync.abs().sum()) == 0 and not band.error()
    ok &= chained and clean
    print(f"4. 96 calls queued back to back equal the same calls run one at a time: {chained}; counters clean: {clean}")
    print("TESTS", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
