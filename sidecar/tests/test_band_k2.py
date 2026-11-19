"""The band with hi2-k2 records and 2-bit scale codes against the band with hi2-k1 records: the same bits.

    python tests/test_band_k2.py --build DIR [--hi2-k1 L01.hi2 --hi2-k2 L01.hi2] [--native CKPT] [--calls 48]
                                 [--engine /engine]          (DIR: librtxband.so and librtxband_k2.so, build.sh)

Two processes run the same seeded work lists on the same experts, one per record format (rtxband.py takes the format
from K31_REC_FMT when it is imported), and this process compares what they wrote:
  - every call's output rows y and the intermediate activations, bit for bit (fused launch and two launches);
  - in each format the device repack (the entry path's kernels) against the torch repack, byte for byte.
Experts: synthetic records with the checkpoint's scale statistics, some with rows that need exception entries (one
and two per row, two in one step), real records of a layer file when both formats of it are given, and with --native
records built from the official checkpoint (L36 e524 and L25 e385: the experts whose rows span five scale values).
The hi2-k2 process also checks what does not fit the band: a row with three exception entries, a status word set.
Needs engine/k31_engine/recfmt.py and engine/tests/k2_testlib.py (--engine).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def role(a):
    sys.path.insert(0, os.path.join(HERE, ".."))
    sys.path.insert(0, os.path.join(a.engine, "tests"))
    os.environ.setdefault("K31_ENGINE_DIR", a.engine)
    import k2_testlib as tl
    from k2_testlib import LAYOUT_K1, LAYOUT_K2, recfmt
    from band import rtxband as rb
    k2 = a.role == "k2"
    assert rb.K2 == k2, "K31_REC_FMT and the role disagree"
    dev = torch.device("cuda")
    G = recfmt.geom(LAYOUT_K1)
    # ---- the experts, as hi2-k1 records
    recs = [tl.synth_records(3, seed=21)]
    hard = tl.synth_records(3, seed=22)
    o2 = G["codes"] + G["s13"]
    s13 = hard[:, G["codes"]:o2].view(3, 28, 6144, 4)
    s2 = hard[:, o2:o2 + G["s2"]].view(3, 24, 3584, 4)
    # record 0: one exception in a w2 row and one in a w13 row; record 1: two in one row (different steps), two in
    # one row and step; record 2: exceptions in the first and last rows and K blocks of both tensors. Every record
    # stays within 16 scale codes (what a hi2-k1 band record holds): an exception goes below its row if above is too far

    def put(t, r, kq, row, j, d):
        m = int(t[r, :, row, :].min())
        t[r, kq, row, j] = m + d if m + d <= 122 else int(t[r, :, row, :].max()) - d - 2

    put(s2, 0, 7, 1000, 2, 5)
    put(s13, 0, 20, 77, 0, 6)
    put(s2, 1, 3, 2222, 1, 4)
    put(s2, 1, 19, 2222, 3, 4)
    put(s13, 1, 11, 4000, 0, 4)
    put(s13, 1, 11, 4000, 2, 4)
    for t, n, kq in ((s13, 6144, 28), (s2, 3584, 24)):
        put(t, 2, 0, 0, 0, 4)
        put(t, 2, kq - 1, n - 1, 3, 5)
    recs.append(hard)
    names = ["synthetic"] * 3 + ["synthetic with exception rows"] * 3
    if a.hi2_k1 and a.hi2_k2:
        idx = [0, 5, 300, 895]
        f = a.hi2_k2 if k2 else a.hi2_k1
        R = LAYOUT_K2["record_bytes"] if k2 else LAYOUT_K1["record_bytes"]
        fd = os.open(f, os.O_RDONLY)
        real = torch.stack([torch.frombuffer(bytearray(os.pread(fd, R, e * R)), dtype=torch.uint8) for e in idx])
        os.close(fd)
        names += ["layer file"] * len(idx)
    else:
        real = None
    if a.native:
        nat = []
        for L, e in ((36, 524), (25, 385)):
            nat.append(tl.records_k1(tl.native_row(tl.ckpt_expert(a.native, L, e)))[0])
        recs.append(torch.stack(nat))
        names += ["checkpoint L36 e524", "checkpoint L25 e385"]
    k1 = torch.cat(recs)
    if k2:
        conv, fits = recfmt.k1_to_k2(k1, LAYOUT_K2)
        assert bool(fits.all())
        nexc = [len(x) for x in recfmt.exceptions(conv, LAYOUT_K2)]
        hi2 = conv
    else:
        hi2, nexc = k1, [0] * k1.shape[0]
    if real is not None:                                   # the layer file's records are in this format already
        n0 = 6
        hi2 = torch.cat((hi2[:n0], real, hi2[n0:]))
        nexc = nexc[:n0] + [0] * real.shape[0] + nexc[n0:]
    n = hi2.shape[0]
    hi2 = hi2.contiguous().to(dev)
    band = rb.Band(os.path.join(a.build, f"librtxband{rb.LIB_SUFFIX}.so"), n, dev)
    band.load(0, hi2)                                      # the torch repack
    torch.cuda.synchronize()
    store_t, base_t = band.store.clone(), band.sbase.clone()
    band.store.fill_(0x5A)
    band.sbase.fill_(0x5A)
    fit_dev = [band.load_device(i, hi2[i].contiguous()) for i in range(n)]      # the entry path's kernels
    torch.cuda.synchronize()
    repack_same = bool(torch.equal(band.store, store_t)) and bool(torch.equal(band.sbase, base_t)) and all(fit_dev)
    out = {"names": names, "repack_same": repack_same, "nexc": nexc, "flags": base_t.cpu().tolist(), "y": [], "aq": [],
           "ybulk": [], "unfit": None}
    # ---- the same seeded work lists in both processes
    g = torch.Generator().manual_seed(1234)
    for c in range(a.calls):
        T = int(torch.randint(1, rb.MAXT + 1, (1,), generator=g))
        ne = int(torch.randint(1, min(rb.MAXE, n) + 1, (1,), generator=g))
        slots = torch.randperm(n, generator=g)[:ne].tolist()
        if c < n:                                          # every expert is used at least once, alone at first
            slots = [c] + [s for s in slots if s != c][:ne - 1]
        w = torch.rand((len(slots), rb.MAXT), generator=g)
        w = torch.where(torch.rand(w.shape, generator=g) < 0.2, torch.zeros_like(w), w)
        x = torch.randn((T, rb.H), generator=g) * float(torch.rand(1, generator=g) * 3 + 0.1)
        flags = rb.FLAGS_RUNNER if c % 5 == 4 else rb.FLAGS_MEGAMOE
        sf_min = rb.SF_MIN_RUNNER if c % 5 == 4 else rb.SF_MIN_MEGAMOE
        xq, xs, _ = rb.mxfp8_encode(x.to(dev), 0)
        xf = band.pack_x(xq.contiguous(), xs.contiguous()).clone()
        call = band.make_call(slots, w, T, flags=flags, sf_min=sf_min)
        y = band.run(call, xf, fused=True).clone()
        aq = band.aq.clone()
        y2 = band.run(call, xf, fused=False).clone()
        assert bool((y == y2).all()), "fused and two launches disagree"
        assert not band.error()
        out["y"].append(y.cpu().view(torch.int32).numpy().copy())
        out["aq"].append(aq[:len(slots) * rb.KQ2 * rb.BREC].cpu().numpy().copy())
    # bulk units (prompt chunks, band/bulk.py): 16 units in one launch, each its own 8 rows in and out
    import ctypes
    V = ctypes.c_void_p
    band.lib.rtxband_run_bulk.argtypes = [ctypes.c_void_p, V, ctypes.c_int, V, V, V]
    for c in range(2):
        slots = [(c * 5 + i) % n for i in range(rb.MAXE)]
        w = torch.rand((rb.MAXE, rb.MAXT), generator=g) + 0.05
        xf = torch.stack([rb.pack_x(*rb.mxfp8_encode(torch.randn((rb.MAXT, rb.H), generator=g).to(dev), 0)[:2])
                          for _ in range(rb.MAXE)]).contiguous()
        call = band.make_call(slots, w, rb.MAXT, flags=rb.FLAGS_MEGAMOE | 1024, sf_min=rb.SF_MIN_MEGAMOE)
        yu = torch.zeros((rb.MAXE * rb.MAXT, rb.H), dtype=torch.float32, device=dev)
        rc = band.lib.rtxband_run_bulk(ctypes.byref(band.c), call.data_ptr(), 1, xf.data_ptr(), yu.data_ptr(),
                                       torch.cuda.current_stream(dev).cuda_stream)
        torch.cuda.synchronize()
        assert rc == 0 and not band.error()
        out["ybulk"].append(yu.cpu().view(torch.int32).numpy().copy())
    if k2:                                                 # what the band's format cannot hold
        bad = tl.synth_records(2, seed=23)
        b2 = bad[:, o2:o2 + G["s2"]].view(2, 24, 3584, 4)
        for kq, j in ((1, 0), (9, 2), (20, 3)):            # three exception entries in one row
            b2[0, kq, 555, j] = b2[0, :, 555, :].min() + 5
        b13 = bad[1, G["codes"]:o2].view(28, 6144, 4)
        for r in range(5):                                 # five in one tile: the hi2-k2 status word is set
            b13[4, 130 + r, 1] = b13[:, 130 + r, :].min() + 5
        conv, fits = recfmt.k1_to_k2(bad, LAYOUT_K2)
        _, _, ok_t = rb.repack(conv.to(dev))
        b = rb.Band(os.path.join(a.build, f"librtxband{rb.LIB_SUFFIX}.so"), 2, dev)
        ok_d = [b.load_device(i, conv[i].contiguous().to(dev)) for i in range(2)]
        out["unfit"] = dict(torch=ok_t.cpu().tolist(), device=ok_d, k2_fits=fits.tolist())
    torch.save(out, a.dump)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi2-k1", default=""); ap.add_argument("--hi2-k2", default="")
    ap.add_argument("--native", default="")
    ap.add_argument("--engine", default=os.path.join(HERE, "..", "..", "engine"))
    ap.add_argument("--calls", type=int, default=48)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--build", default="/work/build")
    ap.add_argument("--role", default=""); ap.add_argument("--dump", default="")
    a = ap.parse_args()
    if a.role:
        return role(a)
    dumps = {}
    for r in ("k1", "k2"):
        dumps[r] = os.path.join(a.tmp, f"band-{r}.pt")
        env = dict(os.environ, K31_REC_FMT="hi2-" + r)
        cmd = [sys.executable, os.path.abspath(__file__), "--role", r, "--build", a.build, "--dump", dumps[r],
               "--engine", a.engine, "--calls", str(a.calls), "--hi2-k1", a.hi2_k1, "--hi2-k2", a.hi2_k2,
               "--native", a.native]
        subprocess.run(cmd, env=env, check=True)
    d1, d2 = (torch.load(dumps[r], weights_only=False) for r in ("k1", "k2"))
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)

    n = len(d1["names"])
    check(f"{n} experts: " + ", ".join(sorted(set(d1['names']))), d1["names"] == d2["names"])
    check("hi2-k1 build: the device repack = the torch repack, every record fits", d1["repack_same"])
    check("hi2-k2 build: the device repack = the torch repack, every record fits", d2["repack_same"])
    ex = [i for i, c in enumerate(d2["nexc"]) if c]
    check(f"hi2-k2: records with exception entries {dict((i, d2['nexc'][i]) for i in ex)} carry the band's flag, the "
          f"others do not", all((d2["flags"][i] == 1) == (d2["nexc"][i] > 0) for i in range(n)) and len(ex) >= 3)
    same_y = sum(int(np.array_equal(u, v)) for u, v in zip(d1["y"], d2["y"]))
    same_a = sum(int(np.array_equal(u, v)) for u, v in zip(d1["aq"], d2["aq"]))
    check(f"{len(d1['y'])} calls (1-8 rows, 1-16 experts, MegaMoE and runner arithmetic): output rows bit-identical in "
          f"{same_y}, intermediate activations in {same_a}", same_y == len(d1["y"]) == len(d2["y"]) and same_a == same_y)
    same_b = sum(int(np.array_equal(u, v)) for u, v in zip(d1["ybulk"], d2["ybulk"]))
    check(f"bulk units (16 units of 8 rows per launch, the prompt path's form): unit rows bit-identical in {same_b} of "
          f"{len(d1['ybulk'])} launches, non-zero", same_b == len(d1["ybulk"]) == 2 and all(int(np.count_nonzero(v)) > 0 for v in d2["ybulk"]))
    u = d2["unfit"]
    check(f"hi2-k2: a row with three exception entries and a record with a status word set do not fit the band "
          f"(torch {u['torch']}, device {u['device']}; the second is not a valid hi2-k2 record either: {u['k2_fits']})",
          u["torch"] == [False, False] and u["device"] == [False, False] and u["k2_fits"] == [True, False])
    print("BAND K2: PASS" if ok else "BAND K2: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
