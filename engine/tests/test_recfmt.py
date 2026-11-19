"""Record format hi2-k2 (k31_engine/recfmt.py) on the CPU: k1 -> k2 -> k1 must give the k1 record back byte for byte.
  1. synthetic records with the checkpoint's scale statistics, with and without rows that need exceptions
  2. the canonical base and the exception entries of hand-made rows; a unit with more exceptions than it holds
  3. (HI2_K1=dir of a hi2-k1 store) real records of L01.hi2
  4. (NATIVE=the official checkpoint) records built here from the checkpoint, among them experts whose rows span
     five scale values (the only ones in the checkpoint: layers 24, 25, 36), and their lo2 complement
usage: test_recfmt.py      (env: HI2_K1, NATIVE, N_REAL=24)"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_testlib as tl  # noqa: E402
from k2_testlib import LAYOUT_K1, LAYOUT_K2, recfmt  # noqa: E402

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)


g = recfmt.geom(LAYOUT_K1)
check("sizes: k1 9,289,728 B, k2 8,527,872 B, packed scales 268,992 B, units 12 x 14,912 + 7 x 12,864, 76 tiles",
      (g["k1_bytes"], g["k2_bytes"], g["p13"] + g["p2"], g["units13"], g["unit13"], g["units2"], g["unit2"],
       g["status_words"]) == (9289728, 8527872, 268992, 12, 14912, 7, 12864, 76) and LAYOUT_K2["record_bytes"] == 8527872
      and LAYOUT_K2["lo2_record_bytes"] == 9289728 and recfmt.geom(LAYOUT_K2) == g)

# 1. synthetic
t0 = time.time()
rec = tl.synth_records(3, seed=1)
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
back = recfmt.k2_to_k1(k2, LAYOUT_K2)
check(f"synthetic records, no wide rows: k1 -> k2 -> k1 exact, no exceptions ({time.time() - t0:.1f} s)",
      torch.equal(back, rec) and bool(okr.all()) and all(len(e) == 0 for e in recfmt.exceptions(k2, LAYOUT_K2))
      and int(recfmt.status(k2, LAYOUT_K2).sum()) == 0)
check("code bytes of a k2 record = the k1 record's", torch.equal(k2[:, :g["codes"]], rec[:, :g["codes"]]))
rec = tl.synth_records(3, seed=2, wide_rows=5)
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
ex = recfmt.exceptions(k2, LAYOUT_K2)
check(f"synthetic records with 5 wide rows per tensor: exact, {[len(e) for e in ex]} exception entries",
      torch.equal(recfmt.k2_to_k1(k2, LAYOUT_K2), rec) and bool(okr.all()) and all(1 <= len(e) <= 10 for e in ex))

# 2. hand-made rows: w2 row 700 of record 0 = {120 x 69, 121 x 24, 119, 122, 123} -> base 119 (window 119..122 holds
#    95, 120..123 holds 95: the lowest wins), one exception (123)
rec = tl.synth_records(1, seed=3)
o2 = g["codes"] + g["s13"]
sf2 = rec[0, o2:o2 + g["s2"]].view(24, 3584, 4)
row = torch.tensor([120] * 69 + [121] * 24 + [119, 122, 123], dtype=torch.uint8)
sf2[:, 700, :] = row.view(24, 4)
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
e = recfmt.exceptions(k2, LAYOUT_K2)[0]
u2 = g["codes"] + g["p13"] + (700 // 512) * g["unit2"]
check("a row of five values: canonical base 119, one exception (w2 row 700, K block 95, byte 123), exact",
      e == [("w2", 700, 95, 123)] and int(k2[0, u2 + 700 % 512]) == 119 and torch.equal(recfmt.k2_to_k1(k2, LAYOUT_K2), rec))
row = torch.tensor([121] * 60 + [120] * 13 + [122] * 10 + [123] * 11 + [124] * 2, dtype=torch.uint8)
sf2[:, 700, :] = row.view(24, 4)
k2, _ = recfmt.k1_to_k2(rec, LAYOUT_K2)
e = recfmt.exceptions(k2, LAYOUT_K2)[0]
check("{120 x 13, 121 x 60, 122 x 10, 123 x 11, 124 x 2}: base 120 (94 in the window), exceptions at K blocks 94, 95",
      e == [("w2", 700, 94, 124), ("w2", 700, 95, 124)] and int(k2[0, u2 + 700 % 512]) == 120
      and torch.equal(recfmt.k2_to_k1(k2, LAYOUT_K2), rec))
# four exceptions in one tile fit, in the tile's own entries; a fifth does not, and the status word says so
sf13 = rec[0, g["codes"]:g["codes"] + g["s13"]].view(28, 6144, 4)
for r in range(4):
    sf13[5, 1024 + 128 + 3 * r, 2] = sf13[:, 1024 + 128 + 3 * r, :].min() + 6
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
ua = g["codes"] + 2 * g["unit13"] + 512
ent = k2[0, ua:ua + 64].view(16, 4).to(torch.int64)
ent = (ent[:, 0] | (ent[:, 1] << 8) | (ent[:, 2] << 16) | (ent[:, 3] << 24)).tolist()
check("four exceptions in rows 1152..1279 (unit 2, tile 1): entries 4..7 of the unit, the others unused, exact",
      bool(okr[0]) and all(x == recfmt.EMPTY for x in ent[:4] + ent[8:]) and [x & 511 for x in ent[4:8]] == [128, 131, 134, 137]
      and torch.equal(recfmt.k2_to_k1(k2, LAYOUT_K2), rec))
sf13[5, 1024 + 128 + 100, 2] = sf13[:, 1024 + 128 + 100, :].min() + 6
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
check("five exceptions in one tile of 128 rows: ok = False, status 1 (the record is refused, not written wrong)",
      not bool(okr[0]) and int(recfmt.status(k2, LAYOUT_K2)[0]) == 1)
# a zero byte (native 0 or 1) is an exception like any other
rec = tl.synth_records(1, seed=4)
rec[0, g["codes"] + 4 * (3 * 6144 + 77) + 1] = 0
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
check("a zero scale byte: one exception, exact", bool(okr[0]) and torch.equal(recfmt.k2_to_k1(k2, LAYOUT_K2), rec)
      and recfmt.exceptions(k2, LAYOUT_K2)[0] == [("w13", 77, 13, 0)])

# the numpy codec (store conversion) writes the same bytes as the torch reference
rec = tl.synth_records(3, seed=7, wide_rows=6)
sf13 = rec[1, g["codes"]:g["codes"] + g["s13"]].view(28, 6144, 4)
for r in range(6):                                   # six exceptions in one tile of record 1: two do not fit
    sf13[2, 4096 + 9 * r, 3] = sf13[:, 4096 + 9 * r, :].max() + 9
k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
sfn = rec[:, g["codes"]:g["codes"] + g["s13"] + g["s2"]].numpy()
tails, over = recfmt.tails_np(sfn, LAYOUT_K2)
check("numpy codec: the same record tails as the torch reference (wide rows, a tile over its entries), and it decodes "
      "the fitting records exactly",
      torch.equal(torch.from_numpy(tails), k2[:, g["codes"]:]) and over.sum(1).tolist() == [0, 2, 0]
      and bool((recfmt.untails_np(tails, LAYOUT_K2)[[0, 2]] == sfn[[0, 2]]).all()))

# 3. real k1 records
d = os.environ.get("HI2_K1", "")
if d and os.path.exists(os.path.join(d, "L01.hi2")):
    n = int(os.environ.get("N_REAL", "24"))
    R = LAYOUT_K1["record_bytes"]
    fd = os.open(os.path.join(d, "L01.hi2"), os.O_RDONLY)
    es = [int(x) for x in torch.linspace(0, 895, n).round().tolist()]
    rec = torch.stack([torch.frombuffer(bytearray(os.pread(fd, R, e * R)), dtype=torch.uint8) for e in es])
    os.close(fd)
    t0 = time.time()
    k2, okr = recfmt.k1_to_k2(rec, LAYOUT_K2)
    t1 = time.time()
    back = recfmt.k2_to_k1(k2, LAYOUT_K2)
    sf = rec[:, g["codes"]:g["codes"] + g["s13"] + g["s2"]]
    check(f"{n} real records of L01.hi2: exact, no exceptions; stored scale bytes {int(sf.min())}..{int(sf.max())} "
          f"(encode {1e3 * (t1 - t0) / n:.0f} ms, decode {1e3 * (time.time() - t1) / n:.0f} ms per record)",
          torch.equal(back, rec) and bool(okr.all()) and sum(len(x) for x in recfmt.exceptions(k2, LAYOUT_K2)) == 0)
else:
    print("skip  real k1 records (HI2_K1 not set)")

# 4. records built from the official checkpoint, with the checkpoint's five-value rows
native = os.environ.get("NATIVE", "")
if native and os.path.exists(os.path.join(native, "model.safetensors.index.json")):
    want = {(36, 524): 8, (36, 186): 7, (24, 742): 2, (25, 385): 1, (24, 33): 1, (1, 7): 0, (92, 895): 0}
    for (L, e), nex in want.items():
        t = tl.ckpt_expert(native, L, e)
        row = tl.native_row(t)
        hi, lo = tl.records_k1(row)
        k2, okr = recfmt.k1_to_k2(hi.unsqueeze(0), LAYOUT_K2)
        ex = recfmt.exceptions(k2, LAYOUT_K2)[0]
        # the hi2 + lo2 code planes give the native code bytes back
        cb = g["codes"]
        h, l = k2[0, :cb].to(torch.int32), lo[:cb].to(torch.int32)
        nat = torch.cat((row["w13"], row["w2"])).view(-1, 2).to(torch.int32)
        n0 = ((h & 3) << 2) | (l & 3) | (((((h >> 2) & 3) << 2) | ((l >> 2) & 3)) << 4)
        n1 = (((h >> 4) & 3) << 2) | ((l >> 4) & 3) | (((((h >> 6) & 3) << 2) | ((l >> 6) & 3)) << 4)
        check(f"layer {L} expert {e} from the checkpoint: k2 exact with {len(ex)} exception entries (expected {nex}); "
              f"hi2-k2 codes + lo2 = the native row",
              torch.equal(recfmt.k2_to_k1(k2, LAYOUT_K2)[0], hi) and bool(okr[0]) and len(ex) == nex
              and torch.equal(n0, nat[:, 0]) and torch.equal(n1, nat[:, 1])
              and torch.equal(lo[cb:cb + g["s13"]], row["w13s"]))
        if nex:
            print(f"      exceptions: {ex}")
else:
    print("skip  records from the checkpoint (NATIVE not set)")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
