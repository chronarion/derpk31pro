"""stager2.py on the GPU: every kernel against the reference codecs, byte for byte, for the four combinations of record
format (hi2-k1, hi2-k2) and row format (int32 scale words, 6-bit scale codes).

  stage     records -> staging rows: code bytes = the table expansion, scale bytes = the record's (k1 bytes or their
            6-bit coding); untouched rows stay as they were; hi2-k1 into plain rows also equals stager.py's kernel
  ring      the two-destination variant (late on / off, ring rows)
  expand    the copy-engine lane's kernel
  merge     hi2 (either format) + lo2 -> the native row (codes and unlowered scales)
  extract   native row -> hi2 record = recfmt's canonical encoding, the whole record (status words and tail included)
Records: synthetic with the checkpoint's scale statistics and rows that need exceptions; with HI2_K1 real records of
L01.hi2; with NATIVE records built from the official checkpoint (experts with five-value rows).
Runs on any GPU with TMA bulk copies: K31_CUDA_ARCH=120a for an RTX PRO 6000 (default 103a, the GB300). The bank is
device memory here (the kernels read whatever address they are given).
usage: test_stager2.py [--time]"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_testlib as tl  # noqa: E402
from k2_testlib import LAYOUT_K1, LAYOUT_K2, recfmt  # noqa: E402
from k31_engine import sf6, stager, stager2  # noqa: E402

dev = torch.device("cuda")
SLOTS = stager2.SLOTS
G = recfmt.geom(LAYOUT_K1)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)


class Mod:
    """The four kernel tensors of `rows` experts as vLLM registers them, every byte = fill; six: 6-bit scale banks."""

    def __init__(self, rows, six, fill=0):
        f4 = fill * 0x01010101
        f4 = f4 - (1 << 32) if f4 >= (1 << 31) else f4
        f1 = fill - 256 if fill >= 128 else fill
        self.fill = fill
        self.w13_weight = torch.full((rows, 6144, 1792), f1, dtype=torch.int8, device=dev)
        self.w2_weight = torch.full((rows, 3584, 1536), f1, dtype=torch.int8, device=dev)
        self.w13_weight_scale = torch.full((rows, 28, 6144), f4, dtype=torch.int32, device=dev).transpose(1, 2)
        self.w2_weight_scale = torch.full((rows, 24, 3584), f4, dtype=torch.int32, device=dev).transpose(1, 2)
        if six:
            self._k31_sf6 = (torch.full((rows, sf6.row_bytes(6144, 28)), fill, dtype=torch.uint8, device=dev),
                             torch.full((rows, sf6.row_bytes(3584, 24)), fill, dtype=torch.uint8, device=dev))

    def row(self, r):
        """(w13 code bytes, w2 code bytes, w13 scale bytes, w2 scale bytes) of row r as stored (views)"""
        out = [stager._row_bytes(self.w13_weight, r), stager._row_bytes(self.w2_weight, r)]
        c = stager2.coded(self)
        out += [stager._row_bytes(self.w13_weight_scale, r), stager._row_bytes(self.w2_weight_scale, r)] if c is None \
            else [c[0][r], c[1][r]]
        return out

    def untouched(self, r):
        return all(int((x != self.fill).sum()) == 0 for x in self.row(r))


def lut_expand(rec):
    """the stager's code expansion (store/build_store.py expand): record bytes -> native bytes"""
    p = [(rec >> (2 * j)) & 3 for j in range(4)]
    n = [(q << 2) | 3 for q in p]
    return torch.stack((n[0] | (n[1] << 4), n[2] | (n[3] << 4)), dim=-1).reshape(-1)


def want_scales(sf_k1, six):
    """k1 scale bytes [s13 + s2] of one record -> the two rows' scale bytes as stored"""
    a, b = sf_k1[:G["s13"]], sf_k1[G["s13"]:]
    if not six:
        return [a, b]
    t13 = a.clone().view(torch.int32).view(1, 28, 6144).permute(0, 2, 1)
    t2 = b.clone().view(torch.int32).view(1, 24, 3584).permute(0, 2, 1)
    return [sf6.encode(t13)[0], sf6.encode(t2)[0]]


def want_row(k1_rec, six):
    """the staging row a hi2-k1 record expands to"""
    return [lut_expand(k1_rec[:G["c13"]]), lut_expand(k1_rec[G["c13"]:G["codes"]])] + \
        want_scales(k1_rec[G["codes"]:G["codes"] + G["s13"] + G["s2"]], six)


def same_row(mod, r, want):
    return all(torch.equal(a, b) for a, b in zip(mod.row(r), want))


# ---------------------------------------------------------------------------------------------------- records
recs = {"synthetic, 4 wide rows per tensor": tl.synth_records(3, seed=11, wide_rows=4).to(dev)}
d = os.environ.get("HI2_K1", "")
if d and os.path.exists(os.path.join(d, "L01.hi2")):
    R = LAYOUT_K1["record_bytes"]
    fd = os.open(os.path.join(d, "L01.hi2"), os.O_RDONLY)
    recs["real records of L01.hi2"] = torch.stack(
        [torch.frombuffer(bytearray(os.pread(fd, R, e * R)), dtype=torch.uint8) for e in (7, 100, 555, 895)]).to(dev)
    os.close(fd)
native = os.environ.get("NATIVE", "")
natives = {}
if native and os.path.exists(os.path.join(native, "model.safetensors.index.json")):
    his = []
    for L, e in ((36, 524), (25, 385), (24, 742), (1, 7)):
        row = tl.native_row(tl.ckpt_expert(native, L, e))
        hi, lo = tl.records_k1(row)
        his.append(hi)
        natives[(L, e)] = ({k: v.to(dev) for k, v in row.items()}, hi.to(dev), lo.to(dev))
    recs["records from the checkpoint (L36 e524, L25 e385, L24 e742: five-value rows; L1 e7)"] = torch.stack(his).to(dev)

TIME = "--time" in sys.argv
lay = {False: LAYOUT_K1, True: LAYOUT_K2}
for k2 in (False, True):
    for six in (False, True):
        tag = f"{'hi2-k2' if k2 else 'hi2-k1'} records, {'6-bit' if six else 'int32'} scale rows"
        for name, k1 in recs.items():
            n = k1.shape[0]
            if k2:
                bank, okr = recfmt.k1_to_k2(k1, LAYOUT_K2)
                assert bool(okr.all())
            else:
                bank = k1
            bank = bank.contiguous()
            nexc = sum(len(x) for x in recfmt.exceptions(bank, LAYOUT_K2)) if k2 else 0
            # stage: records 0..n-1 into slots 3, 0, 15, 8 of the rows 2..17
            mod = Mod(SLOTS + 3, six, fill=0x5A)
            st = stager.Stager(mod, 2, lay[k2])
            assert isinstance(st, stager2.Stager2) == (k2 or six), type(st)
            slots = [3, 0, 15, 8][:n]
            idx = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
            for j, s in enumerate(slots):
                idx[s] = j
            st(bank.data_ptr(), idx)
            torch.cuda.synchronize()
            good = all(same_row(mod, 2 + s, want_row(k1[j], six)) for j, s in enumerate(slots))
            good &= all(mod.untouched(r) for r in range(SLOTS + 3) if r - 2 not in slots)
            check(f"stage   {tag}: {name}" + (f" ({nexc} exception entries)" if k2 else ""), good)
            if not (k2 or six):                           # the same through stager2's kernel
                mod2 = Mod(SLOTS + 3, False, fill=0x5A)
                st2 = stager2.Stager2(mod2, 2, LAYOUT_K1)
                st2(bank.data_ptr(), idx)
                torch.cuda.synchronize()
                check(f"stage   {tag}: stager2's kernel = stager.py's, {name}",
                      all(same_row(mod2, r, mod.row(r)) for r in range(SLOTS + 3)))
            # ring: slot k's record -> shared row k (late) and ring row ring[k]
            for late in (True, False):
                mod = Mod(SLOTS + 8, six, fill=0x33)
                rs = stager.RingStager(mod, 0, SLOTS, lay[k2])
                src = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
                ring = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
                plan = [(1, 0, 5), (6, 1 % n, -1), (9, 2 % n, 0)]             # (slot, record, ring row or -1)
                for s, j, rr in plan:
                    src[s], ring[s] = j, rr
                rs(bank.data_ptr(), src, ring, late)
                torch.cuda.synchronize()
                touched = set()
                good = True
                for s, j, rr in plan:
                    if late:
                        good &= same_row(mod, s, want_row(k1[j], six)); touched.add(s)
                    if rr >= 0:
                        good &= same_row(mod, SLOTS + rr, want_row(k1[j], six)); touched.add(SLOTS + rr)
                good &= all(mod.untouched(r) for r in range(SLOTS + 8) if r not in touched)
                check(f"ring    {tag}, late {int(late)}: {name}", good)
            # expand (the copy-engine lane): records in device memory -> rows 4 .. 4 + n - 1
            mod = Mod(n + 6, six, fill=0x11)
            stager2.expand(bank.data_ptr(), bank.shape[1], n, mod, 4, lay[k2], 64)
            torch.cuda.synchronize()
            good = all(same_row(mod, 4 + j, want_row(k1[j], six)) for j in range(n))
            good &= all(mod.untouched(r) for r in list(range(4)) + [n + 4, n + 5])
            check(f"expand  {tag}: {name}", good)
            del mod
        # merge and extract on native rows from the checkpoint
        for (L, e), (row, hi, lo) in natives.items():
            hi_rec = recfmt.k1_to_k2(hi.unsqueeze(0), LAYOUT_K2)[0][0].contiguous() if k2 else hi
            mod = Mod(3, six, fill=0x77)
            stager.merge(hi_rec.data_ptr(), lo.data_ptr(), mod, 1, lay[k2], blocks=64)
            torch.cuda.synchronize()
            want = [row["w13"], row["w2"]] + want_scales(torch.cat((row["w13s"], row["w2s"])), six)
            good = same_row(mod, 1, want) and mod.untouched(0) and mod.untouched(2)
            check(f"merge   {tag}: layer {L} expert {e}: hi2 + lo2 = the native row", good)
            out = torch.full((lay[k2]["record_bytes"],), 0xEE, dtype=torch.uint8, device=dev)
            stager.extract_hi2(mod, 1, lay[k2], out, blocks=152)
            torch.cuda.synchronize()
            ref = torch.empty_like(out)
            stager.extract_record(mod, 1, lay[k2], ref)
            check(f"extract {tag}: layer {L} expert {e}: the kernel's record = the store's record = the reference "
                  f"encoder's", torch.equal(out, hi_rec[:out.numel()]) and torch.equal(out, ref))
            del mod
        # rows the fast path cannot take, against the reference encoder: a tile with exactly its four exceptions (two in
        # one row, one in the same lane's next row, one in another lane), a zero scale byte, a byte above 127, and a tile
        # with nine exceptions (five do not fit: the status word says so)
        if k2 and natives:
            row, hi, lo = next(iter(natives.values()))
            for case in ("four in a tile, a zero byte, a byte above 127", "nine in a tile"):
                mod = Mod(1, six)
                stager.merge(hi.data_ptr(), lo.data_ptr(), mod, 0, LAYOUT_K1 if not six else LAYOUT_K2, blocks=64)
                torch.cuda.synchronize()
                sc = stager2.row_scales(mod, 0).clone()
                v13 = sc[:G["s13"]].view(28, 6144, 4)
                v2 = sc[G["s13"]:].view(24, 3584, 4)
                if case.startswith("four"):
                    for r, kq, j in ((641, 0, 3), (641, 12, 2), (642, 5, 0), (717, 23, 1)):
                        v2[kq, r, j] = v2[:, r, :].min() + 7
                    v13[9, 3, 2] = 0
                    if not six:                         # a 6-bit row holds 96..159 only
                        v13[20, 5000, 1] = 200
                else:
                    for r in range(9):
                        v13[3, 2048 + 5 * r, 1] = v13[:, 2048 + 5 * r, :].max() + 5
                if six and case.startswith("four"):
                    v13[9, 3, 2] = 97                   # the lowest byte a 6-bit row can hold beside its neighbours: an exception
                if six:
                    mod._k31_sf6[0][0] = sf6.encode(v13.reshape(-1).clone().view(torch.int32).view(1, 28, 6144).permute(0, 2, 1))[0]
                    mod._k31_sf6[1][0] = sf6.encode(v2.reshape(-1).clone().view(torch.int32).view(1, 24, 3584).permute(0, 2, 1))[0]
                else:
                    stager._row_bytes(mod.w13_weight_scale, 0).copy_(v13.reshape(-1))
                    stager._row_bytes(mod.w2_weight_scale, 0).copy_(v2.reshape(-1))
                ref = torch.empty(LAYOUT_K2["record_bytes"], dtype=torch.uint8, device=dev)
                stager.extract_record(mod, 0, LAYOUT_K2, ref)
                st_ref = int(recfmt.status(ref.unsqueeze(0), LAYOUT_K2)[0])
                nex = len(recfmt.exceptions(ref.unsqueeze(0), LAYOUT_K2)[0])
                good = True
                for blocks in (19, 64, 152, 1216):
                    out = torch.full((LAYOUT_K2["record_bytes"],), 0xEE, dtype=torch.uint8, device=dev)
                    stager.extract_hi2(mod, 0, LAYOUT_K2, out, blocks=blocks)
                    torch.cuda.synchronize()
                    good &= torch.equal(out, ref)
                check(f"extract {tag}: {case}: the kernel's record = the reference encoder's at 19 / 64 / 152 / 1216 "
                      f"blocks ({nex} exception entries, status {st_ref})",
                      good and st_ref == (0 if case.startswith("four") else 5))
                if st_ref == 0:                         # and the record stages back into the row it came from
                    mod2 = Mod(SLOTS, six)
                    idx = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
                    idx[4] = 0
                    stager.Stager(mod2, 0, LAYOUT_K2)(ref.data_ptr(), idx)
                    torch.cuda.synchronize()
                    lowered = torch.where(sc > 0, sc - 1, sc)
                    check(f"stage   {tag}: {case}: the staged scale bytes = the row's, lowered by one",
                          all(torch.equal(a_, b_) for a_, b_ in zip(mod2.row(4)[2:], want_scales(lowered, six))))
                    del mod2
                del mod

if TIME:
    print("\ntiming (this GPU, records in device memory; best of 20, us)")
    k1 = next(iter(recs.values()))
    k2r = recfmt.k1_to_k2(k1, LAYOUT_K2)[0].contiguous()

    def best(fn, reps=20):
        b = 1e9
        for _ in range(reps):
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); e.record(); e.synchronize()
            b = min(b, a.elapsed_time(e) * 1e3)
        return b
    for sms in (24, 64):
        for nrec in (1, 4, 16):
            row = []
            for tag, lo_, six, bank in (("k1 stager.py", LAYOUT_K1, False, k1), ("k2", LAYOUT_K2, False, k2r),
                                        ("k1 6-bit", LAYOUT_K1, True, k1), ("k2 6-bit", LAYOUT_K2, True, k2r)):
                mod = Mod(SLOTS, six)
                st = stager.Stager(mod, 0, lo_, sms=sms)
                idx = torch.full((SLOTS,), -1, dtype=torch.int32, device=dev)
                idx[:nrec] = torch.arange(nrec, device=dev) % bank.shape[0]
                st(bank.data_ptr(), idx); torch.cuda.synchronize()
                row.append(f"{tag} {best(lambda: st(bank.data_ptr(), idx)):.0f}")
                del mod, st
            print(f"  stage, {sms} CTAs, {nrec} record(s): " + ", ".join(row))
    if natives:
        row, hi, lo = next(iter(natives.values()))
        hk2 = recfmt.k1_to_k2(hi.unsqueeze(0), LAYOUT_K2)[0][0].contiguous()
        cfgs = (("k1 stager.py", LAYOUT_K1, False, hi), ("k2", LAYOUT_K2, False, hk2),
                ("k1 6-bit", LAYOUT_K1, True, hi), ("k2 6-bit", LAYOUT_K2, True, hk2))
        mods = [Mod(1, six) for _, _, six, _ in cfgs]
        bufs = [torch.empty(lo_["record_bytes"], dtype=torch.uint8, device=dev) for _, lo_, _, _ in cfgs]
        for blocks in (152, 608, 1216):
            m, x = [1e9] * len(cfgs), [1e9] * len(cfgs)
            for rnd in range(4):                          # interleaved rounds: the GPU is shared
                for i, (tag, lo_, six, h) in enumerate(cfgs):
                    m[i] = min(m[i], best(lambda: stager.merge(h.data_ptr(), lo.data_ptr(), mods[i], 0, lo_, blocks=blocks)))
                    x[i] = min(x[i], best(lambda: stager.extract_hi2(mods[i], 0, lo_, bufs[i], blocks=blocks)))
            print(f"  {blocks} blocks: " + "; ".join(f"{c[0]} merge {m[i]:.1f} extract {x[i]:.1f}" for i, c in enumerate(cfgs)))
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
