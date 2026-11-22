"""celane.expand_map (the copy-engine lane pool's landing, Pool.land) on the GPU for the four combinations of record
format (hi2-k1, hi2-k2) and row format (int32 scale words, 6-bit scale codes): records at scattered pool slots ->
scattered rows, byte for byte against the reference expansion; rows not named stay untouched. hi2-k1 into plain
rows goes through celane's map kernel, the other three through stager2's expand in runs of consecutive slots and rows.
Records: synthetic with the checkpoint's scale statistics and rows that need exceptions; with HI2_K1 also real records
of L01.hi2. Runs on any GPU with TMA bulk copies: K31_CUDA_ARCH=120a for an RTX PRO 6000 (default 103a, the GB300).
usage: test_celane_pool_k2.py"""
import os
import sys
import types
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_testlib as tl  # noqa: E402
from k2_testlib import LAYOUT_K1, LAYOUT_K2, recfmt  # noqa: E402
from k31_engine import celane, sf6, stager, stager2  # noqa: E402

dev = torch.device("cuda")
G = recfmt.geom(LAYOUT_K1)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)


class Mod:
    """The four kernel tensors of `rows` experts as vLLM registers them, every byte = fill; six: 6-bit scale banks
    (as tests/test_stager2.py)."""
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
        out = [stager._row_bytes(self.w13_weight, r), stager._row_bytes(self.w2_weight, r)]
        c = stager2.coded(self)
        out += [stager._row_bytes(self.w13_weight_scale, r), stager._row_bytes(self.w2_weight_scale, r)] if c is None \
            else [c[0][r], c[1][r]]
        return out

    def untouched(self, r):
        return all(int((x != self.fill).sum()) == 0 for x in self.row(r))


def lut_expand(rec):
    p = [(rec >> (2 * j)) & 3 for j in range(4)]
    n = [(q << 2) | 3 for q in p]
    return torch.stack((n[0] | (n[1] << 4), n[2] | (n[3] << 4)), dim=-1).reshape(-1)


def want_row(k1_rec, six):
    a = k1_rec[G["codes"]:G["codes"] + G["s13"]]
    b = k1_rec[G["codes"] + G["s13"]:G["codes"] + G["s13"] + G["s2"]]
    if six:
        t13 = a.clone().view(torch.int32).view(1, 28, 6144).permute(0, 2, 1)
        t2 = b.clone().view(torch.int32).view(1, 24, 3584).permute(0, 2, 1)
        a, b = sf6.encode(t13)[0], sf6.encode(t2)[0]
    return [lut_expand(k1_rec[:G["c13"]]), lut_expand(k1_rec[G["c13"]:G["codes"]]), a, b]


recs = {"synthetic, 4 wide rows per tensor": tl.synth_records(3, seed=11, wide_rows=4).to(dev)}
d = os.environ.get("HI2_K1", "")
if d and os.path.exists(os.path.join(d, "L01.hi2")):
    R1 = LAYOUT_K1["record_bytes"]
    fd = os.open(os.path.join(d, "L01.hi2"), os.O_RDONLY)
    recs["real records of L01.hi2"] = torch.stack(
        [torch.frombuffer(bytearray(os.pread(fd, R1, e * R1)), dtype=torch.uint8) for e in (7, 100, 555, 895)]).to(dev)
    os.close(fd)

lay = {False: LAYOUT_K1, True: LAYOUT_K2}
for k2 in (False, True):
    for six in (False, True):
        tag = f"{'hi2-k2' if k2 else 'hi2-k1'} records, {'6-bit' if six else 'int32'} scale rows"
        layout = lay[k2]
        R = int(layout["record_bytes"])
        lane = types.SimpleNamespace(layout=layout, R=R, m=celane._module(), segs=celane.segs_of(layout),
                                     lut=celane._lut16_cached(dev), xblocks=64)
        for name, k1 in recs.items():
            n = k1.shape[0]
            bank = recfmt.k1_to_k2(k1, LAYOUT_K2)[0] if k2 else k1
            # a pool of 12 slots; the records at slots that make runs and gaps: slot -> row
            plan = [(5, 10), (6, 11), (7, 12), (2, 3), (9, 20), (10, 22)][:max(n, 3)]
            pool = torch.full((12, R), 0xEE, dtype=torch.uint8, device=dev)
            for i, (p, _) in enumerate(plan):
                pool[p].copy_(bank[i % n])
            mod = Mod(24, six, fill=0x5A)
            celane.expand_map(lane, pool.data_ptr(), [p for p, _ in plan], [r for _, r in plan], mod)
            torch.cuda.synchronize()
            good = all(all(torch.equal(x, y) for x, y in zip(mod.row(r), want_row(k1[i % n], six)))
                       for i, (_, r) in enumerate(plan))
            named = {r for _, r in plan}
            good &= all(mod.untouched(r) for r in range(24) if r not in named)
            check(f"expand_map {tag}: {name} ({len(plan)} records, slots and rows in runs and gaps)", good)
            del mod, pool
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
