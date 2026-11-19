"""K31_SF6's set-up on any GPU (not the MegaMoE call: tests/fork/sf6_rows_test.py does that on the GB300):
  1. vmm.SharedSlots for 6-bit scale banks: 128 rows of 516,096 B are 31.5 mapping granules, so the rows start half a
     granule into the shared mapping; two layers' tensors share the staging rows and nothing else; plain tensors
     (whole granules) map as before
  2. sf6rows.attach: the launcher's scale tensors claim the uncoded shape over the coded banks, the control block
     (magic, no host rows, word 11 = coded, the clone table's address) and the four coded tensor maps are written
     (cuTensorMapEncodeTiled accepts the banks' geometry), the kernel kwargs carry the block
  3. sf6rows.write_scales (rows loaded from the checkpoint) = sf6.encode of the row's scale tensor; with NATIVE on a
     checkpoint expert, through a stand-in for vLLM's _load_weight written from the layout's formulas
  4. a record staged into a shared row through one layer's stager is the same coded row in the other layer's tensors
usage: test_sf6rows.py      (env: K31_CUDA_ARCH, NATIVE)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2_testlib as tl  # noqa: E402
from k2_testlib import LAYOUT_K1, LAYOUT_K2, recfmt  # noqa: E402
from k31_engine import sf6, sf6rows, stager, stager2, vmm  # noqa: E402

dev = torch.device("cuda")
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)


class Mod:
    """A stand-in for DeepGemmMegaMoEExperts: one-row parameters (as the hook finds them before it binds), a
    _load_weight that lays a checkpoint shard out by the kernel layout's formulas, kernel kwargs."""

    def __init__(self):
        self.w13_weight = torch.zeros((1, 6144, 1792), dtype=torch.int8, device=dev)
        self.w2_weight = torch.zeros((1, 3584, 1536), dtype=torch.int8, device=dev)
        self.w13_weight_scale = torch.zeros((1, 28, 6144), dtype=torch.int32, device=dev).transpose(1, 2)
        self.w2_weight_scale = torch.zeros((1, 24, 3584), dtype=torch.int32, device=dev).transpose(1, 2)
        self._kernel_kwargs = {"activation": "situ"}

    @staticmethod
    def _load_weight(dst, src, shard, is_scale):
        assert is_scale
        kb = dst.shape[1]
        s = src.view(torch.uint8).contiguous().view(-1, kb, 4)                 # [rows, K groups, 4 scale bytes]
        w = s.clone().view(torch.int32).view(-1, kb)                           # one word per K group
        if shard == "w2":                                                     # row 128 b + 32 j + i -> 128 b + 4 i + j
            dst.view(-1, 32, 4, kb).copy_(w.view(-1, 4, 32, kb).transpose(1, 2))
        else:                                # 64 b + 16 j + 8 q + m -> 128 b + 64 q + 32 half + 4 m + j
            half = 0 if shard == "w1" else 1
            dst.view(-1, 2, 2, 8, 4, kb)[:, :, half].copy_(w.view(-1, 4, 2, 8, kb).permute(0, 2, 3, 1, 4))


# ---- 1. the shared-slot mapping
SH = 128
mods = [Mod(), Mod()]                                # (also the device's CUDA context)
g = vmm.granularity(0)
own = [3, 5]
sw = {n: vmm.SharedSlots(getattr(mods[0], n), SH) for n in ("w13_weight", "w2_weight")}
s6 = [vmm.SharedSlots(sf6rows.prototype(getattr(mods[0], n)), SH) for n in ("w13_weight_scale", "w2_weight_scale")]
raw = vmm.SharedSlots(mods[0].w13_weight_scale, SH)
check(f"granule {g} B: the weight and int32 scale tensors' {SH} slots are whole granules (no offset); the 6-bit banks "
      f"start {s6[0].front} and {s6[1].front} B into the shared mapping",
      sw["w13_weight"].front == 0 and sw["w2_weight"].front == 0 and raw.front == 0 and
      s6[0].front == (-SH * 516096) % g and s6[1].front == (-SH * 258048) % g and s6[0].front > 0)
banks = []
for m, n_own in zip(mods, own):
    for n in ("w13_weight", "w2_weight"):
        setattr(m, n, sw[n].tensor(n_own))
    banks.append((s6[0].tensor(n_own), s6[1].tensor(n_own)))
b0, b1 = banks
shapes = b0[0].shape == (SH + own[0], 516096) and b1[1].shape == (SH + own[1], 258048) and b0[0].dtype == torch.uint8
aligned = all(t.data_ptr() % 128 == 0 for b in banks for t in b)
b0[0][5].fill_(0xA5); b0[1][SH - 1].fill_(0x3C); b0[0][SH].fill_(0x11); b0[0][SH + own[0] - 1].fill_(0x12)
b1[0][SH].fill_(0x77); b1[1][SH + own[1] - 1].fill_(0x78)
torch.cuda.synchronize()
alias = bool((b1[0][5] == 0xA5).all()) and bool((b1[1][SH - 1] == 0x3C).all())
own_ok = bool((b0[0][SH] == 0x11).all()) and bool((b0[0][SH + own[0] - 1] == 0x12).all()) and \
    bool((b1[0][SH] == 0x77).all()) and bool((b1[1][SH + own[1] - 1] == 0x78).all()) and \
    bool((b0[0][SH + 1] == 0).all()) and bool((b1[0][SH + 1] == 0).all())
check("6-bit banks of two layers: shapes, 128-byte alignment, the staging rows are one memory, the own rows are each "
      "layer's (fresh rows zero)", shapes and aligned and alias and own_ok)

# ---- 2. attach
for m, b in zip(mods, banks):
    sf6rows.attach(m, b[0], b[1])
m0 = mods[0]
rows0 = SH + own[0]
sp = m0.w13_weight_scale
buf, tbl = m0._k31_sf6_ctl
w64 = buf[:16].view(torch.int64)
tail = tbl[sf6.CLONE_BYTES:]
maps = [tail[i * 128:(i + 1) * 128] for i in range(4)]
check("attach: the scale parameters claim [rows, 6144, 28] / [rows, 3584, 24] int32 with the uncoded strides at the "
      "coded banks' addresses",
      tuple(sp.shape) == (rows0, 6144, 28) and tuple(sp.stride()) == (28 * 6144, 1, 6144) and sp.dtype == torch.int32
      and sp.data_ptr() == b0[0].data_ptr() and m0.w2_weight_scale.data_ptr() == b0[1].data_ptr()
      and tuple(m0.w2_weight_scale.shape) == (rows0, 3584, 24))
check("attach: control block = magic, 0 host rows, word 11 = 1, the clone table's address at words 14-15, zero pick "
      "counts; the four coded tensor maps are written; the kernel kwargs carry the first `rows` words",
      int(buf[0]) == sf6rows.MAGIC and int(buf[1]) == 0 and int(buf[11]) == 1 and int(w64[7]) == tbl.data_ptr()
      and tbl.data_ptr() % 128 == 0 and buf.numel() == 16 + rows0 and int(buf[16:].abs().sum()) == 0
      and all(int(x.to(torch.int64).sum()) > 0 for x in maps) and not torch.equal(maps[2], maps[3])
      and m0._kernel_kwargs["cumulative_local_expert_recv_stats"].numel() == rows0
      and m0._kernel_kwargs["cumulative_local_expert_recv_stats"].data_ptr() == buf.data_ptr()
      and m0._kernel_kwargs["activation"] == "situ" and stager2.coded(m0) is not None)

# ---- 3. rows from the checkpoint
native = os.environ.get("NATIVE", "")
if native and os.path.exists(os.path.join(native, "model.safetensors.index.json")):
    t = tl.ckpt_expert(native, 36, 524)
    row = tl.native_row(t)
    sf6rows.write_scales(m0, SH + 1, t["w1"][1], t["w3"][1], t["w2"][1], dev)
    want13 = sf6.encode(row["w13s"].to(dev).view(torch.int32).view(1, 28, 6144).permute(0, 2, 1))[0]
    want2 = sf6.encode(row["w2s"].to(dev).view(torch.int32).view(1, 24, 3584).permute(0, 2, 1))[0]
    check("write_scales: layer 36 expert 524's scales from the checkpoint into a coded row = the coding of its kernel "
          "scale tensors (formulas of the layout); the neighbouring rows untouched",
          torch.equal(b0[0][SH + 1], want13) and torch.equal(b0[1][SH + 1], want2) and bool((b0[0][SH] == 0x11).all())
          and bool((b0[0][SH + 2] == 0x12).all()))
    back = stager2.row_scales(m0, SH + 1)
    check("the row's scale bytes read back through the coded bank = the checkpoint's",
          torch.equal(back, torch.cat((row["w13s"], row["w2s"])).to(dev)))
else:
    print("skip  rows from the checkpoint (NATIVE not set)")

# ---- 4. a staged record is the same row in every layer
rec = tl.synth_records(2, seed=31, wide_rows=3).to(dev)
k2, fits = recfmt.k1_to_k2(rec, LAYOUT_K2)
k2 = k2.contiguous()
idx = torch.full((stager2.SLOTS,), -1, dtype=torch.int32, device=dev)
idx[3], idx[9] = 0, 1
st = stager.Stager(mods[0], 16, LAYOUT_K2)               # slot group 1: rows 16..31
st(k2.data_ptr(), idx)
torch.cuda.synchronize()
G = recfmt.geom(LAYOUT_K1)


def coded_scales(k1rec):
    sc = k1rec[G["codes"]:G["codes"] + G["s13"] + G["s2"]]
    a = sc[:G["s13"]].clone().view(torch.int32).view(1, 28, 6144).permute(0, 2, 1)
    b = sc[G["s13"]:].clone().view(torch.int32).view(1, 24, 3584).permute(0, 2, 1)
    return sf6.encode(a)[0], sf6.encode(b)[0]


good = True
for slot, j in ((3, 0), (9, 1)):
    w13, w2 = coded_scales(rec[j])
    for b in banks:                                       # the same shared row through either layer's bank
        good &= torch.equal(b[0][16 + slot], w13) and torch.equal(b[1][16 + slot], w2)
    good &= torch.equal(stager._row_bytes(mods[1].w13_weight, 16 + slot), stager._row_bytes(mods[0].w13_weight, 16 + slot))
good &= bool((b0[0][5] == 0xA5).all()) and bool((b0[0][SH] == 0x11).all())
check("hi2-k2 records staged through layer 0's stager into shared rows 19 and 25: the coded rows, seen through both "
      "layers' banks, are the coding of the records' scale bytes; other rows untouched", good and bool(fits.all()))
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
