"""The band's records and arithmetic against Moonshot's official MXFP4 experts.

    python tests/test_native.py --lib LIB --hi2 L01.hi2 --native DIR [--layer 1] [--runner RUNNER_DIR]

For each expert with a file DIR/L{layer}_e{expert}.native (tools/extract_native.py):
  1. the weights decoded from its hi2 record (band/reference.py: the stager's expansion, the stored scale codes, the
     kernel layout's row maps) must equal the official weights under the depth-2 code map (sign and high exponent
     bit kept, magnitude 0.75 or 3.0 block scales: the runner's "planes2" / recon "trick") exactly;
  2. the kernel's output from the band record against the torch reference with those weights (as tests/test_band.py);
  3. for information, the same call computed with the official 4-bit weights: what the 2-bit code map costs.
With --runner, step 1 is also checked against runner/k31/formats.py decode_planes(..., 2, "trick"), and the band's
MXFP8 quantizer against its mx_fake_quant(x, "mxfp8").
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from band import reference as ref                                             # noqa: E402
from band.rtxband import FLAG_PRECISE, FLAGS_RUNNER, H, HI2_REC, I, MAXT, Band, mxfp8_encode, pack_x   # noqa: E402

MAG = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])


def unpack(packed):
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()


def official(packed, scale):
    """Official MXFP4 weight: e2m1 codes (low nibble = even column) x 2^(scale - 127)."""
    c = unpack(packed)
    v = MAG.to(packed.device)[c & 7] * (1.0 - 2.0 * (c >> 3).float())
    return v * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)


def depth2(packed, scale):
    """The depth-2 code map on the official codes: magnitude 0.75 (codes 0-3) or 3.0 (codes 4-7) block scales."""
    c = unpack(packed)
    v = torch.where((c & 7) >= 4, 3.0, 0.75) * (1.0 - 2.0 * (c >> 3).float())
    return v * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True); ap.add_argument("--hi2", required=True)
    ap.add_argument("--native", required=True); ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--runner", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(3)
    files = sorted(glob.glob(os.path.join(a.native, f"L{a.layer:02d}_e*.native")))
    experts = [int(re.search(r"_e(\d+)\.native", f).group(1)) for f in files]
    mm = np.memmap(a.hi2, dtype=np.uint8, mode="r").reshape(-1, HI2_REC)
    recs = torch.from_numpy(np.stack([mm[e] for e in experts])).to(dev)
    shapes = [(I, H // 2), (I, H // 32), (H, I // 2), (H, I // 32), (I, H // 2), (I, H // 32)]
    fmt = None
    if a.runner:
        sys.path.insert(0, a.runner)
        from k31 import formats as fmt
    ok = True
    if fmt is not None:                                  # the band's MXFP8 quantizer against the runner's
        same_q = True
        for sc in (1e-3, 0.1, 1.0, 30.0):
            v = torch.randn(64, H, generator=gen).to(dev) * sc
            v[0, :32] = 448.0 * 2.0 ** torch.arange(-16, 16).to(dev)           # block maxima on the scale boundaries
            same_q &= bool((mxfp8_encode(v, 0)[2] == fmt.mx_fake_quant(v, "mxfp8")).all())
        print(f"mxfp8_encode == runner mx_fake_quant(x, 'mxfp8') on random rows and boundary blocks: {same_q}")
        ok &= same_q
    nat, d2 = [], []
    for j, (e, f) in enumerate(zip(experts, files)):
        raw = torch.from_numpy(np.fromfile(f, dtype=np.uint8)).to(dev)
        t, o = [], 0
        for r, c in shapes:
            t.append(raw[o:o + r * c].view(r, c)); o += r * c
        assert o == raw.numel()
        w1p, w1s, w2p, w2s, w3p, w3s = t
        want = (depth2(w1p, w1s), depth2(w3p, w3s), depth2(w2p, w2s))
        got = ref.decode_hi2(recs[j])
        same = all(bool((g == w).all()) for g, w in zip(got, want))
        line = f"expert {e}: record weights == official weights under the depth-2 map: {same}"
        if fmt is not None:
            run = (fmt.decode_planes(w1p, w1s, 2, "trick", torch.float32), fmt.decode_planes(w3p, w3s, 2, "trick", torch.float32),
                   fmt.decode_planes(w2p, w2s, 2, "trick", torch.float32))
            same_r = all(bool((g == w).all()) for g, w in zip(got, run))
            line += f"; == runner decode_planes(2, 'trick'): {same_r}"
            same &= same_r
        full = (official(w1p, w1s), official(w3p, w3s), official(w2p, w2s))
        mse = [float(((g - w).double() ** 2).sum() / (w.double() ** 2).sum()) for g, w in zip(got, full)]
        line += f"; weight MSE vs official w1 {mse[0]:.4f} w3 {mse[1]:.4f} w2 {mse[2]:.4f}"
        print(line, flush=True)
        ok &= same
        nat.append(full); d2.append(got)
    band = Band(a.lib, len(experts), dev)
    band.load(0, recs)
    n = len(experts)
    print(f"{'T':>1s} {'x scale':>7s} | {'kernel vs depth-2 reference':>28s} | {'depth-2 vs official output: rel err, cosine':>44s}")
    for T, xs_ in ((1, 1.0), (3, 1.0), (8, 1.0), (3, 0.3), (3, 3.0)):
        x = torch.randn(T, H, generator=gen).to(dev) * xs_
        xq, xs, xv = mxfp8_encode(x, 0)
        w = torch.zeros(n, MAXT)
        w[:, :T] = torch.rand(n, T, generator=gen) * 0.2 + 0.02
        call = band.make_call(list(range(n)), w, T, flags=FLAGS_RUNNER, sf_min=0)
        y = band.run(call, pack_x(xq, xs)).clone()[:T].double()
        y2, _ = ref.forward(xv, d2, w, flags=FLAGS_RUNNER, sf_min=0)
        y4, _ = ref.forward(xv, nat, w, flags=FLAGS_RUNNER, sf_min=0)
        rel = float((y - y2).norm() / y2.norm())
        rel4 = float((y2 - y4).norm() / y4.norm())
        cos = float((y2 * y4).sum() / (y2.norm() * y4.norm()))
        print(f"{T:1d} {xs_:7.1f} | {rel:28.2e} | {rel4:30.3f} {cos:13.3f}", flush=True)
        ok &= rel < 1e-2
    print("NATIVE CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
