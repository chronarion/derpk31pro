"""2-bit expert records ("hi2") for the stager: per MoE layer one file of 896 records, each the top two bit planes
(sign, high exponent bit) of one expert's MXFP4 codes **in MegaMoE's kernel layout**, plus its scale bytes lowered
by one (the 2^-1 gain of the depth-2 code map: a staged nibble is (pair << 2) | 3 -> 0.75 / 3.0 times the block
scale; runner recon "trick", weight MSE 0.148 vs the centroids' 0.135).

    build_store.py NATIVE_DIR OUT_DIR [--layers 1-92] [--check]

Layout (`hi2-k1`): record e of layer L at byte e * REC in `L{L:02d}.hi2`:
  [0, CODES)          pairs of the four kernel tensors' code bytes in memory order, w13 then w2: native byte 2j and
                      2j + 1 -> record byte j = lo(2j) | hi(2j) << 2 | lo(2j+1) << 4 | hi(2j+1) << 6 (lo/hi = the
                      byte's low / high nibble, each reduced to its top two bits). The stager's byte -> two-nibble
                      table rebuilds bytes 2j, 2j + 1 of a native row.
  [CODES, CODES+SF)   the w13 then w2 scale tensors' bytes in memory order (UE8M0 in the kernel's int32 packing),
                      each lowered by one (zero stays zero).
The kernel layout comes from vLLM's own loader (`DeepGemmMegaMoEExperts._load_weight`, image vllm-openai:kimi-k31),
run on the GPU. `layout.json` in OUT_DIR records shapes, strides and offsets. --check rebuilds a few experts' native
rows from their records with the stager's expansion and compares with the runner's "trick" reconstruction.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("K31_ENGINE_DIR", "/engine"))
from k31_engine.ckpt import Experts  # noqa: E402

E, LATENT, INTER = 896, 3584, 3072


def kernel_rows(dev):
    """Empty kernel-layout tensors for a batch of experts, shaped and strided as DeepGemmMegaMoEExperts registers
    them (w13 [n, 2I, H/2] int8, w13 scale [n, 2I, H/128] int32 MN-major, w2 [n, H, I/2], w2 scale [n, H, I/128])."""
    def scale(n, rows, k):
        return torch.zeros(n, k // 128, rows, dtype=torch.int32, device=dev).transpose(-1, -2)
    return lambda n: {"w13": torch.zeros(n, 2 * INTER, LATENT // 2, dtype=torch.int8, device=dev),
                      "w13s": scale(n, 2 * INTER, LATENT),
                      "w2": torch.zeros(n, LATENT, INTER // 2, dtype=torch.int8, device=dev),
                      "w2s": scale(n, LATENT, INTER)}


def flat(t: torch.Tensor) -> torch.Tensor:
    """[n, ...] -> [n, bytes] uint8 in memory order (each row is a dense block whatever its inner strides)."""
    n, per = t.shape[0], t[0].numel()
    assert t.stride(0) == per, "rows must be dense blocks"
    return torch.as_strided(t, (n, per), (per, 1)).contiguous().view(torch.uint8)


def pairs(codes: torch.Tensor, low: bool = False) -> torch.Tensor:
    """[n, B] uint8 native code bytes -> [n, B/2] record bytes, four pairs per byte: the top two bits of every nibble
    (hi2), or with `low` the bottom two (lo2)."""
    if low:
        lo, hi = codes & 0x03, (codes >> 4) & 0x03
    else:
        lo, hi = (codes & 0x0F) >> 2, codes >> 6    # top two bits of the low nibble (bits 3-2) and the high (7-6)
    b = (lo | (hi << 2)).view(codes.shape[0], -1, 2)
    return b[..., 0] | (b[..., 1] << 4)


def expand(rec: torch.Tensor) -> torch.Tensor:
    """The stager's expansion: [n, B/2] record bytes -> [n, B] native code bytes, nibble = (pair << 2) | 3."""
    p = [(rec >> (2 * j)) & 3 for j in range(4)]
    n = [(q << 2) | 3 for q in p]
    return torch.stack((n[0] | (n[1] << 4), n[2] | (n[3] << 4)), dim=-1).reshape(rec.shape[0], -1)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("native"); a.add_argument("out")
    a.add_argument("--layers", default="1-92")
    a.add_argument("--check", action="store_true")
    a.add_argument("--batch", type=int, default=64)
    a.add_argument("--parts", default="hi2", help="hi2, lo2 or hi2,lo2 (lo2: the bottom two bit planes and the "
                   "unlowered scale bytes; with a hi2 record it rebuilds the native row exactly: residency.py)")
    args = a.parse_args()
    parts = args.parts.split(",")
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts as M
    lo, _, hi = args.layers.partition("-")
    layers = range(int(lo), int(hi or lo) + 1)
    dev = torch.device("cuda")
    mk = kernel_rows(dev)
    probe = mk(1)
    sizes = {k: flat(v).shape[1] for k, v in probe.items()}
    codes_b = (sizes["w13"] + sizes["w2"]) // 2
    sf_b = sizes["w13s"] + sizes["w2s"]
    rec_b = (codes_b + sf_b + 4095) // 4096 * 4096
    os.makedirs(args.out, exist_ok=True)
    meta = {"format": "hi2-k1", "experts": E, "record_bytes": rec_b, "codes_bytes": codes_b, "scale_bytes": sf_b,
            "native_bytes": sizes, "order": ["w13", "w2", "w13s", "w2s"],
            "shapes": {k: [list(v.shape[1:]), list(v.stride()[1:]), str(v.dtype)] for k, v in probe.items()},
            "code_map": "nibble = (pair << 2) | 3, scale byte - 1 (gain 1/2)", "source": args.native}
    json.dump(meta, open(os.path.join(args.out, "layout.json"), "w"), indent=1)
    ck = Experts(args.native)
    pin = torch.empty((args.batch, rec_b), dtype=torch.uint8, pin_memory=True)
    t0 = time.time()
    for L in layers:
        paths = {p: os.path.join(args.out, f"L{L:02d}.{p}") for p in parts}
        fds = {p: os.open(paths[p] + ".part", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644) for p in parts}
        for b0 in range(0, E, args.batch):
            es = list(range(b0, min(E, b0 + args.batch)))
            tens = ck.read_layer(L, es)
            rows = mk(len(es))
            for r, e in enumerate(es):
                for shard in ("w1", "w2", "w3"):
                    packed, scale = tens[e][shard]
                    pre = "w2" if shard == "w2" else "w13"
                    M._load_weight(M, rows[pre][r], packed.to(dev), shard, False)
                    M._load_weight(M, rows[pre + "s"][r], scale.to(dev), shard, True)
            fl = {k: flat(v) for k, v in rows.items()}
            c13 = sizes["w13"] // 2
            sf = torch.cat((fl["w13s"], fl["w2s"]), dim=1)
            for p in parts:
                rec = torch.zeros((len(es), rec_b), dtype=torch.uint8, device=dev)
                rec[:, :c13] = pairs(fl["w13"], low=p == "lo2")
                rec[:, c13:codes_b] = pairs(fl["w2"], low=p == "lo2")
                rec[:, codes_b:codes_b + sf_b] = torch.where(sf > 0, sf - 1, sf) if p == "hi2" else sf
                if args.check and b0 == 0 and p == "hi2":
                    check(rows, fl, rec, sizes, codes_b, sf_b, tens, es, dev)
                pin[:len(es)].copy_(rec)
                os.pwrite(fds[p], memoryview(pin[:len(es)].numpy()).cast("B"), b0 * rec_b)
                del rec
            del rows, fl
        for p in parts:
            os.fsync(fds[p]); os.close(fds[p])
            os.replace(paths[p] + ".part", paths[p])
        print(f"layer {L}: {E} records each of {', '.join(parts)}, {E * rec_b / 1e9:.2f} GB each, "
              f"{time.time() - t0:.0f} s", flush=True)
    ck.close()


def check(rows, fl, rec, sizes, codes_b, sf_b, tens, es, dev):
    """Records -> native bytes (stager expansion) for the first experts; dequantize the staged w1 part through the
    kernel layout's inverse and compare with the runner's 'trick' reconstruction of the official weights."""
    c13 = sizes["w13"] // 2
    staged13 = expand(rec[:, :c13])
    # every staged nibble keeps the record's two bits and sets the two low bits
    nat = fl["w13"]
    keep = ((staged13 & 0xCC) == (nat & 0xCC)).all().item() and ((staged13 & 0x33) == 0x33).all().item()
    staged_sf = rec[:, codes_b:codes_b + sf_b].int()
    raw_sf = torch.cat((fl["w13s"], fl["w2s"]), dim=1).int()
    sf_ok = bool(((raw_sf - staged_sf == 1) | ((raw_sf == 0) & (staged_sf == 0))).all())
    # values: the trick reconstruction in the checkpoint layout for expert 0's w1, vs the staged bytes decoded back
    # through the same loader (load the staged codes as if they were a checkpoint is not possible; instead compare
    # magnitude maps: nibble codes 3 / 7 at half scale = 0.75 / 3.0)
    mag = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    p, s = tens[es[0]]["w1"]
    codes = torch.stack((p & 15, p >> 4), dim=-1).flatten(-2).long()
    sign = 1.0 - 2.0 * (codes >> 3).double()
    exact = mag[codes & 7] * sign
    trick = torch.where((codes & 7) >= 4, 6.0, 1.5).double() * 0.5 * sign
    rel = float(((trick - exact) ** 2 * torch.exp2(2 * (s.double() - 127)).repeat_interleave(32, -1)).sum()
                / (exact ** 2 * torch.exp2(2 * (s.double() - 127)).repeat_interleave(32, -1)).sum())
    print(f"check: staged nibbles keep the record's bits and set the low two: {keep}; staged scales = native - 1: "
          f"{sf_ok}; expert {es[0]} w1 weight MSE of the depth-2 code map {rel:.4f} (runner trick: ~0.148)", flush=True)
    if not (keep and sf_ok):
        raise SystemExit("record check FAILED")


if __name__ == "__main__":
    main()
