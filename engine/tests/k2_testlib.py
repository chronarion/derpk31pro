"""Shared pieces of the record-format tests (hi2-k1 / hi2-k2, 6-bit scale rows): the kernel layout without vLLM,
records from the official checkpoint on the CPU, and synthetic records with the checkpoint's scale statistics."""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("K31_ENGINE_DIR", "/engine"))
from k31_engine import recfmt  # noqa: E402

H, I = 3584, 3072
LAYOUT_K1 = {"format": "hi2-k1", "experts": 896, "record_bytes": 9289728, "codes_bytes": 8257536,
             "scale_bytes": 1032192, "native_bytes": {"w13": 11010048, "w13s": 688128, "w2": 5505024, "w2s": 344064},
             "order": ["w13", "w2", "w13s", "w2s"],
             "shapes": {"w13": [[6144, 1792], [1792, 1], "torch.int8"], "w13s": [[6144, 28], [1, 6144], "torch.int32"],
                        "w2": [[3584, 1536], [1536, 1], "torch.int8"], "w2s": [[3584, 24], [1, 3584], "torch.int32"]}}
LAYOUT_K2 = recfmt.layout_k2(LAYOUT_K1)


def native_row(t: dict) -> dict:
    """One expert's six checkpoint tensors {"w1"/"w2"/"w3": (packed uint8 [out, in / 2], scale uint8 [out, in / 32])}
    -> the bytes of its row in the four kernel tensors (uint8, memory order), by the layout's formulas alone:
      w13 codes  kernel row 16 (r // 8) + 8 half + r % 8 = checkpoint row r of w1 (half 0) or w3 (half 1)
      w13 scale  [K group k][position P][4]: the scales 4 k .. 4 k + 3 of checkpoint row r = 64 b + 16 j + 8 q + m at
                 P = 128 b + 64 q + 32 half + 4 m + j
      w2 codes   as in the checkpoint
      w2 scale   [k][P][4] with P = 128 b + 4 i + j for checkpoint row r = 128 b + 32 j + i"""
    p1, s1 = t["w1"]
    p3, s3 = t["w3"]
    p2, s2 = t["w2"]
    dev = p1.device
    w13 = torch.empty((I // 8, 2, 8, H // 2), dtype=torch.uint8, device=dev)
    w13[:, 0], w13[:, 1] = p1.view(I // 8, 8, H // 2), p3.view(I // 8, 8, H // 2)
    kq1, kq2 = H // 128, I // 128
    s13 = torch.empty((I // 64, 2, 2, 8, 4, kq1, 4), dtype=torch.uint8, device=dev)        # b, q, half, m, j, k, 4
    for half, s in ((0, s1), (1, s3)):
        s13[:, :, half] = s.view(I // 64, 4, 2, 8, kq1, 4).permute(0, 2, 3, 1, 4, 5)        # b, j, q, m -> b, q, m, j
    s13 = s13.reshape(2 * I, kq1, 4).permute(1, 0, 2)                                       # [k][P][4]
    w2s = s2.view(H // 128, 4, 32, kq2, 4).permute(0, 2, 1, 3, 4).reshape(H, kq2, 4).permute(1, 0, 2)
    return {"w13": w13.reshape(-1), "w2": p2.reshape(-1).clone(), "w13s": s13.reshape(-1).clone(),
            "w2s": w2s.reshape(-1).clone()}


def pairs(codes: torch.Tensor, low: bool = False) -> torch.Tensor:
    """store/build_store.py pairs(): native code bytes [n, B] -> record bytes [n, B / 2]"""
    if low:
        lo, hi = codes & 0x03, (codes >> 4) & 0x03
    else:
        lo, hi = (codes & 0x0F) >> 2, codes >> 6
    b = (lo | (hi << 2)).view(codes.shape[0], -1, 2)
    return b[..., 0] | (b[..., 1] << 4)


def records_k1(row: dict):
    """A native row (native_row) -> (hi2-k1 record, lo2 record), uint8 [9289728] each, as build_store.py writes them."""
    R = LAYOUT_K1["record_bytes"]
    codes = torch.cat((row["w13"], row["w2"])).unsqueeze(0)
    sf = torch.cat((row["w13s"], row["w2s"]))
    hi = torch.zeros(R, dtype=torch.uint8, device=codes.device)
    lo = torch.zeros(R, dtype=torch.uint8, device=codes.device)
    cb = codes.shape[1] // 2
    hi[:cb], lo[:cb] = pairs(codes)[0], pairs(codes, low=True)[0]
    hi[cb:cb + sf.numel()] = torch.where(sf > 0, sf - 1, sf)
    lo[cb:cb + sf.numel()] = sf
    return hi, lo


def ckpt_expert(model_dir: str, layer: int, expert: int) -> dict:
    """One expert's six tensors from the official checkpoint (CPU)."""
    from k31_engine.ckpt import Experts
    ck = Experts(model_dir, workers=2, buf_bytes=64 << 20)
    ck.buf = torch.empty(ck.buf_bytes, dtype=torch.uint8)            # no pinned memory needed here
    t = ck.read_layer(layer, [expert])[expert]
    out = {s: (p.clone(), q.clone()) for s, (p, q) in t.items()}
    ck.close()
    return out


def synth_scales(n: int, g: torch.Generator, wide_rows: int = 0, device="cpu") -> torch.Tensor:
    """Scale bytes of n records in k1 order [n, 1032192] with the checkpoint's statistics: stored bytes 108..123, a
    row's bytes = its base + {0, 1, 2} (mostly two values); `wide_rows` rows per record get a byte 4 above or below."""
    out = []
    for nr, kq in ((6144, 28), (3584, 24)):
        base = torch.randint(108, 121, (n, nr, 1), generator=g)
        off = (torch.rand((n, nr, kq * 4), generator=g) < 0.2).to(torch.int64) + \
            (torch.rand((n, nr, kq * 4), generator=g) < 0.004).to(torch.int64)
        v = base + off
        v[:, :, 0] = base[:, :, 0]                                  # the base occurs in every row
        for i in range(n):
            for r in torch.randint(0, nr, (wide_rows,), generator=g).tolist():
                k = int(torch.randint(1, kq * 4, (1,), generator=g))
                v[i, r, k] = int(base[i, r, 0]) + 4
        out.append(v.to(torch.uint8).view(n, nr, kq, 4).permute(0, 2, 1, 3).reshape(n, -1))
    return torch.cat(out, dim=1).to(device)


def synth_records(n: int, seed: int = 0, wide_rows: int = 0, device="cpu") -> torch.Tensor:
    """n hi2-k1 records: random code bytes, scale bytes as synth_scales."""
    g = torch.Generator().manual_seed(seed)
    rec = torch.zeros((n, LAYOUT_K1["record_bytes"]), dtype=torch.uint8)
    cb = LAYOUT_K1["codes_bytes"]
    rec[:, :cb] = torch.randint(0, 256, (n, cb), dtype=torch.uint8, generator=g)
    rec[:, cb:cb + LAYOUT_K1["scale_bytes"]] = synth_scales(n, g, wide_rows)
    return rec.to(device)
