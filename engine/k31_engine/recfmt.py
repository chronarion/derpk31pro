"""Record formats of the 2-bit store, and the packed scale code of format hi2-k2 (reference implementation, torch).

A record is one expert's top two bit planes of its MXFP4 codes in MegaMoE's kernel layout, followed by its block
scales (UE8M0, one byte per 32 weights of a row, each lowered by one: store/build_store.py). The code bytes are the
same in both formats; they differ in how the scales are stored:

  hi2-k1  the scale bytes as they lie in the kernel tensors: w13 [28 K groups][6144 scale rows][4 bytes], then w2
          [24][3584][4]. 1,032,192 B of a 9,289,728 B record.
  hi2-k2  per scale row one base byte, per scale byte a 2-bit offset from it, and a few exact exceptions.
          268,384 B of a 8,527,872 B record (-8.2%). Lossless: decode(k2) is the k1 record, byte for byte.

Why 2 bits: in Moonshot's checkpoint a row's scales (112 or 96 per row) take two values in 63% of the rows, three in
37%, four in 0.008% and five in 42 rows of 802 million (scale_scan.py over all 82,432 experts); no scale byte is zero.

hi2-k2 scale section, after the code bytes: for w13 then w2, the tensor's scale rows in units of UNIT_ROWS (512)
consecutive rows (12 units for w13, 7 for w2). A unit is self-contained (a staging kernel decodes it from one bulk
copy):
    [UNIT_ROWS base bytes]            base of scale row 512 u + r
    [EXC x uint32 exception entries]  row in unit | K block << 9 | stored byte << 16; unused entries are 0xFFFFFFFF.
                                      Each tile of TILE_ROWS (128) rows owns EXC_TILE (4) of them, entries
                                      [4 g, 4 g + 4) for rows 128 g .. 128 g + 127, sorted by (row, K block)
    [KQ x UNIT_ROWS code bytes]       byte (kq, r): the four K blocks 4 kq .. 4 kq + 3 of row r, block j in bits
                                      2 j .. 2 j + 1: stored byte = base + code (an exception's code is 0)
then one uint32 status word per tile (0, or the number of exceptions the tile could not hold: such a record is not
representable and must not be used), then zeros up to a multiple of 4096. (The checkpoint's worst tile has 2.)

The encoder is canonical (the device encoder in stager2.py writes the same bytes): a row's base is its lowest byte
if the row spans at most 4 values, else the lowest s in [min, max - 3] whose window [s, s + 3] holds the most bytes.
"""
import torch

K1, K2 = "hi2-k1", "hi2-k2"
UNIT_ROWS = 512
TILE_ROWS = 128               # a unit's exception entries are owned by its tiles (one warp encodes a tile)
EXC_TILE = 4                  # exception entries per tile
EXC = EXC_TILE * (UNIT_ROWS // TILE_ROWS)      # per unit
EMPTY = 0xFFFFFFFF
BLOCK = 4096                  # records are whole 4 KiB blocks (the k2 store can share the code blocks of a k1 store)


def geom(layout: dict) -> dict:
    """Sizes of a record's parts from a layout.json (either format)."""
    nb = layout["native_bytes"]
    g = {"c13": nb["w13"] // 2, "c2": nb["w2"] // 2, "s13": nb["w13s"], "s2": nb["w2s"]}
    g["codes"] = g["c13"] + g["c2"]
    sh = layout.get("shapes")
    if sh:                                           # [[rows, K groups], strides, dtype]
        g["n13"], g["kq13"] = sh["w13s"][0]
        g["n2"], g["kq2"] = sh["w2s"][0]
    else:                                            # Kimi K3.1 Pro (older layout files and the tests' dicts)
        g["n13"], g["kq13"], g["n2"], g["kq2"] = 6144, 28, 3584, 24
    assert g["n13"] * g["kq13"] * 4 == g["s13"] and g["n2"] * g["kq2"] * 4 == g["s2"], "scale tensor shapes"
    g["k1_bytes"] = (g["codes"] + g["s13"] + g["s2"] + BLOCK - 1) // BLOCK * BLOCK
    for t in ("13", "2"):
        n, kq = g["n" + t], g["kq" + t]
        assert n % UNIT_ROWS == 0, "scale rows must be whole units"
        g["units" + t] = n // UNIT_ROWS
        g["unit" + t] = UNIT_ROWS + 4 * EXC + kq * UNIT_ROWS
        g["p" + t] = g["units" + t] * g["unit" + t]
    g["status"] = g["codes"] + g["p13"] + g["p2"]    # offset of the status words (one per tile)
    g["status_words"] = (g["n13"] + g["n2"]) // TILE_ROWS
    g["k2_bytes"] = (g["status"] + 4 * g["status_words"] + BLOCK - 1) // BLOCK * BLOCK
    return g


def fmt(layout: dict) -> str:
    return layout.get("format", K1)


def lo2_bytes(layout: dict) -> int:
    """Bytes of a lo2 record (the complement of a hi2 record keeps the hi2-k1 layout in both formats)."""
    return int(layout.get("lo2_record_bytes", layout["record_bytes"]))


def layout_k2(layout_k1: dict) -> dict:
    """The layout.json of a hi2-k2 store with the kernel tensors of `layout_k1`."""
    g = geom(layout_k1)
    out = {k: v for k, v in layout_k1.items() if k not in ("format", "record_bytes", "scale_bytes", "code_map")}
    out.update({
        "format": K2, "record_bytes": g["k2_bytes"], "codes_bytes": g["codes"],
        "scale_bytes": g["p13"] + g["p2"], "lo2_record_bytes": g["k1_bytes"],
        "scales": {"code": "row base + 2-bit offset, exact exceptions", "unit_rows": UNIT_ROWS,
                   "tile_rows": TILE_ROWS, "exceptions_per_tile": EXC_TILE,
                   "w13": {"rows": g["n13"], "k_groups": g["kq13"], "units": g["units13"], "unit_bytes": g["unit13"]},
                   "w2": {"rows": g["n2"], "k_groups": g["kq2"], "units": g["units2"], "unit_bytes": g["unit2"]},
                   "status_offset": g["status"], "status_words": g["status_words"]},
        "code_map": "nibble = (pair << 2) | 3, scale byte - 1 (gain 1/2); scale byte = row base + 2-bit code",
    })
    return out


def _best_base(row: torch.Tensor) -> int:
    """Canonical base of a row that spans more than 4 values: the lowest s in [min, max - 3] whose window holds most."""
    lo, hi = int(row.min()), int(row.max())
    best, base = -1, lo
    for s in range(lo, hi - 2):
        c = int(((row >= s) & (row <= s + 3)).sum())
        if c > best:
            best, base = c, s
    return base


def pack_scales(sf: torch.Tensor, n_rows: int, kq: int):
    """One kernel scale tensor of n records, k1 bytes uint8 [n, kq * n_rows * 4] (memory order [kq][row][4]) ->
    (packed uint8 [n, units * unit_bytes], overflow int32 [n, tiles]: exceptions a tile could not hold)."""
    n = sf.shape[0]
    dev = sf.device
    units = n_rows // UNIT_ROWS
    v = sf.view(n, kq, n_rows, 4).permute(0, 2, 1, 3).reshape(n, n_rows, kq * 4).to(torch.int16)   # [n, row, K block]
    lo, hi = v.amin(-1), v.amax(-1)
    base = lo.clone()
    for i, r in (hi - lo > 3).nonzero().tolist():                   # rare (42 rows of the whole checkpoint)
        base[i, r] = _best_base(v[i, r])
    off = v - base.unsqueeze(-1)
    bad = (off < 0) | (off > 3)
    code = torch.where(bad, torch.zeros_like(off), off).view(n, n_rows, kq, 4)
    cb = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4) | (code[..., 3] << 6)).to(torch.uint8)   # [n, row, kq]
    tiles = n_rows // TILE_ROWS
    exc = torch.full((n, tiles, EXC_TILE), EMPTY, dtype=torch.int64, device=dev)
    over = torch.zeros((n, tiles), dtype=torch.int32, device=dev)
    count = {}
    for i, r, k in bad.nonzero().tolist():                          # sorted by (record, row, K block)
        t = r // TILE_ROWS
        j = count.get((i, t), 0)
        count[(i, t)] = j + 1
        if j < EXC_TILE:
            exc[i, t, j] = (r % UNIT_ROWS) | (k << 9) | (int(v[i, r, k]) << 16)
        else:
            over[i, t] += 1
    ub = UNIT_ROWS + 4 * EXC + kq * UNIT_ROWS
    out = torch.empty((n, units, ub), dtype=torch.uint8, device=dev)
    out[:, :, :UNIT_ROWS] = base.to(torch.uint8).view(n, units, UNIT_ROWS)
    e = exc.view(n, units, EXC, 1)
    out[:, :, UNIT_ROWS:UNIT_ROWS + 4 * EXC] = torch.cat([(e >> (8 * b)) & 0xFF for b in range(4)], -1).to(torch.uint8) \
        .view(n, units, 4 * EXC)
    out[:, :, UNIT_ROWS + 4 * EXC:] = cb.view(n, units, UNIT_ROWS, kq).permute(0, 1, 3, 2).reshape(n, units, kq * UNIT_ROWS)
    return out.view(n, units * ub), over


def unpack_scales(p: torch.Tensor, n_rows: int, kq: int) -> torch.Tensor:
    """The inverse of pack_scales: packed uint8 [n, units * unit_bytes] -> k1 bytes uint8 [n, kq * n_rows * 4]."""
    n = p.shape[0]
    units = n_rows // UNIT_ROWS
    ub = UNIT_ROWS + 4 * EXC + kq * UNIT_ROWS
    q = p.view(n, units, ub)
    base = q[:, :, :UNIT_ROWS].reshape(n, n_rows, 1, 1).to(torch.int16)
    cb = q[:, :, UNIT_ROWS + 4 * EXC:].view(n, units, kq, UNIT_ROWS).permute(0, 1, 3, 2).reshape(n, n_rows, kq, 1) \
        .to(torch.int16)
    sh = torch.tensor([0, 2, 4, 6], dtype=torch.int16, device=p.device)
    v = (base + ((cb >> sh) & 3)).to(torch.uint8)                   # [n, row, kq, 4]
    e = q[:, :, UNIT_ROWS:UNIT_ROWS + 4 * EXC].reshape(n, units, EXC, 4).to(torch.int64)
    e = e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)
    for i, u, j in (e != EMPTY).nonzero().tolist():
        w = int(e[i, u, j])
        r, k, val = u * UNIT_ROWS + (w & 511), (w >> 9) & 127, (w >> 16) & 255
        assert (w & 511) // TILE_ROWS == j // EXC_TILE, "an exception entry outside its tile's slots"
        v[i, r, k // 4, k % 4] = val
    return v.permute(0, 2, 1, 3).reshape(n, kq * n_rows * 4)


def k1_to_k2(rec: torch.Tensor, layout: dict):
    """hi2-k1 records uint8 [n, >= codes + scales] -> (hi2-k2 records uint8 [n, record bytes], ok bool [n]). ok is
    False for a record with a tile of more than EXC_TILE exceptions (its status words say how many did not fit)."""
    g = geom(layout)
    n = rec.shape[0]
    out = torch.zeros((n, g["k2_bytes"]), dtype=torch.uint8, device=rec.device)
    out[:, :g["codes"]] = rec[:, :g["codes"]]
    o = g["codes"]
    p13, ov13 = pack_scales(rec[:, o:o + g["s13"]], g["n13"], g["kq13"])
    p2, ov2 = pack_scales(rec[:, o + g["s13"]:o + g["s13"] + g["s2"]], g["n2"], g["kq2"])
    out[:, o:o + g["p13"]] = p13
    out[:, o + g["p13"]:g["status"]] = p2
    over = torch.cat((ov13, ov2), dim=1)                            # [n, status words]
    st = over.to(torch.int64).unsqueeze(-1)
    out[:, g["status"]:g["status"] + 4 * g["status_words"]] = \
        torch.cat([(st >> (8 * b)) & 0xFF for b in range(4)], -1).to(torch.uint8).view(n, -1)
    return out, over.sum(1) == 0


def k2_to_k1(rec: torch.Tensor, layout: dict) -> torch.Tensor:
    """hi2-k2 records uint8 [n, record bytes] -> the hi2-k1 records uint8 [n, k1 record bytes]."""
    g = geom(layout)
    n = rec.shape[0]
    out = torch.zeros((n, g["k1_bytes"]), dtype=torch.uint8, device=rec.device)
    o = g["codes"]
    out[:, :o] = rec[:, :o]
    out[:, o:o + g["s13"]] = unpack_scales(rec[:, o:o + g["p13"]], g["n13"], g["kq13"])
    out[:, o + g["s13"]:o + g["s13"] + g["s2"]] = unpack_scales(rec[:, o + g["p13"]:g["status"]], g["n2"], g["kq2"])
    return out


def status(rec: torch.Tensor, layout: dict) -> torch.Tensor:
    """int64 [n]: exceptions that did not fit, summed over a k2 record's tiles (0 = the record is exact)."""
    g = geom(layout)
    w = rec[:, g["status"]:g["status"] + 4 * g["status_words"]].reshape(rec.shape[0], -1, 4).to(torch.int64)
    return (w[..., 0] | (w[..., 1] << 8) | (w[..., 2] << 16) | (w[..., 3] << 24)).sum(1)


def exceptions(rec: torch.Tensor, layout: dict) -> list:
    """Per k2 record the list of (tensor, scale row, K block, stored byte) of its exception entries."""
    g = geom(layout)
    out = []
    for i in range(rec.shape[0]):
        ex = []
        o = g["codes"]
        for t in ("13", "2"):
            for u in range(g["units" + t]):
                a = o + u * g["unit" + t] + UNIT_ROWS
                w = rec[i, a:a + 4 * EXC].view(EXC, 4).to(torch.int64)
                w = w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)
                for x in w.tolist():
                    if x != EMPTY:
                        ex.append(("w13" if t == "13" else "w2", u * UNIT_ROWS + (x & 511), (x >> 9) & 127, (x >> 16) & 255))
            o += g["p" + t]
        out.append(ex)
    return out


def scale_view(rec: torch.Tensor, layout: dict) -> torch.Tensor:
    """The k1 scale bytes [n, s13 + s2] of records of either format."""
    g = geom(layout)
    if fmt(layout) == K2:
        rec = k2_to_k1(rec, layout)
    return rec[:, g["codes"]:g["codes"] + g["s13"] + g["s2"]]
