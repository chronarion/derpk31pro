#!/usr/bin/env python3
"""A hi2-k2 record store from a hi2-k1 store (k31_engine/recfmt.py: the same code bytes, the scale bytes as row bases
and 2-bit offsets; 8,527,872 B per record instead of 9,289,728). CPU only, the source is never written.

    convert_store.py SRC_DIR DST_DIR [--layers 1-92] [--workers 8] [--copy] [--lo2 reflink|link|skip]

Per layer, L{L}.hi2 of DST gets every record's code bytes and its packed scales. The code bytes are the source's,
byte for byte, and both stores keep records on 4 KiB blocks, so on a filesystem with reflinks (XFS, Btrfs) they are
shared with the source file (FICLONERANGE): only the 270,336 B after each record's codes are new data, ~22 GB for
Kimi K3.1 Pro's 82,432 experts instead of 703 GB. Without reflinks (or with --copy) the code bytes are copied.
Every record is checked: the scale bytes decoded from what was written must equal the source's, and (copies) the code
bytes are compared. L{L}.lo2 (the complement that promotions merge with, unchanged by the format) is shared with the
source as a reflink, else as a hard link. layout.json describes the new store; L{L}.k2.json lists a layer's records
with exception entries and its range of scale bytes. A record that hi2-k2 cannot hold (a tile of 128 scale rows with
more than 4 exceptions) stops the conversion; none of the checkpoint's is one.
"""
import argparse
import fcntl
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.environ.get("K31_ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from k31_engine import recfmt  # noqa: E402

FICLONE = 0x40049409
FICLONERANGE = 0x4020940D
BATCH = 32


def clone_range(src_fd: int, src_off: int, length: int, dst_fd: int, dst_off: int) -> None:
    fcntl.ioctl(dst_fd, FICLONERANGE, struct.pack("qQQQ", src_fd, src_off, length, dst_off))


def pread_into(fd: int, view, pos: int) -> None:
    view = memoryview(view).cast("B")
    while view:
        k = os.preadv(fd, [view], pos)
        if k <= 0:
            raise IOError(f"short read at {pos}")
        view, pos = view[k:], pos + k


def pwrite_all(fd: int, data, pos: int) -> None:
    view = memoryview(data).cast("B")
    while view:
        k = os.pwrite(fd, view, pos)
        view, pos = view[k:], pos + k


def convert_layer(args):
    src_dir, dst_dir, L, copy, lo2 = args
    t0 = time.time()
    lay1 = json.load(open(os.path.join(src_dir, "layout.json")))
    lay2 = recfmt.layout_k2(lay1)
    g = recfmt.geom(lay1)
    R1, R2, codes, sbytes = lay1["record_bytes"], lay2["record_bytes"], g["codes"], g["s13"] + g["s2"]
    name = f"L{L:02d}.hi2"
    src = os.open(os.path.join(src_dir, name), os.O_RDONLY)
    n = os.fstat(src).st_size // R1
    assert n * R1 == os.fstat(src).st_size, f"{name}: not a whole number of hi2-k1 records"
    part = os.path.join(dst_dir, name + ".part")
    dst = os.open(part, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(dst, n * R2)
    shared = not copy
    if shared:
        try:
            clone_range(src, 0, codes, dst, 0)
        except OSError as ex:
            shared = False
            why = str(ex)
    sf = np.empty((BATCH, sbytes), dtype=np.uint8)
    buf = np.empty(codes, dtype=np.uint8) if not shared else None
    back = np.empty(codes, dtype=np.uint8) if not shared else None
    exc_records, lo_b, hi_b, n_exc = [], 255, 0, 0
    for b0 in range(0, n, BATCH):
        m = min(BATCH, n - b0)
        for i in range(m):
            e = b0 + i
            pread_into(src, sf[i], e * R1 + codes)
            if shared:
                if e:
                    clone_range(src, e * R1, codes, dst, e * R2)
            else:
                pread_into(src, buf, e * R1)
                pwrite_all(dst, buf, e * R2)
        tails, over = recfmt.tails_np(sf[:m], lay2)
        if over.any():
            bad = [b0 + int(i) for i in np.nonzero(over.sum(1))[0]]
            raise RuntimeError(f"layer {L}: records {bad} do not fit hi2-k2 (a tile with more than "
                               f"{recfmt.EXC_TILE} exceptions)")
        if not np.array_equal(recfmt.untails_np(tails, lay2), sf[:m]):
            raise RuntimeError(f"layer {L}: records {b0}..{b0 + m - 1}: the packed scales do not decode to the source's")
        for i in range(m):
            pwrite_all(dst, tails[i], (b0 + i) * R2 + codes)
        for i, c in enumerate(_exc_counts(tails, g)):
            if c:
                exc_records.append(b0 + i)
                n_exc += int(c)
        lo_b, hi_b = min(lo_b, int(sf[:m].min())), max(hi_b, int(sf[:m].max()))
    os.fsync(dst)
    # read back what the file now holds: every record's tail, and the code bytes of copies
    tail_len = R2 - codes
    rb = np.empty((BATCH, tail_len), dtype=np.uint8)
    for b0 in range(0, n, BATCH):
        m = min(BATCH, n - b0)
        for i in range(m):
            pread_into(dst, rb[i], (b0 + i) * R2 + codes)
            pread_into(src, sf[i], (b0 + i) * R1 + codes)
            if not shared:
                pread_into(src, buf, (b0 + i) * R1)
                pread_into(dst, back, (b0 + i) * R2)
                if not np.array_equal(buf, back):
                    raise RuntimeError(f"layer {L} record {b0 + i}: copied code bytes differ")
        if not np.array_equal(recfmt.untails_np(rb[:m], lay2), sf[:m]):
            raise RuntimeError(f"layer {L}: records {b0}..{b0 + m - 1}: the written scales do not decode to the source's")
    if shared:                         # shared blocks are the source's by construction; spot-check three records
        chk = np.empty(codes, dtype=np.uint8)
        ref = np.empty(codes, dtype=np.uint8)
        for e in sorted({0, n // 2, n - 1}):
            pread_into(src, ref, e * R1)
            pread_into(dst, chk, e * R2)
            if not np.array_equal(ref, chk):
                raise RuntimeError(f"layer {L} record {e}: shared code bytes differ")
    os.close(dst)
    os.close(src)
    os.replace(part, os.path.join(dst_dir, name))
    how_lo = "none"
    if lo2 != "skip":
        s_lo, d_lo = os.path.join(src_dir, f"L{L:02d}.lo2"), os.path.join(dst_dir, f"L{L:02d}.lo2")
        if os.path.exists(s_lo) and not os.path.exists(d_lo):
            how_lo = share_file(s_lo, d_lo, lo2)
        elif os.path.exists(d_lo):
            how_lo = "present"
    json.dump({"layer": L, "records": n, "exception_records": exc_records, "exception_entries": n_exc,
               "scale_range": [lo_b, hi_b], "code_bytes": "shared with the source (reflink)" if shared else "copied"},
              open(os.path.join(dst_dir, f"L{L:02d}.k2.json"), "w"))
    return (L, n, shared, how_lo, len(exc_records), n_exc, lo_b, hi_b, time.time() - t0, None if shared or copy else why)


def _exc_counts(tails, g):
    """Exception entries in use per record of a batch of tails."""
    n = tails.shape[0]
    out = np.zeros(n, dtype=np.int64)
    o = 0
    for t in ("13", "2"):
        u = tails[:, o:o + g["p" + t]].reshape(n, g["units" + t], g["unit" + t])
        e = np.ascontiguousarray(u[:, :, recfmt.UNIT_ROWS:recfmt.UNIT_ROWS + 4 * recfmt.EXC]).view("<u4")
        out += (e != recfmt.EMPTY).reshape(n, -1).sum(1)
        o += g["p" + t]
    return out


def share_file(src: str, dst: str, how: str) -> str:
    """dst = the same bytes as src without copying them: a reflink (how = reflink, falling back to a hard link) or a
    hard link."""
    if how == "reflink":
        s = os.open(src, os.O_RDONLY)
        d = os.open(dst + ".part", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            fcntl.ioctl(d, FICLONE, s)
            os.close(d)
            os.replace(dst + ".part", dst)
            return "reflink"
        except OSError:
            os.close(d)
            os.unlink(dst + ".part")
        finally:
            os.close(s)
    os.link(src, dst)
    return "hard link"


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("src"); a.add_argument("dst")
    a.add_argument("--layers", default="1-92")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--copy", action="store_true", help="copy the code bytes instead of sharing them (no reflinks)")
    a.add_argument("--lo2", default="reflink", choices=["reflink", "link", "skip"])
    args = a.parse_args()
    lay1 = json.load(open(os.path.join(args.src, "layout.json")))
    if recfmt.fmt(lay1) != recfmt.K1:
        raise SystemExit(f"{args.src} is {recfmt.fmt(lay1)}, not hi2-k1")
    if os.path.realpath(args.src) == os.path.realpath(args.dst):
        raise SystemExit("the new store must be another directory")
    os.makedirs(args.dst, exist_ok=True)
    lay2 = recfmt.layout_k2(lay1)
    lay2["converted_from"] = lay1.get("format", "hi2-k1")
    lo, _, hi = args.layers.partition("-")
    layers = [L for L in range(int(lo), int(hi or lo) + 1)
              if os.path.exists(os.path.join(args.src, f"L{L:02d}.hi2")) and
              not os.path.exists(os.path.join(args.dst, f"L{L:02d}.k2.json"))]
    print(f"{args.src} ({lay1['record_bytes']:,} B per record) -> {args.dst} ({lay2['record_bytes']:,} B): "
          f"{len(layers)} layers to do", flush=True)
    t0, tot = time.time(), [0, 0, 0]
    with ProcessPoolExecutor(args.workers) as ex:
        for L, n, shared, how_lo, nrec, nexc, lo_b, hi_b, dt, why in ex.map(
                convert_layer, [(args.src, args.dst, L, args.copy, args.lo2) for L in layers]):
            tot[0] += n; tot[1] += nrec; tot[2] += nexc
            print(f"layer {L}: {n} records, code bytes {'shared (reflink)' if shared else 'copied'}"
                  + (f" (no reflink: {why})" if why else "") + f", lo2 {how_lo}; {nrec} records with exception entries "
                  f"({nexc} entries); scale bytes {lo_b}..{hi_b}; {dt:.0f} s, {time.time() - t0:.0f} s in all", flush=True)
    json.dump(lay2, open(os.path.join(args.dst, "layout.json"), "w"), indent=1)
    print(f"DONE: {tot[0]} records converted and verified, {tot[1]} with exception entries ({tot[2]} entries); "
          f"layout.json written", flush=True)


if __name__ == "__main__":
    main()
