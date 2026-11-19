"""Routed experts straight from Moonshot's checkpoint (the serving model is a stripped copy without them).

`Experts(model_dir)` reads the index and each shard's safetensors header once; `read_layer(layer, experts)` returns,
per expert, the six checkpoint tensors {"w1"/"w2"/"w3": (weight_packed uint8 [out, in/2], weight_scale uint8
[out, in/32])} as CPU views into one pinned staging buffer. The byte ranges of a layer's experts are merged when the
gap between them is under `gap` and read front to back with os.pread from `workers` threads (NVMe reads at
~14 GB/s on a Gen5 drive want a few requests in flight); the buffer is reused, so the views are valid until the next read.
"""
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import torch

SHARDS = ("w1", "w2", "w3")
PARTS = ("weight_packed", "weight_scale")
_DT = {"U8": torch.uint8, "I8": torch.int8, "BF16": torch.bfloat16, "F32": torch.float32, "F8_E8M0": torch.uint8}


def _name(layer: int, e: int, shard: str, part: str) -> str:
    return f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{shard}.{part}"


class Experts:
    BATCH = 96              # experts per read_layer call that fit the staging buffer (17.5 MB each, plus gaps)

    def __init__(self, model_dir: str, workers: int = 8, gap: int = 16 << 20, piece: int = 64 << 20,
                 buf_bytes: int = 2 << 30):
        self.dir = model_dir
        self.wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}            # shard -> (data offset, {name: (dtype, shape, start, end)})
        self.workers, self.gap, self.piece = workers, gap, piece
        self.buf_bytes = buf_bytes
        self.buf = None
        self.pool = ThreadPoolExecutor(workers)
        self.fds = {}

    def _header(self, shard: str):
        h = self.hdr.get(shard)
        if h is None:
            with open(os.path.join(self.dir, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(n))
            meta.pop("__metadata__", None)
            h = self.hdr[shard] = (8 + n, {k: (v["dtype"], v["shape"], *v["data_offsets"]) for k, v in meta.items()})
        return h

    def _fd(self, shard: str) -> int:
        fd = self.fds.get(shard)
        if fd is None:
            fd = self.fds[shard] = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
            if getattr(self, "fadv_random", False):  # the residency's promotion readers: no read-ahead (K31_FADV_RANDOM)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        return fd

    def read_layer(self, layer: int, experts) -> dict:
        names = [_name(layer, e, s, p) for e in experts for s in SHARDS for p in PARTS]
        spans = []                                                   # (shard, abs start, abs end, name)
        for n in names:
            shard = self.wm[n]
            base, meta = self._header(shard)
            dt, shape, a, b = meta[n]
            spans.append((shard, base + a, base + b, n))
        spans.sort()
        runs = []                                                    # merged (shard, start, end, [(name, start, end)])
        for sh, a, b, n in spans:
            if runs and runs[-1][0] == sh and a - runs[-1][2] <= self.gap:
                runs[-1][2] = max(runs[-1][2], b)
                runs[-1][3].append((n, a, b))
            else:
                runs.append([sh, a, b, [(n, a, b)]])
        total = sum(r[2] - r[1] for r in runs)
        offs, o = [], 0
        for r in runs:
            offs.append(o)
            o += (r[2] - r[1] + 4095) // 4096 * 4096
        if o > self.buf_bytes:
            raise ValueError(f"layer {layer}: {len(experts)} experts need {o >> 20} MiB of staging, more than "
                             f"{self.buf_bytes >> 20} MiB: read fewer at a time (Experts.BATCH)")
        if self.buf is None:      # one allocation for the process (torch's pinned allocator rounds up to 2^n)
            self.buf = torch.empty(self.buf_bytes, dtype=torch.uint8, pin_memory=True)
        mv = memoryview(self.buf.numpy())
        jobs = []
        for r, ro in zip(runs, offs):
            for p0 in range(r[1], r[2], self.piece):
                p1 = min(r[2], p0 + self.piece)
                jobs.append((r[0], p0, p1, ro + p0 - r[1]))

        def rd(job):
            sh, p0, p1, dst = job
            fd, view, pos = self._fd(sh), mv[dst:dst + p1 - p0], p0
            while view:
                k = os.preadv(fd, [view], pos)
                if k <= 0:
                    raise IOError(f"short read in {sh} at {pos}")
                view, pos = view[k:], pos + k

        list(self.pool.map(rd, jobs))
        out = {}
        for r, ro in zip(runs, offs):
            for n, a, b in r[3]:
                shard = r[0]
                dt, shape, _, _ = self._header(shard)[1][n]
                t = self.buf[ro + a - r[1]:ro + b - r[1]].view(_DT[dt]).view(shape)
                out[n] = t
        res = {}
        for e in experts:
            res[e] = {s: (out[_name(layer, e, s, "weight_packed")], out[_name(layer, e, s, "weight_scale")]) for s in SHARDS}
        self.last_bytes = total
        return res

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}
        self.pool.shutdown()
