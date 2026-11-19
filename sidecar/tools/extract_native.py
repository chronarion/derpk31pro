"""Raw official MXFP4 tensors of a few routed experts, from Moonshot's checkpoint (standard library only).

    python tools/extract_native.py MODEL_DIR LAYER OUT_DIR EXPERT [EXPERT ...]

Writes OUT_DIR/L{layer:02d}_e{expert:03d}.native = w1 packed [3072, 1792], w1 scale [3072, 112], w2 packed [3584, 1536],
w2 scale [3584, 96], w3 packed, w3 scale (17,547,264 B), the checkpoint's bytes as they are.
"""
import json
import os
import struct
import sys

model, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
experts = [int(v) for v in sys.argv[4:]]
wm = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
hdr = {}


def header(shard):
    if shard not in hdr:
        with open(os.path.join(model, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        meta.pop("__metadata__", None)
        hdr[shard] = (8 + n, meta)
    return hdr[shard]


os.makedirs(out, exist_ok=True)
for e in experts:
    blob = b""
    for shard_id in ("w1", "w2", "w3"):
        for part in ("weight_packed", "weight_scale"):
            name = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{shard_id}.{part}"
            shard = wm[name]
            base, meta = header(shard)
            a, b = meta[name]["data_offsets"]
            fd = os.open(os.path.join(model, shard), os.O_RDONLY)
            data = os.pread(fd, b - a, base + a)
            os.close(fd)
            assert len(data) == b - a
            print(f"{name}: {meta[name]['dtype']} {meta[name]['shape']} {len(data)} B")
            blob += data
    path = os.path.join(out, f"L{layer:02d}_e{e:03d}.native")
    open(path, "wb").write(blob)
    print(path, len(blob))
