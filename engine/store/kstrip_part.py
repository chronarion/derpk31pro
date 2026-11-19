#!/usr/bin/env python3
"""A stripped (no routed experts) and truncated Kimi K3.1 Pro for exercising the serving path: the non-expert tensors of
layers < N plus the embeddings, final norms, head, projector and vision tower, with the config cut to N layers (as
tomcupr's kstrip.py --truncate does). Unlike `kstrip.py --truncate`, it needs only the shards of the first N layers
and the three tail shards, so it can run before the full checkpoint is local.

    kstrip_part.py MODEL_DIR OUT_DIR N [--workers 4]
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.environ.get("KSTRIP_DIR", "/proj/third_party/tomcupr"))
from kstrip import EXPERT, LAYER, write  # noqa: E402

if __name__ == "__main__":     # spawn workers re-import this file: only the parent does the work
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("out"); p.add_argument("n", type=int); p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))
    wm = index["weight_map"]
    groups = {}
    for name in wm:
        if EXPERT.search(name):
            continue
        m = LAYER.search(name) if name.startswith("language_model.") else None
        if m and int(m.group(1)) >= a.n:
            continue
        groups.setdefault(f"layer-{int(m.group(1)):02d}.safetensors" if m else "rest.safetensors", []).append(name)
    new_map = {}
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for out, names in ex.map(write, [(a.model, a.out, o, ns, wm) for o, ns in sorted(groups.items())]):
            new_map.update({n: out for n in names})
            print(out, len(names), flush=True)
    json.dump({"metadata": index.get("metadata", {}), "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    for fn in os.listdir(a.model):
        s = os.path.join(a.model, fn)
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn.startswith("."):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(a.out, fn), dirs_exist_ok=True)
        elif fn == "config.json":
            cfg = json.load(open(s))
            t = cfg["text_config"]
            t.pop("quantization_config", None)
            t["num_hidden_layers"] = a.n
            for sect, key in (("linear_attn_config", "full_attn_layers"), ("linear_attn_config", "kda_layers"),
                              ("dsa_config", "dsa_layers")):
                t[sect][key] = [i for i in t[sect][key] if i <= a.n]          # these lists count layers from 1
            json.dump(cfg, open(os.path.join(a.out, fn), "w"), indent=2)
        else:
            shutil.copy(s, os.path.join(a.out, fn))
    print(f"done: {a.out}, {a.n} layers, {len(new_map)} tensors", flush=True)
