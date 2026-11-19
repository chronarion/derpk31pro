"""Build every Kimi K3.1 Pro decoder layer from Moonshot's K3 reference modeling code on the meta device (no weights,
no GPU) and compare its parameters with the official checkpoint's tensor list: names, shapes, and which layers are
KDA and which MLA (the DSA layers; their indexer tensors are not in the K3 reference and are skipped, as kstream.py
does: at <= 2,048 tokens the top-2,048 selection keeps every token).

usage: check_structure.py REF_SRC_DIR MODEL_META_DIR   (MODEL_META_DIR holds config.json and tensors.json)
"""
import importlib
import json
import re
import shutil
import sys
import tempfile

import torch
import transformers.utils.generic as tf_generic
import transformers.utils.output_capturing as tf_capture

if not hasattr(tf_generic, "OutputRecorder"):        # transformers 5.x moved it; the K3 reference imports the old path
    tf_generic.OutputRecorder = tf_capture.OutputRecorder

PREFIX = "language_model.model."


def load_reference(ref_src):
    pkg_root = tempfile.mkdtemp()
    shutil.copytree(ref_src, pkg_root + "/k3ref")
    sys.path.insert(0, pkg_root)
    return (importlib.import_module("k3ref.modeling_kimi_linear"),
            importlib.import_module("k3ref.configuration_kimi_k3"))


def checkpoint_layers(tensors):
    """name -> shape per layer, with MXFP4 experts folded back to one logical [out, in] weight."""
    layers = {}
    for name, (_dtype, shape, _nbytes) in tensors.items():
        m = re.match(re.escape(PREFIX) + r"layers\.(\d+)\.(.*)", name)
        if not m:
            continue
        key = m.group(2)
        if ".indexer." in key or key.endswith(".weight_scale"):
            continue
        if key.endswith(".weight_packed"):
            key, shape = key[: -len("_packed")], [shape[0], shape[1] * 2]
        layers.setdefault(int(m.group(1)), {})[key] = list(shape)
    return layers


def main(ref_src, meta_dir):
    ref, cfgmod = load_reference(ref_src)
    config = cfgmod.KimiLinearConfig(**json.load(open(f"{meta_dir}/config.json"))["text_config"])
    config._attn_implementation = "eager"
    tensors = json.load(open(f"{meta_dir}/tensors.json"))
    have = checkpoint_layers(tensors)
    kinds, bad = {}, 0
    for i in range(config.num_hidden_layers):
        with torch.device("meta"):
            layer = ref.KimiDecoderLayer(config, i)
        want = {k: list(p.shape) for k, p in layer.named_parameters()}
        got = have.get(i, {})
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        wrong = sorted(k for k in set(want) & set(got) if want[k] != got[k])
        kind = "KDA" if layer.is_linear_attn else "MLA"
        kinds.setdefault(kind, []).append(i)
        if missing or extra or wrong:
            bad += 1
            print(f"layer {i} ({kind}): missing {missing[:3]} extra {extra[:3]} shape {wrong[:3]}", flush=True)
        del layer
    for kind, ids in kinds.items():
        print(f"{kind}: {len(ids)} layers, first {ids[:6]}, last {ids[-3:]}")
    print("indexer tensors in the checkpoint (not in the K3 reference):",
          sum(1 for n in tensors if ".indexer." in n))
    top = [n[len(PREFIX):] for n in tensors if n.startswith(PREFIX) and ".layers." not in n]
    print("top-level tensors:", sorted(top))
    print("layers with mismatches:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
