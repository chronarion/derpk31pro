"""Does a sequence's output depend on its batch neighbours? One layer of the runner's model on cut X in three batches:
[X] alone, [A, X] and [C, X] (and [X, A]). Prints max |difference| of X's output (and of the KDA / attention output)
between the layouts. Any nonzero difference beyond GEMM rounding means information leaks across the batch.

usage: python3 tests/leak_test.py MODEL_DIR REF_SRC CORPUS.pt LAYERS(comma list, e.g. 0,3) [--x 27 --a 24 --c 31]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31 import stream                                   # noqa: E402
from k31.policy import Universe                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("ref_src")
ap.add_argument("corpus")
ap.add_argument("layers")
ap.add_argument("--x", type=int, default=27)
ap.add_argument("--a", type=int, default=24)
ap.add_argument("--c", type=int, default=31)
a = ap.parse_args()
dev, dtype = torch.device("cuda"), torch.bfloat16
ref, cfgmod = stream.load_reference(a.ref_src)
config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"])
config._attn_implementation = "eager"
shards = stream.Shards(a.model_dir)
toks = torch.load(a.corpus, weights_only=False)["tokens"]
T = toks.shape[1]
H, E = config.hidden_size, config.num_experts
emb = shards.read([stream.PREFIX + "embed_tokens.weight"], 4)[stream.PREFIX + "embed_tokens.weight"]
mask = stream.causal_mask(T, dev, dtype)
layouts = {"X": [a.x], "A,X": [a.a, a.x], "C,X": [a.c, a.x], "X,A": [a.x, a.a]}
pos_of_x = {"X": 0, "A,X": 1, "C,X": 1, "X,A": 0}
for i in [int(v) for v in a.layers.split(",")]:
    tensors = shards.read(shards.by_layer[i], 4)
    lay = stream.build_layer(config, ref, i)
    stream.load_dense(lay, tensors, i, dev, dtype)
    if hasattr(lay, "block_sparse_moe"):
        moe = lay.block_sparse_moe
        moe.packed, moe.universe, moe.layer = stream.load_experts(tensors, i, E, dev), Universe(name="native"), i
        moe.stats = torch.zeros((stream.NTREAT, config.num_experts_per_token), dtype=torch.int64)
    del tensors
    cap = {}
    hk = lay.self_attn.register_forward_hook(lambda m, inp, out: cap.__setitem__("attn", out[0] if isinstance(out, tuple) else out))
    res = {}
    with torch.no_grad():
        for name, cuts in layouts.items():
            x = emb[toks[cuts]].to(dev, dtype)          # layer input: the embeddings (enough to test the layer's mixing)
            B = x.shape[0]
            br = x.new_zeros(B * T, 0, H)
            y, _ = lay(x, attention_mask=None if lay.is_linear_attn else mask, block_residual=br)
            p = pos_of_x[name]
            res[name] = (y[p].float().cpu(), cap["attn"][p].float().cpu())
    hk.remove()
    base_y, base_a = res["X"]
    kind = "KDA" if config.is_kda_layer(i) else "MLA"
    for name in ("A,X", "C,X", "X,A"):
        y, at = res[name]
        dy, da = (y - base_y).abs(), (at - base_a).abs()
        first = int((da.amax(-1) > 0).nonzero()[0]) if (da > 0).any() else -1
        print(f"layer {i} {kind}: X in [{name}] vs alone: output max|d| {dy.max():.3e} (rel {dy.norm() / base_y.norm():.2e}), "
              f"attention max|d| {da.max():.3e} (rel {da.norm() / base_a.norm():.2e}), first differing position {first}",
              flush=True)
    del lay
    torch.cuda.empty_cache()
