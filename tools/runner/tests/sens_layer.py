"""Where does the runner amplify tiny differences? Two copies of the same input (real tokens) go through layers
0..L-1 of the runner's model; copy B starts from the embeddings with a quarter of the elements moved by one bf16 ulp
(up or down at random). For every layer and op this prints the relative change between the copies
(||a - b|| / ||a||) of the op's input and output, the router's flip rate by rank, and the MXFP8 rounding flip rate of
the expert inputs. A healthy layer passes a small change through at roughly its own size.

usage (in the runner image, one GPU):
  python3 tests/sens_layer.py MODEL_DIR REF_SRC CORPUS.pt LAYERS [--act mxfp8|bf16] [--cuts 0,2]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31 import stream                                   # noqa: E402
from k31.formats import mx_fake_quant                     # noqa: E402
from k31.policy import Universe                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("ref_src")
ap.add_argument("corpus")
ap.add_argument("layers", type=int)
ap.add_argument("--act", default="mxfp8")
ap.add_argument("--cuts", default="0,2")
ap.add_argument("--readers", type=int, default=4)
a = ap.parse_args()
dev, dtype = torch.device("cuda"), torch.bfloat16
torch.manual_seed(0)

ref, cfgmod = stream.load_reference(a.ref_src)
import json                                               # noqa: E402
config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"])
config._attn_implementation = "eager"
shards = stream.Shards(a.model_dir)
tok = torch.load(a.corpus, weights_only=False)["tokens"][[int(c) for c in a.cuts.split(",")]]
B, T = tok.shape
H, E = config.hidden_size, config.num_experts
emb = shards.read([stream.PREFIX + "embed_tokens.weight"], a.readers)[stream.PREFIX + "embed_tokens.weight"]
h = emb[tok].float()
del emb
xa = h.to(dtype)
# copy B: a quarter of the (nonzero) elements moved by one bf16 ulp, up or down at random
b16 = xa.view(torch.int16)
r = torch.randint(0, 8, b16.shape)
step = torch.where(r == 0, 1, torch.where(r == 1, -1, 0)).to(torch.int16) * (b16 != 0)
xb = (b16 + step).view(dtype)
mask = stream.causal_mask(T, dev, dtype)
xs = [xa.to(dev), xb.to(dev)]
blocks = [[], []]
u = Universe(name="native", act=a.act)


def rel(p, q):
    p, q = p.float(), q.float()
    return ((p - q).norm() / p.norm().clamp_min(1e-30)).item()


cap = {}


def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        i0 = inp[0] if inp and torch.is_tensor(inp[0]) else None
        cap.setdefault(name, []).append((None if i0 is None else i0.detach().clone(), o.detach().clone()))
    return f


print(f"cuts {a.cuts}, act {a.act}; input change {rel(xs[0], xs[1]):.2e}", flush=True)
for i in range(a.layers):
    tensors = shards.read(shards.by_layer[i], a.readers)
    lay = stream.build_layer(config, ref, i)
    stream.load_dense(lay, tensors, i, dev, dtype)
    moe_layer = hasattr(lay, "block_sparse_moe")
    if moe_layer:
        moe = lay.block_sparse_moe
        moe.packed, moe.universe, moe.layer = stream.load_experts(tensors, i, E, dev), u, i
    del tensors
    names = ["input_layernorm", "self_attn", "post_attention_layernorm"]
    names += ["block_sparse_moe", "block_sparse_moe.routed_expert_down_proj", "block_sparse_moe.routed_expert_norm",
              "block_sparse_moe.shared_experts"] if moe_layer else ["mlp"]
    if hasattr(lay.self_attn, "o_norm"):
        names += ["self_attn.o_norm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.f_b_proj",
                  "self_attn.b_proj"]
    hs = [lay.get_submodule(n).register_forward_hook(hook(n)) for n in names]
    is_block = i % config.attn_res_block_size == 0
    outs, recs = [], []
    with torch.no_grad():
        for s in range(2):
            x = xs[s]
            br = torch.stack([b.view(B * T, H) for b in blocks[s]], dim=1) if blocks[s] else x.new_zeros(B * T, 0, H)
            if moe_layer:
                moe.rec = []
                moe.stats = torch.zeros((stream.NTREAT, config.num_experts_per_token), dtype=torch.int64)
            y, br2 = lay(x, attention_mask=None if lay.is_linear_attn else mask, block_residual=br)
            if moe_layer:
                recs.append(moe.rec[-1][0].long())
            outs.append(y)
            if is_block:
                blocks[s].append(br2[:, -1].view(B, T, H))
    for hk in hs:
        hk.remove()
    line = [f"layer {i:2d} {'MoE' if moe_layer else 'dense'} {'KDA' if config.is_kda_layer(i) else 'MLA'}: "
            f"in {rel(xs[0], xs[1]):.2e} out {rel(outs[0], outs[1]):.2e}"]
    for n in names:
        (ia, oa), (ib, ob) = cap[n]
        line.append(f"{n.split('.')[-1]} in {rel(ia, ib) if ia is not None else float('nan'):.1e} "
                    f"out {rel(oa, ob):.1e}")
    if moe_layer:
        ra, rb = recs
        same = (ra.unsqueeze(-1) == rb.unsqueeze(-2)).any(-1)          # [n, K]: A's pick at rank r is in B's set
        flips = 1 - same.float().mean(0)
        line.append("router flips by rank " + " ".join(f"{100 * v:.2f}" for v in flips.tolist()) + " %")
        ha = cap["block_sparse_moe.routed_expert_down_proj"][0][1]
        hb = cap["block_sparse_moe.routed_expert_down_proj"][1][1]
        qa, qb = mx_fake_quant(ha, "mxfp8"), mx_fake_quant(hb, "mxfp8")
        line.append(f"mxfp8(h) changed elements {100 * (qa != qb).float().mean():.2f}% (h changed "
                    f"{100 * (ha != hb).float().mean():.2f}%), rel {rel(qa, qb):.2e} vs h {rel(ha, hb):.2e}")
    print("\n   ".join(line), flush=True)
    cap.clear()
    xs = outs
    del lay
    if moe_layer:
        del moe
    torch.cuda.empty_cache()
