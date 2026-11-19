"""Time one real decoder layer of the runner on a GPU: the whole layer per universe, then the MoE block's parts under
the PyTorch profiler (top operations by GPU and CPU time).
usage: profile_layer.py MODEL_DIR REF_SRC TOKENS_PT [LAYER]"""
import json
import sys
import time

import torch

sys.path.insert(0, "/proj/runner")
from k31 import stream  # noqa: E402
from k31.policy import Universe  # noqa: E402

model_dir, ref_src, tok_path = sys.argv[1:4]
i = int(sys.argv[4]) if len(sys.argv) > 4 else 1
ref, cfgm = stream.load_reference(ref_src)
config = cfgm.KimiLinearConfig(**json.load(open(f"{model_dir}/config.json"))["text_config"])
config._attn_implementation = "eager"
shards = stream.Shards(model_dir)
t = time.time()
tensors = shards.read(shards.by_layer[i], 4)
print(f"read layer {i}: {sum(v.numel() * v.element_size() for v in tensors.values()) / 1e9:.1f} GB in "
      f"{time.time() - t:.1f} s", flush=True)
layer = stream.build_layer(config, ref, i)
stream.load_dense(layer, tensors, i, "cuda", torch.bfloat16)
packed = stream.load_experts(tensors, i, config.num_experts, "cuda")
del tensors
moe = layer.block_sparse_moe
moe.packed = packed
tokens = torch.load(tok_path)["tokens"]
emb = shards.read([stream.PREFIX + "embed_tokens.weight"])[stream.PREFIX + "embed_tokens.weight"]
x = emb[tokens].to("cuda", torch.bfloat16)
del emb
B, T, H = x.shape
br = x.reshape(B * T, 1, H).clone()
for u in (Universe(name="native"), Universe(name="hot-cut", experts="hotcut", hot=125, strong=5)):
    moe.universe = u
    for rep in range(3):
        torch.cuda.synchronize()
        t = time.time()
        y, _ = layer(x, attention_mask=None, block_residual=br)
        torch.cuda.synchronize()
        print(f"{u.name} layer forward {rep}: {time.time() - t:.2f} s", flush=True)
    xn = layer.post_attention_layernorm(x)
    torch.cuda.synchronize()
    t = time.time()
    moe(xn)
    torch.cuda.synchronize()
    print(f"{u.name} MoE block alone: {time.time() - t:.2f} s", flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        moe(xn)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12), flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=8), flush=True)
