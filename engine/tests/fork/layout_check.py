"""derpk31pro: what an expert row looks like in memory on the kimi-k31 image (the kernel layout vLLM now loads into),
checked three ways for one expert at K3.1's geometry:
  1. vLLM's loader on the GPU (DeepGemmMegaMoEExperts.weight_loader into w13_weight / w13_weight_scale / w2_weight /
     w2_weight_scale): the reference bytes.
  2. the same loader code on the CPU, writing straight into a pinned host row with the bank's strides
     (DeepGemmMegaMoEExperts._load_weight on a pinned tensor): how the engine can build host rows with no GPU
     transform. Must equal 1 byte for byte.
  3. explicit index formulas (no vLLM code), the contract a store builder can follow. Must equal 1.
Then the 6-bit scale codes of the row (Derpseek's sf6.py): size and decode(encode) == the uncoded scales.
Needs the GB300 only for (1) (the module refuses to build without an SM100 device); a few hundred MB.
env: SF6_PY (directory of sf6.py; skipped when absent)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k31_fork_common as c  # noqa: E402

H, I = c.H, c.I
g = torch.Generator(device="cpu").manual_seed(5)
shards = {}
for shard, rows, cols in (("w1", I, H), ("w3", I, H), ("w2", H, I)):
    shards[shard] = (torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g),
                     torch.randint(119, 124, (rows, cols // 32), dtype=torch.uint8, generator=g))

# 1. vLLM's loader on the GPU
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts  # noqa: E402

exp = c.make_experts(1, torch.Generator(device=c.dev).manual_seed(0), 64, "r5.layout.experts")
params = dict(exp.named_parameters())
for shard, (w, s) in shards.items():
    pp = "w2" if shard == "w2" else "w13"
    for suffix, t in (("weight", w), ("weight_scale", s)):
        p = params[f"{pp}_{suffix}"]
        p.weight_loader(p, t, f"layers.0.block_sparse_moe.experts.{pp}_{suffix}", expert_id=0, shard_id=shard)
b4 = [t.cpu() for t in c.bank(exp)]
names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
for n, t in zip(names, b4):
    print(f"{n}: shape {tuple(t.shape)} {t.dtype}, strides {tuple(t.stride())} (elements), "
          f"{t[0].numel() * t.element_size()} B per expert", flush=True)
print(f"one expert: {c.row_bytes(b4)} B", flush=True)

# 2. the loader's code on the CPU, into a pinned host row with the bank's strides
host = [torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, pin_memory=True) for t in b4]
for shard, (w, s) in shards.items():
    k = 2 if shard == "w2" else 0
    exp._load_weight(host[k][0], w, shard, False)
    exp._load_weight(host[k + 1][0], s, shard, True)
ok2 = all(torch.equal(a, b) for a, b in zip(host, b4))
print(f"2. DeepGemmMegaMoEExperts._load_weight on a pinned CPU row: {'BYTES EQUAL the GPU-loaded row' if ok2 else 'DIFFERS'}",
      flush=True)

# 3. explicit formulas
# weights: kernel row R of w13 = checkpoint row r of w1 (gate) or w3 (up): R = 16 * (r // 8) + 8 * half + r % 8; w2 as is
w13 = torch.empty(2 * I, H // 2, dtype=torch.uint8)
r = torch.arange(I)
for half, shard in enumerate(("w1", "w3")):
    w13[16 * (r // 8) + 8 * half + r % 8] = shards[shard][0]
ok_w13 = torch.equal(w13.view(torch.int8), b4[0][0])
ok_w2 = torch.equal(shards["w2"][0].view(torch.int8), b4[2][0])


def scale_mem(n_rows, kb, src_by_pos):
    """memory of one expert's scales: [kb K blocks][n_rows positions] int32; byte j of the word at (k, P) is the scale
    of position P for the K block's j-th group of 32 weights"""
    mem = torch.empty(kb, n_rows, 4, dtype=torch.uint8)
    for pos, row_scales in src_by_pos:                        # row_scales: [rows, kb * 4]
        mem[:, pos] = row_scales.view(-1, kb, 4).permute(1, 0, 2)
    return mem.view(kb, n_rows * 4).contiguous().view(torch.int32)


# L1 scales: checkpoint row r = 64 b + 16 j + 8 q + m of half (w1 0, w3 1) sits at position 128 b + 64 q + 32 half + 4 m + j
b, j, q, m = r // 64, (r % 64) // 16, (r % 16) // 8, r % 8
s13 = scale_mem(2 * I, H // 128, [(128 * b + 64 * q + 32 * half + 4 * m + j, shards[sh][1]) for half, sh in enumerate(("w1", "w3"))])
# L2 scales: checkpoint row r = 128 b + 32 j + i sits at position 128 b + 4 i + j
r2 = torch.arange(H)
s2 = scale_mem(H, I // 128, [(128 * (r2 // 128) + 4 * (r2 % 32) + (r2 % 128) // 32, shards["w2"][1])])
# the registered scale tensors are [N, KB] views (strides (1, N)) of [KB][N] memory
ok_s13 = torch.equal(s13, b4[1][0].t().contiguous())
ok_s2 = torch.equal(s2, b4[3][0].t().contiguous())
print(f"3. explicit formulas: w13_weight {ok_w13}, w2_weight {ok_w2}, w13_weight_scale {ok_s13}, w2_weight_scale {ok_s2}", flush=True)
ok = ok2 and ok_w13 and ok_w2 and ok_s13 and ok_s2

sf6_dir = os.environ.get("SF6_PY", "/d/engine/derpseek_engine")
if os.path.exists(os.path.join(sf6_dir, "sf6.py")):
    sys.path.insert(0, sf6_dir)
    import sf6
    for k in (1, 3):
        coded = sf6.encode(b4[k])
        back = sf6.decode(coded, b4[k].shape[1], b4[k].shape[2])
        same = torch.equal(back, b4[k])
        ok = ok and same
        print(f"sf6 (E0 {sf6.E0}, layout {sf6.LAYOUT}) {names[k]}: {coded.shape[1]} B per expert coded "
              f"({b4[k][0].numel() * 4} uncoded), decode(encode) {'equal' if same else 'DIFFERS'}", flush=True)
print("layout check:", "ALL EQUAL" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
