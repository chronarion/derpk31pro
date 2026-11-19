"""GPU test of K31_DSA_ROW_SLICE: one DSA layer's sparse attention (vLLM's MLAAttention.forward_impl with FlashInfer's
TRT-LLM sparse MLA, FP8 KV cache, Kimi K3.1 Pro's shapes: 96 heads, 128 + 64 query dims, 512-dim latent, 128-dim
values, top-2,048) on random inputs, unsliced against row slices, bit for bit.

The batch is a mixed step of 8,192 rows: two decode requests of 5 rows (contexts 4,000 and 9,000), then three prompt
chunks (4,990 rows after 6,000 cached tokens, 2,892 after 3,000, and 300 rows of a new request, whose rows see fewer
than 2,048 keys: their top-k rows end in -1). Random distinct causal top-k indices per row, random FP8 cache, random
BF16 queries and kv_b_proj weights.
  1. forward_impl unsliced, twice: equal (the kernels' own determinism; the slices cannot be compared otherwise).
  2. Through the K31_DSA_ROW_SLICE wrapper at 4,096 / 2,048 / 1,024 / 1,000 / 512 rows: output bit-equal to (1).
  3. Diagnostics per kernel (lines "DIAG", not part of the verdict): the W_UK and W_UV batched GEMMs, the FP8 query
     quantization, the top-k index conversion and the TRT-LLM sparse MLA kernel, each sliced against whole.
  4. Memory: the call's transient peak (torch allocator, beyond the query and the output) unsliced and at 1,024 rows;
     the sliced one must be below half the unsliced one (expected ~2.9 GiB against ~0.4).
  5. Time per call (CUDA events, mean of 5 after warm-up), unsliced and sliced: information only.

Run (from engine/; the GB300 alone, ~6 GB of HBM, a few minutes with Triton's first compiles; the script ends itself
after 1,200 s, and the container has a name to kill if the docker client is interrupted):
    docker run --rm --name k31-dsa-slice-test --network none --device nvidia.com/gpu=$GB --ulimit core=0 \
        --user $(id -u):$(id -g) -e HOME=/tmp -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo --cpuset-cpus 56-57 \
        -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_dsa_slice_gpu.py
Exit code 0 and "DSA ROW SLICE GPU: PASS" = checks 1, 2 and 4 passed.
"""
import os
import socket
import sys
import threading
import time
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
os.environ.setdefault("K31_FLAGS_FILE", "/nonexistent/k31-flags.json")      # flags from the environment only
fails = []
T_MAX, H, NOPE, ROPE, L, V, TOPK, BS = 8192, 96, 128, 64, 512, 128, 2048, 64


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def diag(name, ok, detail=""):
    print(f"DIAG {'equal' if ok else 'DIFFERS'}: {name} {detail}", flush=True)


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _timeout():
    print("DSA ROW SLICE GPU: FAIL (timeout after 1,200 s)", flush=True)
    os._exit(3)


_watchdog = threading.Timer(1200, _timeout)
_watchdog.daemon = True
_watchdog.start()
t_start = time.time()
dev = torch.device("cuda")
print(f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
from vllm.config import (CacheConfig, CompilationConfig, CompilationMode, SchedulerConfig,  # noqa: E402
                         VllmConfig, set_current_vllm_config)
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment  # noqa: E402

cfg = VllmConfig(cache_config=CacheConfig(cache_dtype="fp8", block_size=BS),
                 scheduler_config=SchedulerConfig.default_factory(max_model_len=65536, max_num_batched_tokens=T_MAX,
                                                                  max_num_seqs=4),
                 compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
ctx = set_current_vllm_config(cfg)
ctx.__enter__()
init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{free_port()}",
                             local_rank=0, backend="nccl")
ensure_model_parallel_initialized(1, 1)
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

init_workspace_manager(dev)                    # as the GPU worker does before building the model (prefill backends)

import vllm.model_executor.layers.attention.mla_attention as M  # noqa: E402
import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as F  # noqa: E402
from vllm.model_executor.layers.attention.sparse_mla_attention import SparseMLAPrefillMetadata  # noqa: E402
from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402
from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index  # noqa: E402

from k31_engine import rowslice  # noqa: E402

ORIG = M.MLAAttention.forward_impl            # vLLM's own, before the wrapper goes on
g = torch.Generator(device="cpu").manual_seed(31)
topk_buf = torch.full((T_MAX, TOPK), -1, dtype=torch.int32, device=dev)
indexer = types.SimpleNamespace(topk_indices_buffer=topk_buf)    # the impl resolves the buffer through its indexer
old_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    with torch.device(dev):
        kv_b = ColumnParallelLinear(L, H * (NOPE + V), bias=False, params_dtype=torch.bfloat16, quant_config=None,
                                    prefix="model.layers.3.self_attn.kv_b_proj")
        w = torch.randn(tuple(kv_b.weight.shape), generator=g, dtype=torch.float32, device="cpu") * 0.02
        kv_b.weight.data.copy_(w.to(torch.bfloat16))
        attn = M.MLAAttention(num_heads=H, scale=(NOPE + ROPE) ** -0.5, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE,
                              v_head_dim=V, q_lora_rank=1536, kv_lora_rank=L, kv_b_proj=kv_b,
                              cache_config=cfg.cache_config, quant_config=None, prefix="model.layers.3.self_attn.attn",
                              attn_backend=F.FlashInferMLASparseTRTLLMBackend, use_sparse=True, indexer=indexer,
                              topk_indices_buffer=topk_buf)
        attn.process_weights_after_loading(torch.bfloat16)
finally:
    torch.set_default_dtype(old_dtype)
F._reserve_kv_counters(dev, T_MAX, F.trtllm_gen_query_heads(H))   # as the metadata builder does before serving
print(f"layer: {type(attn.impl).__name__}, kv cache {attn.kv_cache_dtype}, prims_ts {attn.impl.use_prims_ts}, "
      f"W_UK_T {tuple(attn.W_UK_T.shape)}, W_UV {tuple(attn.W_UV.shape)} ({time.time() - t_start:.0f} s)", flush=True)

# ---------------------------------------------------------------------------------------------------- the batch
reqs = [(4000, 5), (9000, 5), (6000, 4990), (3000, 2892), (0, 300)]      # (cached tokens, rows this step)
n_dec, n_dec_tok = 2, 10
T = sum(n for _, n in reqs)
assert T == T_MAX
seqs = [c + n for c, n in reqs]
blocks = [-(-s // BS) for s in seqs]
NB = sum(blocks) + 16
perm = torch.randperm(NB, generator=g)
bt = torch.zeros(len(reqs), max(blocks), dtype=torch.int32)
o = 0
for r, nb in enumerate(blocks):
    bt[r, :nb] = perm[o:o + nb].to(torch.int32)
    o += nb
bt = bt.to(dev)
kv = (torch.randn(NB, BS, L + ROPE, generator=g) * 0.5).to(torch.float8_e4m3fn).to(dev)
rid = torch.cat([torch.full((n,), r, dtype=torch.int32) for r, (_, n) in enumerate(reqs)]).to(dev)
pos = torch.cat([torch.arange(c, c + n) for c, n in reqs])                 # each row's position in its request
for a in range(0, T, 512):                                                  # distinct causal top-k per row
    b = min(T, a + 512)
    vis = pos[a:b] + 1                                                      # keys a row may see
    width = int(vis.max())
    r = torch.rand(b - a, width, generator=g)
    col = torch.arange(width)
    r = torch.where(col[None, :] < vis[:, None], r, -1.0)
    idx = r.topk(min(TOPK, width), dim=1).indices.to(torch.int32)
    idx = torch.where(idx < vis[:, None], idx, -1)
    row = torch.full((b - a, TOPK), -1, dtype=torch.int32)
    row[:, :idx.shape[1]] = idx
    topk_buf[a:b] = row.to(dev)
q = torch.randn(T, H, NOPE + ROPE, generator=g).to(torch.bfloat16).to(dev)
kc = torch.randn(T, L, generator=g).to(torch.bfloat16).to(dev)
kp = torch.randn(T, 1, ROPE, generator=g).to(torch.bfloat16).to(dev)
qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor([n for _, n in reqs]), 0)), dtype=torch.int32, device=dev)
prefill = SparseMLAPrefillMetadata(block_table=bt[n_dec:], query_start_loc=qsl[n_dec:] - n_dec_tok,
                                   max_query_len=max(n for _, n in reqs[n_dec:]), use_dense_mha=False)
md = F.FlashInferMLASparseMetadata(
    num_reqs=len(reqs), max_query_len=max(n for _, n in reqs), max_seq_len=max(seqs), num_actual_tokens=T,
    query_start_loc=qsl, slot_mapping=torch.full((T,), -1, dtype=torch.int64, device=dev), num_decodes=n_dec,
    num_decode_tokens=n_dec_tok, num_prefills=len(reqs) - n_dec, block_table=bt, req_id_per_token=rid,
    seq_lens=torch.tensor(seqs, dtype=torch.int32, device=dev), block_size=BS, topk_tokens=TOPK,
    prefill_max_seq_len=max(seqs[n_dec:]), prefill=prefill)
attn.impl.record_logical_topk_ready()
print(f"batch: {T} rows, {len(reqs)} requests ({n_dec} decode), {NB} cache blocks; all-MQA path: "
      f"{not attn._use_sparse_mha(md)} ({time.time() - t_start:.0f} s)", flush=True)
check("the batch takes the all-MQA sparse path (as a prompt chunk with > 2,048 tokens of context)",
      not attn._use_sparse_mha(md))


def new_out():
    return torch.full((T, H * V), float("nan"), dtype=torch.bfloat16, device=dev)


def unsliced():
    out = new_out()
    ORIG(attn, q, kc, kp, kv, md, output=out)
    return out


def sliced(rows):
    os.environ["K31_DSA_ROW_SLICE"] = str(rows)
    out = new_out()
    attn.forward_impl(q, kc, kp, kv, md, output=out)
    os.environ.pop("K31_DSA_ROW_SLICE")
    return out


# ---------------------------------------------------------------------------------------------- 1. determinism
ref = unsliced()
ref2 = unsliced()
torch.cuda.synchronize()
check("unsliced forward_impl twice: equal (deterministic kernels)", torch.equal(ref, ref2),
      f"(max abs {maxdiff(ref, ref2):.3e}; finite {bool(torch.isfinite(ref).all())})")

# ------------------------------------------------------------------------------------------------- 2. slices
rowslice.install_dsa(M)
for rows in (4096, 2048, 1024, 1000, 512):
    out = sliced(rows)
    n = len(rowslice.plan(T, rowslice._dsa_rows(attn.impl, rows)))
    check(f"K31_DSA_ROW_SLICE={rows}: {n} slices, output bit-equal to the unsliced call", torch.equal(out, ref),
          "" if torch.equal(out, ref) else f"(max abs {maxdiff(out, ref):.3e}, rows differing "
                                           f"{int((out != ref).any(1).sum())})")
check("the wrapper sliced (its one log line was written)", rowslice._DSA["logged"])

# --------------------------------------------------------------------------------------------- 3. diagnostics
sl = rowslice.plan(T, 1000)
qn = q[..., :NOPE].transpose(0, 1)
full = torch.bmm(qn, attn.W_UK_T)
parts = torch.cat([torch.bmm(qn[:, a:b], attn.W_UK_T) for a, b in sl], 1)
diag("W_UK batched GEMM (96 x T x 128 @ 128 x 512)", torch.equal(full, parts), f"(max abs {maxdiff(full, parts):.3e})")
ql = full.transpose(0, 1).contiguous()
qpe = q[..., NOPE:].contiguous()
qf = attn._decode_concat_quant_fp8_op(ql, qpe, attn._q_scale)
qf_parts = torch.cat([attn._decode_concat_quant_fp8_op(ql[a:b], qpe[a:b], attn._q_scale) for a, b in sl])
diag("FP8 query quantization", torch.equal(qf.view(torch.uint8), qf_parts.view(torch.uint8)))
phys, cnt = triton_convert_req_index_to_global_index(rid, bt, topk_buf[:T], BLOCK_SIZE=BS, BLOCK_STRIDE_ROWS=BS,
                                                     NUM_TOPK_TOKENS=TOPK, return_valid_counts=True)
pp = [triton_convert_req_index_to_global_index(rid[a:b], bt, topk_buf[a:b], BLOCK_SIZE=BS, BLOCK_STRIDE_ROWS=BS,
                                               NUM_TOPK_TOKENS=TOPK, return_valid_counts=True) for a, b in sl]
diag("top-k index conversion", torch.equal(phys, torch.cat([p for p, _ in pp])) and
     torch.equal(cnt, torch.cat([c for _, c in pp])))
kv8 = kv.view(torch.float8_e4m3fn)
o_full, _ = attn.impl._forward_mqa_kernel(qf, kv8, phys, cnt, layer=attn, block_size=BS, is_decode=False)
o_parts = torch.cat([attn.impl._forward_mqa_kernel(qf[a:b], kv8, phys[a:b], cnt[a:b], layer=attn, block_size=BS,
                                                   is_decode=False)[0] for a, b in sl])
diag("TRT-LLM sparse MLA kernel", torch.equal(o_full, o_parts), f"(max abs {maxdiff(o_full, o_parts):.3e})")
x = o_full.reshape(T, H, L).transpose(0, 1)
vf = torch.bmm(x, attn.W_UV)
vp = torch.cat([torch.bmm(x[:, a:b], attn.W_UV) for a, b in sl], 1)
diag("W_UV batched GEMM (96 x T x 512 @ 512 x 128)", torch.equal(vf, vp), f"(max abs {maxdiff(vf, vp):.3e})")
del full, parts, ql, qf, qf_parts, phys, cnt, pp, o_full, o_parts, x, vf, vp

# ------------------------------------------------------------------------------------------------- 4. memory


def transient(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    out = new_out()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(out)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def run_unsliced(out):
    ORIG(attn, q, kc, kp, kv, md, output=out)


def run_sliced(rows):
    def f(out):
        os.environ["K31_DSA_ROW_SLICE"] = str(rows)
        attn.forward_impl(q, kc, kp, kv, md, output=out)
        os.environ.pop("K31_DSA_ROW_SLICE")
    return f


m_full = transient(run_unsliced)
m_slices = {rows: transient(run_sliced(rows)) for rows in (2048, 1024, 512)}
print("memory: transient of one call beyond its query and output (GiB): unsliced %.2f; %s" % (
    m_full, ", ".join(f"{r} rows {m:.2f}" for r, m in m_slices.items())), flush=True)
check("memory: at 1,024-row slices below half the unsliced transient", m_slices[1024] < 0.5 * m_full,
      f"({m_slices[1024]:.2f} vs {m_full:.2f} GiB)")

# --------------------------------------------------------------------------------------------------- 5. time


def timed(fn, n=5):
    out = new_out()
    for _ in range(2):
        fn(out)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn(out)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


t_full = timed(run_unsliced)
times = {rows: timed(run_sliced(rows)) for rows in (4096, 2048, 1024)}
print("time per call (ms): unsliced %.3f; %s" % (t_full, ", ".join(f"{r} rows {t:.3f}" for r, t in times.items())),
      flush=True)

ctx.__exit__(None, None, None)
print(f"DSA ROW SLICE GPU: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)} ({time.time() - t_start:.0f} s)",
      flush=True)
os._exit(1 if fails else 0)            # no wait on the process group's threads at exit
