"""Row slices for long prompt chunks (8,192 rows): a DSA layer's sparse attention and MegaMoE's symmetric buffer.

K31_DSA_ROW_SLICE=<rows> (runtime flag, flags.py; 0 = off, the default). Once a request's context passes 2,048 tokens,
vLLM's MLAAttention.forward_impl sends every row of a prompt chunk through the sparse MQA path in one call, with
~380 KB per row of temporaries beside the query and the output: the latent query in BF16 [T, 96, 576] and FP8, the FP8
query padded to 128 heads, the 128-head latent output [T, 128, 512] BF16 and the physical top-k rows [T, 2048] int32.
At 8,192 rows that is 2.9 GiB (3.4 GiB with the query and the output), the largest transient of a prompt forward.
A row of that path reads only its own query, its own top-k row (the model-level topk_indices_buffer[t]) and its
request id (req_id_per_token[t]); the cache and the block table are shared. So a call of more than <rows> rows runs as
balanced slices of at most <rows> rows, each through vLLM's own forward_impl with the slice's metadata (a shallow copy
with num_actual_tokens, num_decode_tokens 0 and req_id_per_token[a:b]) and the impl's top-k buffer seen from the
slice's first row. The kernels and their inputs per row are the same; the output is meant to be bit-identical (the
GPU test tests/test_dsa_slice_gpu.py and the in-server check below say whether it is).
Only calls that take the all-MQA path are sliced: sparse impl, no DCP / PCP / ShardQ / HiSparse, no quantized output,
not capturing, and a prefill that does not take dense or masked MHA (contexts of 2,048 tokens or less run dense MHA,
which is not sliced). Everything else goes to vLLM's forward_impl unchanged.
K31_DSA_ROW_SLICE_CHECK=n (runtime): the first n sliced calls of each layer also run unsliced, twice (into scratch
outputs: 3.4 GiB more at 8,192 rows); one line each: bit-equal or the largest difference, and whether two unsliced
runs agree (the kernel's own determinism).

K31_MEGAMOE_ROWS=<rows> (start-up; 0 = off). MegaMoE's symmetric buffer is sized for max_num_tokens, which vLLM sets
to max_num_batched_tokens: 2,243 MiB at 8,192 against 1,563 at 4,096 and 1,232 at 2,048 (start-up logs, 506 expert
rows). With this set, every MoE layer's DeepGemmMegaMoEExperts is built for <rows> tokens and every call of more rows
(prompt chunks, the profile run) runs in balanced row slices of at most <rows> rows, as K31_PROMPT_SUB does inside the
prompt path. The value is raised to 256 if smaller (decode and graph-sized calls are never sliced).
K31_MEGAMOE_ROWS_CHECK=n (start-up): the buffer keeps its full size, calls above <rows> are still sliced, and the
first n of them per layer also run unsliced, twice; one line each, as above.
"""
import copy
import os
import sys

import torch

from . import flags

_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
MIN_SLICE = 256          # smallest slice size accepted (smaller values are raised to this)
_DSA: dict = {"logged": False, "checks": {}}
_MOE: dict = {"rows": 0, "check": 0, "logged": False, "checks": {}}


def plan(T: int, rows: int) -> list:
    """Balanced row slices [(a, b), ...] covering [0, T): n = ceil(T / rows) slices of floor(T / n) or one more row,
    so every slice holds at most `rows` and at least rows // 2 rows. One slice (0, T) when T <= rows or rows <= 0."""
    if rows <= 0 or T <= rows:
        return [(0, T)]
    n = -(-T // rows)
    base, extra = divmod(T, n)
    out, a = [], 0
    for i in range(n):
        b = a + base + (1 if i < extra else 0)
        out.append((a, b))
        a = b
    return out


# ----------------------------------------------------------------------------------- DSA: MLAAttention.forward_impl
_KW = frozenset(("output", "output_scale", "output_block_scale", "quant_group_size", "quant_scale_ue8m0",
                 "quant_col_major", "quant_tma_aligned", "q_dcp_replicated"))


def _dsa_rows(impl, rows: int) -> int:
    """The slice size for this impl: at least MIN_SLICE, and every slice (>= rows // 2 rows) larger than the index
    group's decode workspace, so each slice converts its top-k rows into a fresh tensor (the large-call path of
    SparseMLAIndexGroup._convert_once) and never into the group's shared buffer on its side stream."""
    phys = getattr(getattr(impl, "index_group", None), "physical_topk_indices", None)
    floor = 2 * int(phys.shape[0]) + 2 if phys is not None else 0
    return max(rows, MIN_SLICE, floor)


def sliceable(layer, q, md, args, kwargs, rows: int) -> int:
    """The call's row count T if forward_impl may run it in row slices of `rows` (> 0), else 0."""
    if rows <= 0 or md is None or args or not kwargs.keys() <= _KW:
        return 0
    out = kwargs.get("output")
    if out is None or kwargs.get("output_scale") is not None or kwargs.get("output_block_scale") is not None or \
            kwargs.get("q_dcp_replicated") is not None:
        return 0
    impl = getattr(layer, "impl", None)
    if not getattr(impl, "is_sparse", False) or getattr(impl, "dcp_world_size", 1) != 1 or \
            getattr(layer, "use_pcp", False) or getattr(layer, "use_shardq", False) or \
            getattr(layer, "hisparse_cache", None) is not None or getattr(md, "shardq", None) is not None:
        return 0
    if not (hasattr(impl, "_indexer") and hasattr(impl, "_topk_indices_buffer")):
        return 0                                     # SharedTopkIndicesBuffer's fields: the buffer view is swapped
    T = int(getattr(md, "num_actual_tokens", 0) or 0)
    if T <= _dsa_rows(impl, rows) or q.shape[0] < T or out.shape[0] < T or out.dtype != q.dtype:
        return 0
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return 0
    rid = getattr(md, "req_id_per_token", None)
    buf = impl.topk_indices_buffer
    if rid is None or rid.shape[0] < T or buf is None or buf.shape[0] < T:
        return 0
    use_mha = getattr(layer, "_use_sparse_mha", None)
    nd = int(getattr(md, "num_decode_tokens", 0) or 0)
    if use_mha is None or (T - nd > 0 and use_mha(md)):   # prefill rows on dense or masked MHA: not sliced
        return 0
    return T


def run_slices(orig, layer, q, k_c_normed, k_pe, kv_cache, md, kwargs, T: int, rows: int):
    """forward_impl over balanced row slices of at most `rows` rows into kwargs["output"] (rows [0, T))."""
    impl = layer.impl
    output = kwargs["output"]
    rest = {k: v for k, v in kwargs.items() if k != "output"}
    buf = impl.topk_indices_buffer                       # the model-level buffer the indexer wrote, row t = q row t
    rid = md.req_id_per_token
    saved = (impl._indexer, impl._topk_indices_buffer)
    try:
        for a, b in plan(T, _dsa_rows(impl, rows)):
            m = copy.copy(md)
            m.num_actual_tokens = b - a
            m.num_decode_tokens = 0                      # every row of the call takes the MQA path, as unsliced
            m.req_id_per_token = rid[a:b]
            impl._indexer, impl._topk_indices_buffer = None, buf[a:]   # forward_mqa reads buffer[:rows] = rows a..
            orig(layer, q[a:b], k_c_normed[a:b], k_pe[a:b], kv_cache, m, output=output[a:b], **rest)
    finally:
        impl._indexer, impl._topk_indices_buffer = saved
    return output


def _dsa_check(orig, layer, q, k_c_normed, k_pe, kv_cache, md, kwargs, T: int, rows: int) -> None:
    out = kwargs["output"][:T]
    rest = {k: v for k, v in kwargs.items() if k != "output"}
    ref = torch.empty_like(out)
    ref2 = torch.empty_like(out)
    orig(layer, q, k_c_normed, k_pe, kv_cache, md, output=ref, **rest)
    orig(layer, q, k_c_normed, k_pe, kv_cache, md, output=ref2, **rest)
    same, det = bool(torch.equal(out, ref)), bool(torch.equal(ref, ref2))
    diff = 0.0 if same else float((out.float() - ref.float()).abs().max())
    n = len(plan(T, _dsa_rows(layer.impl, rows)))
    _LOG(f"DSA row slice check {getattr(layer, 'layer_name', '?')}: {T} rows in {n} slices of <= "
         f"{_dsa_rows(layer.impl, rows)}: " + ("bit-equal to the unsliced call" if same else
                                                 f"DIFFERS from the unsliced call (max abs {diff:.3e})")
         + f"; two unsliced runs {'agree' if det else 'DIFFER (the kernel is not deterministic)'}")
    del ref, ref2


def install_dsa(module=None) -> bool:
    """Wrap MLAAttention.forward_impl once; the wrapper reads K31_DSA_ROW_SLICE per call (0: vLLM's call as is)."""
    if module is None:
        import importlib
        module = importlib.import_module("vllm.model_executor.layers.attention.mla_attention")
    cls = getattr(module, "MLAAttention", None)
    if cls is None or getattr(cls, "_k31_rowslice", False):
        return False
    orig = cls.forward_impl

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *args, **kwargs):
        if attn_metadata is None or getattr(attn_metadata, "num_actual_tokens", 0) <= MIN_SLICE:
            return orig(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *args, **kwargs)
        rows = flags.num("K31_DSA_ROW_SLICE", 0)
        T = sliceable(self, q, attn_metadata, args, kwargs, rows)
        if not T:
            return orig(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *args, **kwargs)
        out = run_slices(orig, self, q, k_c_normed, k_pe, kv_cache, attn_metadata, kwargs, T, rows)
        if not _DSA["logged"]:
            _DSA["logged"] = True
            _LOG(f"DSA row slices (K31_DSA_ROW_SLICE={rows}): {T} rows of {getattr(self, 'layer_name', '?')} in "
                 f"{len(plan(T, _dsa_rows(self.impl, rows)))} slices of <= {_dsa_rows(self.impl, rows)}")
        n_check = flags.num("K31_DSA_ROW_SLICE_CHECK", 0)
        name = getattr(self, "layer_name", "?")
        if n_check > 0 and _DSA["checks"].get(name, 0) < n_check:
            _DSA["checks"][name] = _DSA["checks"].get(name, 0) + 1
            _dsa_check(orig, self, q, k_c_normed, k_pe, kv_cache, attn_metadata, kwargs, T, rows)
        return out

    forward_impl._k31_orig = orig
    cls.forward_impl = forward_impl
    cls._k31_rowslice = True
    return True


# ------------------------------------------------------------------------------------- MegaMoE: rows per call, buffer
def moe_config() -> tuple:
    """(rows, check) from K31_MEGAMOE_ROWS / K31_MEGAMOE_ROWS_CHECK, read once at start-up (the buffer's size)."""
    if not _MOE.get("read"):
        _MOE["read"] = True
        rows = int(os.environ.get("K31_MEGAMOE_ROWS", "0") or 0)
        _MOE["rows"] = max(rows, MIN_SLICE) if rows > 0 else 0
        _MOE["check"] = int(os.environ.get("K31_MEGAMOE_ROWS_CHECK", "0") or 0) if rows > 0 else 0
    return _MOE["rows"], _MOE["check"]


def megamoe_kw(vllm_config, kw: dict) -> dict:
    """DeepGemmMegaMoEExperts' constructor arguments with max_num_tokens capped at K31_MEGAMOE_ROWS (unless in check
    mode, which keeps the full buffer so that the unsliced call still fits)."""
    rows, check = moe_config()
    if rows <= 0 or check > 0:
        return kw
    full = kw.get("max_num_tokens")
    if full is None:
        full = getattr(getattr(vllm_config, "scheduler_config", None), "max_num_batched_tokens", None)
    if full is None or int(full) <= rows:
        return kw
    kw = dict(kw)
    kw["max_num_tokens"] = rows
    if not _MOE["logged"]:
        _MOE["logged"] = True
        _LOG(f"MegaMoE built for {rows} tokens per call (K31_MEGAMOE_ROWS) instead of {full}: calls of more rows run "
             f"in row slices")
    return kw


def megamoe_forward(orig):
    """DeepGemmMegaMoEExperts.forward with calls of more than K31_MEGAMOE_ROWS rows in balanced row slices; `orig`
    itself when the flag is off."""
    rows, check = moe_config()
    if rows <= 0:
        return orig

    def forward(mod, hidden_states, topk_weights, topk_ids, shared_experts=None):
        T = hidden_states.shape[0]
        if T <= rows or (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()):
            return orig(mod, hidden_states, topk_weights, topk_ids, shared_experts)
        y = moe_slices(orig, mod, hidden_states, topk_weights, topk_ids, shared_experts, rows)
        key = id(mod)
        if check > 0 and _MOE["checks"].get(key, 0) < check:
            _MOE["checks"][key] = _MOE["checks"].get(key, 0) + 1
            full = orig(mod, hidden_states, topk_weights, topk_ids, shared_experts)
            full2 = orig(mod, hidden_states, topk_weights, topk_ids, shared_experts)
            same, det = bool(torch.equal(y, full)), bool(torch.equal(full, full2))
            diff = 0.0 if same else float((y.float() - full.float()).abs().max())
            verdict = "bit-equal to the unsliced call" if same else \
                f"DIFFERS from the unsliced call (max abs {diff:.3e})"
            _LOG(f"MegaMoE row slice check {getattr(mod, 'prefix', '?')}: {T} rows in {len(plan(T, rows))} slices: "
                 f"{verdict}; two unsliced runs {'agree' if det else 'DIFFER (the kernel is not deterministic)'}")
        return y

    forward._k31_orig = orig
    return forward


def moe_slices(orig, mod, x, w, ids, shared_experts, rows: int):
    """One MegaMoE call over [T] rows as balanced slices of at most `rows`; each slice's output (a fresh tensor) is
    copied into one [T, hidden] result, so only one slice's output is alive beside it. The forward context's padding
    mask follows the slice (prompt_path._PaddingSlice)."""
    from .prompt_path import _PaddingSlice
    T = x.shape[0]
    y = None
    for a, b in plan(T, rows):
        with _PaddingSlice(a, b):
            o = orig(mod, x[a:b], w[a:b], ids[a:b].contiguous(), shared_experts)
        if y is None:
            y = torch.empty((T,) + tuple(o.shape[1:]), dtype=o.dtype, device=o.device)
        y[a:b].copy_(o)
        del o
    return y
