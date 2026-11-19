"""k31_engine.rowslice and mempeak on CPU: the slice plan, the DSA forward_impl wrapper, the MegaMoE row slices and
the memory-peak bookkeeping, with fakes that read their inputs the way vLLM's code does (forward_mqa:
topk_indices_buffer[:rows] and req_id_per_token[:rows] of the metadata it is given; MegaMoE: the forward context's
padding mask). The kernels' own row independence is the GPU test's part (tests/test_dsa_slice_gpu.py).

Run in the engine image, CPU only, one core (from engine/):
    docker run --rm --network none --cpuset-cpus 56 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_rowslice.py
Exit 0 and "ROWSLICE: PASS" = every check passed:
  1. plan(T, rows): contiguous cover of [0, T), at most `rows` and at least rows // 2 rows per slice, ceil(T / rows)
     slices; one slice when T <= rows.
  2. DSA wrapper: sliced output == unsliced output (exactly) for T across slice and request boundaries, mixed batches
     (decode rows first) and several slice sizes; each slice's call sees num_actual_tokens = its rows,
     num_decode_tokens 0, its own req_id rows and the top-k buffer from its first row; the impl's buffer fields are
     restored (also after an exception); every ineligible call reaches vLLM's forward_impl once with the original
     arguments; K31_DSA_ROW_SLICE read per call through flags (control file); the check mode's line.
  3. MegaMoE: megamoe_kw caps max_num_tokens (not in check mode); megamoe_forward slices calls above the cap with the
     padding mask sliced along, output == unsliced; unchanged when off.
  4. mempeak (K31_MEM_PEAK) on a fake allocator: hooks on every decoder layer's attention and MLP, one line for a
     step at or above the threshold only, the step's peak and the largest module peak per kind with its layer.
"""
import importlib
import io
import json
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
FLAGS_FILE = os.path.join(tempfile.mkdtemp(), "k31-flags.json")
os.environ["K31_FLAGS_FILE"] = FLAGS_FILE
for k in ("K31_DSA_ROW_SLICE", "K31_DSA_ROW_SLICE_CHECK", "K31_MEGAMOE_ROWS", "K31_MEGAMOE_ROWS_CHECK"):
    os.environ.pop(k, None)

# a stand-in for vllm.forward_context (prompt_path._PaddingSlice imports it at call time)
_ctx = types.SimpleNamespace(is_padding=None)
_fc = types.ModuleType("vllm.forward_context")
_fc.is_forward_context_available = lambda: True
_fc.get_forward_context = lambda: _ctx
sys.modules.setdefault("vllm", types.ModuleType("vllm"))
sys.modules["vllm.forward_context"] = _fc

from k31_engine import flags, rowslice  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def set_flags(d):
    with open(FLAGS_FILE + ".tmp", "w") as f:
        json.dump(d, f)
    os.replace(FLAGS_FILE + ".tmp", FLAGS_FILE)
    st = os.stat(FLAGS_FILE)
    os.utime(FLAGS_FILE, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a new mtime even within one tick
    flags.refresh()


# ---------------------------------------------------------------------------------------------------------- 1. plan
ok = True
for T in (1, 255, 256, 257, 1000, 2048, 2049, 4097, 8191, 8192, 8193, 16384, 30000):
    for rows in (256, 1000, 1024, 2048, 4096):
        p = rowslice.plan(T, rows)
        sizes = [b - a for a, b in p]
        cover = p[0][0] == 0 and p[-1][1] == T and all(p[i][1] == p[i + 1][0] for i in range(len(p) - 1))
        if T <= rows:
            ok &= p == [(0, T)]
        else:
            ok &= cover and len(p) == -(-T // rows) and max(sizes) <= rows and min(sizes) >= rows // 2 and \
                max(sizes) - min(sizes) <= 1
check("plan: balanced cover, sizes in [rows // 2, rows], ceil(T/rows) slices", ok)
check("plan: rows <= 0 is one slice", rowslice.plan(5000, 0) == [(0, 5000)])


# ------------------------------------------------------------------------------------------------ 2. the DSA wrapper
class Indexer:
    def __init__(self, buf):
        self.topk_indices_buffer = buf


class Impl:
    """SharedTopkIndicesBuffer's resolution (sparse_mla_attention.py) and an index group with a decode workspace."""
    is_sparse = True
    dcp_world_size = 1

    def __init__(self, buf):
        self._indexer = Indexer(buf)
        self._topk_indices_buffer = None
        self.index_group = types.SimpleNamespace(physical_topk_indices=torch.empty(21, 8, dtype=torch.int32))

    @property
    def topk_indices_buffer(self):
        if self._indexer is not None:
            return self._indexer.topk_indices_buffer
        return self._topk_indices_buffer


class Md:
    """The fields of SparseMLACommonMetadata that forward_impl / forward_mqa read on the all-MQA path."""

    def __init__(self, T, nd, rid, block_table):
        self.num_actual_tokens, self.num_decode_tokens, self.num_decodes, self.num_prefills = T, nd, 1, 1
        self.req_id_per_token, self.block_table, self.block_size, self.shardq = rid, block_table, 64, None
        self.prefill = object()


CALLS = []


class MLAAttention:
    """vLLM's MLAAttention reduced to what the all-MQA path does with its inputs, per row: q's row, its top-k row from
    impl.topk_indices_buffer[:n] and its block-table row through req_id_per_token[:n]."""
    use_pcp = use_shardq = False
    hisparse_cache = None

    def __init__(self, buf, use_mha=False, fail_at=None):
        self.impl = Impl(buf)
        self.layer_name = "model.layers.3.self_attn.attn"
        self.use_mha, self.fail_at = use_mha, fail_at

    def _use_sparse_mha(self, md):
        return self.use_mha

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output=None, output_scale=None,
                     output_block_scale=None, q_dcp_replicated=None, **kw):
        md = attn_metadata
        n = md.num_actual_tokens
        topk = self.impl.topk_indices_buffer[:n]
        rid = md.req_id_per_token[:n]
        CALLS.append((n, md.num_decode_tokens, int(topk[0, 0]), int(rid[0]), q.data_ptr(), output.data_ptr()))
        if self.fail_at is not None and len(CALLS) == self.fail_at:
            raise RuntimeError("injected")
        bt = md.block_table[rid.long()]                                  # [n, blocks]
        phys = bt.gather(1, (topk.clamp_min(0) // 64).long()) * 64 + topk.clamp_min(0) % 64
        phys = torch.where(topk >= 0, phys, -1)
        val = kv_cache[phys.clamp_min(0)].float() * (topk >= 0).unsqueeze(-1)   # [n, K, D]
        att = (q[:n].float().unsqueeze(1) * val).sum(1)                    # [n, D]
        output[:n] = (att + k_c_normed[:n].float().sum(-1, keepdim=True)).to(output.dtype)
        return output


fake_mod = types.SimpleNamespace(MLAAttention=MLAAttention)
check("install_dsa wraps forward_impl once", rowslice.install_dsa(fake_mod) and not rowslice.install_dsa(fake_mod))
wrapped = MLAAttention.forward_impl
orig = wrapped._k31_orig

g = torch.Generator().manual_seed(7)
K, D, NB = 8, 16, 512


def case(T, nd, reqs):
    """T rows of `reqs` requests (decode rows first, nd of them), a top-k buffer longer than T, a block table."""
    buf = torch.randint(-1, 64 * 40, (T + 37, K), generator=g, dtype=torch.int32)
    rid = torch.sort(torch.randint(0, reqs, (T,), generator=g)).values.to(torch.int32)
    rid_full = torch.cat([rid, torch.full((50,), 99, dtype=torch.int32)])      # the builder's buffer is longer
    bt = torch.randperm(NB, generator=g)[:reqs * 40].view(reqs, 40).to(torch.int32)
    bt = torch.cat([bt, torch.zeros(100 - reqs, 40, dtype=torch.int32)])
    kv = torch.randn(NB * 64, D, generator=g).to(torch.bfloat16)
    q = torch.randn(T + 5, D, generator=g).to(torch.bfloat16)                 # padded rows past T
    kc = torch.randn(T + 5, 4, generator=g).to(torch.bfloat16)
    kp = torch.randn(T + 5, 1, 2, generator=g).to(torch.bfloat16)
    return buf, Md(T, nd, rid_full, bt), q, kc, kp, kv


def run(layer, T, args, sliced_rows):
    buf, md, q, kc, kp, kv = args
    out = torch.full((T + 5, D), 7.0, dtype=torch.bfloat16)
    set_flags({"K31_DSA_ROW_SLICE": sliced_rows})
    CALLS.clear()
    wrapped(layer, q, kc, kp, kv, md, output=out)
    return out, list(CALLS)


for T, nd, reqs, rows in ((3000, 0, 1, 1024), (8192, 0, 2, 2048), (8192, 5, 3, 1024), (5000, 20, 4, 300),
                          (4097, 0, 1, 4096), (1500, 3, 2, 256)):
    args = case(T, nd, reqs)
    layer = MLAAttention(args[0])
    ref, c0 = run(layer, T, args, 0)
    out, c1 = run(layer, T, args, rows)
    sl = rowslice.plan(T, rowslice._dsa_rows(layer.impl, rows))
    want = [(b - a, 0, int(args[0][a, 0]), int(args[1].req_id_per_token[a])) for a, b in sl]
    check(f"DSA T={T} decode={nd} reqs={reqs} rows={rows}: sliced == unsliced",
          len(c0) == 1 and torch.equal(out, ref) and bool((out[T:] == 7.0).all()),
          f"({len(c1)} slices)")
    check(f"DSA T={T} rows={rows}: each slice's metadata, top-k rows and request rows",
          [c[:4] for c in c1] == want and c0[0][:2] == (T, nd))
    check(f"DSA T={T} rows={rows}: impl buffer fields restored",
          layer.impl._indexer is not None and layer.impl._topk_indices_buffer is None and
          layer.impl.topk_indices_buffer is args[0])

# ineligible calls reach vLLM's forward_impl once, with the very same arguments
args = case(3000, 0, 2)
for label, mk, kwx in (("flag off", lambda: MLAAttention(args[0]), None),
                       ("prefill on MHA", lambda: MLAAttention(args[0], use_mha=True), None),
                       ("DCP", lambda: MLAAttention(args[0]), "dcp"),
                       ("quantized output", lambda: MLAAttention(args[0]), "scale"),
                       ("T <= rows", lambda: MLAAttention(args[0]), "small")):
    layer = mk()
    kw = {}
    if kwx == "dcp":
        layer.impl.dcp_world_size = 2
    if kwx == "scale":
        kw["output_scale"] = torch.ones(1)
    rows = 0 if label == "flag off" else (4096 if kwx == "small" else 1024)
    set_flags({"K31_DSA_ROW_SLICE": rows})
    out = torch.zeros(3005, D, dtype=torch.bfloat16)
    CALLS.clear()
    wrapped(layer, args[2], args[3], args[4], args[5], args[1], output=out, **kw)
    check(f"DSA ineligible ({label}): one unsliced call", len(CALLS) == 1 and CALLS[0][:2] == (3000, 0) and
          CALLS[0][4] == args[2].data_ptr() and CALLS[0][5] == out.data_ptr())

# the profile run (no metadata) and decode-size calls never look at the flag
set_flags({"K31_DSA_ROW_SLICE": 64})
CALLS.clear()
layer = MLAAttention(args[0])
small = Md(40, 0, args[1].req_id_per_token, args[1].block_table)
wrapped(layer, args[2], args[3], args[4], args[5], small, output=torch.zeros(3005, D, dtype=torch.bfloat16))
check("DSA decode-size call (40 rows) unsliced even with a small flag", len(CALLS) == 1 and CALLS[0][0] == 40)

# an exception inside a slice still restores the buffer fields
layer = MLAAttention(args[0], fail_at=2)
set_flags({"K31_DSA_ROW_SLICE": 1024})
try:
    wrapped(layer, args[2], args[3], args[4], args[5], args[1], output=torch.zeros(3005, D, dtype=torch.bfloat16))
    raised = False
except RuntimeError:
    raised = True
check("DSA exception in a slice: raised, buffer fields restored",
      raised and layer.impl._indexer is not None and layer.impl._topk_indices_buffer is None)

# the environment's value when the control file has no key; the control file's when it has one
os.environ["K31_DSA_ROW_SLICE"] = "1024"
set_flags({})
layer = MLAAttention(args[0])
CALLS.clear()
wrapped(layer, args[2], args[3], args[4], args[5], args[1], output=torch.zeros(3005, D, dtype=torch.bfloat16))
n_env = len(CALLS)
set_flags({"K31_DSA_ROW_SLICE": "0"})
CALLS.clear()
wrapped(layer, args[2], args[3], args[4], args[5], args[1], output=torch.zeros(3005, D, dtype=torch.bfloat16))
check("DSA flag: environment 1024 -> 3 slices, control file 0 -> 1 call", n_env == 3 and len(CALLS) == 1,
      f"({n_env}, {len(CALLS)})")
os.environ.pop("K31_DSA_ROW_SLICE")

# check mode: one line, bit-equal on a deterministic fake
buf_err = io.StringIO()
keep, sys.stderr = sys.stderr, buf_err
try:
    set_flags({"K31_DSA_ROW_SLICE": 1024, "K31_DSA_ROW_SLICE_CHECK": 1})
    layer = MLAAttention(args[0])
    rowslice._DSA["checks"].clear()
    for _ in range(2):
        wrapped(layer, args[2], args[3], args[4], args[5], args[1], output=torch.zeros(3005, D, dtype=torch.bfloat16))
finally:
    sys.stderr = keep
lines = [ln for ln in buf_err.getvalue().splitlines() if "row slice check" in ln]
check("DSA check mode: one line per layer up to n, bit-equal, deterministic",
      len(lines) == 1 and "bit-equal" in lines[0] and "agree" in lines[0], lines[0] if lines else "(no line)")
set_flags({})

# ------------------------------------------------------------------------------------------------------- 3. MegaMoE
os.environ["K31_MEGAMOE_ROWS"] = "1000"
importlib.reload(rowslice)
kw = rowslice.megamoe_kw(types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8192)),
                         {"max_num_tokens": 8192, "top_k": 16})
kw2 = rowslice.megamoe_kw(types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8192)),
                          {"top_k": 16})
kw3 = rowslice.megamoe_kw(None, {"max_num_tokens": 512})
check("MegaMoE kw: max_num_tokens capped at K31_MEGAMOE_ROWS", kw["max_num_tokens"] == 1000 and
      kw2["max_num_tokens"] == 1000 and kw3["max_num_tokens"] == 512 and kw["top_k"] == 16)

SEEN = []


def moe_orig(mod, x, w, ids, shared_experts=None):
    """Row-wise stand-in for MegaMoE, with the padding mask's rows zeroed as VLLM_MOE_SKIP_PADDING does."""
    T = x.shape[0]
    p = _ctx.is_padding
    if p is not None:
        p = p[:T]
    SEEN.append((T, None if p is None else int(p.sum())))
    y = x.float() * w.float().sum(1, keepdim=True) + ids.float().sum(1, keepdim=True)
    if p is not None:
        y = torch.where(p.unsqueeze(1), 0.0, y)
    return y.to(torch.bfloat16)


fwd = rowslice.megamoe_forward(moe_orig)
x = torch.randn(4100, 64, generator=g).to(torch.bfloat16)
w = torch.rand(4100, 16, generator=g)
ids = torch.randint(-1, 500, (4100, 16), generator=g, dtype=torch.int32)
_ctx.is_padding = torch.rand(4100, generator=g) < 0.1
SEEN.clear()
ref = moe_orig(None, x, w, ids)
SEEN.clear()
y = fwd(None, x, w, ids)
check("MegaMoE: 4,100 rows in 5 slices of <= 1,000, == unsliced (padding mask sliced along)",
      torch.equal(y, ref) and [s[0] for s in SEEN] == [820] * 5 and
      sum(s[1] for s in SEEN) == int(_ctx.is_padding.sum()), f"{SEEN}")
check("MegaMoE: padding mask restored", _ctx.is_padding.shape[0] == 4100)
SEEN.clear()
fwd(None, x[:1000], w[:1000], ids[:1000])
check("MegaMoE: a call of <= rows is one call", [s[0] for s in SEEN] == [1000])
os.environ["K31_MEGAMOE_ROWS_CHECK"] = "1"
importlib.reload(rowslice)
kw4 = rowslice.megamoe_kw(None, {"max_num_tokens": 8192})
check("MegaMoE check mode keeps the full buffer", kw4["max_num_tokens"] == 8192)
os.environ.pop("K31_MEGAMOE_ROWS")
os.environ.pop("K31_MEGAMOE_ROWS_CHECK")
importlib.reload(rowslice)
check("MegaMoE off: forward and kw unchanged", rowslice.megamoe_forward(moe_orig) is moe_orig and
      rowslice.megamoe_kw(None, {"max_num_tokens": 8192})["max_num_tokens"] == 8192)
os.environ["K31_MEGAMOE_ROWS"] = "64"
importlib.reload(rowslice)
check("MegaMoE: rows below 256 raised to 256", rowslice.moe_config()[0] == 256)
os.environ.pop("K31_MEGAMOE_ROWS")

# ------------------------------------------------------------------------- 4. mempeak's bookkeeping (fake allocator)
A = {"alloc": 10 * 2 ** 30, "peak": 0, "res": 12 * 2 ** 30, "rpeak": 0}


def _use(n):
    A["alloc"] += n
    A["peak"] = max(A["peak"], A["alloc"])
    A["res"] = max(A["res"], A["alloc"] + 2 ** 30)
    A["rpeak"] = max(A["rpeak"], A["res"])


def _reset():
    A["peak"], A["rpeak"] = A["alloc"], A["res"]


fake_cuda = {"memory_allocated": lambda: A["alloc"], "max_memory_allocated": lambda: A["peak"],
             "memory_reserved": lambda: A["res"], "max_memory_reserved": lambda: A["rpeak"],
             "reset_peak_memory_stats": _reset, "mem_get_info": lambda: (20 * 2 ** 30, 250 * 2 ** 30)}
saved = {k: getattr(torch.cuda, k) for k in fake_cuda}
for k, v in fake_cuda.items():
    setattr(torch.cuda, k, v)


class KimiK3DSAAttention(torch.nn.Module):
    def forward(self, x):
        _use(3 * 2 ** 30)            # a 3 GiB transient
        _use(-3 * 2 ** 30)
        _use(2 ** 20)                # keeps 1 MiB (its output)
        return x


class KimiK3DeltaAttention(KimiK3DSAAttention):
    def forward(self, x):
        _use(2 * 2 ** 30)
        _use(-2 * 2 ** 30)
        return x


class KimiMoE(torch.nn.Module):
    def forward(self, x):
        _use(2 ** 29)
        _use(-2 ** 29)
        return x


class KimiDecoderLayer(torch.nn.Module):
    def __init__(self, i, dsa):
        super().__init__()
        self.layer_idx = i
        self.self_attn = KimiK3DSAAttention() if dsa else KimiK3DeltaAttention()
        self.mlp = KimiMoE()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


try:
    os.environ["K31_MEM_PEAK"] = "1024"
    from k31_engine import mempeak  # noqa: E402
    model = torch.nn.Sequential(*[KimiDecoderLayer(i, i % 4 == 3) for i in range(8)])
    n_hooks = mempeak.install(model)
    err = io.StringIO()
    keep, sys.stderr = sys.stderr, err
    try:
        mempeak.step_begin(40)                   # a decode step: no line
        model(torch.zeros(1))
        mempeak.step_end()
        mempeak.step_begin(8192)
        _use(2 ** 30)                            # 1 GiB outside the layers (kept)
        model(torch.zeros(1))
        mempeak.step_end()
    finally:
        sys.stderr = keep
    lines = [ln for ln in err.getvalue().splitlines() if "memory peak" in ln]
    ok = len(lines) == 1 and "step of 8192 tokens" in lines[0] and "DSA +3.00 (layer 3" in lines[0] and \
        "KDA +2.00 (layer 0" in lines[0] and "MoE +0.50" in lines[0] and "peak 14.00 (+4.00)" in lines[0]
    check("mempeak: hooks on 16 modules, one line for the 8,192-token step only, peaks by kind", n_hooks == 16 and ok,
          lines[0] if lines else "(no line)")
finally:
    for k, v in saved.items():
        setattr(torch.cuda, k, v)
    os.environ.pop("K31_MEM_PEAK", None)

print(f"ROWSLICE: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
