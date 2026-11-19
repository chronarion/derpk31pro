#!/usr/bin/env python3
"""derpk31pro layer-streaming runner for Kimi K3.1 Pro.

One pass over the official checkpoint, one decoder layer at a time, advances several "universes" (model variants)
over the same token windows; at the end each universe's logits are compared with universe 0's (the published model)
in float64 over the full vocabulary, and universe 0's routing (expert ids and weights, by rank) is saved for the
routing studies.

Built on tools/kstream.py of tomcupr/dgx-station-recipes (MIT; third_party/tomcupr): the same method (Moonshot's
Kimi K3 reference modules for everything but the routed experts; hidden states and Attention-Residual block states
of every universe kept on the host; DSA = dense MLA at <= 2,048 tokens, where its top-2,048 selection keeps every
token). Changes:
  - the experts stay MXFP4 codes on the GPU and are decoded per expert and treatment when used, so variants cost no
    memory; per-pick treatments (k31/policy.py) include a state-dependent hot-set policy;
  - routed-expert input activations optionally in MXFP8 (the official W4A8 arithmetic); non-expert matrices
    optionally in MXFP8 or FP6 (k31/formats.py);
  - full-vocabulary KLD per token against universe 0 (float64 log-softmax), top-1 agreement and next-token NLL;
  - tensors read as memmap views of the shards with a page-cache prefetch of the next layer by several readers
    (file-backed pages; safetensors get_tensor copies stayed resident in an earlier streaming run);
  - K3.1 details: KDA A_log is stored padded to 128 (96 heads + zeros) and is sliced; the DSA indexer tensors are
    skipped; the router bias, A_log and dt_bias stay float32 (as in vLLM), everything else bfloat16 (as kstream).

usage: python3 -m k31.stream MODEL_DIR TOKENS.pt OUT_DIR --ref REF_SRC [--universes U.json] [--layers N]
"""
import argparse
import concurrent.futures as cf
import copy
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from k31.formats import MAG, decode_stack, mx_fake_quant, plane_tables
from k31.policy import CUT, DEFER, NATIVE, NTREAT, PDEFER, PLANES, WDEFER, as_dicts, load_universes, treatments

PREFIX = "language_model.model."
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}
KEEP_F32 = ("e_score_correction_bias", "A_log", "dt_bias")


def load_reference(ref_src, patch=None):
    """Import Moonshot's K3 reference modeling code from a copy (it uses relative imports) under transformers 5.x."""
    import transformers.utils.generic as tf_generic
    import transformers.utils.output_capturing as tf_capture
    if not hasattr(tf_generic, "OutputRecorder"):
        tf_generic.OutputRecorder = tf_capture.OutputRecorder
    root = tempfile.mkdtemp(prefix="k3ref-")
    shutil.copytree(ref_src, os.path.join(root, "k3ref"))
    sys.path.insert(0, root)
    ref = importlib.import_module("k3ref.modeling_kimi_linear")
    cfg = importlib.import_module("k3ref.configuration_kimi_k3")
    import inspect
    from transformers import masking_utils
    params = inspect.signature(masking_utils.create_causal_mask).parameters
    if "input_embeds" not in params and not getattr(ref.create_causal_mask, "_k31_shim", False):
        orig = masking_utils.create_causal_mask        # 5.x: inputs_embeds, no cache_position (full forward only)

        def create_causal_mask(*args, input_embeds=None, cache_position=None, **kwargs):
            if input_embeds is not None:
                kwargs["inputs_embeds"] = input_embeds
            if "allow_is_causal_skip" in params:
                kwargs["allow_is_causal_skip"] = False    # keep an explicit mask for eager attention
            return orig(*args, **kwargs)
        create_causal_mask._k31_shim = True
        ref.create_causal_mask = create_causal_mask
    if not getattr(ref._apply_attn_res, "_k31_hook", False):
        orig_mix = ref._apply_attn_res
        ref._k31_inject = None

        def _apply_attn_res(prefix_sum, block_residual, proj, norm):
            # half-layer landing: late shares join the layer's running sum (or a stored block) right before the
            # mixing that builds its MoE input, so the MoE and everything after see them; the attention did not
            inj = ref._k31_inject
            if inj is not None and proj is inj["proj"]:
                ref._k31_inject = None
                if inj["prefix"] is not None:
                    prefix_sum.add_(inj["prefix"].reshape(prefix_sum.shape).to(prefix_sum.dtype))
                for j, d in inj["blocks"]:
                    block_residual[:, j].add_(d.reshape(-1, d.shape[-1]).to(block_residual.dtype))
            return orig_mix(prefix_sum, block_residual, proj, norm)
        _apply_attn_res._k31_hook = True
        ref._apply_attn_res = _apply_attn_res
    if patch is not None:
        patch(ref)
    return ref, cfg


class Shards:
    """Tensor access over the safetensors shards: headers parsed once, tensors read into host memory in parallel."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.headers, self.lock = {}, threading.Lock()
        self.by_layer = {}
        for k in self.map:
            m = re.match(re.escape(PREFIX) + r"layers\.(\d+)\.", k)
            if m:
                self.by_layer.setdefault(int(m.group(1)), []).append(k)

    def wait_for(self, fn, timeout=6 * 3600):
        """Block until shard `fn` exists: a copy that fills the model directory while a pass runs renames each file
        into place only when it is complete (rsync), so the runner can follow right behind it."""
        path, t0, said = os.path.join(self.dir, fn), time.time(), False
        while not os.path.exists(path):
            if not said:
                print(f"waiting for {fn}", flush=True)
                said = True
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{path} did not appear in {timeout} s")
            time.sleep(10)

    def info(self, name):
        fn = self.map[name]
        self.wait_for(fn)
        with self.lock:
            if fn not in self.headers:
                with open(os.path.join(self.dir, fn), "rb") as fh:
                    n = int.from_bytes(fh.read(8), "little")
                    self.headers[fn] = (8 + n, json.loads(fh.read(n)))
        base, h = self.headers[fn]
        e = h[name]
        return fn, base + e["data_offsets"][0], base + e["data_offsets"][1], e["dtype"], e["shape"]

    def read(self, names, workers=1, piece=64 << 20, gap=16 << 20):
        """{name: CPU tensor}. The tensors' byte ranges are merged per shard (holes up to `gap` read through) and each
        merged range is read front to back in `piece`-sized preads into one buffer; tensors are views of it (cloned
        when their offset is not aligned to their element size). `workers` shards are read at once. Measured on
        a network share (SMB, ~100 MB/s): one sequential stream reaches the link rate, while thousands of per-tensor reads
        interleaved by 4 threads reached ~65 MB/s; memmap page faults fetch one page per round trip."""
        infos = {n: self.info(n) for n in names}
        by_file = {}
        for n, (fn, a, b, dt, shape) in infos.items():
            by_file.setdefault(fn, []).append((a, b, n))

        def one_file(fn):
            items = sorted(by_file[fn])
            ranges = []                                          # [start, end, [(a, b, name), ...]]
            for a, b, n in items:
                if ranges and a <= ranges[-1][1] + gap:
                    ranges[-1][1] = max(ranges[-1][1], b)
                    ranges[-1][2].append((a, b, n))
                else:
                    ranges.append([a, b, [(a, b, n)]])
            out = {}
            fd = os.open(os.path.join(self.dir, fn), os.O_RDONLY)
            try:
                for start, end, members in ranges:
                    buf = torch.empty(end - start, dtype=torch.uint8)
                    mv = memoryview(buf.numpy())
                    off = 0
                    while off < end - start:
                        k = os.preadv(fd, [mv[off:off + piece]], start + off)
                        if k <= 0:
                            raise IOError(f"short read: {fn} at {start + off}")
                        off += k
                    for a, b, n in members:
                        _, _, _, dt, shape = infos[n]
                        t = buf[a - start:b - start]
                        size = torch.empty((), dtype=DT[dt]).element_size()
                        if (a - start) % size:
                            t = t.clone()
                        out[n] = t.view(DT[dt]).view(shape)
            finally:
                os.close(fd)
            return out

        result = {}
        with cf.ThreadPoolExecutor(max(1, workers)) as ex:
            for part in ex.map(one_file, list(by_file)):
                result.update(part)
        return result


class StreamMoE(torch.nn.Module):
    """The reference KimiSparseMoeBlock with routed experts as MXFP4 codes set from outside, and per-pick treatments.
    Experts are decoded CHUNK at a time per weight version (one kernel per matrix and chunk on a GPU)."""

    CHUNK = 32

    def __init__(self, config, ref):
        super().__init__()
        self.top_k, self.E = config.num_experts_per_token, config.num_experts
        self.scaling, self.renorm_router = config.routed_scaling_factor, config.moe_renormalize
        assert config.moe_router_activation_func == "sigmoid" and getattr(config, "num_expert_group", 1) == 1
        self.gate = ref.KimiMoEGate(config)
        self.shared_experts = ref.KimiMLP(config=config,
                                          intermediate_size=config.moe_intermediate_size * config.num_shared_experts)
        d = config.routed_expert_hidden_size
        self.routed_expert_down_proj = torch.nn.Linear(config.hidden_size, d, bias=False)
        self.routed_expert_up_proj = torch.nn.Linear(d, config.hidden_size, bias=False)
        assert config.latent_moe_use_norm
        self.routed_expert_norm = ref.KimiRMSNorm(d, eps=config.rms_norm_eps)
        beta, linear_beta = ref._get_situ_activation_params(config)
        self.act = ref.SituAndMul(beta=beta, linear_beta=linear_beta)
        self.packed = None        # {"w1"/"w2"/"w3": (packed [E, out, in/2] uint8, scale [E, out, in/32] uint8)}
        self.universe = None
        self.rec = None           # list of (ids int16 [n, K], weights fp16 [n, K]) by rank, when recording
        self.stats = None         # int64 [NTREAT, K]: picks per treatment and rank
        self.deferred = None      # this call's late share (DEFER picks, WDEFER corrections), added by the runner later
        self.mean_acc = None      # (sum [E, D] fp32, count [E]): universe 0's native expert outputs, for --save-means
        self.mean_tab = None      # [E, D] mean expert outputs of this layer (cut_sub = "mean")

    def route(self, x):
        """ids, weights [N, K], sorted by routing weight (descending): the reference gate's arithmetic."""
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.gate.e_score_correction_bias.float().unsqueeze(0)
        idx = torch.topk(choice, k=self.top_k, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if self.top_k > 1 and self.renorm_router:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        w = w * self.scaling
        order = w.argsort(dim=-1, descending=True, stable=True)
        return idx.gather(1, order), w.gather(1, order)

    def tables(self, version):
        """{w1, w2, w3: [E, 8] magnitude tables} for a weight version, ("native",) or ("planes", depth, recon):
        computed once per layer and shared by every universe (they share self.packed)."""
        cache = self.packed.setdefault("tables", {})
        if version not in cache:
            if version[0] == "native":
                t = MAG.float().to(self.packed["w1"][0].device).expand(self.E, 8).contiguous()
                cache[version] = {m: t for m in ("w1", "w2", "w3")}
            else:
                cache[version] = {m: plane_tables(*self.packed[m], version[1], version[2]) for m in ("w1", "w2", "w3")}
        return cache[version]

    def forward(self, hidden_states):
        shape = hidden_states.shape
        B, T = shape[0], shape[1]
        x = hidden_states.reshape(-1, shape[-1])
        ids, w = self.route(x)
        u = self.universe
        tr = treatments(u, ids.view(B, T, -1), w.view(B, T, -1), self.E, getattr(self, "layer", None)).view(-1, self.top_k)
        if self.rec is not None:
            self.rec.append((ids.to(torch.int16).cpu(), w.to(torch.float16).cpu()))
        if self.stats is not None:
            for t in range(NTREAT):
                self.stats[t] += (tr == t).sum(0).cpu()
        alive = tr != CUT
        w_cut = torch.where(alive, 0.0, w) if (u.cut_sub == "mean" and self.mean_tab is not None) else None
        if u.renorm:
            total = w.sum(-1, keepdim=True)
            w = w * alive * (total / (w * alive).sum(-1, keepdim=True).clamp_min(1e-20))
        else:
            w = w * alive
        h = self.routed_expert_down_proj(x)
        hq = mx_fake_quant(h, u.act)
        key = ids.reshape(-1).long() * NTREAT + tr.reshape(-1).long()
        key = torch.where(alive.reshape(-1), key, torch.full_like(key, NTREAT * self.E))   # cut picks sort last
        order = key.argsort(stable=True)
        bounds = torch.searchsorted(key[order], torch.arange(NTREAT * self.E + 1, device=x.device)).tolist()
        tok = order // self.top_k
        wflat = w.reshape(-1)
        y = torch.zeros(h.shape, dtype=torch.float32, device=x.device)
        late = None
        versions = {"native": ("native",), "planes": ("planes", u.depth, u.recon)}
        for c in range(0, self.E, self.CHUNK):
            ce = min(self.E, c + self.CHUNK)
            groups = [(e, t) for e in range(c, ce) for t in (NATIVE, PLANES, DEFER, WDEFER, PDEFER)
                      if bounds[e * NTREAT + t] != bounds[e * NTREAT + t + 1]]
            if not groups:
                continue
            need = set()
            for _, t in groups:
                need |= {"planes"} if t in (PLANES, PDEFER) else {"native"} if t in (NATIVE, DEFER) else {"planes", "native"}
            dec = {}
            for v in need:
                tab = self.tables(versions[v])
                dec[v] = [decode_stack(self.packed[m][0][c:ce], self.packed[m][1][c:ce], tab[m][c:ce], h.dtype)
                          for m in ("w1", "w2", "w3")]

            def run_expert(v, e, x_in):
                w1, w2, w3 = (d[e - c] for d in dec[v])
                z = self.act(torch.cat((x_in @ w1.T, x_in @ w3.T), dim=-1))
                return mx_fake_quant(z, u.act) @ w2.T

            for e, t in groups:
                k = e * NTREAT + t
                a, b = bounds[k], bounds[k + 1]
                sel = tok[a:b]
                wt = wflat[order[a:b]].unsqueeze(-1)
                z_in = hq[sel]
                if t == WDEFER:                           # the base now, the correction later
                    base = run_expert("planes", e, z_in)
                    y.index_add_(0, sel, base.float() * wt)
                    late = torch.zeros_like(y) if late is None else late
                    late.index_add_(0, sel, (run_expert("native", e, z_in).float() - base.float()) * wt)
                    continue
                z = run_expert("planes" if t in (PLANES, PDEFER) else "native", e, z_in)
                if self.mean_acc is not None and t == NATIVE:
                    self.mean_acc[0][e] += z.float().sum(0)
                    self.mean_acc[1][e] += z.shape[0]
                if t in (DEFER, PDEFER):
                    late = torch.zeros_like(y) if late is None else late
                    late.index_add_(0, sel, z.float() * wt)
                else:
                    y.index_add_(0, sel, z.float() * wt)
            del dec
        if w_cut is not None:                    # cut picks: routing weight x the expert's mean output
            for k in range(self.top_k):
                y += w_cut[:, k:k + 1].float() * self.mean_tab[ids[:, k].long()].float()
        out = self.routed_expert_up_proj(self.routed_expert_norm(y.to(h.dtype)))
        self.deferred = None
        if late is not None:
            # the latent MoE normalizes the sum of its picks before the up-projection, so a deferred pick's share is
            # what the complete sum gives minus what the sum without it gave
            full = self.routed_expert_up_proj(self.routed_expert_norm((y + late).to(h.dtype)))
            self.deferred = (full - out).view(shape)
        return out.view(shape) + self.shared_experts(hidden_states)


def build_layer(config, ref, i):
    orig = ref.KimiSparseMoeBlock
    ref.KimiSparseMoeBlock = lambda c: StreamMoE(c, ref)
    try:
        with torch.device("meta"):                      # no random init: load_dense supplies every parameter
            layer = ref.KimiDecoderLayer(config, i)
    finally:
        ref.KimiSparseMoeBlock = orig
    assert (i >= config.first_k_dense_replace) == hasattr(layer, "block_sparse_moe")
    return layer.eval()


def set_param(module, name, t):
    *path, leaf = name.split(".")
    for p in path:
        module = getattr(module, p)
    module._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)


def load_dense(layer, tensors, i, device, dtype):
    """Every parameter of layer i except the routed experts, from the layer's tensors (A_log un-padded)."""
    pre = f"{PREFIX}layers.{i}."
    names = [n for n in tensors if ".block_sparse_moe.experts." not in n and ".self_attn.indexer." not in n]
    have = {n[len(pre):]: n for n in names}
    want = dict(layer.named_parameters())
    missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    if missing or extra:
        raise RuntimeError(f"layer {i}: missing {missing[:5]} unexpected {extra[:5]}")
    for k, p in want.items():
        t = tensors[have[k]]
        if k.endswith("A_log") and t.shape != p.shape:            # K3.1: [96 heads] padded to 128 with zeros
            n = p.shape[0]
            assert t.dim() == 1 and t.shape[0] > n and not t[n:].any(), f"layer {i} {k}: {tuple(t.shape)}"
            t = t[:n]
        if t.shape != p.shape:
            raise RuntimeError(f"layer {i}: {k} is {tuple(t.shape)}, expected {tuple(p.shape)}")
        keep32 = k.split(".")[-1] in KEEP_F32 and t.dtype == torch.float32
        set_param(layer, k, t.to(device=device, dtype=t.dtype if keep32 else dtype))
    left = [n for n, p in layer.named_parameters() if p.is_meta]
    assert not left, f"layer {i}: parameters not loaded: {left[:5]}"


def load_experts(tensors, i, E, device):
    pre = f"{PREFIX}layers.{i}.block_sparse_moe.experts."
    out = {}
    for name in ("w1", "w2", "w3"):
        packed = torch.stack([tensors.pop(f"{pre}{e}.{name}.weight_packed") for e in range(E)]).to(device)
        scale = torch.stack([tensors.pop(f"{pre}{e}.{name}.weight_scale") for e in range(E)]).to(device)
        out[name] = (packed, scale)
    return out


def _fp8_block_weight(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    """vLLM's fp8_per_block weights: E4M3 with one float32 scale (amax / 448) per 128 x 128 block, dequantized."""
    o, i = w.shape
    po, pi = -o % block, -i % block
    x = torch.nn.functional.pad(w.float(), (0, pi, 0, po))
    xb = x.view(x.shape[0] // block, block, x.shape[1] // block, block)
    s = (xb.abs().amax(dim=(1, 3), keepdim=True) / 448.0).clamp_min(1e-12)
    q = (xb / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    return q.view(x.shape)[:o, :i].to(w.dtype)


def _fp8_token_group(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """vLLM's dynamic activation quantization for fp8_per_block: E4M3 per token and group of 128 inputs, float32
    scale amax / 448, dequantized."""
    shape = x.shape
    xf = x.float().reshape(-1, shape[-1] // group, group)
    s = (xf.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return ((xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).reshape(shape).to(x.dtype)


class _FP8BlockLinear(torch.nn.Module):
    """A Linear whose input is quantized per token-group of 128 (its weight already block-quantized)."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return torch.nn.functional.linear(_fp8_token_group(x), self.lin.weight, self.lin.bias)


def _int8_group_weight(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    """INT8 weights, symmetric, one float scale (amax / 127) per output row and group of 128 inputs, dequantized."""
    o, i = w.shape
    x = w.float().view(o, i // group, group)
    s = (x.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
    return ((x / s).round().clamp(-127, 127) * s).view(o, i).to(w.dtype)


class _F32Linear(torch.nn.Module):
    """The same Linear computed in float32 and rounded to the input dtype once: a different, equally valid summation
    order for every output (the implementation noise floor)."""

    def __init__(self, lin, noise=False):
        super().__init__()
        self.w = lin.weight.data.float()
        self.b = None if lin.bias is None else lin.bias.data.float()
        self.noise = noise

    def forward(self, x):
        y = torch.nn.functional.linear(x.float(), self.w, self.b)
        if self.noise:          # stochastic rounding to bf16: every output moves by up to one bf16 ulp, unbiased
            bits = y.view(torch.int32)
            y = (bits + torch.randint_like(bits, 0, 1 << 16)).bitwise_and(-65536).view(torch.float32)
        return y.to(x.dtype)


def dense_copy(layer, fmt, keep=""):
    """A copy of a layer (before its experts attach) with every non-expert matrix in `fmt` (MX blocks of 32 along the
    input; "fp8b128" = vLLM's fp8_per_block: 128 x 128 weight blocks and per-token 128-groups of the inputs, float
    scales), except the router and the AttnRes mixing projections."""
    if fmt == "map":
        return _dense_map_copy(layer, keep)
    if fmt == "nvfp4":
        return _dense_map_copy(layer, ",".join(f"{k}:bf16" for k in keep.split(",") if k) + ",*:nvfp4")
    q = copy.deepcopy(layer)
    if fmt in ("f32", "bf16noise", "fp8b128w", "int8g128"):
        # every Linear outside the routed experts (shared experts included), except the router and AttnRes mixing
        swaps = []
        for n, m in q.named_modules():
            if not isinstance(m, torch.nn.Linear) or n.endswith(("gate", "res_proj")) or "experts" in n.split(".") \
                    or _kept(n, keep):
                continue
            if fmt in ("f32", "bf16noise"):
                swaps.append(n)
            elif m.in_features % 128 == 0:
                m.weight.data = (_fp8_block_weight if fmt == "fp8b128w" else _int8_group_weight)(m.weight.data)
        for n in swaps:
            parent = q.get_submodule(n.rsplit(".", 1)[0]) if "." in n else q
            setattr(parent, n.rsplit(".", 1)[-1], _F32Linear(q.get_submodule(n), noise=fmt == "bf16noise"))
        return q
    if fmt == "fp8b128":
        swaps = []
        for n, m in q.named_modules():
            if isinstance(m, torch.nn.Linear) and m.in_features % 128 == 0 and \
                    not n.endswith(("gate", "res_proj")) and "experts" not in n and not _kept(n, keep):
                m.weight.data = _fp8_block_weight(m.weight.data)
                swaps.append(n)
        for n in swaps:
            parent = q.get_submodule(n.rsplit(".", 1)[0]) if "." in n else q
            setattr(parent, n.rsplit(".", 1)[-1], _FP8BlockLinear(q.get_submodule(n)))
        return q
    for n, p in q.named_parameters():
        if p.ndim == 2 and p.shape[-1] % 32 == 0 and not n.endswith(("gate.weight", "res_proj.weight")) and \
                not _kept(n, keep):
            p.data = mx_fake_quant(p.data, fmt)
    return q


def _dense_map_copy(layer, spec):
    """dense "map": per Linear (outside the routed experts, the router and the AttnRes mixing), the format of the first
    `substring:fmt` pair of `spec` whose substring is in its name ("*" = every other one); fmt in bf16 | fp8b128 (W and
    A, vLLM's fp8_per_block) | fp8b128w | mxfp4 | fp6 | mxfp8 | nvfp4 (weights only, BF16 inputs)."""
    from .formats import mx_fake_quant, nvfp4_fake_quant
    pairs = [p.rsplit(":", 1) for p in spec.split(",") if p]
    q = copy.deepcopy(layer)
    swaps = []
    for n, m in q.named_modules():
        if not isinstance(m, torch.nn.Linear) or n.endswith(("gate", "res_proj")) or "experts" in n.split("."):
            continue
        fmt = next((f for pat, f in pairs if pat != "*" and pat in n), None) or \
            next((f for pat, f in pairs if pat == "*"), "bf16")
        k = m.in_features
        if fmt == "bf16":
            continue
        if fmt in ("fp8b128", "fp8b128w") and k % 128 == 0:
            m.weight.data = _fp8_block_weight(m.weight.data)
            if fmt == "fp8b128":
                swaps.append(n)
        elif fmt in ("mxfp4", "fp6", "mxfp8") and k % 32 == 0:
            m.weight.data = mx_fake_quant(m.weight.data, fmt)
        elif fmt == "nvfp4" and k % 16 == 0:
            m.weight.data = nvfp4_fake_quant(m.weight.data)
    for n in swaps:
        parent = q.get_submodule(n.rsplit(".", 1)[0]) if "." in n else q
        setattr(parent, n.rsplit(".", 1)[-1], _FP8BlockLinear(q.get_submodule(n)))
    return q


def _kept(name, keep):
    return bool(keep) and any(k and k in name for k in keep.split(","))


class Store:
    """Host buffers for the universes' hidden and block states: plain RAM tensors, or (with a directory) memmaps of
    files on local disk, unlinked as soon as they are mapped. Their pages are file-backed, so the kernel writes them
    back and reclaims them under memory pressure (many universes fit beside other work), and a buffer's disk space
    is freed when the runner drops it."""

    NP = {2: np.uint16, 4: np.uint32}

    def __init__(self, directory=None):
        self.dir, self.n = directory, 0
        if directory:
            os.makedirs(directory, exist_ok=True)

    def new(self, shape, dtype):
        if not self.dir:
            return torch.empty(shape, dtype=dtype)
        self.n += 1
        path = os.path.join(self.dir, f"buf{os.getpid()}-{self.n}.bin")
        m = np.memmap(path, dtype=self.NP[torch.empty((), dtype=dtype).element_size()], mode="w+", shape=tuple(shape))
        os.unlink(path)
        return torch.from_numpy(m).view(dtype)

    def copy(self, t):
        out = self.new(t.shape, t.dtype)
        out.copy_(t)
        return out


@torch.no_grad()
def pregate(ref, lay, x, br, k):
    """Predict this MoE layer's expert picks from its input alone (what the previous layer hands over): the layer's
    own MoE-input mixing and norm, but without this layer's attention output, which a one-layer-ahead prefetch
    cannot know. Returns the top-k expert ids by the router's choice score, [B*T, k]."""
    B, T, H = x.shape
    xin = x.reshape(B * T, H)
    mixed = ref._apply_attn_res(xin.clone(), br, lay.mlp_res_proj, lay.mlp_res_norm) if br.shape[1] > 0 else xin
    r = lay.post_attention_layernorm(mixed.view(B, T, H)).reshape(B * T, H)
    g = lay.block_sparse_moe.gate
    score = F.linear(r.float(), g.weight.float()).sigmoid() + g.e_score_correction_bias.float().unsqueeze(0)
    return score.topk(k, dim=-1).indices


def causal_mask(T, device, dtype):
    return torch.full((T, T), torch.finfo(dtype).min, device=device, dtype=dtype).triu(1)[None, None]


def memory_line():
    st = dict(l.split(":", 1) for l in open("/proc/self/status") if l.startswith(("RssAnon", "RssFile")))
    av = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    g = lambda k: int(st[k].split()[0]) / 2 ** 20  # noqa: E731
    return f"rss anon {g('RssAnon'):.1f} file {g('RssFile'):.1f} GiB, host avail {av / 2 ** 20:.0f} GiB"


@torch.no_grad()
def run(model_dir, tokens, out_dir, ref_src, universes, n_layers=None, batch=8, device="cuda", dtype=torch.bfloat16,
        record=True, readers=1, patch=None, log=print, state_dir=None, dump_logits=None, save_means=None, means=None,
        dump_hidden=None):
    """tokens: int64 [N, T]. Returns (final hidden [S, N, T, H] on the host or None, results dict)."""
    ref, cfgmod = load_reference(ref_src, patch)
    config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(model_dir, "config.json")))["text_config"])
    config._attn_implementation = "eager"
    shards = Shards(model_dir)
    L = n_layers or config.num_hidden_layers
    N, T = tokens.shape
    H, E, K = config.hidden_size, config.num_experts, config.num_experts_per_token
    assert T <= config.dsa_config["index_topk"], "sparse attention would no longer equal dense MLA"
    S = len(universes)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(as_dicts(universes), open(os.path.join(out_dir, "universes.json"), "w"), indent=1)
    for u in universes:
        log(u.describe())

    emb = shards.read([PREFIX + "embed_tokens.weight"], readers)[PREFIX + "embed_tokens.weight"]
    h0 = emb[tokens].to(dtype)
    del emb
    store = Store(state_dir)
    hidden = [store.copy(h0) for _ in range(S)]
    blocks = [[] for _ in range(S)]
    del h0
    route_ids = torch.zeros((L, N * T, K), dtype=torch.int16) if record else None
    pg_k = (16, 24, 32)                   # pre-gating: next-layer prediction, recall of the true picks by rank
    pg_hits = {k: torch.zeros((L, K), dtype=torch.int64) for k in pg_k} if record else None
    pg_rows = torch.zeros(L, dtype=torch.int64)
    route_w = torch.zeros((L, N * T, K), dtype=torch.float16) if record else None
    stats = torch.zeros((S, L, NTREAT, K), dtype=torch.int64)
    drift = torch.zeros((S, L))
    pending = [[] for _ in range(S)]      # per universe: late shares {"arrive": layer, "origin": layer, "t": [N,T,H]}
    bs = config.attn_res_block_size
    mask = causal_mask(T, device, dtype)
    pool = cf.ThreadPoolExecutor(1)
    pre = pool.submit(shards.read, shards.by_layer[0], readers)
    t00 = time.time()
    for i in range(L):
        t0 = time.time()
        tensors = pre.result()
        t_wait = time.time() - t0
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if i + 1 < L:
            pre = pool.submit(shards.read, shards.by_layer[i + 1], readers)
        layer = build_layer(config, ref, i)
        load_dense(layer, tensors, i, device, dtype)
        moe_layer = hasattr(layer, "block_sparse_moe")
        copies = {"bf16": layer}
        for fmt, keep in sorted({(u.dense, u.dense_keep) for u in universes if u.dense != "bf16"}):
            copies[(fmt, keep)] = dense_copy(layer, fmt, keep)          # before the experts attach
        packed = load_experts(tensors, i, E, device) if moe_layer else None
        del tensors
        t_load = time.time() - t0
        is_block = i % config.attn_res_block_size == 0
        mean_tab = None
        if save_means:
            os.makedirs(save_means, exist_ok=True)
        for s, u in enumerate(universes):
            lay = copies["bf16"] if u.dense == "bf16" else copies[(u.dense, u.dense_keep)]
            if moe_layer:
                moe = lay.block_sparse_moe
                moe.packed, moe.universe, moe.layer = packed, u, i
                moe.rec = [] if (record and s == 0) else None
                moe.stats = torch.zeros((NTREAT, K), dtype=torch.int64)
                D = config.routed_expert_hidden_size
                moe.mean_acc = ((torch.zeros((E, D), dtype=torch.float32, device=device),
                                 torch.zeros(E, dtype=torch.int64, device=device)) if (save_means and s == 0) else None)
                moe.mean_tab = None
                if means and u.cut_sub == "mean":
                    if mean_tab is None:
                        mean_tab = torch.load(os.path.join(means, f"means-L{i}.pt"), weights_only=False)["mean"].to(device)
                    moe.mean_tab = mean_tab
            new_h = store.new(hidden[s].shape, hidden[s].dtype)
            new_block = store.new(hidden[s].shape, hidden[s].dtype) if is_block else None
            new_carry = None
            arriving = [c for c in pending[s] if c["arrive"] == i]
            for a in range(0, N, batch):
                x = hidden[s][a:a + batch].to(device)
                B = x.shape[0]
                if blocks[s]:
                    br = torch.stack([b[a:a + batch].to(device).view(B * T, H) for b in blocks[s]], dim=1)
                else:
                    br = x.new_zeros(B * T, 0, H)
                pred = None
                if record and s == 0 and moe_layer:   # what a prefetch issued at this layer's start could know
                    pred = pregate(ref, lay, x, br, min(max(pg_k), E))
                early = [c for c in arriving if c["land"] == "mlp"]
                if early:                                 # half-layer landing: hand the shares to the mixing hook
                    inj = {"proj": lay.mlp_res_proj, "prefix": None, "blocks": []}
                    for c in early:
                        lb = c["t"][a:a + batch].to(device)
                        stored_at = bs * (c["origin"] // bs + 1)
                        if i < stored_at:
                            inj["prefix"] = lb if inj["prefix"] is None else inj["prefix"] + lb
                        else:
                            inj["blocks"].append((c["origin"] // bs + 1, lb))
                    ref._k31_inject = inj
                y, br2 = lay(x, attention_mask=None if lay.is_linear_attn else mask, block_residual=br)
                assert ref._k31_inject is None, "the half-layer landing hook did not fire"
                if pred is not None:
                    actual = moe.rec[-1][0].to(device).long()            # this batch's true picks, by rank
                    for k in pg_k:
                        hit = (actual.unsqueeze(-1) == pred[:, :k].unsqueeze(1)).any(-1)
                        pg_hits[k][i] += hit.sum(0).cpu()
                    pg_rows[i] += actual.shape[0]

                done = br2[:, -1].view(B, T, H) if is_block else None   # the block stored at this layer
                for c in arriving:                        # late shares land where their origin layer's output went
                    lb = c["t"][a:a + batch].to(device)
                    stored_at = bs * (c["origin"] // bs + 1)               # the layer that stores the origin's block
                    if c["land"] == "mlp":                # already added inside the layer (before its MoE); only a
                        if i > stored_at:                 # block stored earlier needs its host copy completed too
                            blocks[s][c["origin"] // bs + 1][a:a + batch] += lb.cpu()
                        continue
                    if i < stored_at:
                        y = y + lb                        # its block is still the running sum
                    elif i == stored_at:
                        done = done + lb                  # its block is being stored now
                    else:                                 # stored earlier: complete the stored state in place
                        blocks[s][c["origin"] // bs + 1][a:a + batch] += lb.cpu()
                if moe_layer and moe.deferred is not None:
                    if new_carry is None:
                        new_carry = store.new(hidden[s].shape, hidden[s].dtype)
                        new_carry.zero_()
                    new_carry[a:a + batch] = moe.deferred.cpu()
                    moe.deferred = None
                new_h[a:a + batch] = y.cpu()
                if is_block:
                    new_block[a:a + batch] = done.cpu()
            hidden[s] = new_h
            pending[s] = [c for c in pending[s] if c["arrive"] != i]
            if new_carry is not None:
                pending[s].append({"arrive": i + u.late, "origin": i, "t": new_carry, "land": u.land})
            if is_block:
                blocks[s].append(new_block)
            if moe_layer and moe.mean_acc is not None:
                sm, cn = moe.mean_acc
                torch.save({"mean": (sm / cn.clamp_min(1).unsqueeze(1)).to(torch.bfloat16).cpu(), "count": cn.cpu()},
                           os.path.join(save_means, f"means-L{i}.pt"))
                moe.mean_acc = None
            if moe_layer:
                stats[s, i] = moe.stats
                if moe.rec is not None:
                    route_ids[i] = torch.cat([r[0] for r in moe.rec])
                    route_w[i] = torch.cat([r[1] for r in moe.rec])
                moe.packed = moe.rec = None
            if s:
                drift[s, i] = ((hidden[s].float() - hidden[0].float()).norm() / hidden[0].float().norm()).item()
        del layer, copies, packed
        if str(device) != "cpu":
            torch.cuda.empty_cache()
        log(f"layer {i:2d} {'MoE' if moe_layer else 'dense'} {'KDA' if config.is_kda_layer(i) else 'MLA'} "
            f"read-wait {t_wait:5.1f}s load {t_load:5.1f}s total {time.time() - t0:6.1f}s "
            f"({nbytes / 1e9:.1f} GB read) drift {' '.join(f'{d:.4f}' for d in drift[1:, i].tolist())} | "
            f"{memory_line()}")
        torch.save(dict(layer=i, stats=stats, drift=drift, names=[u.name for u in universes]),
                   os.path.join(out_dir, "stats.pt"))
    pool.shutdown()
    if record:
        torch.save(dict(ids=route_ids, w=route_w, shape=(N, T)), os.path.join(out_dir, "route.pt"))
        rows = pg_rows.clamp_min(1).unsqueeze(1).double()
        rec = {k: pg_hits[k].double() / rows for k in pg_k}                # [L, K] recall per layer and rank
        torch.save(dict(recall=rec, rows=pg_rows, note="next-layer pre-gating from the layer input without its "
                        "attention output; recall of the true picks by rank for a predicted top-k"),
                   os.path.join(out_dir, "pregate.pt"))
        moe_l = pg_rows > 0
        for k in pg_k:
            r = rec[k][moe_l]
            log(f"pre-gating, predicted top-{k}: recall ranks 1-5 {r[:, :5].mean():.3f}, ranks 6-16 "
                f"{r[:, 5:].mean():.3f}, all {r.mean():.3f}")
    if L < config.num_hidden_layers:
        log(f"done: {L} layers (partial: no logits) in {time.time() - t00:.0f}s")
        return None, None
    for s in range(S):                    # late shares still on the way land before the output mixing
        for c in pending[s]:
            stored_at = bs * (c["origin"] // bs + 1)
            target = blocks[s][c["origin"] // bs + 1] if stored_at <= L - 1 else hidden[s]
            for n in range(N):
                target[n].add_(c["t"][n])
        pending[s] = []
    final = store.new((S, N, T, H), dtype)
    tail = shards.read([PREFIX + n for n in ("output_attn_res_norm.weight", "output_attn_res_proj.weight", "norm.weight")]
                       + ["language_model.lm_head.weight"], readers)
    res_norm = ref.KimiRMSNorm(H, eps=config.rms_norm_eps).to(device)
    res_norm.weight.data = tail[PREFIX + "output_attn_res_norm.weight"].to(device, dtype)
    proj = torch.nn.Linear(H, 1, bias=False).to(device)
    proj.weight.data = tail[PREFIX + "output_attn_res_proj.weight"].to(device, dtype)
    norm = ref.KimiRMSNorm(H, eps=config.rms_norm_eps).to(device)
    norm.weight.data = tail[PREFIX + "norm.weight"].to(device, dtype)
    for s in range(S):
        for a in range(0, N, batch):
            x = hidden[s][a:a + batch].to(device)
            B = x.shape[0]
            br = torch.stack([b[a:a + batch].to(device).view(B * T, H) for b in blocks[s]], dim=1)
            y = ref._apply_attn_res(x.view(B * T, H), br, proj, res_norm).view(B, T, H)
            final[s, a:a + batch] = norm(y).cpu()
    head = tail["language_model.lm_head.weight"].to(device, dtype)
    del tail
    if dump_hidden:                     # universe 0's final hidden states (after the output norm), bf16 bits [N, T, H]:
        hd = np.lib.format.open_memmap(dump_hidden, mode="w+", dtype=np.uint16, shape=(N, T, H))   # the engine's
        for n in range(N):                                                                      # exact-KL reference
            hd[n] = final[0, n].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        hd.flush()
        del hd
    res = score(final, tokens, head, device, dump=dump_logits)
    res["names"] = [u.name for u in universes]
    torch.save(res, os.path.join(out_dir, "kld.pt"))
    summary = summarize(res, universes)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    for line in summary["lines"]:
        log(line)
    log(f"done: {L} layers, {S} universes, {N * T} tokens in {time.time() - t00:.0f}s")
    return final, res


REF_K = 64


def _coarse_kl(lp0, lp, k=REF_K):
    """The engine harness's estimator: KL over the tokens in both top-k lists plus one rest bucket (a lower bound)."""
    v0, i0 = lp0.topk(k, dim=-1)
    v1, i1 = lp.topk(k, dim=-1)
    m = i0.unsqueeze(2) == i1.unsqueeze(1)
    has = m.any(2)
    e_at = torch.where(m, v1.unsqueeze(1), torch.tensor(float("-inf"), dtype=v1.dtype, device=v1.device)).amax(2)
    pr = torch.where(has, v0.exp(), 0.0)
    pe = torch.where(has, e_at.exp(), 0.0)
    rr = (1 - pr.sum(-1)).clamp_min(1e-300)
    re_ = (1 - pe.sum(-1)).clamp_min(1e-300)
    return ((pr * (torch.where(has, v0, 0.0) - torch.where(has, e_at, 0.0))).sum(-1)
            + rr * (rr.log() - re_.log())).clamp_min(0)


@torch.no_grad()
def score(final, tokens, head, device, dump=None):
    """Per token: KL(universe 0 || universe s) in float64 over the full vocabulary, top-1 agreement with universe 0,
    and each universe's NLL of the next token (perplexity sanity)."""
    S, N, T, _ = final.shape
    kl = torch.zeros((S, N, T), dtype=torch.float64)
    top1 = torch.zeros((S, N, T), dtype=torch.bool)
    nll = torch.zeros((S, N, T - 1), dtype=torch.float64)
    # universe 0's top REF_K log-probabilities per position: the reference an engine's own logits are scored against
    # later (engine/harness); KL(ref || engine) over these tokens plus one bucket for the rest of the mass
    ref_lp = torch.zeros((N, T, REF_K), dtype=torch.float32)
    ref_id = torch.zeros((N, T, REF_K), dtype=torch.int32)
    kl_top = torch.zeros((S, N, T), dtype=torch.float64)       # the harness estimator, for calibration
    full = None
    if dump:                                                   # universe 0's full log-probabilities, bf16 [N, T, V]
        V = head.shape[0]
        full = np.lib.format.open_memmap(dump, mode="w+", dtype=np.uint16, shape=(N, T, V))
    head = head.float()                                        # logits in fp32 (a bf16 GEMM rounds them to ~0.06)
    for n in range(N):
        nxt = tokens[n, 1:].to(device)
        lp0 = F.log_softmax((final[0, n].to(device).float() @ head.T).double(), dim=-1)
        a0 = lp0.argmax(-1)
        p0 = lp0.exp()
        v, i = lp0.topk(REF_K, dim=-1)
        ref_lp[n], ref_id[n] = v.float().cpu(), i.int().cpu()
        if full is not None:
            full[n] = lp0.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
        for s in range(S):
            lp = lp0 if s == 0 else F.log_softmax((final[s, n].to(device).float() @ head.T).double(), dim=-1)
            kl[s, n] = (p0 * (lp0 - lp)).sum(-1).cpu()
            kl_top[s, n] = _coarse_kl(lp0, lp).cpu()
            top1[s, n] = (lp.argmax(-1) == a0).cpu()
            nll[s, n] = -lp[:-1].gather(1, nxt.unsqueeze(1)).squeeze(1).cpu()
            del lp
    if full is not None:
        full.flush()
    return dict(kl=kl, top1=top1, nll=nll, ref_lp=ref_lp, ref_id=ref_id, kl_top64=kl_top)


def summarize(res, universes, buckets=((0, 256), (256, 1024), (1024, 2048))):
    kl, top1, nll = res["kl"], res["top1"], res["nll"]
    T = kl.shape[2]
    out = {"lines": [], "universes": {}}
    for s, u in enumerate(universes):
        d = {"kld": kl[s].mean().item(), "top1": top1[s].float().mean().item(),
             "ppl": float(torch.exp(nll[s].mean())), "by_position": {}}
        for a, b in buckets:
            if a < T:
                d["by_position"][f"{a}-{min(b, T)}"] = kl[s, :, a:min(b, T)].mean().item()
        out["universes"][u.name] = d
        pos = " ".join(f"{k} {v:.4f}" for k, v in d["by_position"].items())
        out["lines"].append(f"{u.name:24s} KLD {d['kld']:.5f} top-1 {d['top1']:.4f} ppl {d['ppl']:.3f} | by position {pos}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir")
    p.add_argument("tokens")
    p.add_argument("out_dir")
    p.add_argument("--ref", required=True, help="directory with Moonshot's modeling_kimi_linear.py and its config")
    p.add_argument("--universes", help="JSON list of universes (k31/policy.py); universe 0 = the published model")
    p.add_argument("--layers", type=int, help="stop after this many layers (no logits)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--readers", type=int, default=1, help="shards read at once, one sequential stream each")
    p.add_argument("--no-record", action="store_true", help="do not save universe 0's routing")
    p.add_argument("--device", default="cuda")
    p.add_argument("--state-dir", help="keep the universes' hidden states in unlinked memmap files here (local disk)")
    p.add_argument("--dump-logits", help="write universe 0's full log-probabilities (bf16 bits, [N, T, V] .npy) here: "
                   "the reference for the engine's exact KL (k31_engine/force.py)")
    p.add_argument("--dump-hidden", help="write universe 0's final hidden states (bf16 bits, [N, T, H] .npy) here: the "
                   "compact exact-KL reference (the engine rebuilds the logits with its BF16 LM head)")
    p.add_argument("--save-means", help="write universe 0's mean routed-expert outputs per layer (means-L<i>.pt) here")
    p.add_argument("--means", help="mean-output tables (from --save-means) for universes with cut_sub = mean")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"),
                   help="activations and dense weights (float32: a precise reference; experts are decoded to it)")
    p.add_argument("--precise", action="store_true",
                   help="no reduced-precision GEMM reductions and no TF32 (cuBLAS split-K sums in fp32)")
    p.add_argument("--cpu-standins", action="store_true",
                   help="plain-PyTorch stand-ins for fla's kernels (tests/cpu_standins.py): loader smoke tests on CPU")
    a = p.parse_args()
    tokens = torch.load(a.tokens)
    tokens = tokens["tokens"] if isinstance(tokens, dict) else tokens
    patch = None
    if a.cpu_standins:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
        import cpu_standins
        patch = cpu_standins.patch
    if a.precise:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(f"dtype {a.dtype}, precise {a.precise}", flush=True)
    run(a.model_dir, tokens, a.out_dir, a.ref, load_universes(a.universes), a.layers, a.batch, device=a.device,
        dtype=getattr(torch, a.dtype), record=not a.no_record, readers=a.readers, patch=patch,
        log=lambda s: print(s, flush=True), state_dir=a.state_dir, dump_logits=a.dump_logits,
        save_means=a.save_means, means=a.means, dump_hidden=a.dump_hidden)


if __name__ == "__main__":
    main()
