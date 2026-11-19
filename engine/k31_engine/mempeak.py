"""K31_MEM_PEAK=<rows> (0 = off): GB300 memory peaks of the serving steps with at least <rows> scheduled tokens (prompt
chunks), to check the 8,192-row chunk's memory budget on a running server.

Set at start-up, the forward hooks go on every decoder layer's attention and MLP / MoE module (none otherwise); the
row threshold is then a runtime flag (flags.py), 0 switching the lines off. One line per such step: the torch
allocator's peak over the step against its allocation at the step's start, how far its reserved memory grew (what it
had to take from the free HBM; free memory at the start beside it), and the largest peak inside one module of each
kind (DSA / KDA attention, dense MLA, MoE, dense MLP) with its layer. Host-side allocator statistics only: no device
synchronization, nothing on the GPU. Allocations outside torch's allocator (cuBLAS / DeepGEMM workspaces, registered
buffers) are not seen; the KV pool, weights and persistent buffers are part of the step's starting allocation.
"""
import sys

import torch

from . import flags

_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
G = 2 ** 30
_S: dict = {"installed": False, "on": False, "n": 0}
_KIND = {"KimiK3DSAAttention": "DSA", "KimiK3DeltaAttention": "KDA", "MultiHeadLatentAttention": "MLA",
         "KimiMoE": "MoE", "KimiMLP": "MLP"}


def _fold() -> None:
    _S["peak"] = max(_S["peak"], torch.cuda.max_memory_allocated())
    _S["rpeak"] = max(_S["rpeak"], torch.cuda.max_memory_reserved())


def _pre(mod, args) -> None:
    if not _S["on"]:
        return
    _fold()
    torch.cuda.reset_peak_memory_stats()
    mod._k31_mbase = torch.cuda.memory_allocated()


def _post(mod, args, out) -> None:
    if not _S["on"]:
        return
    p = torch.cuda.max_memory_allocated()
    _fold()
    d = p - getattr(mod, "_k31_mbase", p)
    kind = mod._k31_mkind
    cur = _S["mods"].get(kind)
    if cur is None or d > cur[0]:
        _S["mods"][kind] = (d, mod._k31_mlayer, p)
    torch.cuda.reset_peak_memory_stats()


def install(model) -> int:
    """Hooks on every decoder layer's self_attn and mlp; K31_MEM_PEAK must be set (> 0) at start-up."""
    if _S["installed"] or flags.num("K31_MEM_PEAK", 0) <= 0:
        return 0
    n = 0
    for _, m in model.named_modules():
        if type(m).__name__ != "KimiDecoderLayer":
            continue
        idx = getattr(m, "layer_idx", -1)
        for sub in (getattr(m, "self_attn", None), getattr(m, "mlp", None)):
            if sub is None:
                continue
            sub._k31_mkind = _KIND.get(type(sub).__name__, type(sub).__name__)
            sub._k31_mlayer = idx
            sub.register_forward_pre_hook(_pre)
            sub.register_forward_hook(_post)
            n += 1
    _S["installed"] = True
    _LOG(f"memory peaks (K31_MEM_PEAK): hooks on {n} attention / MLP modules; one line per step of >= "
         f"{flags.num('K31_MEM_PEAK', 0)} tokens")
    return n


def step_begin(ntok: int) -> None:
    rows = flags.num("K31_MEM_PEAK", 0)
    _S["on"] = _S["installed"] and rows > 0 and ntok >= rows
    if not _S["on"]:
        return
    free, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    _S.update(ntok=ntok, free=free, base=torch.cuda.memory_allocated(), rbase=torch.cuda.memory_reserved(), peak=0,
              rpeak=0, mods={})


def step_end() -> None:
    if not _S["on"]:
        return
    _fold()
    _S["on"] = False
    _S["n"] += 1
    base, rbase = _S["base"], _S["rbase"]
    mods = sorted(_S["mods"].items(), key=lambda kv: -kv[1][0])
    parts = ", ".join(f"{k} +{d / G:.2f} (layer {li}, at {p / G:.2f})" for k, (d, li, p) in mods)
    _LOG(f"memory peak, step of {_S['ntok']} tokens: allocated {base / G:.2f} GiB at the start, peak "
         f"{_S['peak'] / G:.2f} (+{(_S['peak'] - base) / G:.2f}); reserved {rbase / G:.2f} -> peak "
         f"{_S['rpeak'] / G:.2f} (+{max(0, _S['rpeak'] - rbase) / G:.2f}) with {_S['free'] / G:.2f} GiB free at the "
         f"start; largest module peaks over their own start (GiB): {parts or 'none'}")
