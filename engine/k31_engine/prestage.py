"""K31_PRESTAGE=1 (start-up flag, mode staged): decode pre-staging. A layer's staged records are copied while its
attention runs, instead of between its router and its MegaMoE call.

Without it, a decode MoE call is serial on the main stream: router -> top-k -> unitk plan -> stager (the strong
misses' 2-bit records read from pinned Grace memory over C2C and expanded into the layer's 16 staging rows, ~30 us
per record) -> MegaMoE. Without the RTX band the stager sits on the critical path for ~4 ms per step.

With it, at the end of a decoder layer's _pre_attn_norm (the layer's inputs are final: the running prefix sum and
the stored block states) a side stream predicts the layer's picks and stages them while the attention runs:
  1. the MoE input as if this layer's attention output were zero: vLLM's attn_res over the stored blocks and the
     prefix sum without a delta (a pure read), or over the blocks and a zero prefix in a block-write layer (whose
     prefix restarts with the attention output; the zero source only scales the mix, which the output RMSNorm
     undoes); the main stream waits for this read before _post_attn_norm adds the attention output to the prefix
     sum in place (`read` event)
  2. the layer's router GEMV and grouped top-k on it (the model's own gate and fused_grouped_topk)
  3. unitk's plain planner on the predicted picks with ranks < strong + K31_PRESTAGE_EXTRA (default 0) as
     candidates, at most 16 - K31_PRESTAGE_RESERVE slots (default 12: free slots stay for the misses), no residency
     side effects -> the pre list (`plan` event)
  4. the layer's stager on the pre list into the layer's own slot group (`copy` event)
At the layer's MoE call (hook._forward, the one-round branch) unitk runs in PRE + FILL mode on the real picks:
candidates whose record was staged ahead use its slot, the others take the free slots and are staged now (the late
list, usually empty, through the same stager on the main stream), and main waits for `copy` before MegaMoE. Every
real candidate is still staged when the free slots suffice (they do unless a call has more than 4 misses), so the
experts computed are those of the plain path, plus the picks of staged-ahead experts that K31_STAGE_ALL serves.

The slot group of layer L (L % 8) is free while L's attention runs: layer L - 8 used it last and has finished. The
side stream's work joins the main stream at the layer's own MoE call (inside the decode CUDA graph). Only the
one-round decode calls (T <= 16 rows) of layers with staging (strong > 0) are pre-staged.
"""
import os

import torch

from . import unitk

EXTRA = int(os.environ.get("K31_PRESTAGE_EXTRA", "0"))
# K31_PRESTAGE_SMS: CTAs of the side stream's stager (0 = the layer's own stager, K31_STAGER_SMS). Its TMA reads from
# Grace memory hold their SMs for the whole copy; at 24 they starve the attention's clustered GEMVs beside them (trace
# 11-19: int8 GEMV 18 -> 88 us), so the copy ahead runs slower on fewer SMs instead
SMS = int(os.environ.get("K31_PRESTAGE_SMS", "0"))
RESERVE = int(os.environ.get("K31_PRESTAGE_RESERVE", "4"))
assert 0 <= RESERVE < unitk.SLOTS
_S: dict = {"stream": None, "zero": None, "n": 0, "taken": 0}
_PEND: dict = {}            # layer idx -> this step's pre-stage: T, events, the pre list, references kept


def _stream(dev):
    if _S["stream"] is None:
        _S["stream"] = torch.cuda.Stream(device=dev)
    return _S["stream"]


def _zero(like: torch.Tensor) -> torch.Tensor:
    z = _S["zero"]
    if z is None or z.shape[0] < like.shape[0] or z.shape[1] != like.shape[1] or z.dtype != like.dtype or \
            z.device != like.device:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("K31_PRESTAGE: the zero prefix must exist before graph capture")
        z = torch.zeros((max(16, like.shape[0]), like.shape[1]), dtype=like.dtype, device=like.device)
        _S["zero"] = z
    return z[:like.shape[0]]


def prepare(dev, hidden: int, dtype=torch.bfloat16) -> None:
    """At bind: the side stream and the zero prefix (16 rows), so graph capture allocates neither."""
    _stream(dev)
    if _S["zero"] is None:
        _S["zero"] = torch.zeros((16, hidden), dtype=dtype, device=dev)


def join_stale() -> None:
    """Join and drop any pre-stage a MoE call did not take (keeps every side branch joined inside a graph)."""
    if not _PEND:
        return
    main = torch.cuda.current_stream()
    for p in _PEND.values():
        main.wait_event(p["copy"])
    _PEND.clear()


def fork(layer, st, moe, prefix_sum, residual, T: int, attn_res, fgt, slots: int, rowcap=None) -> None:
    """Called at the end of layer's _pre_attn_norm for a decode call of T rows; queues the side stream's work."""
    join_stale()
    if st.strong <= 0 or T > 16 or st.stager is None or residual is None or prefix_sum is None:
        return
    if not (torch.cuda.is_current_stream_capturing() or T * st.strong <= slots):
        return                                       # hook._forward takes its eager branch: nothing to pre-stage
    main = torch.cuda.current_stream()
    side = _stream(main.device)
    side.wait_stream(main)
    bw = bool(layer.is_block_write_layer)
    with torch.cuda.stream(side):
        prefix = _zero(prefix_sum) if bw else prefix_sum
        x = attn_res(prefix, None, residual, layer.mlp_res_norm.weight, layer.mlp_res_proj.weight.squeeze(0),
                     layer.post_attention_layernorm.weight, num_blocks=layer.prev_valid_blocks + int(bw),
                     block_write_idx=-1, eps=layer.mlp_res_norm.variance_epsilon,
                     output_norm_eps=layer.post_attention_layernorm.variance_epsilon)
        read = torch.cuda.Event()
        read.record(side)
        logits, _ = moe.gate(x)
        w, ids = fgt(hidden_states=x, gating_output=logits, topk=moe.experts.top_k,
                     renormalize=moe.moe_renormalize, e_score_correction_bias=moe.gate.e_score_correction_bias.data,
                     num_expert_group=moe.num_expert_group, topk_group=moe.topk_group,
                     scoring_func=moe.moe_router_activation_func, routed_scaling_factor=moe.routed_scaling_factor)
        out, pre = st.pre_bufs
        unitk.plan(ids, w, st.row_map, st.bank_map, min(unitk.SP, st.strong + EXTRA), st.slot_base, out=out,
                   stage=pre, resmax=16, allrows=False, cap=unitk.SLOTS - RESERVE, rowcap=rowcap)
        plan = torch.cuda.Event()
        plan.record(side)
        (getattr(st, "pre_stager", None) or st.stager)(st.bank_ptr, pre)
        copy = torch.cuda.Event()
        copy.record(side)
    _PEND[st.idx] = {"T": T, "read": read, "plan": plan, "copy": copy, "pre": pre, "keep": (x, logits, w, ids)}
    _S["n"] += 1
    if _S["n"] in (1, 92, 1000, 100000):
        import sys
        sys.stderr.write(f"K31 K31_PRESTAGE: fork {_S['n']} (layer {st.idx}, {T} rows, strong {st.strong}, capturing "
                         f"{torch.cuda.is_current_stream_capturing()}); taken so far {_S['taken']}\n")


def wait_read(idx: int) -> None:
    """Main stream, before _post_attn_norm: the side stream has read the prefix sum (which is about to change)."""
    p = _PEND.get(idx)
    if p is not None:
        torch.cuda.current_stream().wait_event(p["read"])


def take(st, T: int):
    """hook._forward: this layer's pre-stage for a call of T rows (main waits for the pre list), or None."""
    p = _PEND.pop(st.idx, None)
    if p is None:
        return None
    main = torch.cuda.current_stream()
    if p["T"] != T:                                  # not this call's: join it, stage the plain way
        main.wait_event(p["copy"])
        return None
    main.wait_event(p["plan"])
    _S["taken"] += 1
    return p
