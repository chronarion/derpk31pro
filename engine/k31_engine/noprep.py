"""K31_MEGAMOE_NOPREP=1: MegaMoE's inputs without vLLM's prep kernel, for decode-size calls.

vLLM's DeepGemmMegaMoEExperts.forward launches `_prepare_megamoe_inputs_kernel` right before every MegaMoE call (the
last thing before MegaMoE in every layer window: 6.0 us after the latent down in windows that stage nothing, 3.0 after a
stage, R3 trace). It quantizes the latent rows to MXFP8 (E4M3 codes, one power-of-two scale byte per 32 values) into the
symmetric buffer's x / x_sf, and copies the call's row ids (int64) and routing weights (float32) into its topk_idx /
topk_weights (padding rows unrouted). Both halves move into kernels that run anyway:
  x half      the latent down-projection, w8a16's split-K W8A16 GEMM (BN 32 = one scale group per program): the program
              that writes a tile's bf16 output also writes the tile's FP8 codes and scale bytes, computed from those
              bf16 values by vLLM's own Triton function (_quantize_fp8_groups, the prep kernel's): prep's bytes.
  index half  the plan kernel (csrc/noprep_plan.cu, one CTA): vLLM's sigmoid grouped top-k for one group with its
              arithmetic (K31_NOPREP_TOPK=1, the default: it replaces vLLM's top-k launch after the router GEMV; 0 keeps
              vLLM's top-k kernel), unitk's plan (row ids, stage list, residency inputs) and the routing inputs in
              MegaMoE's layout.
MegaMoE is then launched directly (the arguments of vLLM's forward without the prep step). MegaMoE gets the same bytes
(tests/test_noprep.py on the GPU; at bind the first eligible MoE layer is checked against vLLM's kernels on random rows,
K31_NOPREP_CHECK=1, and any difference turns the path off, or only the top-k part, with one log line).
The plan kernel follows K31_STAGE_ALL=1 (unitk's ALLROWS rule): every non-resident pick of an expert that
holds a slot uses the slot. The flag is read at start-up: decode calls are captured in CUDA graphs.

Scope: the one-round decode path (T <= 16) with K31_SHARED_SIDE=3 (the hook's deferred latent-down join, where this
module computes the latent down), mode staged, no deferral, MegaMoE MXFP4 (MXFP8 dispatch) without a fused shared
expert, EPLB or a capture hook, the latent down on the split-K W8A16 kernel (not Marlin, K31_W8_A8=0). Every other call
keeps vLLM's prep (which rewrites all four inputs), so mixing paths within a step is safe.
"""
import os
import sys

import torch
import triton
import triton.language as tl

from . import w8a16

try:                                    # the prep kernel's quantization, a Triton function (a stand-in without a GPU)
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import _quantize_fp8_groups
except Exception:                       # noqa: BLE001
    _quantize_fp8_groups = None

MODE = os.environ.get("K31_MEGAMOE_NOPREP", "0") == "1"
TOPK = os.environ.get("K31_NOPREP_TOPK", "1") == "1"
CHECK = os.environ.get("K31_NOPREP_CHECK", "1") == "1"
PDL = os.environ.get("K31_PDL", "0") == "1"
STAGE_ALL = os.environ.get("K31_STAGE_ALL", "0") == "1"     # unitk.plan's ALLROWS default
E, K, LATENT, HIDDEN, GROUP = 896, 16, 3584, 7168, 32
ACTIVE = [MODE]                 # cleared at bind if the check finds a difference
TOPK_ON = [TOPK]
_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
_CALL: dict = {"idx": -1}       # the decode call in flight: set at its router, taken by the experts' forward
_STATIC: dict = {}              # layer idx -> reason it is not eligible ("" = eligible), static checks
_VIEWS: dict = {}               # layer idx -> MegaMoE's input views (x, x_sf, topk_idx, topk_weights) or None
_SEEN = {"calls": 0, "direct": 0}


# ------------------------------------------------------------------------------ x half: the latent down's epilogue
@triton.jit
def _mx_store(out, rm, rn, pn, M, N, xq_ptr, xq_sm, xs_ptr, xs_sm, xs_sk, BM: tl.constexpr, BN: tl.constexpr):
    """MXFP8 of a finished [BM, BN] bf16 tile (BN / 32 scale groups of each row) into MegaMoE's x / x_sf, by vLLM's
    _quantize_fp8_groups (UE8M0, group 32) as the prep kernel calls it: codes at x[row, col], the scale byte of group
    g at byte g % 4 of x_sf's int32 word (row, g // 4) (little-endian packing, four groups per word)."""
    fp8, _qs, exp = _quantize_fp8_groups(out.to(tl.float32), BM, BN, 32, True)
    tl.store(xq_ptr + rm[:, None] * xq_sm + rn[None, :], fp8, mask=(rm[:, None] < M) & (rn[None, :] < N))
    G: tl.constexpr = BN // 32
    gi = pn * G + tl.arange(0, G)
    xs8 = xs_ptr.to(tl.pointer_type(tl.uint8))
    off = (rm[:, None] * xs_sm + (gi[None, :] // 4) * xs_sk) * 4 + gi[None, :] % 4
    tl.store(xs8 + off, exp.to(tl.uint8), mask=(rm[:, None] < M) & (gi[None, :] * 32 < N))


@triton.jit
def _w8a16_mx(x_ptr, q_ptr, s_ptr, ws_ptr, cnt_ptr, y_ptr, M, N, K, NG, sxm, xq_ptr, xq_sm, xs_ptr, xs_sm, xs_sk,
              BM: tl.constexpr, BN: tl.constexpr, KCH: tl.constexpr, SPLIT: tl.constexpr):
    """w8a16._w8a16 with W16 = NORM = RES = A8 = 0, statement for statement (the bf16 output is the same), plus the
    MXFP8 epilogue in the program that writes each output tile."""
    tl.extra.cuda.gdc_wait()                     # the previous kernel's writes (x, a reused workspace) are visible
    tl.extra.cuda.gdc_launch_dependents()        # the next kernel may launch and run up to its own wait
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rm = tl.program_id(2) * BM + tl.arange(0, BM)
    rg = tl.arange(0, 128)
    nm = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in tl.range(0, KCH // 128):
        k0 = pk * KCH + g * 128
        km = (k0 + rg) < K
        x = tl.load(x_ptr + rm[:, None] * sxm + k0 + rg[None, :], mask=(rm[:, None] < M) & km[None, :], other=0.0)
        w = tl.load(q_ptr + rn[:, None] * K + k0 + rg[None, :], mask=nm[:, None] & km[None, :], other=0.0)
        part = tl.dot(x, tl.trans(w.to(tl.bfloat16)))                 # [BM, BN] float32
        sc = tl.load(s_ptr + rn * NG + (k0 // 128), mask=nm & ((k0 // 128) < NG), other=0.0).to(tl.float32)
        acc += part * sc[None, :]
    if SPLIT == 1:
        out = acc.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & nm[None, :])
        _mx_store(out, rm, rn, pn, M, N, xq_ptr, xq_sm, xs_ptr, xs_sm, xs_sk, BM, BN)
    else:
        tl.store(ws_ptr + (pk * BM + rm[:, None]) * N + rn[None, :], acc, mask=(rm[:, None] < M) & nm[None, :])
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
        if old == SPLIT - 1:                                              # the last slice of this N tile: sum in order
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for j in tl.range(0, SPLIT):
                tot += tl.load(ws_ptr + (j * BM + rm[:, None]) * N + rn[None, :],
                               mask=(rm[:, None] < M) & nm[None, :], other=0.0, cache_modifier=".cg")
            out = tot.to(y_ptr.dtype.element_ty)
            tl.store(y_ptr + rm[:, None] * N + rn[None, :], out, mask=(rm[:, None] < M) & nm[None, :])
            _mx_store(out, rm, rn, pn, M, N, xq_ptr, xq_sm, xs_ptr, xs_sm, xs_sk, BM, BN)
            tl.atomic_xchg(cnt_ptr + pn, 0, sem="relaxed", scope="gpu")


def linear_mx(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, xq: torch.Tensor, xsf: torch.Tensor) -> torch.Tensor:
    """w8a16.linear(x, q, s) for M <= 16 rows (same config, grid, workspace and bf16 output), and the output's MXFP8
    into xq (float8_e4m3fn [>= M, N], unit column stride) and xsf (int32 [>= M, N / 128], four scale bytes per word)."""
    M, Kd = x.shape
    N = q.shape[0]
    assert M <= 16 and q.shape[1] == Kd and s.shape == (N, Kd // w8a16.GK) and x.stride(1) == 1 and \
        (M == 1 or x.stride(0) >= Kd)
    assert xq.dtype == torch.float8_e4m3fn and xq.stride(1) == 1 and xsf.dtype == torch.int32 and \
        xq.shape[0] >= M and xsf.shape[0] >= M and xq.shape[1] == N and xsf.shape[1] * 128 == N
    cfg = w8a16.config(M, N, Kd)
    BN, KCH, SPLIT = cfg[:3]
    nw, ns = (cfg[3], cfg[4]) if len(cfg) > 3 else (4, 3)
    assert BN % GROUP == 0 and N % BN == 0, f"w8a16 config {cfg}: BN must be whole scale groups"
    sxm = x.stride(0) if M > 1 else Kd
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    nt = triton.cdiv(N, BN)
    ws, cnt = w8a16._workspace(x.device, torch.cuda.current_stream().cuda_stream, SPLIT, 16, N, nt)
    _w8a16_mx[(nt, SPLIT)](x, q, s, ws, cnt, out, M, N, Kd, Kd // w8a16.GK, sxm, xq, xq.stride(0), xsf, xsf.stride(0),
                           xsf.stride(1), BM=16, BN=BN, KCH=KCH, SPLIT=SPLIT, num_warps=nw, num_stages=ns,
                           launch_pdl=w8a16.PDL)
    return out


# ------------------------------------------------------------------------------ index half: the plan kernel
_MOD: list = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "noprep_plan.cu")
        # no fast math: vLLM's top-k (_moe_C) divides and adds with IEEE rounding and keeps denormals
        _MOD.append(load(name="k31_noprep_plan", sources=[src], verbose=False,
                         extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_103a,code=sm_103a"]))
    return _MOD[0]


def _ptr(t) -> int:
    return 0 if t is None else t.data_ptr()


def plan_kernel(T: int, row_map, bank_map, strong: int, S: int, resmax: int, mm_idx, mm_w, *, logits=None, bias=None,
                renorm: bool = True, scale: float = 1.0, ids=None, w=None, score=None, lam: float = 1.0, stats=None,
                pend=None, pad=None, allrows: bool = None):
    """One launch. With logits (float32 [T, 896] contiguous) and bias: vLLM's top-k in the kernel; else ids (int32
    [T, 16]) and w (float32 [T, 16]) from vLLM's top-k. Writes MegaMoE's mm_idx (int64 [>= T, 16]) / mm_w (float32
    [>= T, 16]) rows 0..T-1 (pad: bool [T] or None, vLLM's is_padding) and the residency inputs when score is given
    (unitk.plan's score / lam / stats / pend). allrows (default K31_STAGE_ALL): unitk's ALLROWS rule. Returns
    (weights, expert ids, row ids, stage list); the first two are the inputs when the top-k is not in the kernel."""
    allrows = STAGE_ALL if allrows is None else bool(allrows)
    use_topk = logits is not None
    dev = row_map.device
    rows = torch.empty((T, K), dtype=torch.int32, device=dev)
    stage = torch.empty(16, dtype=torch.int32, device=dev)
    if use_topk:
        assert logits.dtype == torch.float32 and logits.is_contiguous() and tuple(logits.shape) == (T, E)
        assert bias.dtype == torch.float32 and bias.numel() == E and bias.is_contiguous()
        out_w = torch.empty((T, K), dtype=torch.float32, device=dev)
        out_ids = torch.empty((T, K), dtype=torch.int32, device=dev)
    else:
        out_ids = ids if ids.dtype == torch.int32 and ids.is_contiguous() else ids.int().contiguous()
        out_w = w if w.dtype == torch.float32 and w.is_contiguous() else w.float().contiguous()
    assert mm_idx.dtype == torch.int64 and mm_w.dtype == torch.float32 and mm_idx.stride(1) == 1 and mm_w.stride(1) == 1
    assert pad is None or (pad.dtype == torch.bool and pad.dim() == 1)
    acc = score is not None and pend is not None
    pid, pw, pt = pend if acc else (None, None, None)
    ptrs = [_ptr(logits), _ptr(bias), 0 if use_topk else _ptr(out_ids), 0 if use_topk else _ptr(out_w), _ptr(row_map),
            _ptr(bank_map), _ptr(score), _ptr(stats if score is not None else None), _ptr(pid), _ptr(pw), _ptr(pt),
            _ptr(pad), _ptr(out_w) if use_topk else 0, _ptr(out_ids) if use_topk else 0, _ptr(rows), _ptr(stage),
            _ptr(mm_idx), _ptr(mm_w)]
    ints = [T, int(strong), int(S), int(resmax), int(bool(renorm)), K, mm_idx.stride(0), mm_w.stride(0),
            pad.stride(0) if pad is not None else 0, int(use_topk), int(PDL), int(allrows)]
    _module().plan(ptrs, ints, float(scale), float(lam ** T))      # unitk: decay = float(lam ** T) as a float32
    return out_w, out_ids, rows, stage


# ------------------------------------------------------------------------------ eligibility, the call protocol
def _static(moe, st) -> str:
    """Why this layer cannot take the path ("" = it can); checks that do not touch MegaMoE's symmetric buffer."""
    mod = moe.experts
    fmt = getattr(mod, "weight_format", None)
    if fmt is None or fmt.kernel != "fp8_fp4_mega_moe" or fmt.name != "mxfp4":
        return f"MegaMoE format {getattr(fmt, 'name', None)}"
    hq = fmt.hidden_quant
    if hq is None or hq.dtype != torch.float8_e4m3fn or hq.scale.group_shape.col != GROUP or \
            hq.scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        return f"dispatch quantization {hq}"
    if mod.has_fused_shared_experts or mod.shared_spec is not None:
        return "fused shared expert"
    if int(mod.hidden_size) != LATENT or int(mod.top_k) != K:
        return "shape"
    down = getattr(moe, "routed_expert_down_proj", None)
    if down is None or getattr(down, "bias", None) is not None:
        return "no latent down-projection (or a biased one)"
    if any(getattr(down, a, None) is not None for a in ("_k31_c8", "_k31_i8", "_k31_m4", "_k31_dg4")):
        return "latent down not on W8A16"
    w8 = getattr(down, "_k31_w8", None)
    if w8 is None or not isinstance(w8[0], torch.Tensor) or w8[1] is None or w8a16.A8 != 0:
        return "latent down not on the split-K W8A16 kernel (Marlin, other format or K31_W8_A8)"
    q = w8[0]
    if tuple(q.shape) != (LATENT, HIDDEN):
        return f"latent down shape {tuple(q.shape)}"
    BN = w8a16.config(16, LATENT, HIDDEN)[0]
    if BN % GROUP or LATENT % BN:
        return f"latent down config BN {BN}"
    return ""


def _views(mod, idx: int):
    """MegaMoE's four routing inputs of the layer's symmetric buffer (the hook's shared buffer view), checked once."""
    if idx not in _VIEWS:
        v = None
        try:
            sb = mod.get_symm_buffer()
            x, xsf, ti, tw = sb.x, sb.x_sf, sb.topk_idx, sb.topk_weights
            if x.dtype == torch.float8_e4m3fn and x.dim() == 2 and x.shape[1] == LATENT and x.stride(1) == 1 and \
                    xsf.dtype == torch.int32 and xsf.dim() == 2 and xsf.shape[1] * 128 == LATENT and \
                    ti.dtype == torch.int64 and ti.dim() == 2 and ti.shape[1] == K and ti.stride(1) == 1 and \
                    tw.dtype == torch.float32 and tw.dim() == 2 and tw.shape[1] == K and tw.stride(1) == 1 and \
                    min(x.shape[0], xsf.shape[0], ti.shape[0], tw.shape[0]) >= 16:
                v = (x, xsf, ti, tw)
            else:
                _LOG(f"noprep: layer {idx}: MegaMoE's input views are not the expected layout "
                     f"(x {x.dtype} {tuple(x.shape)}, x_sf {xsf.dtype} {tuple(xsf.shape)}, topk_idx {ti.dtype}, "
                     f"topk_weights {tw.dtype}): prep stays")
        except Exception as ex:                  # noqa: BLE001
            _LOG(f"noprep: layer {idx}: no MegaMoE input views ({ex}): prep stays")
        _VIEWS[idx] = v
    return _VIEWS[idx]


def _padding(T: int):
    """vLLM's forward: the forward context's is_padding[:T] when VLLM_MOE_SKIP_PADDING, else None; False = a mask this
    module does not read (not bool)."""
    from vllm import envs
    if not envs.VLLM_MOE_SKIP_PADDING:
        return None
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    p = get_forward_context().is_padding
    if p is None:
        return None
    p = p[:T]
    return p if p.dtype == torch.bool and p.dim() == 1 else False


def begin(moe, h, allowed: bool, slots: int) -> bool:
    """At the router of a decode-size call (hook._router_down_nojoin): whether it takes the path. The predicate is the
    hook's one-round branch, so the experts' forward of this call always takes that branch."""
    _CALL["idx"] = -1
    if not (ACTIVE[0] and allowed) or h.dim() != 2:
        return False
    mod = getattr(moe, "experts", None)
    st = getattr(mod, "_k31", None)
    if st is None or st.row_map is None:
        return False
    T = h.shape[0]
    if not (1 <= T <= 16 and (torch.cuda.is_current_stream_capturing() or T * st.strong <= slots)):
        return False
    if st.idx not in _STATIC:
        _STATIC[st.idx] = _static(moe, st)
    if _STATIC[st.idx] or mod.capture_fn is not None or mod.eplb_state.logical_to_physical_map is not None:
        return False
    views = _views(mod, st.idx)
    pad = _padding(T)
    if views is None or pad is False or h.dtype != torch.bfloat16:
        return False
    _CALL.update(idx=st.idx, T=T, x=False, index=False, plan=None, pad=pad, views=views)
    _SEEN["calls"] += 1
    return True


def route(moe, h, logits, lam: float):
    """K31_NOPREP_TOPK: in place of vLLM's grouped top-k after the router GEMV: the plan kernel (top-k + unitk + the
    index half). Returns the router's (weights, ids) as vLLM's top-k would, or None (vLLM's top-k runs)."""
    c = _CALL
    if c["idx"] < 0 or not TOPK_ON[0] or logits.dtype != torch.float32 or not logits.is_contiguous() or \
            tuple(logits.shape) != (c["T"], E):
        return None
    if not (moe.num_expert_group == 1 and moe.topk_group == 1 and moe.moe_router_activation_func == "sigmoid"):
        return None
    bias = moe.gate.e_score_correction_bias.data
    if bias.dtype != torch.float32 or not bias.is_contiguous():
        return None
    st = moe.experts._k31
    _, _, ti, tw = c["views"]
    w, ids, rows, stage = plan_kernel(c["T"], st.row_map, st.bank_map, st.strong, st.slot_base, st.resmax, ti, tw,
                                      logits=logits, bias=bias, renorm=moe.moe_renormalize,
                                      scale=moe.routed_scaling_factor, score=st.score, lam=lam, stats=st.stats,
                                      pend=getattr(st, "pend", None), pad=c["pad"])
    c["plan"] = (rows, stage)
    c["index"] = True
    return w, ids


def latent_down(moe, h):
    """On the latent down's stream, in place of routed_expert_down_proj(h): the same bf16 output (fp6.apply's W8A16
    branch) plus MegaMoE's x / x_sf. None = not this call (the caller runs the layer as before)."""
    c = _CALL
    if c["idx"] < 0:
        return None
    down = moe.routed_expert_down_proj
    q, sc, N = down._k31_w8
    x2 = h.reshape(-1, h.shape[-1])
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) < x2.shape[1]):
        x2 = x2.contiguous()
    xq, xsf, _, _ = c["views"]
    y = linear_mx(x2, q, sc, xq, xsf)
    c["x"] = True
    return y.reshape(*h.shape[:-1], N)


def take(st, T: int):
    """The experts' forward of the call begun at its router (None if this call did not begin)."""
    c = _CALL
    if c["idx"] != st.idx or c["T"] != T:
        return None
    c["idx"] = -1
    return dict(c)


def plan(st, nop: dict, topk_ids, topk_weights, lam: float):
    """unitk.plan's (row ids, stage list) for a begun call: the plan kernel's (top-k in the kernel), or the plan kernel
    now on vLLM's top-k outputs, which also writes the index half."""
    if nop["plan"] is not None:
        return nop["plan"]
    _, _, ti, tw = nop["views"]
    _, _, rows, stage = plan_kernel(nop["T"], st.row_map, st.bank_map, st.strong, st.slot_base, st.resmax, ti, tw,
                                    ids=topk_ids, w=topk_weights, score=st.score, lam=lam, stats=st.stats,
                                    pend=getattr(st, "pend", None), pad=nop["pad"])
    nop["index"] = True
    return rows, stage


def megamoe(mod, nop: dict, hidden_states, shared_experts=None):
    """DeepGemmMegaMoEExperts.forward without the prep step: the launch with the same arguments (MXFP4 weights,
    no fused shared expert, the layer's symmetric buffer). None = run vLLM's forward."""
    if not (nop["x"] and nop["index"]) or shared_experts is not None or hidden_states.shape[0] != nop["T"]:
        return None
    from vllm.utils.deep_gemm import _import_deep_gemm
    dg = _import_deep_gemm()
    mod._synchronize_first_launch()
    y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    getattr(dg, mod.weight_format.kernel)(y, mod.l1_weights, mod.l2_weights, mod.get_symm_buffer(),
                                          fast_math=mod.fast_math, **dict(mod._kernel_kwargs))
    _SEEN["direct"] += 1
    if _SEEN["direct"] in (1, 92, 10000):
        _LOG(f"noprep: MegaMoE without the prep kernel, {_SEEN['direct']} calls (begun {_SEEN['calls']}), capturing "
             f"{torch.cuda.is_current_stream_capturing()}")
    return y


# ------------------------------------------------------------------------------ bind: the check against vLLM
def _cmp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of differing bytes."""
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def unitk_has_allrows() -> bool:
    """This tree's unitk.plan takes `allrows` (K31_STAGE_ALL)."""
    import inspect
    from . import unitk
    return "allrows" in inspect.signature(unitk.plan).parameters


def _unitk_plan(*args, allrows: bool, **kw):
    """unitk.plan with the given ALLROWS mode (a tree without the mode has only allrows = False)."""
    from . import unitk
    if unitk_has_allrows():
        return unitk.plan(*args, allrows=allrows, **kw)
    assert not allrows, "this tree's unitk has no ALLROWS mode"
    return unitk.plan(*args, **kw)


def _fp8_zeros(rows: int, dev):
    return torch.zeros(rows, LATENT, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)


@torch.no_grad()
def check_layer(moe, st, dev, trials=(1, 2, 3, 5, 8, 13, 16), seed: int = 0) -> dict:
    """One MoE layer on random rows, against vLLM's kernels: the latent down + prep's x half, and (vLLM's top-k ->
    unitk -> prep's index half, with a padding mask) against the plan kernel with and without the top-k. Returns the
    differing byte counts per part ({"x": n, "topk": n, "plan": n})."""
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, sc, _ = moe.routed_expert_down_proj._k31_w8
    bias = moe.gate.e_score_correction_bias.data
    bad = {"x": 0, "topk": 0, "plan": 0}
    for T in trials:
        h = (torch.randn(T, HIDDEN, generator=g) * 2.0).to(dev, torch.bfloat16)
        if T > 2:
            h[1] *= 0                                   # an all-zero row: amax below 1e-4 in every group
            h[2] *= 1e-6
        xq = _fp8_zeros(16, dev)
        xsf = torch.zeros(16, LATENT // 128, dtype=torch.int32, device=dev)
        y = linear_mx(h, q, sc, xq, xsf)
        y_ref = w8a16.linear(h, q, sc)
        logits = (torch.randn(T, E, generator=g) * 3.0).to(dev)
        if T > 1:
            logits[1, 100:300] = logits[1, 100]          # ties in the selection (broken by the lower id)
        w_ref, ids_ref = fused_grouped_topk(h, logits, K, moe.moe_renormalize, bias, 1, 1, "sigmoid",
                                            moe.routed_scaling_factor)
        rows_ref, stage_ref = _unitk_plan(ids_ref, w_ref, st.row_map, st.bank_map, st.strong, st.slot_base,
                                          resmax=st.resmax, allrows=STAGE_ALL)
        pad = torch.zeros(T, dtype=torch.bool, device=dev)
        if T > 3:
            pad[-1] = True
        xr = _fp8_zeros(16, dev)
        xsr = torch.zeros(16, LATENT // 128, dtype=torch.int32, device=dev)
        ir = torch.zeros(16, K, dtype=torch.int64, device=dev)
        wr = torch.zeros(16, K, dtype=torch.float32, device=dev)
        prepare_megamoe_inputs(y_ref, w_ref, rows_ref.to(ids_ref.dtype), xr[:T], xsr[:T], ir[:T], wr[:T],
                               is_padding=pad)
        bad["x"] += _cmp(y, y_ref) + _cmp(xq[:T], xr[:T]) + _cmp(xsf[:T], xsr[:T])
        for use_topk in (True, False):
            ti = torch.zeros(16, K, dtype=torch.int64, device=dev)
            tw = torch.zeros(16, K, dtype=torch.float32, device=dev)
            if use_topk:
                w, ids, rows, stage = plan_kernel(T, st.row_map, st.bank_map, st.strong, st.slot_base, st.resmax, ti,
                                                  tw, logits=logits, bias=bias, renorm=moe.moe_renormalize,
                                                  scale=moe.routed_scaling_factor, pad=pad, allrows=STAGE_ALL)
                bad["topk"] += _cmp(w, w_ref) + _cmp(ids, ids_ref) + _cmp(rows, rows_ref) + _cmp(stage, stage_ref) + \
                    _cmp(ti[:T], ir[:T]) + _cmp(tw[:T], wr[:T])
            else:
                _, _, rows, stage = plan_kernel(T, st.row_map, st.bank_map, st.strong, st.slot_base, st.resmax, ti,
                                                tw, ids=ids_ref, w=w_ref, pad=pad, allrows=STAGE_ALL)
                bad["plan"] += _cmp(rows, rows_ref) + _cmp(stage, stage_ref) + _cmp(ti[:T], ir[:T]) + \
                    _cmp(tw[:T], wr[:T])
    torch.cuda.synchronize(dev)
    return bad


def bind(model, dev, log=_LOG) -> None:
    """After the hook's bind: which MoE layers can take the path, and (K31_NOPREP_CHECK) the first of them checked
    against vLLM's kernels; a difference in the x half or the plan turns the path off, one in the top-k only the
    top-k part."""
    if not ACTIVE[0]:
        return
    if STAGE_ALL and not unitk_has_allrows():
        ACTIVE[0] = False
        log("noprep: K31_STAGE_ALL=1 but this tree's unitk has no ALLROWS mode to check the plan kernel against: "
            "the prep kernel stays")
        return
    clash = [f for f in ("K31_LATE", "K31_CPU_STAGE") if os.environ.get(f, "0") == "1"]
    if clash:                                    # the Grace pool tiers extend unitk's plan of these calls (cpu_n, ...)
        ACTIVE[0] = False
        log(f"noprep: {', '.join(clash)}=1 extends unitk's plan of decode calls, which the plan kernel does not do: "
            f"the prep kernel stays")
        return
    moes = {}
    for _, m in model.named_modules():
        st = getattr(getattr(m, "experts", None), "_k31", None)
        if st is not None and hasattr(m, "routed_expert_down_proj") and hasattr(m, "gate"):
            moes[st.idx] = m
    for idx, m in moes.items():
        _STATIC[idx] = _static(m, m.experts._k31)
    ok = [i for i in sorted(moes) if not _STATIC[i]]
    reasons = sorted({r for r in _STATIC.values() if r})
    if not ok:
        ACTIVE[0] = False
        log(f"noprep: no MoE layer can take the path ({'; '.join(reasons) or 'no MoE layer found'}): prep stays")
        return
    if _quantize_fp8_groups is None:
        ACTIVE[0] = False
        log("noprep: vLLM's _quantize_fp8_groups is not importable: prep stays")
        return
    note = ""
    if CHECK:
        moe = moes[ok[0]]
        bad = check_layer(moe, moe.experts._k31, dev)
        if bad["x"] or bad["plan"]:
            ACTIVE[0] = False
            log(f"noprep: CHECK FAILED on layer {ok[0]} against vLLM's latent down + prep / top-k + unitk + prep "
                f"(differing bytes {bad}): the prep kernel stays")
            return
        if bad["topk"] and TOPK_ON[0]:
            TOPK_ON[0] = False
            note = f"; the plan kernel's top-k differed from vLLM's ({bad['topk']} bytes): vLLM's top-k kernel stays"
        else:
            note = f"; layer {ok[0]} checked bit-equal to vLLM's latent down + prep and (top-k ->) unitk + prep" + \
                ("" if not bad["topk"] else " (the in-kernel top-k, unused here, differed)")
    where = "the top-k in the plan kernel" if TOPK_ON[0] else "vLLM's top-k kernel before the plan kernel"
    log(f"noprep: MegaMoE's inputs without the prep kernel (K31_MEGAMOE_NOPREP) on {len(ok)} of {len(moes)} MoE layers"
        f" (decode-size calls of the one-round path): x half in the latent down's epilogue, index half in the plan "
        f"kernel, {where}" + (f"; not eligible: {'; '.join(reasons)}" if reasons else "") + note)
