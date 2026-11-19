"""Shared pieces of the MegaMoE fork tests (derpk31pro: the vLLM 0.32 port's tests carried to the kimi-k31 image).

What changed against the vLLM 0.32 port's tests:
  - weights: no `transform_weights`. The routed weights are created and loaded by vLLM's own
    `DeepGemmMegaMoEExperts` (its `_create_weights` and `weight_loader`: every checkpoint shard is written straight
    into the kernel layout), fed with synthetic shards named as `KimiLinearModel.load_weights` names them.
  - `prepare_megamoe_inputs` comes from `vllm.model_executor.layers.fused_moe.prepare_megamoe`.
  - the kernel call takes the module's `l1_weights` / `l2_weights` and its `_kernel_kwargs` (SiTU alpha 4, beta 25).
  - the fork's headers: either bind-mounted over the image's two files (MODE=fork with the run scripts: the engine's
    deployment form) or laid over a copy of the include tree with `_C.init(copy)` (FORK_VIA=copy, needed for DEFINES).

Geometry: Kimi K3.1's routed experts (latent 3584, intermediate 3072, 16 picks). Random FP4 nibbles (every e2m1 code
is a number) and scale bytes 119..123 (K3.1's range)."""
import hashlib
import os
import shutil
import subprocess
import types

import torch

H, I, K = 3584, 3072, 16
SITU_ALPHA, SITU_BETA = 4.0, 25.0
MAGIC = 0x53505244              # "DRPS" = derpseek_host::kMagic
HDR = 16                        # control block header words; per-row pick counts follow
CLONE_BYTES = 160 * 4 * 128     # derpseek_host::kMaxSMs CTAs x 4 maps x 128 B
KERNEL_REL = "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh"
SCHED_REL = "include/deep_gemm/scheduler/mega_moe.cuh"
GRAPH_CALLS = int(os.environ.get("GRAPH_CALLS", "10"))     # calls per timed graph replay (the cap, see timeit)
REPLAY_US = float(os.environ.get("REPLAY_US", "1500"))     # a timed replay is kept under this many microseconds
dev = "cuda"
# Diagnostic variants of a kernel header (VARIANT=name): (text, replacement) pairs, each text exactly once in the file
VARIANTS = {
    # one scheduling queue per task kind instead of one per locality domain (die) of the GPU: DeepGEMM 2.8.0's
    # schedule. On the image's stock kernel: what the per-die queues are worth at this geometry
    "one-domain": [("L2_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 ? kNumDeviceLocalityDomains : 1;",
                    "L2_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 ? 1 : 1;")],
    # the other split of the 64512 registers between the non-epilogue and the epilogue warps (more than 64 rows):
    # DeepGEMM 2.8.0's 88 / 160, which Derpseek's fork was tuned on; the image has 72 / 168
    "regs-88-160": [("kUseMoreEpilogueRegisters ? 40 : 72;", "kUseMoreEpilogueRegisters ? 40 : 88;"),
                    ("kUseMoreEpilogueRegisters ? 208 : 168;", "kUseMoreEpilogueRegisters ? 208 : 160;")],
}


def _ctl_early(src):
    """Mutation for ctl_percall_test.py: the fork's control-block reads back at the kernel's start, above the
    dependency wait (where Derpseek's fork and the vLLM 0.32 port have them)."""
    a = src.index("    // Derpseek exact Grace tier: the control block (see derpseek_host). With it, the per-row token counts")
    end_line = "    const uint32_t drp_sf6_word = drp_sf6 ? derpseek_sf6::kE0Word : 0u;"
    b = src.index(end_line, a)
    b = src.index("\n", b) + 1
    block, src = src[a:b], src[:a] + src[b:]
    anchor = "    // Workspaces and Buffer\n"
    assert src.count(anchor) == 1
    return src.replace(anchor, block + "\n" + anchor)


VARIANT_FUNCS = {"ctl-early": _ctl_early}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.current.sm",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def import_dg():
    """DeepGEMM as vLLM imports it, with the headers the mode asks for. Returns (dg, description).
    MODE=stock: the image's headers, untouched (asserts they are not overlaid).
    MODE=fork, FORK_VIA=mount (default): the run script bind-mounted the overlay over the image's two headers; checked
        here against FORK (the overlay directory, mounted read-only).
    MODE=fork, FORK_VIA=copy: the include tree is copied to /tmp, the two headers from FORK laid over it, DEFINES
        ("NAME=VALUE,NAME2=VALUE2") prepended to the kernel header as #define lines, and the JIT pointed at the copy.
    STOCK_DEFINES: the same for the image's own kernel header (diagnostic variants of the stock kernel).
    VARIANT: a named text substitution in the kernel header of the copy (VARIANTS below; implies the copy)."""
    from vllm.utils.deep_gemm import _import_deep_gemm
    dg = _import_deep_gemm()
    mode = os.environ.get("MODE", "stock")
    via = os.environ.get("FORK_VIA", "mount")
    root = os.path.dirname(os.path.abspath(dg.__file__))
    fork = os.environ.get("FORK", "/overlay")
    defines = [d for d in os.environ.get("DEFINES" if mode == "fork" else "STOCK_DEFINES", "").split(",") if d]
    variant = os.environ.get("VARIANT", "")
    img = {rel: md5(os.path.join(root, rel)) for rel in (KERNEL_REL, SCHED_REL)}
    if mode == "stock" and not defines and not variant:
        ovl = os.path.join(fork, KERNEL_REL)
        assert not os.path.exists(ovl) or md5(ovl) != img[KERNEL_REL], "MODE=stock but the overlay is mounted over the image"
        desc = f"stock headers of the image (kernel md5 {img[KERNEL_REL][:8]}, scheduler {img[SCHED_REL][:8]})"
        if os.environ.get("SCHED_OVERLAY") == "1":     # run_fork_test.sh MODE=sched
            assert md5(os.path.join(fork, SCHED_REL)) == img[SCHED_REL], "MODE=sched: the scheduler is not the overlay's"
            desc = (f"the image's kernel header (md5 {img[KERNEL_REL][:8]}) with the overlay's scheduler mounted over the "
                    f"image's (md5 {img[SCHED_REL][:8]})")
        else:
            ovl_s = os.path.join(fork, SCHED_REL)
            assert not os.path.exists(ovl_s) or md5(ovl_s) != img[SCHED_REL], "MODE=stock but the overlay's scheduler is mounted"
    elif mode == "fork" and via == "mount" and not variant:
        assert not defines, "DEFINES needs FORK_VIA=copy"
        for rel in (KERNEL_REL, SCHED_REL):
            assert md5(os.path.join(fork, rel)) == img[rel], f"MODE=fork FORK_VIA=mount: {rel} is not the overlay's file"
        desc = (f"fork headers bind-mounted over the image's (kernel md5 {img[KERNEL_REL][:8]}, scheduler "
                f"{img[SCHED_REL][:8]})")
    else:
        copy = "/tmp/k31-fork-root"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(os.path.join(root, "include"), os.path.join(copy, "include"))
        if mode == "fork":
            for rel in (KERNEL_REL, SCHED_REL):
                shutil.copyfile(os.path.join(fork, rel), os.path.join(copy, rel))
        if defines:
            p = os.path.join(copy, KERNEL_REL)
            src = open(p).read()
            head = "".join(f"#define {d.split('=')[0]} {d.split('=', 1)[1] if '=' in d else ''}\n" for d in defines)
            open(p, "w").write(head + src)
        if variant:
            p = os.path.join(copy, KERNEL_REL)
            src = open(p).read()
            if variant in VARIANT_FUNCS:
                src = VARIANT_FUNCS[variant](src)
            for old, new in VARIANTS.get(variant, ()):
                assert src.count(old) == 1, f"VARIANT {variant}: {old!r} occurs {src.count(old)} times in the {mode} kernel header"
                src = src.replace(old, new)
            open(p, "w").write(src)
        dg._C.init(copy)
        desc = (f"{mode} headers in a copy of the include tree at {copy} (kernel md5 {md5(os.path.join(copy, KERNEL_REL))[:8]}, "
                f"scheduler {md5(os.path.join(copy, SCHED_REL))[:8]}; defines {defines or 'none'}; variant {variant or 'none'})")
    if "PDL" in os.environ:           # vLLM enables PDL on DeepGEMM when it serves (vllm/utils/deep_gemm.py, _lazy_init)
        dg.set_pdl(os.environ["PDL"] == "1")
    if int(os.environ.get("NSMS", "0")):
        dg.set_num_sms(int(os.environ["NSMS"]))
    return dg, desc + f"; pdl {dg.get_pdl()}; num_sms {dg.get_num_sms()}"


def init_group(port=29577):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group("nccl", rank=0, world_size=1)
    return dist.group.WORLD


def make_experts(n, gen, tmax, prefix, scale_lo=119, scale_hi=123, n_fill=None):
    """n routed experts in the kernel layout, through vLLM's code path: a `DeepGemmMegaMoEExperts` built as `KimiMoE`
    builds it (MXFP4, SiTU alpha 4 beta 25), filled by its `weight_loader` from synthetic checkpoint shards
    (`experts.{e}.{w1,w2,w3}.{weight_packed,weight_scale}`: uint8 FP4 pairs [rows, cols / 2] and uint8 UE8M0 scales
    [rows, cols / 32]) under the parameter names `load_weights` maps them to."""
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (MEGA_MOE_MXFP4, DeepGemmMegaMoEExperts,
                                                                         MegaMoEActivation)
    cfg = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=tmax),
                                compilation_config=types.SimpleNamespace(static_forward_context={}))
    with torch.device(dev):
        exp = DeepGemmMegaMoEExperts(cfg, num_experts=n, num_local_experts=n, experts_start_idx=0, top_k=K,
                                     hidden_size=H, intermediate_size=I,
                                     activation=MegaMoEActivation("situ", alpha=SITU_ALPHA, beta=SITU_BETA),
                                     weight_format=MEGA_MOE_MXFP4, max_num_tokens=tmax, prefix=prefix)
    params = dict(exp.named_parameters())
    for e in range(n if n_fill is None else n_fill):      # n_fill = 0: the caller loads the rows itself
        for shard, pp, rows, cols in (("w1", "w13", I, H), ("w3", "w13", I, H), ("w2", "w2", H, I)):
            w = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device=dev, generator=gen)
            s = torch.randint(scale_lo, scale_hi + 1, (rows, cols // 32), dtype=torch.uint8, device=dev, generator=gen)
            for suffix, t in (("weight", w), ("weight_scale", s)):
                p = params[f"{pp}_{suffix}"]
                p.weight_loader(p, t, f"layers.0.block_sparse_moe.experts.{pp}_{suffix}", expert_id=e, shard_id=shard)
    return exp


def bank(exp):
    """[w13_weight, w13_weight_scale, w2_weight, w2_weight_scale] as the launcher gets them"""
    return [exp.l1_weights[0].data, exp.l1_weights[1].data, exp.l2_weights[0].data, exp.l2_weights[1].data]


def row_bytes(t4):
    return sum(t[0].numel() * t.element_size() for t in t4)


_TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4", torch.float32: "<f4"}


class _Arr:
    def __init__(self, t, rows):
        self.__cuda_array_interface__ = {"shape": (rows,) + tuple(t.shape[1:]), "typestr": _TS[t.dtype],
                                         "data": (t.data_ptr(), False),
                                         "strides": tuple(s * t.element_size() for s in t.stride()), "version": 3}


def spanning(t, rows):
    """Derpseek's `exact.spanning`: a tensor over t's memory that claims `rows` rows (only t's own are backed)."""
    assert t.stride(0) == t[0].numel(), "rows must be dense"
    out = torch.as_tensor(_Arr(t, rows), device=t.device)
    return out if out.dtype == t.dtype else out.view(t.dtype)


def pinned_like(t, rows=None):
    """Pinned host copy of t's first `rows` rows with t's strides (the host rows of a bank)."""
    rows = t.shape[0] if rows is None else rows
    h = torch.empty_strided((rows,) + tuple(t.shape[1:]), t.stride(), dtype=t.dtype, pin_memory=True)
    h.copy_(t[:rows])
    return h


def make_ids(T, n_rows, nd, valid, gen, kind="cyc", lo=0):
    """[T, 16] int64 routing over rows lo..lo+n_rows-1. cyc: `valid` picks per row cycling over exactly
    `nd` distinct rows (distinct within a row), the other picks -1 (cut). rnd: every row draws `valid` distinct rows
    of a pool of `nd` at random and the picks sit at random slots. cutrow: cyc with the middle row cut entirely (a
    token whose picks are all cut). valid = 0: every pick cut (a call with no task at all)."""
    ids = torch.full((T, K), -1, dtype=torch.int64, device=dev)
    if valid == 0 or nd == 0:
        return ids
    pool = torch.randperm(n_rows, device=dev, generator=gen)[:nd] + lo
    if kind in ("cyc", "cutrow"):
        ids[:, :valid] = pool[(torch.arange(T, device=dev)[:, None] * valid + torch.arange(valid, device=dev)[None, :]) % nd]
        if kind == "cutrow":
            ids[T // 2] = -1
    else:
        for t in range(T):
            pick = pool[torch.randperm(nd, device=dev, generator=gen)[:valid]]
            slot = torch.randperm(K, device=dev, generator=gen)[:valid]
            ids[t, slot] = pick
    return ids


def make_inputs(T, ids, gen):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=gen)
    w = torch.softmax(torch.randn(T, K, device=dev, generator=gen), -1).float()
    return x, torch.where(ids >= 0, w, torch.zeros_like(w))


def _graph(fn, calls):
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(calls):
            fn()
    return gr


def _replays(gr, calls, reps):
    out = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        gr.replay()
        en.record()
        en.synchronize()
        out.append(st.elapsed_time(en) * 1000 / calls)
    out.sort()
    return out


def timeit(fn, reps=40, calls=None):
    """us per call: (min, median) over `reps` timed replays of a CUDA graph of back-to-back calls (the served form; the
    minimum filters the GPU's other tenants). Calls per replay: GRAPH_CALLS (10) while a replay stays under
    REPLAY_US (1500 us), fewer for longer calls (never under 2): a replay of 3-4 ms seldom escapes the other tenants'
    time slices, which added 80-90 us per call to 10 x 370 us replays here (the earlier port's "anomaly" at 76 and 128
    distinct experts) and nothing to 2 x 370 us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()                                  # JIT and allocations before the capture
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    if calls is None:
        calls = GRAPH_CALLS
        probe = _graph(fn, 2)
        probe.replay()
        torch.cuda.synchronize()
        est = _replays(probe, 2, 7)[0]
        calls = max(2, min(GRAPH_CALLS, int(REPLAY_US / max(est, 1.0))))
        del probe
    gr = _graph(fn, calls)
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    out = _replays(gr, calls, reps)
    return out[0], out[len(out) // 2]


def new_ctl(rows, d=0, lane=0, host=None, desc_tbl=None):
    """A control block for a call with `rows` kernel rows: int32 [HDR + rows] on the device. Returns (buffer, the view
    of `rows` words the launcher is given as cumulative_local_expert_recv_stats).
    words: 0 magic, 1 host rows D, 2-9 the four host base addresses (int64: L1 weights, L1 scales, L2 weights, L2
    scales), 10 host-lane clusters, 11 6-bit scale codes (bit 0), 12-13 the lane's claim counters (the kernel zeroes
    them), 14-15 the clone table's device address (int64), 16.. cumulative picks per kernel row."""
    buf = torch.zeros(HDR + rows, dtype=torch.int32, device=dev)
    buf[0] = MAGIC
    buf[1] = d
    buf[10] = lane
    if host is not None:
        buf[:HDR].view(torch.int64)[1:5] = torch.tensor([h.data_ptr() for h in host], dtype=torch.int64, device=dev)
    if desc_tbl is not None:
        assert desc_tbl.data_ptr() % 128 == 0
        buf[:HDR].view(torch.int64)[7] = desc_tbl.data_ptr()
    return buf, buf[:rows]


def bits(y):
    return y.view(torch.int16)


def same(a, b):
    return bool(torch.equal(bits(a), bits(b)))
