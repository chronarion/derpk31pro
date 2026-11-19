"""derpk31pro fork gate (a) and the native part of gate (d): the MegaMoE fork (overlay/megamoe_k31) against the
kimi-k31 image's stock SiTU kernel at Kimi K3.1's routed-expert geometry on the GB300. Ported from the earlier vLLM 0.32 port's microbenchmark.

One run is one arm: MODE=stock (the image's headers) or MODE=fork (the overlay's two headers over them). Both arms
build the same seeded experts, activations and routing; each dumps every case's output and `cmp_dumps.py` compares the
bytes (a fork run also compares with stock dumps already in DUMP). Within an arm the call is repeated in the forms the
engine uses and compared bit for bit with the plain call:
  none   no cumulative_local_expert_recv_stats (what vLLM's forward passes)
  zero   a zeroed int32 [E] stats tensor (stock semantics: cumulative picks per row at words 0..E-1; dumped too)
  ctl0   (fork only) Derpseek's control block with no host row: magic, D = 0, lane LANE; picks counted at words 16..
Call time: CUDA-graph replays of back-to-back calls (k31_fork_common.timeit: 10 calls per replay while a replay stays
under 1.5 ms, fewer above; min / median of 40), with and without vLLM's prepare kernel.

env: MODE, ARM (name of the arm in dumps and lines; default MODE), E (rows, <= 512), TMAX (64), CASES
("T:nd:valid[:rnd|cutrow],..."), DUMP (directory), LANE (4), PDL (0|1; unset: DeepGEMM's default), TIMING (1), and
k31_fork_common.import_dg's FORK, FORK_VIA, DEFINES / STOCK_DEFINES, VARIANT
usage: MODE=fork E=128 DUMP=/out/dump python3 mb_fork.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k31_fork_common as c  # noqa: E402

MODE = os.environ.get("MODE", "stock")
ARM = os.environ.get("ARM", MODE)        # the arm's name in dump files and output lines (variants of a mode)
DUMP = os.environ.get("DUMP", "")
E = int(os.environ.get("E", "128"))
TMAX = int(os.environ.get("TMAX", "64"))
LANE = int(os.environ.get("LANE", "4"))
TIMING = os.environ.get("TIMING", "1") == "1"
dev = c.dev
g = torch.Generator(device=dev).manual_seed(0)

print(f"device {torch.cuda.get_device_name(0)}; mode {MODE}; arm {ARM}; nvidia-smi before: {c.smi()}", flush=True)
dg, desc = c.import_dg()
print(f"headers: {desc}", flush=True)
group = c.init_group()
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs  # noqa: E402

t0 = time.time()
exp = c.make_experts(E, g, TMAX, "r5.mb.experts")
l1, l2 = exp.l1_weights, exp.l2_weights
kw = dict(exp._kernel_kwargs)
b4 = c.bank(exp)
row_bytes = c.row_bytes(b4)
print(f"{E} experts loaded through DeepGemmMegaMoEExperts.weight_loader in {time.time() - t0:.1f} s; "
      f"{row_bytes / 1e6:.2f} MB per expert as the kernel reads it; "
      + "; ".join(f"{n} {tuple(t.shape)} {t.dtype} strides {tuple(t.stride())} contiguous {t.is_contiguous()}"
                  for n, t in zip(("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"), b4))
      + f"; kernel kwargs {kw}; alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB (peak "
      f"{torch.cuda.max_memory_allocated() / 2**30:.1f})", flush=True)
symm = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, c.K, c.H, c.I, activation="situ", mma_type="fp8xfp4")
print(f"symmetric buffer {symm.buffer.numel() / 2**30:.2f} GiB for {symm.num_max_tokens_per_rank} tokens per rank", flush=True)


def mega(T, stats=None):
    y = torch.empty(T, c.H, dtype=torch.bfloat16, device=dev)
    if stats is None:
        dg.fp8_fp4_mega_moe(y, l1, l2, symm, fast_math=True, **kw)
    else:
        dg.fp8_fp4_mega_moe(y, l1, l2, symm, cumulative_local_expert_recv_stats=stats, fast_math=True, **kw)
    return y


# (T, distinct experts, valid picks per row, kind): the earlier port's cases (the K3 trace's distinct counts per row count, the
# calls that serve part of a 4-row step, all-distinct), then T = 16 and 64 and random routing with cut picks
CASES = [(1, 16, 16, "cyc"), (1, 8, 8, "cyc"), (2, 27, 16, "cyc"), (3, 36, 16, "cyc"), (4, 44, 16, "cyc"),
         (5, 52, 16, "cyc"), (8, 76, 16, "cyc"), (4, 14, 8, "cyc"), (4, 18, 8, "cyc"), (4, 24, 10, "cyc"),
         (4, 64, 16, "cyc"), (16, 128, 16, "cyc"), (1, 16, 13, "rnd"), (4, 30, 12, "rnd"), (16, 100, 11, "rnd"),
         (64, 128, 16, "cyc"), (64, min(E, 400), 16, "rnd"), (64, min(E, 512), 9, "rnd"),
         (4, 20, 16, "cutrow"), (1, 0, 0, "cyc"), (4, 0, 0, "cyc")]       # a token with every pick cut; calls with no task
only = os.environ.get("CASES")
if only:
    CASES = [(int(v[0]), int(v[1]), int(v[2]), v[3] if len(v) > 3 else "cyc") for v in (cs.split(":") for cs in only.split(","))]
fit = []
bad = 0
for T, nd, valid, kind in CASES:
    # routing and activations of a case come from the case's own seed: a dump is comparable across runs whatever CASES was
    gc = torch.Generator(device=dev).manual_seed(1000003 * T + 1009 * nd + 17 * valid + ("cyc", "rnd", "cutrow").index(kind))
    ids = c.make_ids(T, E, nd, valid, gc, kind)
    x, w = c.make_inputs(T, ids, gc)
    tag = f"E{E}-T{T}-nd{nd}-v{valid}" + ("" if kind == "cyc" else f"-{kind}")
    expect = torch.bincount(ids[ids >= 0].flatten(), minlength=E).to(torch.int32)

    def prep():
        prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])

    def both():
        prep()
        return mega(T)

    y = both()
    torch.cuda.synchronize()
    notes = []
    # zero: stock semantics of the stats argument
    stats = torch.zeros(E, dtype=torch.int32, device=dev)
    prep()
    yz = mega(T, stats)
    torch.cuda.synchronize()
    ok_z = c.same(y, yz) and bool(torch.equal(stats, expect))
    notes.append(f"zero-stats call {'BIT-EQUAL, counts right' if ok_z else 'DIFFERS'}")
    bad += not ok_z
    if MODE == "fork":
        buf, ctl = c.new_ctl(E, d=0, lane=LANE)
        prep()
        yc = mega(T, ctl)
        torch.cuda.synchronize()
        ok_c = (c.same(y, yc) and bool(torch.equal(buf[c.HDR:], expect)) and int(buf[12]) == 0 and int(buf[13]) == 0
                and int(buf[0]) == c.MAGIC and int(buf[1]) == 0)
        notes.append(f"ctl0 call {'BIT-EQUAL, counts at words 16.. right' if ok_c else 'DIFFERS'}")
        bad += not ok_c
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        inputs = {"ids": ids.cpu(), "x": x.cpu().view(torch.int16).clone(), "w": w.cpu()}
        torch.save({"y": y.cpu(), "stats": stats.cpu(), **inputs}, os.path.join(DUMP, f"{ARM}-{tag}.pt"))
        ref = os.path.join(DUMP, f"stock-{tag}.pt")
        if ARM != "stock" and os.path.exists(ref):
            r = torch.load(ref)
            if not all(torch.equal(r[k], v) for k, v in inputs.items()):
                notes.append("the stock dump has other inputs (NOT COMPARED)")
                bad += 1
            else:
                eq = bool(torch.equal(r["y"].view(torch.int16), y.cpu().view(torch.int16))) and bool(torch.equal(r["stats"], stats.cpu()))
                notes.append(f"against the stock dump {'BIT-EQUAL' if eq else 'DIFFERS'}")
                bad += not eq
    ok = bool(torch.isfinite(y.float()).all())
    nz = float((y.float() != 0).float().mean())
    cut_rows = (ids < 0).all(1)                       # a token with no pick gets zeros, every other token something
    ok_cut = bool((y[cut_rows].float() == 0).all()) and bool((y[~cut_rows].float() != 0).any(1).all())
    if not ok_cut:
        notes.append("rows of cut tokens WRONG")
        bad += 1
    bm = int(dg.get_block_m_for_mega_moe(1, E, symm.num_max_tokens_per_rank, T, c.K, "fp8xfp4"))
    line = (f"{ARM} E={E} T={T:2d} distinct {nd:3d} picks/row {valid:2d} {kind} (block_m {bm}, finite {ok}, nonzero {nz:.3f}): "
            + "; ".join(notes))
    if TIMING:
        tb = c.timeit(both)
        prep()
        tm = c.timeit(lambda: mega(T))
        tp = c.timeit(prep)
        mb = nd * row_bytes / 1e6
        line += (f" | prepare+mega {tb[0]:7.1f} us min / {tb[1]:7.1f} med | mega {tm[0]:7.1f} / {tm[1]:7.1f} | prepare "
                 f"{tp[0]:5.1f} / {tp[1]:5.1f} | {mb:6.0f} MB -> {mb / tm[0]:5.2f} TB/s, {tm[0] / max(nd, 1):5.2f} us per expert")
        if MODE == "fork":
            prep()
            tc = c.timeit(lambda: mega(T, ctl))
            line += f" | mega with ctl0 {tc[0]:7.1f} / {tc[1]:7.1f}"
        if kind == "cyc" and valid == 16 and T <= 5 and nd <= 52:      # the earlier port's five points
            fit.append((nd, tm[0]))
    print(line + f"; smi {c.smi()}", flush=True)
if len(fit) >= 2:
    n = len(fit)
    sx, sy = sum(a for a, _ in fit), sum(b for _, b in fit)
    sxx, sxy = sum(a * a for a, _ in fit), sum(a * b for a, b in fit)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    print(f"{ARM} E={E} fit over T = 1..5 with 16 picks per row ({', '.join(f'{a}: {b:.1f}' for a, b in fit)}): "
          f"{(sy - slope * sx) / n:.1f} us per call + {slope:.2f} us per distinct expert (mega, min)", flush=True)
print(f"done ({ARM}, E={E}): {'ALL EQUAL' if bad == 0 else f'{bad} MISMATCHES'} within this run; peak alloc "
      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; nvidia-smi after: {c.smi()}", flush=True)
sys.exit(1 if bad else 0)
