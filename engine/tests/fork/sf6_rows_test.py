"""K31_SF6 on the GB300: MegaMoE on rows with 6-bit scale codes, set up as the engine sets them up, against the same
rows uncoded.

One bank of E experts is loaded through vLLM's DeepGemmMegaMoEExperts. Its coded twin shares the weight tensors and
has 6-bit scale banks, attached with the engine's own k31_engine/sf6rows.attach (no host rows, the fork's control
block in cumulative_local_expert_recv_stats, the clone table with the coded maps):
  1. coded == uncoded, bit for bit: T = 1, 4, 16, 64; routing over few and many rows, cut picks, a token with every
     pick cut, a call with no pick at all; the picks per row land at words 16.. of the control block
  2. rows written by the engine's writers: 2-bit records (hi2-k1 and hi2-k2, cut out of the bank's own rows) staged
     into rows of both banks (stager.py into int32 scales, stager2.py into 6-bit scales): the calls that use those
     rows agree bit for bit; then promotion merges put the native rows back (the coded bank is again the coding of
     the uncoded one) and an extract from a coded row gives the uncoded row's record
  3. call time, uncoded against coded, as CUDA-graph replays (k31_fork_common.timeit): min / median us per call
env: E (128), TMAX (64), TS_EXACT (1,4,16,64), TIMING (1), TS_TIME (1:16,2:27,4:44,4:64,16:96: rows:distinct experts
of the timed calls), PDL, NSMS; k31_fork_common.import_dg's MODE (fork) etc. ENGINE (/engine): the engine tree.
Prompt-sized calls: TMAX=4096 TS_EXACT=256,1024,4096 TS_TIME=1024:128,4096:128. Run by tests/fork/run_sf6_rows.sh.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k31_fork_common as c  # noqa: E402

ENGINE = os.environ.get("ENGINE", "/engine")
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(ENGINE, "tests"))
os.environ.setdefault("K31_ENGINE_DIR", ENGINE)
from k31_engine import recfmt, sf6, sf6rows, stager, stager2  # noqa: E402
from k2_testlib import LAYOUT_K1, LAYOUT_K2  # noqa: E402

E = int(os.environ.get("E", "128"))
TMAX = int(os.environ.get("TMAX", "64"))
TIMING = os.environ.get("TIMING", "1") == "1"
dev = c.dev
g = torch.Generator(device=dev).manual_seed(0)
assert os.environ.get("MODE", "fork") == "fork", "6-bit scale codes exist in the fork only"

dg, desc = c.import_dg()
sf6rows.check_fork(os.path.dirname(os.path.abspath(dg.__file__)) if os.environ.get("FORK_VIA", "mount") == "mount"
                   else "/tmp/k31-fork-root")
print(f"device {torch.cuda.get_device_name(0)}; headers: {desc}; nvidia-smi before: {c.smi()}", flush=True)
group = c.init_group(29593)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs  # noqa: E402

exp = c.make_experts(E, g, TMAX, "sf6rows.experts")
b4 = c.bank(exp)                                    # w13 codes, w13 scales, w2 codes, w2 scales (uncoded)
symm = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, c.K, c.H, c.I, activation="situ", mma_type="fp8xfp4")


class Rows:
    """What the engine's row writers take: the four kernel tensors of a module."""


un = Rows()                                         # the uncoded bank
un.w13_weight, un.w13_weight_scale, un.w2_weight, un.w2_weight_scale = b4
tw = Rows()                                         # its coded twin: the same weight tensors, 6-bit scale banks
tw.w13_weight, tw.w13_weight_scale, tw.w2_weight, tw.w2_weight_scale = b4
tw._kernel_kwargs = dict(exp._kernel_kwargs)
c13, c2 = sf6.encode(b4[1]), sf6.encode(b4[3])
assert torch.equal(sf6.decode(c13[:2], b4[1].shape[1], b4[1].shape[2]), b4[1][:2]), "sf6 reference decode"
sf6rows.attach(tw, c13, c2)
buf = tw._k31_sf6_ctl[0]
print(f"{E} rows; scales per row {c13.shape[1] + c2.shape[1]} B coded, {(b4[1][0].numel() + b4[3][0].numel()) * 4} "
      f"uncoded; launcher scale tensors of the coded call: {tuple(tw.w13_weight_scale.shape)} strides "
      f"{tuple(tw.w13_weight_scale.stride())}; control block {buf.numel()} words", flush=True)


def call(x, ids, w, coded, T):
    prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, c.H, dtype=torch.bfloat16, device=dev)
    if coded:
        dg.fp8_fp4_mega_moe(y, (tw.w13_weight, tw.w13_weight_scale), (tw.w2_weight, tw.w2_weight_scale), symm,
                            fast_math=True, **tw._kernel_kwargs)
    else:
        dg.fp8_fp4_mega_moe(y, (b4[0], b4[1]), (b4[2], b4[3]), symm, fast_math=True, **exp._kernel_kwargs)
    return y


bad = 0
# ---- 1. coded == uncoded
cases = []
for T in (int(t) for t in os.environ.get("TS_EXACT", "1,4,16,64").split(",")):
    nd = min(E, 8 * T if T <= 4 else 96)
    cases.append((T, c.make_ids(T, E, nd, 16, g, "cyc"), f"{nd} experts, 16 picks per token"))
    cases.append((T, c.make_ids(T, E, min(E, 16 * T), 16, g, "rnd"), "random over all rows"))
    cases.append((T, c.make_ids(T, E, min(E, 16 * T), 11, g, "rnd"), "random over all rows, 5 picks cut"))
    if T >= 4:
        cases.append((T, c.make_ids(T, E, min(E, 8 * T), 16, g, "cutrow"), "one token with every pick cut"))
cases.append((4, c.make_ids(4, E, 0, 0, g), "no pick at all"))
for T, ids, what in cases:
    w_raw = torch.softmax(torch.randn(T, c.K, device=dev, generator=g), -1).float()
    x = torch.randn(T, c.H, dtype=torch.bfloat16, device=dev, generator=g)
    w = torch.where(ids >= 0, w_raw, torch.zeros_like(w_raw))
    ya = call(x, ids, w, False, T)
    buf[sf6rows.HDR:].zero_()
    yb = call(x, ids, w, True, T)
    torch.cuda.synchronize()
    eq = c.same(ya, yb)
    counts = bool(torch.equal(buf[sf6rows.HDR:], torch.bincount(ids[ids >= 0].flatten(), minlength=E).to(torch.int32)))
    fin = bool(torch.isfinite(ya.float()).all())
    bad += (not eq) + (not counts) + (not fin)
    print(f"T={T:2d} {what}: coded against uncoded {'BIT-EQUAL' if eq else 'DIFFERS'}; counts at words 16.. "
          f"{'right' if counts else 'WRONG'}; finite {fin}", flush=True)
print(f"1. coded rows: {'ALL EQUAL' if bad == 0 else f'{bad} FAILURES'} ({len(cases)} cases)", flush=True)

# ---- 2. rows written by the engine's writers
bad2 = 0
assert E >= 100, "the test stages into rows 64..79 and reads rows up to 99"
src_rows, dst_rows = [5, 17, 40, 99], [64, 65, 70, 79]          # records of rows src are staged into rows dst (slots)
keep = [[t[r].clone() for t in b4] for r in dst_rows]            # the uncoded rows that staging will overwrite
keep6 = [(c13[r].clone(), c2[r].clone()) for r in dst_rows]
hi = torch.empty((len(src_rows), LAYOUT_K1["record_bytes"]), dtype=torch.uint8, device=dev)
lo = torch.empty_like(hi)
for j, r in enumerate(src_rows):
    stager.extract_record(un, r, LAYOUT_K1, hi[j])
    stager.extract_record(un, r, LAYOUT_K1, lo[j], low=True)
hi_k2, fits = recfmt.k1_to_k2(hi, LAYOUT_K2)
assert bool(fits.all())
hi_k2 = hi_k2.contiguous()
SL = stager2.SLOTS
for name, lay, bank in (("hi2-k1", LAYOUT_K1, hi), ("hi2-k2", LAYOUT_K2, hi_k2)):
    idx = torch.full((SL,), -1, dtype=torch.int32, device=dev)
    for j, d in enumerate(dst_rows):
        idx[d - 64] = j
    stager.Stager(un, 64, lay)(bank.data_ptr(), idx)             # into int32 scales (stager.py, or stager2.py for k2)
    st6 = stager.Stager(tw, 64, lay)
    assert isinstance(st6, stager2.Stager2)
    st6(bank.data_ptr(), idx)                                    # the same records into the coded bank's rows
    torch.cuda.synchronize()
    rows_ok = all(torch.equal(c13[d], sf6.encode(b4[1][d:d + 1])[0]) and torch.equal(c2[d], sf6.encode(b4[3][d:d + 1])[0])
                  for d in dst_rows)
    nbad = 0
    for T in (1, 4, 16):
        ids = c.make_ids(T, 64, max(16, min(64, 8 * T)), 16, g, "rnd")     # rows below the staged ones, then those
        ids[:, :len(dst_rows)] = torch.tensor(dst_rows, device=dev)[torch.randperm(len(dst_rows), device=dev, generator=g)]
        w_raw = torch.softmax(torch.randn(T, c.K, device=dev, generator=g), -1).float()
        x = torch.randn(T, c.H, dtype=torch.bfloat16, device=dev, generator=g)
        ya, yb = call(x, ids, w_raw, False, T), call(x, ids, w_raw, True, T)
        torch.cuda.synchronize()
        nbad += not c.same(ya, yb)
    bad2 += (not rows_ok) + nbad
    print(f"2. {name} records staged into rows {dst_rows} of both banks: the coded rows are the coding of the uncoded "
          f"ones: {rows_ok}; calls on them coded against uncoded: {'BIT-EQUAL' if nbad == 0 else f'{nbad} DIFFER'}", flush=True)
# promotion merges put the native rows of src back into dst (both banks), then dst rows must equal src rows
for j, (r, d) in enumerate(zip(src_rows, dst_rows)):
    stager.merge(hi_k2[j].data_ptr(), lo[j].data_ptr(), un, d, LAYOUT_K2)
    stager.merge(hi_k2[j].data_ptr(), lo[j].data_ptr(), tw, d, LAYOUT_K2)
torch.cuda.synchronize()
merged = all(torch.equal(b4[i][d], b4[i][r]) for r, d in zip(src_rows, dst_rows) for i in range(4)) and \
    all(torch.equal(c13[d], c13[r]) and torch.equal(c2[d], c2[r]) for r, d in zip(src_rows, dst_rows))
out6 = torch.empty(LAYOUT_K2["record_bytes"], dtype=torch.uint8, device=dev)
stager.extract_hi2(tw, dst_rows[0], LAYOUT_K2, out6, blocks=1216)
torch.cuda.synchronize()
extracted = torch.equal(out6, hi_k2[0])
bad2 += (not merged) + (not extracted)
print(f"2. merges of hi2-k2 + lo2 into both banks give the source rows back: {merged}; the record extracted from a "
      f"coded row = the uncoded row's hi2-k2 record: {extracted}", flush=True)
for (r, saved), (s13, s2) in zip(zip(dst_rows, keep), keep6):  # restore the bank
    for t, v in zip(b4, saved):
        t[r].copy_(v)
    c13[r].copy_(s13); c2[r].copy_(s2)
print(f"2. engine writers: {'ALL RIGHT' if bad2 == 0 else f'{bad2} FAILURES'}", flush=True)
bad += bad2

# ---- 3. call time
if TIMING:
    for T, nd in (tuple(int(v) for v in p.split(":")) for p in os.environ.get("TS_TIME", "1:16,2:27,4:44,4:64,16:96").split(",")):
        ids = c.make_ids(T, E, nd, 16, g, "cyc")
        x = torch.randn(T, c.H, dtype=torch.bfloat16, device=dev, generator=g)
        w = torch.softmax(torch.randn(T, c.K, device=dev, generator=g), -1).float()
        eq = c.same(call(x, ids, w, False, T), call(x, ids, w, True, T))
        bad += not eq
        res = {}
        for rnd in range(2):                                     # interleaved
            for coded in (False, True):
                t = c.timeit(lambda: call(x, ids, w, coded, T), reps=30)
                res[coded] = min(res.get(coded, (1e9, 1e9)), t)
        print(f"3. T={T} {nd} distinct experts: uncoded {res[False][0]:6.1f}/{res[False][1]:6.1f} us, 6-bit rows "
              f"{res[True][0]:6.1f}/{res[True][1]:6.1f} us ({res[True][0] - res[False][0]:+.1f} us min)"
              f"{'' if eq else ' DIFFERS'}", flush=True)
print(f"done: {'ALL EQUAL' if bad == 0 else f'{bad} FAILURES'}; peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; "
      f"nvidia-smi after: {c.smi()}", flush=True)
sys.exit(0 if bad == 0 else 1)
