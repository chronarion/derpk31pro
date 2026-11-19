"""E1: do the GB300's copy engines add bandwidth beside the stager, and what does the prompt path's
copy-engine lane (k31_engine/celane.py, K31_PROMPT_CELANE) make of it per round?

Record bytes per second (9,289,728 B per record), each number the median of REPS (default 7) runs:
  A. the stager alone: a round of 128 records in 8 launches of 16, as prompt_path stages it (K31_STAGER_SMS SMs per
     launch, default 24; PDL as K31_PDL)
  B. the copy engines alone: 128 records into an HBM scratch: one stream with one copy or 128 copies; 2 and 4 streams
  C. E1: the stager on 128 - l records while 1 or 2 copy streams move l other records, l = 16 .. 96, all queued
     behind one start event: the wall, the total GB/s, each side's own end
  D. the link's total: the stager on 128 records and 1 or 2 copy streams on 128 others at once
  E. the expand kernel alone (raw records in HBM -> native rows): 16 / 32 / 64 / 128 records
  F. the lane as the prompt path runs it: celane.rounds over 4 rounds (512 records) against prompt_path's stager-only
     rounds, with a MegaMoE stand-in per round (HBM reads of the 128 rows on all SMs, ~0.4 ms; MOE_REPS repeats it)
     and without one; shares 0.125 - 1, 1 / 2 copy streams, 1 / 2 buffers, the expand after the stager or beside it
     (XSIDE); the rows after the last round compared with the stager's (fingerprints of 8 KiB blocks: a gate)
  G. the copies beside the MegaMoE stand-in: its slowdown with 1 and 2 copy streams running
The last lines are the summary: E1's answer (the best total GB/s of C / D against A) and the best lane setting of F
with its ms per round against the stager's, as the K31_PROMPT_CELANE_* values to try in the server.
GB300 only, with nothing else on the GPU and the host quiet (a measurement: no builds beside it); ~5 GB of HBM, ~3.6 GB
of pinned host memory, about two minutes after the first compile. From the repository root:
    mkdir -p build/torch_ext
    docker run --rm --name k31-celane-probe --device nvidia.com/gpu=$GB --network none --ulimit memlock=-1 \
        --ulimit core=0 --cpuset-cpus 58-59 --user $(id -u):$(id -g) -e HOME=/tmp -e K31_PDL=1 \
        -v "$PWD/engine":/engine:ro -v "$PWD/build/torch_ext":/torch_ext -e TORCH_EXTENSIONS_DIR=/torch_ext \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/bench_celane.py
Exit code 0 unless a correctness gate fails (F's rows differ from the stager's): 1.
"""
import mmap
import os
import statistics
import sys
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
sys.path.insert(0, os.path.join(os.environ.get("ENGINE", "/engine"), "store"))
from build_store import flat, kernel_rows  # noqa: E402
from k31_engine import celane  # noqa: E402
from k31_engine import flags  # noqa: E402
from k31_engine import prompt_path as pp  # noqa: E402
from k31_engine.stager import SLOTS, Stager  # noqa: E402

REPS = int(os.environ.get("REPS", "7"))
MOE_REPS = int(os.environ.get("MOE_REPS", "1"))
SMS = int(os.environ.get("K31_STAGER_SMS_EAGER", "0")) or int(os.environ.get("K31_STAGER_SMS", "24"))
dev = torch.device("cuda")
torch.zeros(1, device=dev)
main = torch.cuda.current_stream()
fails = []

mk = kernel_rows(dev)
probe = mk(1)
sizes = {k: flat(v).shape[1] for k, v in probe.items()}
codes_b = (sizes["w13"] + sizes["w2"]) // 2
sf_b = sizes["w13s"] + sizes["w2s"]
lay = {"record_bytes": (codes_b + sf_b + 4095) // 4096 * 4096, "codes_bytes": codes_b, "scale_bytes": sf_b,
       "native_bytes": sizes}
del probe
R = lay["record_bytes"]
ROW = sum(sizes.values())                                 # bytes of one native row
NB = 384
buf = mmap.mmap(-1, NB * R)
bank = torch.frombuffer(buf, dtype=torch.uint8).view(NB, R)
gen = torch.Generator(device=dev).manual_seed(1)
for a in range(0, NB, 64):
    n = min(64, NB - a)
    bank[a:a + n].copy_(torch.randint(0, 256, (n, R), generator=gen, dtype=torch.uint8, device=dev).cpu())
assert int(torch.cuda.cudart().cudaHostRegister(bank.data_ptr(), bank.numel(), 0x3)) == 0, "cudaHostRegister failed"
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor  # noqa: E402
bank_dev = get_accelerator_view_from_cpu_tensor(bank)
SRC = bank.data_ptr()
mod = types.SimpleNamespace()
rows = mk(128)
mod.w13_weight, mod.w2_weight = rows["w13"], rows["w2"]
mod.w13_weight_scale, mod.w2_weight_scale = rows["w13s"], rows["w2s"]
TENSORS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
stagers = [Stager(mod, SLOTS * g, lay, sms=SMS) for g in range(8)]
scratch = torch.empty((128, R), dtype=torch.uint8, device=dev)
M = celane._module()
props = torch.cuda.get_device_properties(dev)
GB = R / 1e9
print(f"{props.name}, {props.multi_processor_count} SMs, {M.attr(40)} async copy engines; record {R} B, native row "
      f"{ROW} B; stager {SMS} SMs per launch, K31_PDL={os.environ.get('K31_PDL', '0')}; median of {REPS}", flush=True)


def row_view(t):
    per = t[0].numel()
    return torch.as_strided(t, (t.shape[0], per), (per, 1)).view(torch.uint8)


def fingerprint():
    return torch.cat([row_view(getattr(mod, t))[:128].view(torch.int64).reshape(128, -1, 1024).sum(-1)
                      for t in TENSORS], dim=1)


def stand_in(*_):
    """The MegaMoE call's stand-in: HBM reads of the 128 rows (2.25 GB) on all SMs, MOE_REPS times."""
    for _ in range(MOE_REPS):
        for t in TENSORS:
            row_view(getattr(mod, t))[:128].view(torch.int64).sum()
    return torch.zeros(16, 8, device=dev)


def ev():
    e = torch.cuda.Event(enable_timing=True)
    return e


def gate():
    """A short spin on the main stream, then the start event: everything queued after it starts together."""
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000)
    e = ev()
    e.record(main)
    return e


def prep(slots):
    """Device stage lists of the 8 slot groups (rows 0 .. len - 1, -1 past the slots), each with "holds a slot";
    made before any timing."""
    lst = torch.full((SLOTS * 8,), -1, dtype=torch.int32)
    lst[:len(slots)] = torch.tensor(slots, dtype=torch.int32)
    lst = lst.to(dev)
    return [(lst[SLOTS * g:SLOTS * (g + 1)].contiguous(), g * SLOTS < len(slots)) for g in range(8)]


def stage(ls, all_groups=True):
    """The stager's launches: all 8 as prompt_path's rounds make them, or only the groups that hold a slot."""
    for g, (lst, holds) in enumerate(ls):
        if all_groups or holds:
            stagers[g](bank_dev.data_ptr(), lst)


streams = [torch.cuda.Stream(dev) for _ in range(4)]


def copies(slots, n_streams, start, per_record=False):
    """Copies of the slots into the scratch (rows 0 ..) on n_streams streams after the start event -> end events."""
    ends = []
    for k, (a, e) in enumerate(celane.parts(len(slots), n_streams)):
        if a == e:
            continue
        s = streams[k]
        s.wait_event(start)
        with torch.cuda.stream(s):
            if per_record:
                tr = []
                for i in range(a, e):
                    tr += [i * R, slots[i] * R, R]
            else:
                tr = celane.runs(slots[a:e], R, a * R)
            M.copy(scratch.data_ptr(), SRC, tr)
            x = ev()
            x.record(s)
        ends.append(x)
    return ends


def med(xs):
    return statistics.median(xs)


# ------------------------------------------------------------------------------------------------ A
lA = prep(list(range(128)))
stage(lA)
ts = []
for _ in range(REPS):
    s0 = gate()
    stage(lA)
    e = ev()
    e.record(main)
    e.synchronize()
    ts.append(s0.elapsed_time(e))
tA = med(ts)
bwA = 128 * GB / (tA * 1e-3)
print(f"A. stager alone, 128 records in 8 launches: {tA:.2f} ms, {bwA:.0f} GB/s", flush=True)

# ------------------------------------------------------------------------------------------------ B
print("B. copy engines alone, 128 records into HBM:", flush=True)
bwB = {}
for n_streams, per_record in ((1, False), (1, True), (2, False), (4, False)):
    ts = []
    for _ in range(REPS):
        s0 = gate()
        ends = copies(list(range(128, 256)), n_streams, s0, per_record)
        for x in ends:
            main.wait_event(x)
        e = ev()
        e.record(main)
        e.synchronize()
        ts.append(s0.elapsed_time(e))
    t = med(ts)
    bwB[(n_streams, per_record)] = 128 * GB / (t * 1e-3)
    print(f"  {n_streams} stream(s), {'one copy per record' if per_record else 'one copy per stream'}: {t:.2f} ms, "
          f"{bwB[(n_streams, per_record)]:.0f} GB/s", flush=True)

# ------------------------------------------------------------------------------------------------ C, D
print("C. E1: the stager on 128 - l records, the copy engines on l others, together:", flush=True)
best_c = (0.0, None)
for n_streams in (1, 2):
    for l in (16, 32, 48, 64, 80, 96):
        ls = prep(list(range(128 - l)))
        walls, ts_s, ts_c = [], [], []
        for _ in range(REPS):
            s0 = gate()
            stage(ls, all_groups=False)
            es = ev()
            es.record(main)
            ends = copies(list(range(128, 128 + l)), n_streams, s0)
            for x in ends:
                main.wait_event(x)
            e = ev()
            e.record(main)
            e.synchronize()
            walls.append(s0.elapsed_time(e))
            ts_s.append(s0.elapsed_time(es))
            ts_c.append(max(s0.elapsed_time(x) for x in ends))
        w, a, c = med(walls), med(ts_s), med(ts_c)
        bw = 128 * GB / (w * 1e-3)
        if bw > best_c[0]:
            best_c = (bw, (n_streams, l))
        print(f"  {n_streams} stream(s), l = {l:3d}: wall {w:.2f} ms = {bw:.0f} GB/s ({bw / bwA:.2f}x the stager "
              f"alone); stager part {a:.2f} ms ({(128 - l) * GB / (a * 1e-3):.0f} GB/s), copies {c:.2f} ms "
              f"({l * GB / (c * 1e-3):.0f} GB/s)", flush=True)
print("D. the stager on 128 records and the copy engines on 128 others, all at once:", flush=True)
best_d = 0.0
for n_streams in (1, 2):
    walls, ts_s, ts_c = [], [], []
    for _ in range(REPS):
        s0 = gate()
        stage(lA)
        es = ev()
        es.record(main)
        ends = copies(list(range(128, 256)), n_streams, s0)
        for x in ends:
            main.wait_event(x)
        e = ev()
        e.record(main)
        e.synchronize()
        walls.append(s0.elapsed_time(e))
        ts_s.append(s0.elapsed_time(es))
        ts_c.append(max(s0.elapsed_time(x) for x in ends))
    w, a, c = med(walls), med(ts_s), med(ts_c)
    best_d = max(best_d, 256 * GB / (w * 1e-3))
    print(f"  {n_streams} stream(s): wall {w:.2f} ms = {256 * GB / (w * 1e-3):.0f} GB/s; stager {a:.2f} ms "
          f"({128 * GB / (a * 1e-3):.0f} GB/s), copies {c:.2f} ms ({128 * GB / (c * 1e-3):.0f} GB/s)", flush=True)

# ------------------------------------------------------------------------------------------------ E
print("E. the expand kernel alone (HBM -> HBM):", flush=True)
lane = celane.Lane(1.0, lay, dev, lanes=1, bufs=1)
lane.begin()
lane.issue(0, SRC, list(range(128)))
lane.land(0, mod, 0)
for n in (16, 32, 64, 128):
    ts = []
    for _ in range(REPS):
        lane.copied[0] = ([], n)
        s0 = gate()
        lane.land(0, mod, 0)
        e = ev()
        e.record(main)
        e.synchronize()
        ts.append(s0.elapsed_time(e))
    t = med(ts)
    print(f"  {n} records: {t * 1e3:.0f} us ({n * (R + ROW) / (t * 1e-3) / 1e12:.2f} TB/s of reads + writes)",
          flush=True)
del lane
torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------------------ F
N_ROUNDS = 4
n_kept = 128 * N_ROUNDS
kept_h = [i % NB for i in range(n_kept)]
kept = torch.tensor(kept_h, dtype=torch.long, device=dev)
place = torch.full((16, 16), -1, dtype=torch.long, device=dev)
nat = torch.full((16, 16), -1, dtype=torch.long, device=dev)
x = torch.zeros(16, 8, device=dev)
w = torch.zeros(16, 16, device=dev)
st = types.SimpleNamespace(idx=1, mod=mod, stagers=stagers, bank_ptr=bank_dev.data_ptr(), bank_cpu=bank, layout=lay)


def no_moe(*_):
    return torch.zeros(16, 8, device=dev)


def by_stager(orig):
    y = None
    for r in range(N_ROUNDS):
        pp._stage_round(st, kept, r)
        ids = torch.where((place >= 0) & (place // 128 == r), place % 128, nat if r == 0 else torch.full_like(nat, -1))
        out = pp._megamoe(orig, mod, x, w, ids, None, torch.int32)
        y = out if y is None else y + out
    return y


def run(fn):
    ts = []
    for _ in range(REPS):
        s0 = gate()
        fn()
        e = ev()
        e.record(main)
        e.synchronize()
        ts.append(s0.elapsed_time(e))
    return med(ts) / N_ROUNDS


ok_rows = True
res = {}
for moe_name, orig in (("stand-in", stand_in), ("none", no_moe)):
    by_stager(orig)
    torch.cuda.synchronize()
    fp_ref = fingerprint()
    t0 = run(lambda: by_stager(orig))
    print(f"F. 4 rounds of 128 records, MegaMoE {moe_name}: stager alone {t0:.2f} ms per round", flush=True)
    res[(moe_name, None)] = t0
    for xside in (False, True):
        flags._over["K31_PROMPT_CELANE_XSIDE"] = str(int(xside))      # a runtime flag, read by celane.rounds
        for lanes in (1, 2):
            for bufs in (1, 2):
                for share in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0):
                    ln = celane.Lane(share, lay, dev, lanes=lanes, bufs=bufs)

                    def go():
                        return celane.rounds(orig, mod, st, x, w, None, kept, kept_h, place, nat, n_kept, torch.int32,
                                             share, lane=ln)
                    go()
                    torch.cuda.synchronize()
                    same = torch.equal(fingerprint(), fp_ref)
                    ok_rows &= same
                    t = run(go)
                    res[(moe_name, (share, lanes, bufs, xside))] = t
                    print(f"  share {share:5.3f}, {lanes} stream(s), {bufs} buffer(s){', XSIDE' if xside else '':7s}: "
                          f"{t:.2f} ms per round ({t0 / t:.2f}x){'' if same else '  ROWS DIFFER'}", flush=True)
                    del ln
                    torch.cuda.empty_cache()
flags._over.pop("K31_PROMPT_CELANE_XSIDE", None)
if not ok_rows:
    fails.append("F: lane rows differ from the stager's")

# ------------------------------------------------------------------------------------------------ G
print("G. the MegaMoE stand-in beside the copy engines:", flush=True)


def time_moe(k=5):
    ts = []
    for _ in range(k):
        a, b = ev(), ev()
        a.record(main)
        stand_in()
        b.record(main)
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return med(ts)


alone = time_moe()
print(f"  alone: {alone:.3f} ms", flush=True)
for n_streams in (1, 2):
    s0 = gate()
    ends = []
    for _ in range(3):                                    # ~3 x 128 records of copies keep the engines busy
        ends += copies(list(range(128, 256)), n_streams, s0)
    beside = time_moe()
    for x_ in ends:
        x_.synchronize()
    print(f"  with {n_streams} copy stream(s) running: {beside:.3f} ms ({100 * (beside / alone - 1):+.1f}%)",
          flush=True)

# ------------------------------------------------------------------------------------------------ summary
print("SUMMARY", flush=True)
bw1, bw2 = bwB[(1, False)], bwB[(2, False)]
print(f"  E1: stager alone {bwA:.0f} GB/s; copy engines alone {bw1:.0f} (1 stream) / {bw2:.0f} (2 streams); stager + "
      f"copies {best_c[0]:.0f} GB/s at its best split (streams, l = {best_c[1]}), {best_c[0] / bwA:.2f}x; both full "
      f"{best_d:.0f} GB/s", flush=True)
for moe_name in ("stand-in", "none"):
    t0 = res[(moe_name, None)]
    cfg, t = min(((k[1], v) for k, v in res.items() if k[0] == moe_name and k[1] is not None), key=lambda kv: kv[1])
    share, lanes, bufs, xside = cfg
    print(f"  lane, MegaMoE {moe_name}: best share {share}, {lanes} stream(s), {bufs} buffer(s)"
          f"{', XSIDE' if xside else ''}: {t:.2f} ms per round against {t0:.2f} ({t0 / t:.2f}x)", flush=True)
    if moe_name == "stand-in":
        if t0 / t >= 1.02:
            print(f"  try: K31_PROMPT_CELANE={share} K31_PROMPT_CELANE_LANES={lanes} K31_PROMPT_CELANE_BUFS={bufs}"
                  + (" K31_PROMPT_CELANE_XSIDE=1" if xside else ""), flush=True)
        else:
            print("  the lane does not pay here (< 2% per round): leave K31_PROMPT_CELANE off", flush=True)
print(f"BENCH CELANE: {'rows equal' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(0 if not fails else 1)
