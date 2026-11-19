"""The copy-engine lane of the prompt path (k31_engine/celane.py, K31_PROMPT_CELANE) on the GB300: the rows it lands
must be byte-equal to the stager's, and the prompt path must hand MegaMoE the same rows with it as without it.

1. expand: n raw records in an HBM scratch -> rows row0 .. row0 + n - 1 of kernel-layout tensors, against the stager
   (stager.Stager) staging the same records from pinned memory into a second set of rows: all four tensors byte-equal
   (n = 1, 3, 16, 37, 128 at row offsets 0, 5, 91, 0), the other rows untouched, and record 0 against the reference
   expansion (store/build_store.expand). Real records of layer 1 if the record store is mounted at /hi2 (its
   layout.json must equal the layout computed here), random bytes otherwise.
2. copies: Lane.issue of contiguous and scattered slot lists on 1 and 2 copy streams: the scratch holds the bank's
   records byte for byte, one cudaMemcpyAsync per run of consecutive slots.
3. the prompt path: prompt_path.forward_prompt on a synthetic layer (896 experts, 105 native rows, a bank of 520
   records, 1,024 prompt rows), the MegaMoE call replaced by a fingerprint (sums of 8 KiB blocks of the four tensors'
   bytes) of the shared rows that call reads (its ids < 128; with K31_PROMPT_AHEAD the other half is being staged
   beside the call, by design), the row list and the ids. The plan runs with integer pick amplitudes: CUDA's
   index_add_ sums in no fixed order, and the study's amplitudes (multiples of 0.001) make records tie exactly, so with
   them the valve's order changes from run to run whatever the lane does. A control per block: two runs without the
   lane must agree. With the lane (shares 0.125 / 0.25 / 0.5 / 0.625 / 0.75 / 1, 1 and 2 copy streams, 1 and 2
   buffers, the expand beside the stager (XSIDE) or after it; rounds and half-rounds (K31_PROMPT_AHEAD, where XSIDE
   has no effect and must change nothing); all switched through the runtime flags as a running server's control file
   switches them, so the lane is rebuilt and released along the way) every call must see the same fingerprints, rows
   and ids as with the lane off, and y must be bit-equal; every failing setting is listed. Bank modes "all" (5 rounds,
   the slots in order: few copies) and "valve" (3 rounds, scattered slots: one copy per record). A scratch that does
   not fit (an impossible buffer count): the stager stages every record, the same calls as without the lane.
4. the in-server check (K31_PROMPT_CELANE_CHECK=1): check_rows finds 0 differing rows after a lane unit, and exactly 1
   after one byte of a lane row is flipped; forward_prompt with CHECK on runs and logs.
5. timing (informative, not a gate): one layer's forward_prompt with 4 full rounds and the fingerprint as the MegaMoE
   stand-in (~0.4 ms of HBM reads per call), the lane off against shares 0.125-0.5.
Needs ~7 GB of HBM and ~5 GB of pinned host memory; nothing else on the GB300. From the repository root:
    docker run --rm --name k31-celane-test --device nvidia.com/gpu=$GB --network none --ulimit memlock=-1 \
        --ulimit core=0 --cpuset-cpus 58-59 --user $(id -u):$(id -g) -e HOME=/tmp -e K31_PDL=1 \
        -v "$PWD/engine":/engine:ro -v "$PWD/build/torch_ext":/torch_ext -e TORCH_EXTENSIONS_DIR=/torch_ext \
        -v /models/k31-hi2:/hi2:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_celane.py
(mkdir -p build/torch_ext first; without the /hi2 mount part 1 uses random records). The first run compiles the
stager and the lane (~1 min each). Exit code 0 = PASS.
"""
import json
import mmap
import os
import random
import sys
import time
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
sys.path.insert(0, os.path.join(os.environ.get("ENGINE", "/engine"), "store"))
from build_store import expand, flat, kernel_rows  # noqa: E402
from k31_engine import celane  # noqa: E402
from k31_engine import flags  # noqa: E402
from k31_engine import prompt_path as pp  # noqa: E402
from k31_engine.stager import SLOTS, Stager  # noqa: E402

dev = torch.device("cuda")
torch.zeros(1, device=dev)
fails = []


def setflag(name, value=None):
    """A runtime flag as the server's control file sets it (None: back to the environment's value)."""
    if value is None:
        flags._over.pop(name, None)
    else:
        flags._over[name] = str(value)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------------------------------- layout and bank
mk = kernel_rows(dev)
probe = mk(1)
sizes = {k: flat(v).shape[1] for k, v in probe.items()}
codes_b = (sizes["w13"] + sizes["w2"]) // 2
sf_b = sizes["w13s"] + sizes["w2s"]
lay = {"record_bytes": (codes_b + sf_b + 4095) // 4096 * 4096, "codes_bytes": codes_b, "scale_bytes": sf_b,
       "native_bytes": sizes}
del probe
HI2 = os.environ.get("K31_HI2", "/hi2")
real = os.path.exists(os.path.join(HI2, "layout.json"))
if real:
    stored = json.load(open(os.path.join(HI2, "layout.json")))
    check("layout.json of the record store = the layout computed here",
          all(stored[k] == lay[k] for k in ("record_bytes", "codes_bytes", "scale_bytes", "native_bytes")))
R = lay["record_bytes"]
G = 520                                                   # bank records (part 3 needs > 4 rounds' worth)
buf = mmap.mmap(-1, G * R)
bank = torch.frombuffer(buf, dtype=torch.uint8).view(G, R)
gen = torch.Generator(device=dev).manual_seed(31)
for a in range(0, G, 64):                                 # random bytes, made on the GPU (fast), into the pinned bank
    b = min(G, a + 64)
    bank[a:b].copy_(torch.randint(0, 256, (b - a, R), generator=gen, dtype=torch.uint8, device=dev).cpu())
if real:                                                  # the first 48 slots: real records of layer 1
    fd = os.open(os.path.join(HI2, "L01.hi2"), os.O_RDONLY)
    for j in range(48):
        os.preadv(fd, [memoryview(bank[j].numpy())], (7 + 11 * j) * R)
    os.close(fd)
assert int(torch.cuda.cudart().cudaHostRegister(bank.data_ptr(), bank.numel(), 0x3)) == 0, "cudaHostRegister failed"
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor  # noqa: E402
bank_dev = get_accelerator_view_from_cpu_tensor(bank)
print(f"bank: {G} records of {R} B ({'48 real records of layer 1, ' if real else ''}random bytes otherwise), pinned; "
      f"device view at the host address: {bank_dev.data_ptr() == bank.data_ptr()}", flush=True)


def module():
    m = types.SimpleNamespace()
    rows = mk(128)
    m.w13_weight, m.w2_weight = rows["w13"], rows["w2"]
    m.w13_weight_scale, m.w2_weight_scale = rows["w13s"], rows["w2s"]
    return m


TENSORS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


def row_view(t):
    """[rows, bytes] uint8 view in memory order (rows are dense blocks whatever the inner strides)."""
    per = t[0].numel()
    return torch.as_strided(t, (t.shape[0], per), (per, 1)).view(torch.uint8)


def fill(m, v):
    for n in TENSORS:
        row_view(getattr(m, n)).fill_(v)


def stage_rows(stagers, rows_slots):
    """The stager: {row: bank slot} into the rows (one launch per slot group that holds any)."""
    for g in range(8):
        lst = torch.full((SLOTS,), -1, dtype=torch.int32)
        for r, s in rows_slots.items():
            if r // SLOTS == g:
                lst[r % SLOTS] = s
        if (lst >= 0).any():
            stagers[g](bank_dev.data_ptr(), lst.to(dev))


# ------------------------------------------------------------------------------------------- 1. expand
ma, mb = module(), module()
stg_b = [Stager(mb, SLOTS * g, lay) for g in range(8)]
lane = celane.Lane(1.0, lay, dev, lanes=1, bufs=1)
ok_all, ok_ref = True, True
for n, row0, s0 in ((1, 0, 0), (3, 5, 10), (16, 91, 20), (37, 91, 3), (128, 0, 200)):
    n = min(n, 128 - row0)
    slots = list(range(s0, s0 + n))
    fill(ma, 0x5A)
    fill(mb, 0x5A)
    lane.begin()
    lane.issue(0, bank.data_ptr(), slots)
    lane.land(0, ma, row0)
    stage_rows(stg_b, {row0 + i: s for i, s in enumerate(slots)})
    torch.cuda.synchronize()
    same = all(torch.equal(row_view(getattr(ma, t)), row_view(getattr(mb, t))) for t in TENSORS)
    outside = [r for r in range(128) if not row0 <= r < row0 + n]
    untouched = all(bool((row_view(getattr(ma, t))[outside] == 0x5A).all()) for t in TENSORS) if outside else True
    ok_all &= same and untouched
    rec = bank[slots[0]].to(dev).unsqueeze(0)
    c13 = sizes["w13"] // 2
    want = (expand(rec[:, :c13])[0], expand(rec[:, c13:codes_b])[0], rec[0, codes_b:codes_b + sizes["w13s"]],
            rec[0, codes_b + sizes["w13s"]:codes_b + sf_b])
    ok_ref &= all(torch.equal(row_view(getattr(ma, t))[row0], wv) for t, wv in zip(TENSORS, want))
    print(f"  {n} record(s) at row {row0}: lane rows {'=' if same else '!='} stager rows, other rows "
          f"{'untouched' if untouched else 'CHANGED'}", flush=True)
check("expand: rows byte-equal to the stager's, other rows untouched", ok_all)
check("expand: record 0 of each case = the reference expansion (build_store.expand) and its scale bytes", ok_ref)
del lane

# ------------------------------------------------------------------------------------------- 2. copies
ok, ncp = True, True
for lanes in (1, 2):
    ln = celane.Lane(1.0, lay, dev, lanes=lanes, bufs=2)
    for u, slots in enumerate(([40, 41, 42, 43], [7, 3, 4, 5, 100, 101, 9, 300, 301, 302, 519], list(range(64, 128)))):
        before = ln.n_copies
        ln.issue(u, bank.data_ptr(), slots)
        for s in ln.streams:
            s.synchronize()
        b = u % ln.bufs
        ok &= all(torch.equal(ln.scratch[b][i], bank_dev[s]) for i, s in enumerate(slots))
        want = sum(len(celane.runs(slots[a:e], R)) // 3 for a, e in celane.parts(len(slots), lanes))
        ncp &= ln.n_copies - before == want
        ln.copied.pop(u)
    del ln
check("copies: the scratch holds the bank's records byte for byte (1 and 2 copy streams, 2 buffers)", ok)
check("copies: one cudaMemcpyAsync per run of consecutive slots", ncp)
del ma, mb, stg_b
torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------------- 3. the prompt path
E, K, T, H = 896, 16, 1024, 105
rng = random.Random(3)
perm = list(range(E))
rng.shuffle(perm)
row_map = torch.full((E,), -1, dtype=torch.int32)
bank_map = torch.full((E,), -1, dtype=torch.int32)
order = list(range(G))
rng.shuffle(order)
for i, e in enumerate(perm[:H]):
    row_map[e] = 128 + i
for j, e in enumerate(perm[H:H + G]):
    bank_map[e] = order[j]
topk_ids = torch.tensor([rng.sample(range(E), K) for _ in range(T)], dtype=torch.int32, device=dev)
topk_w = torch.tensor([sorted((rng.random() for _ in range(K)), reverse=True) for _ in range(T)], device=dev)
x = torch.zeros(T, 8, device=dev)
mod = module()
st = types.SimpleNamespace(idx=1, mod=mod, row_map=row_map.to(dev), bank_map=bank_map.to(dev), band_map=None,
                           band_ok=True, bank_n=G, bank=list(range(G)), strong=3, stagers=None,
                           bank_ptr=bank_dev.data_ptr(), bank_cpu=bank, layout=lay, cut_mean=None)
BLK = 1024                                                # int64 words per fingerprint block (8 KiB)
pp.AMP = tuple(float(round(a * 1000)) for a in pp.AMP)    # integer amplitudes: exact record masses in any summation
pp._C.clear()                                             # order, so the valve's plan is the same in every run


def fingerprint(m):
    out = []
    for t in TENSORS:
        v = row_view(getattr(m, t))[:128].view(torch.int64)
        out.append(v.reshape(128, -1, BLK).sum(-1))
    return torch.cat(out, dim=1)


calls = []


def fake_moe(m, x_, w_, ids, shared):
    """MegaMoE's stand-in: the fingerprints of the shared rows this call reads (its ids < 128), those rows, its ids."""
    sel = torch.unique(ids[(ids >= 0) & (ids < 128)].long())
    calls.append((fingerprint(mod)[sel], sel, ids.clone()))
    return torch.zeros(x_.shape[0], 8, device=dev) + ids.float().sum(1, keepdim=True)


def forward(cfg, ahead=False, xside=False):
    """forward_prompt once -> (calls, y); cfg None = the lane off, else (share, lanes, bufs)."""
    calls.clear()
    fill(mod, 0)
    setflag("K31_PROMPT_AHEAD", int(ahead))
    setflag("K31_PROMPT_CELANE_XSIDE", int(xside))
    share, lanes, bufs = cfg if cfg is not None else (0, 1, 1)
    setflag("K31_PROMPT_CELANE", share)                 # the lane is built, rebuilt or released by forward_prompt
    setflag("K31_PROMPT_CELANE_LANES", lanes)
    setflag("K31_PROMPT_CELANE_BUFS", bufs)
    y = pp.forward_prompt(mod, st, x, topk_w, topk_ids, None, orig=fake_moe, res=None)
    torch.cuda.synchronize()
    if cfg is not None:
        ln = celane._LANE[0]
        assert (ln.share, ln.lanes, ln.bufs) == (share, lanes, bufs), "the lane was not rebuilt for the new flags"
    else:
        assert not celane._LANE, "the lane switched off was not released"
    return list(calls), y


def same(a, b):
    return len(a[0]) == len(b[0]) and all(all(torch.equal(u, v) for u, v in zip(p, q))
                                          for p, q in zip(a[0], b[0])) and torch.equal(a[1], b[1])


for mode in ("all", "valve"):
    setflag("K31_PROMPT_BANK", mode)
    for ahead in (False, True):
        ref = forward(None, ahead=ahead)
        check(f"control, bank {mode}, {'half-rounds' if ahead else 'rounds'}: two runs without the lane agree",
              same(forward(None, ahead=ahead), ref))
        bad = []
        n = 0
        for share in (0.125, 0.25, 0.5, 0.625, 0.75, 1.0):
            for lanes in (1, 2):
                for bufs in (1, 2):
                    for xside in (False, True):
                        got = forward((share, lanes, bufs), ahead=ahead, xside=xside)
                        n += 1
                        if not same(got, ref):
                            bad.append(f"share {share} lanes {lanes} bufs {bufs} xside {xside}")
        check(f"prompt path, bank {mode}, {'half-rounds' if ahead else 'rounds'}: {len(ref[0])} MegaMoE calls see "
              f"the same rows and ids with the lane ({n} configurations), y bit-equal", not bad,
              f"({len(bad)} failing: {'; '.join(bad)})" if bad else "")
setflag("K31_PROMPT_AHEAD")
setflag("K31_PROMPT_CELANE_XSIDE")
# no room for the lane's scratch (an impossible buffer count): the stager stages every record, the same calls
setflag("K31_PROMPT_BANK", "all")
ref = forward(None)
setflag("K31_PROMPT_CELANE", 0.25)
setflag("K31_PROMPT_CELANE_BUFS", 100000)
calls.clear()
fill(mod, 0)
y_oom = pp.forward_prompt(mod, st, x, topk_w, topk_ids, None, orig=fake_moe, res=None)
torch.cuda.synchronize()
check("no room for the lane's scratch: the stager stages every record, the same calls as without the lane",
      same((list(calls), y_oom), ref) and not celane._LANE and celane._FAILED[0] == (0.25, 1, 100000))
celane._FAILED[0] = None
setflag("K31_PROMPT_CELANE_BUFS")
setflag("K31_PROMPT_CELANE", 0)

# ------------------------------------------------------------------------------------------- 4. the check mode
setflag("K31_PROMPT_BANK")
st.stagers = [Stager(mod, SLOTS * g, lay) for g in range(8)]
fill(mod, 0)
ln = celane.Lane(0.5, lay, dev, lanes=2, bufs=1)
kept = torch.tensor(list(range(100, 228)), dtype=torch.long, device=dev)     # places 0..127 -> slots 100..227
h = 128 - celane.take(0.5, 128)
celane.stage_head(st, kept, 0, h, 0)
ln.begin()
ln.issue(0, bank.data_ptr(), kept[h:].tolist())
ln.land(0, mod, h)
torch.cuda.synchronize()
n0 = celane.check_rows(st, kept, 0, 0, h, 128)
mod.w2_weight.view(mod.w2_weight.shape[0], -1)[h + 3, 777] ^= 1             # one bit of one lane row
n1 = celane.check_rows(st, kept, 0, 0, h, 128)
n2 = celane.check_rows(st, kept, 0, 0, h, 128)                              # the stager restored that row
check("check_rows: 0 rows differ after a lane unit, 1 after one flipped bit, 0 once the stager restored it",
      (n0, n1, n2) == (0, 1, 0), f"({n0}, {n1}, {n2})")
setflag("K31_PROMPT_CELANE_CHECK", 1)
celane._CHECKED.clear()
try:
    got = forward((0.25, 1, 1))
    check("forward_prompt with K31_PROMPT_CELANE_CHECK on: runs, checked the first lane unit",
          celane._CHECKED.get(1) == 1)
except RuntimeError as e:
    check("forward_prompt with K31_PROMPT_CELANE_CHECK on", False, f"({e})")
setflag("K31_PROMPT_CELANE_CHECK")
del ln

# ------------------------------------------------------------------------------------------- 5. timing
st.bank_n, st.bank = 512, list(range(512))
bank_map4 = torch.where(bank_map >= 512, torch.full_like(bank_map, -1), bank_map)
st.bank_map = bank_map4.to(dev)                         # 512 records, nearly all hit: 4 full rounds
st.stagers = None


def stand_in(m, x_, w_, ids, shared):
    fingerprint(mod)                                    # ~2.25 GB of HBM reads: about a staged round's MegaMoE call
    return torch.zeros(x_.shape[0], 8, device=dev)


def timed(cfg, reps=5):
    forward(cfg)                                        # sets the flags, builds the lane, warms up
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pp.forward_prompt(mod, st, x, topk_w, topk_ids, None, orig=stand_in, res=None)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


calls_off = forward(None)[0]
print(f"timing, one layer, {len(calls_off)} rounds of the bank's records (MegaMoE stand-in: the fingerprint):",
      flush=True)
base = timed(None)
print(f"  stager alone: {base:.2f} ms", flush=True)
for share in (0.125, 0.25, 0.375, 0.5):
    for lanes, bufs in ((1, 1), (2, 1), (2, 2)):
        t = timed((share, lanes, bufs))
        print(f"  lane share {share}, {lanes} copy stream(s), {bufs} buffer(s): {t:.2f} ms ({base / t:.2f}x)",
              flush=True)
print(f"CELANE GPU: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(0 if not fails else 1)
