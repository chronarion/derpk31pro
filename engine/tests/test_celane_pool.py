"""The lane's pool with copies ahead of the plan (k31_engine/celane.py class Pool, K31_PROMPT_SPEC) on the GB300.

1. expand_map: records in scattered slots of the scratch -> scattered rows of kernel-layout tensors, against the
   stager staging the same records from pinned memory into the same rows of a second module: all four tensors
   byte-equal, every other row untouched (several tables, 1 to 128 records).
2. the prompt path over three synthetic layers (896 experts each, their own resident rows and bank maps over one
   pinned bank of 700 records), two forwards per run, the MegaMoE call replaced by a fingerprint of the shared rows
   it reads (as tests/test_celane.py). Runs: the lane off; the lane on; the lane with the pool (K31_PROMPT_SPEC=1) at
   1 / 2 / 3 buffers, 1 / 2 copy streams, the expand beside the stager or after it. Every call of every run must see
   the same fingerprints, rows and ids as with the lane off, and y must be bit-equal. Two row counts: 1,024 rows
   (every record hit: the copies made ahead are the first units' lane records) and 48 rows (about half the records
   hit: most copies made ahead are not needed and the missing ones are copied after the plan).
3. the same with a residency rewriting bank slots between layers the way residency.py does (unmap; new bytes into the
   pinned slot; map to another expert with the slot's generation raised): a record copied ahead of such a rewrite
   must not be used. The reference (lane off) sees the same rewrites at the same points. A control: with the pool's
   ownership check replaced by "accept everything" the run must differ from the reference (so the case does exercise
   the check).
4. the pool's counters: with every record hit and no rewrites, the first `bufs` units' lane records of every layer
   call but the first come from copies made ahead.
5. timing (informative, not a gate): a layer's rounds with a ~6 ms stand-in for the next layer's dense phase before
   the plan, the pool against the lane without it.
Needs ~9 GB of HBM and ~7 GB of pinned host memory; nothing else on the GB300. From the repository root:
    docker run --rm --name k31-celane-pool-test --device nvidia.com/gpu=$GB --network none --ulimit memlock=-1 \
        --ulimit core=0 --cpuset-cpus 58-59 --user $(id -u):$(id -g) -e HOME=/tmp -e K31_PDL=1 \
        -v "$PWD/engine":/engine:ro -v "$PWD/build/torch_ext":/torch_ext -e TORCH_EXTENSIONS_DIR=/torch_ext \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_celane_pool.py
The first run compiles the stager and the lane (~1 min each). Exit code 0 = PASS.
"""
import mmap
import os
import random
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
sys.path.insert(0, os.path.join(os.environ.get("ENGINE", "/engine"), "store"))
from build_store import flat, kernel_rows  # noqa: E402
from k31_engine import celane  # noqa: E402
from k31_engine import flags  # noqa: E402
from k31_engine import prompt_path as pp  # noqa: E402
from k31_engine.stager import SLOTS, Stager  # noqa: E402

dev = torch.device("cuda")
torch.zeros(1, device=dev)
fails = []
T0 = time.time()


def setflag(name, value=None):
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
R = lay["record_bytes"]
G = 700                                                   # bank records (six units)
buf = mmap.mmap(-1, G * R)
bank = torch.frombuffer(buf, dtype=torch.uint8).view(G, R)
gen = torch.Generator(device=dev).manual_seed(41)
for a in range(0, G, 64):
    b = min(G, a + 64)
    bank[a:b].copy_(torch.randint(0, 256, (b - a, R), generator=gen, dtype=torch.uint8, device=dev).cpu())
bank0 = None                                              # (the rewrites of part 3 restore from fresh random bytes)
assert int(torch.cuda.cudart().cudaHostRegister(bank.data_ptr(), bank.numel(), 0x3)) == 0, "cudaHostRegister failed"
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor  # noqa: E402
bank_dev = get_accelerator_view_from_cpu_tensor(bank)
print(f"bank: {G} records of {R} B, random bytes, pinned ({time.time() - T0:.0f} s)", flush=True)
TENSORS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


def module():
    m = types.SimpleNamespace()
    rows = mk(128)
    m.w13_weight, m.w2_weight = rows["w13"], rows["w2"]
    m.w13_weight_scale, m.w2_weight_scale = rows["w13s"], rows["w2s"]
    return m


def row_view(t):
    per = t[0].numel()
    return torch.as_strided(t, (t.shape[0], per), (per, 1)).view(torch.uint8)


def fill(m, v):
    for n in TENSORS:
        row_view(getattr(m, n)).fill_(v)


def stage_rows(stagers, rows_slots):
    for g in range(8):
        lst = torch.full((SLOTS,), -1, dtype=torch.int32)
        for r, s in rows_slots.items():
            if r // SLOTS == g:
                lst[r % SLOTS] = s
        if (lst >= 0).any():
            stagers[g](bank_dev.data_ptr(), lst.to(dev))


# ------------------------------------------------------------------------------------------- 1. expand_map
ma, mb = module(), module()
stg_b = [Stager(mb, SLOTS * g, lay) for g in range(8)]
lane = celane.Lane(0.75, lay, dev, lanes=2, bufs=2)
celane.set_mode(lane, True)
pool = lane.pool
rng = random.Random(5)
ok_all = True
for n in (1, 2, 17, 96, 128, 60):
    n = min(n, pool.N)
    slots = rng.sample(range(G), n)                       # bank slots
    ps = rng.sample(range(pool.N), n)                     # pool slots, scattered
    rows = rng.sample(range(128), n)                      # rows, scattered
    fill(ma, 0x5A)
    fill(mb, 0x5A)
    pool.live = set()
    pool._copy(bank.data_ptr(), list(zip(ps, slots)))
    pool.live = set(ps)
    pool.land(ma, list(zip(ps, rows)))
    stage_rows(stg_b, dict(zip(rows, slots)))
    torch.cuda.synchronize()
    same = all(torch.equal(row_view(getattr(ma, t)), row_view(getattr(mb, t))) for t in TENSORS)
    outside = [r for r in range(128) if r not in set(rows)]
    untouched = all(bool((row_view(getattr(ma, t))[outside] == 0x5A).all()) for t in TENSORS) if outside else True
    ok_all &= same and untouched and not pool.live
    print(f"  {n} record(s) from scattered pool slots into scattered rows: {'=' if same else '!='} the stager's rows, "
          f"other rows {'untouched' if untouched else 'CHANGED'}", flush=True)
check("expand_map: rows byte-equal to the stager's, other rows untouched", ok_all)
del lane, pool, ma, mb, stg_b
torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------------- 2-4. the prompt path
E, K, H = 896, 16, 60
NL = 3
BLK = 1024
pp.AMP = tuple(float(round(a * 1000)) for a in pp.AMP)
pp._C.clear()
mod = module()


class FakeRes:
    def __init__(self, idx, owners):
        self.idx = list(idx)
        self.bank_owner = np.stack(owners).astype(np.int64)
        self.bank_gen = np.zeros_like(self.bank_owner)


def make_layers(seed):
    """Three layers over the one bank: each maps its own G_l experts to the first G_l slots in a shuffled order."""
    r = random.Random(seed)
    layers, owners = {}, []
    for li, Gl in enumerate((700, 300, 520)):
        idx = li + 1
        perm = list(range(E))
        r.shuffle(perm)
        row_map = torch.full((E,), -1, dtype=torch.int32)
        bank_map = torch.full((E,), -1, dtype=torch.int32)
        order = list(range(Gl))
        r.shuffle(order)
        own = np.full(G, -1, dtype=np.int64)
        for i, e in enumerate(perm[:H]):
            row_map[e] = 128 + i
        for j, e in enumerate(perm[H:H + Gl]):
            bank_map[e] = order[j]
            own[order[j]] = e
        layers[idx] = types.SimpleNamespace(
            idx=idx, mod=mod, row_map=row_map.to(dev), bank_map=bank_map.to(dev), band_map=None, band_ok=True,
            bank_n=Gl, bank=list(range(Gl)), strong=3, stagers=None, bank_ptr=bank_dev.data_ptr(), bank_cpu=bank,
            layout=lay, cut_mean=None, spare=perm[H + Gl:])
        owners.append(own)
    return layers, FakeRes(sorted(layers), owners)


def fingerprint(m):
    out = []
    for t in TENSORS:
        v = row_view(getattr(m, t))[:128].view(torch.int64)
        out.append(v.reshape(128, -1, BLK).sum(-1))
    return torch.cat(out, dim=1)


calls = []


def fake_moe(m, x_, w_, ids, shared):
    sel = torch.unique(ids[(ids >= 0) & (ids < 128)].long())
    calls.append((fingerprint(mod)[sel], sel, ids.clone()))
    return torch.zeros(x_.shape[0], 8, device=dev) + ids.float().sum(1, keepdim=True)


def rewrite(layers, res, r, n):
    """n bank slots of random layers: unmap, new bytes, map to a spare expert with the generation raised (what a
    landed refill does; here the bytes are written by the host at once, after a device sync)."""
    torch.cuda.synchronize()
    for _ in range(n):
        idx = r.choice(sorted(layers))
        st = layers[idx]
        j = res.idx.index(idx)
        slot = r.randrange(st.bank_n)
        old = int(res.bank_owner[j, slot])
        if old < 0 or not st.spare:
            continue
        e = st.spare.pop(0)
        st.spare.append(old)
        st.bank_map[old] = -1
        res.bank_owner[j, slot] = -1
        bank[slot].copy_(torch.randint(0, 256, (R,), generator=torch.Generator().manual_seed(r.randrange(1 << 30)),
                                       dtype=torch.uint8))
        # (the three layers share one bank here, so the slot's bytes changed for the other layers too: their
        # generation of the slot is raised, as if their record had been refilled as well)
        for idx2, st2 in layers.items():
            if idx2 == idx:
                continue
            j2 = res.idx.index(idx2)
            if slot < st2.bank_n and res.bank_owner[j2, slot] >= 0:
                res.bank_gen[j2, slot] += 1
        st.bank_map[e] = slot
        res.bank_owner[j, slot] = e
        res.bank_gen[j, slot] += 1


def run(cfg, T, rewrites=0, seed=7, accept_all=False):
    """Two forwards over the three layers -> (calls, ys, pool counters). cfg None = the lane off, else
    (share, lanes, bufs, xside, spec)."""
    torch.cuda.synchronize()
    celane._LANE.clear()                                  # a fresh lane (and pool, and counters) for every run
    pp._LANE_ON[0] = False
    layers, res = make_layers(11)
    r = random.Random(seed)
    g2 = torch.Generator().manual_seed(seed)
    calls.clear()
    fill(mod, 0)
    setflag("K31_PROMPT_AHEAD", 0)
    setflag("K31_PROMPT_BANK", "all")
    share, lanes, bufs, xside, spec = cfg if cfg is not None else (0, 1, 1, False, False)
    setflag("K31_PROMPT_CELANE_XSIDE", int(xside))
    setflag("K31_PROMPT_CELANE", share)
    setflag("K31_PROMPT_CELANE_LANES", lanes)
    setflag("K31_PROMPT_CELANE_BUFS", bufs)
    setflag("K31_PROMPT_SPEC", int(spec))
    orig_claim = celane.Pool.claim
    if accept_all:
        def claim_all(self, st, resv):
            sp, self.spec = self.spec, None
            return {} if sp is None or sp["idx"] != st.idx else {b: p for p, b in enumerate(sp["slots"])}
        celane.Pool.claim = claim_all
    ys = []
    order = sorted(layers)
    try:
        for f in range(2):
            for li, idx in enumerate(order):
                st = layers[idx]
                if rewrites:
                    rewrite(layers, res, r, rewrites)
                ids = torch.stack([torch.randperm(E, generator=g2)[:K] for _ in range(T)]).to(torch.int32).to(dev)
                w = torch.rand(T, K, generator=g2).sort(dim=1, descending=True).values.to(dev)
                x = torch.zeros(T, 8, device=dev)
                nxt = layers[order[(li + 1) % NL]]
                ys.append(pp.forward_prompt(mod, st, x, w, ids, None, orig=fake_moe, res=None, resv=[res],
                                            next_st=nxt))
        torch.cuda.synchronize()
    finally:
        celane.Pool.claim = orig_claim
    stats = {}
    if cfg is not None and spec and celane._LANE and getattr(celane._LANE[0], "pool", None) is not None:
        stats = dict(celane._LANE[0].pool.n)
    out = (list(calls), ys, stats)
    # the bank's bytes back to their start, so every run begins from the same bank
    if rewrites:
        torch.cuda.synchronize()
        gen_ = torch.Generator(device=dev).manual_seed(41)
        for a in range(0, G, 64):
            b = min(G, a + 64)
            bank[a:b].copy_(torch.randint(0, 256, (b - a, R), generator=gen_, dtype=torch.uint8, device=dev).cpu())
    return out


def same(a, b):
    return len(a[0]) == len(b[0]) and all(all(torch.equal(u, v) for u, v in zip(p, q))
                                          for p, q in zip(a[0], b[0])) \
        and len(a[1]) == len(b[1]) and all(torch.equal(u, v) for u, v in zip(a[1], b[1]))


CFGS = [(0.75, 2, 2, True, True), (0.75, 1, 2, False, True), (0.75, 2, 1, True, True), (1.0, 1, 3, True, True),
        (0.5, 2, 3, False, True), (0.25, 2, 2, True, True), (1.0, 2, 1, True, True), (1.0, 2, 2, True, True)]
for T in (1024, 48):
    ref = run(None, T)
    ref2 = run(None, T)
    check(f"{T} rows: two runs without the lane agree ({len(ref[0])} MegaMoE calls)", same(ref, ref2))
    lane_on = run((0.75, 2, 2, True, False), T)
    check(f"{T} rows: the lane without the pool = the lane off", same(lane_on, ref))
    bad = []
    used = []
    for cfg in CFGS:
        got = run(cfg, T)
        if not same(got, ref):
            bad.append(str(cfg[:4]))
        used.append(f"{got[2].get('used', 0)}/{got[2].get('lane', 0)}")
    check(f"{T} rows: the pool (copies ahead of the plan) = the lane off, call by call and y bit-equal "
          f"({len(CFGS)} settings; lane records found in the pool: {', '.join(used)})", not bad,
          f"(differs at {'; '.join(bad)})" if bad else "")
    print(f"  ({time.time() - T0:.0f} s)", flush=True)

# 3. rewrites between layers
for T in (1024, 48):
    ref = run(None, T, rewrites=12)
    ref2 = run(None, T, rewrites=12)
    check(f"{T} rows, 12 bank slots rewritten before every layer: two runs without the lane agree", same(ref, ref2))
    bad, stale = [], []
    for cfg in CFGS[:4]:
        got = run(cfg, T, rewrites=12)
        if not same(got, ref):
            bad.append(str(cfg[:4]))
        stale.append(str(got[2].get("stale", 0)))
    check(f"{T} rows with rewrites: the pool = the lane off ({len(CFGS[:4])} settings; records copied ahead and "
          f"dropped: {', '.join(stale)})", not bad, f"(differs at {'; '.join(bad)})" if bad else "")
    if T == 1024:
        broken = run(CFGS[0], T, rewrites=12, accept_all=True)
        check("control: with the ownership check off the same run differs from the reference (stale bytes are "
              "seen)", not same(broken, ref))

# 4. counters
got = run((0.75, 2, 2, True, True), 1024)
layers_n = got[2].get("layers", 0)
exp = 0
for f in range(2):
    for li, Gl in enumerate((700, 300, 520)):
        if f == 0 and li == 0:
            continue
        m_left = Gl
        for u in range(2):
            m = max(0, min(128, m_left))
            exp += celane.take(0.75, m)
            m_left -= 128
check("every record hit, no rewrites: the first two units' lane records of every layer call but the first were "
      "copied ahead", layers_n == 6 and got[2].get("used", -1) == exp and got[2].get("stale", -1) == 0,
      f"(used {got[2].get('used')} of {got[2].get('lane')} lane records, expected {exp}; {layers_n} layer calls)")

# ------------------------------------------------------------------------------------------- 5. timing
try:
    layers, res = make_layers(11)
    T = 1024
    g2 = torch.Generator().manual_seed(3)
    ids = torch.stack([torch.randperm(E, generator=g2)[:K] for _ in range(T)]).to(torch.int32).to(dev)
    w = torch.rand(T, K, generator=g2).sort(dim=1, descending=True).values.to(dev)
    x = torch.zeros(T, 8, device=dev)
    busy = torch.randn(1024, 1024, device=dev)
    for _ in range(20):
        busy @ busy
    c0, c1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    c0.record()
    for _ in range(50):
        busy @ busy
    c1.record()
    torch.cuda.synchronize()
    per_ms = c0.elapsed_time(c1) / 50
    n_busy = max(1, round(6.0 / max(per_ms, 1e-3)))
    print(f"  dense stand-in: {n_busy} matrix products of {per_ms * 1e3:.0f} us each", flush=True)

    def dense_stand_in():
        """GPU work on the main stream for about 6 ms, standing in for a layer's dense phase."""
        for _ in range(n_busy):
            busy @ busy

    for name, cfg in (("lane, no pool", (0.75, 2, 2, True, False)), ("pool", (0.75, 2, 2, True, True)),
                      ("lane of whole units, no pool", (1.0, 2, 2, True, False)),
                      ("pool, whole units, 1 buffer", (1.0, 2, 1, True, True)),
                      ("pool, whole units, 2 buffers", (1.0, 2, 2, True, True)),
                      ("pool, share 0.75, 1 buffer", (0.75, 2, 1, True, True))):
        share, lanes, bufs, xside, spec = cfg
        setflag("K31_PROMPT_CELANE_XSIDE", int(xside))
        setflag("K31_PROMPT_CELANE", share)
        setflag("K31_PROMPT_CELANE_LANES", lanes)
        setflag("K31_PROMPT_CELANE_BUFS", bufs)
        setflag("K31_PROMPT_SPEC", int(spec))
        calls.clear()
        order = sorted(layers)
        ts = []
        for rep in range(6):
            for li, idx in enumerate(order):
                st = layers[idx]
                nxt = layers[order[(li + 1) % NL]]
                dense_stand_in()
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record()
                pp.forward_prompt(mod, st, x, w, ids, None, orig=fake_moe, res=None, resv=[res], next_st=nxt)
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()
                ts.append((idx, e0, e1))
            calls.clear()
        torch.cuda.synchronize()
        per = {}
        for idx, e0, e1 in ts[NL:]:                      # the first pass warms up
            per.setdefault(idx, []).append(e0.elapsed_time(e1))
        print(f"  timing, {name}: expert step ms per layer (bank 700 / 300 / 520 records, after a ~6 ms dense "
              f"stand-in): " + ", ".join(f"{sum(v) / len(v):.2f}" for _, v in sorted(per.items())), flush=True)
except Exception as ex:                                   # noqa: BLE001  (informative only)
    print(f"  timing skipped: {type(ex).__name__}: {ex}", flush=True)

print(f"CELANE POOL: {'PASS' if not fails else f'FAIL ({len(fails)} checks, first: ' + '; '.join(fails[:3]) + ')'} "
      f"({time.time() - T0:.0f} s)")
sys.exit(0 if not fails else 1)
