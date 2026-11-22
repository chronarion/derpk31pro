"""The lane's pool with copies ahead of the plan (k31_engine/celane.py class Pool, K31_PROMPT_SPEC), CPU only.

The real Lane and Pool run on simulated streams (test_celane_cpu.py's Sim): copies, expands, stager launches and
MegaMoE calls are queued per stream with their event records and waits and executed in random legal orders, so a
missing wait or a race shows up. A sequence of prompt forwards crosses several MoE layers, each with its own bank;
after a layer's rounds the pool copies the next layer's first lane records ahead. A fake residency rewrites bank
slots the way residency.py does: at a layer's start (a tick) it may unmap slots (owner -1, the expert's picks no
longer reach the slot), a writer stream then replaces the slot's bytes once the main stream has passed the unmap, and
a later tick maps the slot to its new expert (owner set, generation + 1).

1. pair_runs against a brute-force expansion.
2. With the pool on, every MegaMoE call must read rows that hold the bytes the bank holds for its picks' experts (as
   of the layer's plan), and the calls (ids, rows read) must equal those of the lane without the pool on the same
   sequence of forwards and rewrites: same rows, same rounds. Varied: buffers 1-3, copy streams 1-2, shares, the
   expand beside the stager or after it, row counts from a few rows (few records hit) to enough for every record,
   bank sizes from under one unit to six units, with and without a residency, 6 seeds of stream order each.
3. The same with the ownership check switched off (Pool.claim accepting every record copied ahead): the cases with
   rewrites must FAIL (a call reads stale bytes), which shows that test 2 can see the hazard the check guards.
4. Speculation is used: with every record hit and no rewrites, the lane records of the first `bufs` units of every
   layer but the first come from copies made ahead of the plan.
Run in the engine image, CPU only (from the repository root):
    docker run --rm --network none --cpuset-cpus 58 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD/engine":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_celane_pool_cpu.py
Exit 0 and "CELANE POOL CPU: PASS" = all cases agree.
"""
import os
import random
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_celane_cpu as base  # noqa: E402  (Sim, SimStream, SimEvent, setflag; it also patches torch.cuda)
from k31_engine import celane  # noqa: E402
from k31_engine import prompt_path as pp  # noqa: E402

E, K, R = 896, 16, 128
REC = base.SIM_REC
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------------------------------- 1. pair_runs
def part1(rng):
    ok = True
    for _ in range(2000):
        n = rng.randrange(0, 40)
        ps = sorted(rng.sample(range(300), n)) if rng.random() < 0.5 else list(range(7, 7 + n))
        bs = sorted(rng.sample(range(700), n)) if rng.random() < 0.5 else list(range(40, 40 + n))
        pairs = list(zip(ps, bs))
        tr = celane.pair_runs(pairs, 16)
        got = []
        for i in range(0, len(tr), 3):
            d, s_, nb = tr[i:i + 3]
            ok &= nb > 0 and nb % 16 == 0
            got += [(d // 16 + k, s_ // 16 + k) for k in range(nb // 16)]
        ok &= got == pairs
        brk = sum(1 for i in range(1, n) if ps[i] != ps[i - 1] + 1 or bs[i] != bs[i - 1] + 1)
        ok &= len(tr) // 3 == (brk + 1 if n else 0)
    check("pair_runs: every pair once, in order, one copy per run of consecutive pool and bank slots", ok)


# ------------------------------------------------------------------------------------------- the simulation
class FakeRes:
    """The fields of residency.Residency the pool reads: idx, bank_owner, bank_gen."""

    def __init__(self, layers):
        self.idx = sorted(layers)
        G = max(len(st.bank) for st in layers.values())
        self.bank_owner = np.full((len(self.idx), G), -1, dtype=np.int64)
        self.bank_gen = np.zeros((len(self.idx), G), dtype=np.int64)


class World:
    """Layers with their banks, the bank's bytes (one content id per slot), the shared rows and the scratch."""

    def __init__(self, rng, sizes, H):
        self.rng = rng
        self.layers = {}
        self.content = {}                # (layer, slot) -> content id of the bytes the bank holds
        self.next_id = [1000]
        self.rows = torch.full((R,), -1, dtype=torch.int64)
        self.scratch = {}
        self.pending = []                # rewrites whose bytes have been queued: (layer, slot, expert, event, tick)
        self.n_tick = 0
        for li, G in enumerate(sizes):
            idx = li + 1
            perm = list(range(E))
            rng.shuffle(perm)
            row_map = torch.full((E,), -1, dtype=torch.int32)
            bank_map = torch.full((E,), -1, dtype=torch.int32)
            slots = list(range(G))
            rng.shuffle(slots)
            for i, e in enumerate(perm[:H]):
                row_map[e] = R + i
            for j, e in enumerate(perm[H:H + G]):
                bank_map[e] = slots[j]
                self.content[(idx, slots[j])] = self._id()
            st = types.SimpleNamespace(idx=idx, mod=None, row_map=row_map, bank_map=bank_map, band_map=None,
                                       band_ok=True, bank_n=G, bank=list(range(G)), strong=1, stagers=None,
                                       bank_ptr=base.SIM_BANK + (idx << 40), bank_cpu=None, layout=base.SIM_LAYOUT,
                                       cut_mean=None, spare=perm[H + G:])
            self.layers[idx] = st
        self.res = FakeRes(self.layers)
        for j, idx in enumerate(self.res.idx):
            st = self.layers[idx]
            for e in range(E):
                b = int(st.bank_map[e])
                if b >= 0:
                    self.res.bank_owner[j, b] = e

    def _id(self):
        self.next_id[0] += 1
        return self.next_id[0]

    def layer_of(self, src):
        return (src - base.SIM_BANK) >> 40


class PoolMod:
    """celane's extension on the simulated streams. Copies and the stager read the bank's bytes when they run."""

    def __init__(self, world):
        self.w = world

    def copy(self, dst, src, triples):
        w = self.w
        layer = w.layer_of(src)
        items = []
        for i in range(0, len(triples), 3):
            d, s_, nb = triples[i:i + 3]
            items += [(dst + d + k * REC, s_ // REC + k) for k in range(nb // REC)]

        def work():
            for addr, slot in items:
                w.scratch[addr] = w.content.get((layer, slot), -3)       # -3: a slot that never held a record
        base.SIM[0].cur.push(work)
        return len(triples) // 3

    def expand(self, src, rec_bytes, n, bases, strides, row0, segs, lut, blocks):
        w = self.w
        addrs = [src + i * REC for i in range(n)]

        def work():
            for i, a in enumerate(addrs):
                w.rows[row0 + i] = w.scratch.get(a, -2)
        base.SIM[0].cur.push(work)

    def expand_map(self, src, rec_bytes, src_idx, dst_row, bases, strides, segs, lut, blocks):
        w = self.w
        assert rec_bytes == REC and len(src_idx) == len(dst_row) <= 128
        assert len(set(dst_row)) == len(dst_row) and all(0 <= r_ < R for r_ in dst_row)
        pairs = [(src + p * REC, r_) for p, r_ in zip(src_idx, dst_row)]

        def work():
            for a, r_ in pairs:
                w.rows[r_] = w.scratch.get(a, -2)                        # -2: a scratch slot never copied
        base.SIM[0].cur.push(work)


def tick(world, st, rate, sim):
    """A residency tick: map the rewrites submitted two ticks ago (their bytes have landed: the simulated streams are
    drained first, at the same points in every run, so the world does not depend on the engine's own syncs), then
    unmap some slots of any layer and queue their new bytes on the writer stream, after the main stream's unmap (as
    _bank_job waits for the event recorded after the unmap)."""
    w, res = world, world.res
    w.n_tick += 1
    due = [x for x in w.pending if w.n_tick - x[4] >= 2]
    if due:
        sim.drain()
        for (idx, slot, e, ev, _) in due:                # 0. landed refills: map
            assert ev.query()
            lst = w.layers[idx]
            j = res.idx.index(idx)
            lst.bank_map[e] = slot
            res.bank_owner[j, slot] = e
            res.bank_gen[j, slot] += 1
        w.pending = [x for x in w.pending if w.n_tick - x[4] < 2]
    if rate <= 0:
        return
    for idx, lst in w.layers.items():
        j = res.idx.index(idx)
        for slot in range(lst.bank_n):
            if w.rng.random() >= rate or res.bank_owner[j, slot] < 0 or not lst.spare:
                continue
            old = int(res.bank_owner[j, slot])
            lst.bank_map[old] = -1                       # unmap: no later plan reaches the slot through `old`
            res.bank_owner[j, slot] = -1
            lst.spare.append(old)
            e = lst.spare.pop(0)
            new_id = w._id()
            ev_u = base.SimEvent()
            ev_u.record(sim.main)                        # after the unmap: no queued step can still read the slot
            w.writer.wait_event(ev_u)
            w.writer.push(lambda idx=idx, slot=slot, new_id=new_id: w.content.__setitem__((idx, slot), new_id))
            ev_w = base.SimEvent()
            ev_w.record(w.writer)
            w.pending.append((idx, slot, e, ev_w, w.n_tick))


def run_seq(seed, sizes, T_list, H, share, lanes, bufs, xside, spec, rate, forwards=3, between=4,
            break_claim=False, spec_rows=0):
    """`forwards` prompt forwards over the layers -> [(layer, rows read, read list, ids)] per MegaMoE call, pool."""
    rng = random.Random(seed * 7919 + 13)
    world = World(rng, sizes, H)
    sim = base.Sim(seed)
    base.SIM[0] = sim
    torch.cuda.current_stream = lambda device=None: sim.cur
    torch.cuda.Stream = lambda device=None, **kw: base.SimStream(sim)
    torch.cuda.Event = lambda **kw: base.SimEvent()
    torch.cuda.stream = base._sim_stream
    torch.cuda.synchronize = lambda device=None: sim.drain()
    pp._AH["stream"], celane._XS["stream"] = None, None
    celane._LANE.clear()
    celane._J.clear()
    pp._LANE_ON[0] = False
    celane._MOD[:] = [PoolMod(world)]
    world.writer = base.SimStream(sim)
    calls = []
    orig_claim = celane.Pool.claim
    if break_claim:
        def claim_all(self, st, resv):
            sp, self.spec = self.spec, None
            if sp is None or sp["idx"] != st.idx:
                return {}
            return {b: p for p, b in enumerate(sp["slots"])}
        celane.Pool.claim = claim_all

    def stager_of(st):
        def mk(g):
            def f(bank_ptr, lst):
                vals = lst.tolist()

                def work():
                    for k, v in enumerate(vals):
                        if v >= 0:
                            world.rows[pp.SLOTS * g + k] = world.content.get((st.idx, v), -3)
                base.SIM[0].cur.push(work)
            return f
        return [mk(g) for g in range(pp.GROUPS)]

    for st in world.layers.values():
        st.stagers = stager_of(st)
        st.mod = types.SimpleNamespace(**{n: torch.zeros(R, 16, dtype=torch.uint8) for n in
                                          ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")})
    base.setflag("K31_PROMPT_AHEAD", 0)
    base.setflag("K31_PROMPT_BANK", "all")
    base.setflag("K31_PROMPT_CELANE_XSIDE", int(xside))
    base.setflag("K31_PROMPT_CELANE", share)
    base.setflag("K31_PROMPT_CELANE_LANES", lanes)
    base.setflag("K31_PROMPT_CELANE_BUFS", bufs)
    base.setflag("K31_PROMPT_SPEC", int(spec))
    base.setflag("K31_PROMPT_SPEC_ROWS", spec_rows)      # 0: copies ahead whatever the forward's rows (the cases
    order = sorted(world.layers)                         # here have 12-200 rows; the default is 256)
    bad = []
    try:
        for f in range(forwards):
            T = T_list[f % len(T_list)]
            if f:                                        # between two steps: ticks without a prompt forward (decode
                for _ in range(between):                 # steps, idle time); a slot may be unmapped, rewritten and
                    tick(world, None, rate * 4, sim)     # mapped again before the copy made ahead of it is used
                    if rng.random() < 0.5:
                        sim.drain()
            for li, idx in enumerate(order):
                st = world.layers[idx]
                tick(world, st, rate, sim)
                ids = torch.tensor([rng.sample(range(E), K) for _ in range(T)], dtype=torch.int32)
                wts = torch.tensor([sorted((rng.random() for _ in range(K)), reverse=True) for _ in range(T)])
                want = torch.tensor([[world.content.get((idx, int(st.bank_map[int(e)])), -9)
                                      if int(st.bank_map[int(e)]) >= 0 else -9 for e in row] for row in ids])

                def orig(mod, x, w_, ids_, shared, idx=idx, want=want):
                    ids_c = ids_.clone()
                    staged = (ids_c >= 0) & (ids_c < R)
                    sel = torch.unique(ids_c[staged].long())

                    def work():
                        got = world.rows[ids_c.long().clamp(0, R - 1)]
                        if not torch.equal(got[staged], want[staged]):
                            bad.append(f"layer {idx}: a MegaMoE call read a row that does not hold its pick's record")
                        calls.append((idx, world.rows[sel].clone(), sel, ids_c))
                    base.SIM[0].cur.push(work)
                    return torch.zeros(x.shape[0], 4, dtype=torch.float64) + ids_c.double().sum(1, keepdim=True)

                nxt = world.layers[order[(li + 1) % len(order)]]
                pp.forward_prompt(st.mod, st, torch.zeros(T, 4), wts, ids, None, orig=orig, res=None,
                                  resv=[world.res] if rate >= 0 else [], next_st=nxt)
                if rng.random() < 0.5:
                    sim.drain()                          # sometimes the GPU catches up between layers
        sim.drain()
    finally:
        celane.Pool.claim = orig_claim
    pool = celane._LANE[0].pool if celane._LANE and getattr(celane._LANE[0], "pool", None) is not None else None
    stats = dict(pool.n) if pool is not None else {}
    return calls, bad, stats


def same(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2])
                                    and torch.equal(x[3], y[3]) for x, y in zip(a, b))


CASES = [
    # (bank sizes per layer, rows per forward, native rows)
    ((60, 128, 200, 300, 517, 640), (200,), 20),
    ((212, 677, 232, 44, 460), (160, 12), 40),
    ((500, 500, 500), (64,), 60),
    ((129, 257, 385, 96), (120, 40), 30),
]


def part2():
    n, failed = 0, []
    for ci, (sizes, T_list, H) in enumerate(CASES):
        for rate in (-1.0, 0.0, 0.02, 0.2):              # -1: no residency at all; 0: one that never rewrites
            for seed in range(6):
                ref, ref_bad, _ = run_seq(seed, sizes, T_list, H, 0.75, 1, 2, False, False, rate)
                if ref_bad:
                    failed.append(f"reference case {ci} rate {rate} seed {seed}: {ref_bad[0]}")
                    continue
                cfgs = [(0.75, 2, 2, True), (0.75, 1, 2, False), (0.5, 2, 1, True), (1.0, 1, 3, True),
                        (0.25, 2, 2, False), (0.625, 2, 3, False)]
                for share, lanes, bufs, xside in cfgs:
                    n += 1
                    try:
                        calls, bad, _ = run_seq(seed, sizes, T_list, H, share, lanes, bufs, xside, True, rate)
                    except AssertionError as e:
                        failed.append(f"case {ci} rate {rate} seed {seed} share {share} lanes {lanes} bufs {bufs} "
                                      f"xside {xside}: {e}")
                        continue
                    if bad or not same(calls, ref):
                        failed.append(f"case {ci} rate {rate} seed {seed} share {share} lanes {lanes} bufs {bufs} "
                                      f"xside {xside}: {bad[0] if bad else 'calls differ from the lane without the pool'}")
    check(f"the pool on simulated streams: every call reads its records, the same rows and calls as without it "
          f"({n} runs)", not failed, f"({len(failed)} failing: {'; '.join(failed[:5])})" if failed else "")


def part3():
    caught = runs = 0
    for ci, (sizes, T_list, H) in enumerate(CASES):
        for seed in range(6):
            runs += 1
            try:
                _, bad, _ = run_seq(seed, sizes, T_list, H, 0.75, 2, 2, True, True, 0.2, break_claim=True)
            except AssertionError:
                bad = ["assert"]
            caught += bool(bad)
    check(f"without the ownership check the rewrites are caught as stale reads ({caught} of {runs} runs fail)",
          caught >= runs // 2)


def part4():
    ok, detail = True, []
    for bufs in (1, 2, 3):
        sizes, T = (640, 640, 640, 640), 1000           # 1,000 rows x 16 picks: every record hit
        _, bad, st = run_seq(1, sizes, (T,), 20, 0.75, 2, bufs, True, True, 0.0, forwards=2, between=0)
        layers = st.get("layers", 0)
        want = bufs * celane.take(0.75, R) * (layers - 1)        # every layer but the very first claims its copies
        ok &= not bad and layers == 8 and st.get("used", 0) == want and st.get("stale", 0) == 0
        detail.append(f"bufs {bufs}: {st.get('used', 0)} of {st.get('lane', 0)} lane records ahead (want {want})")
    check("every record hit, no rewrites: the first units' lane records come from copies made ahead", ok,
          "(" + "; ".join(detail) + ")")
    # K31_PROMPT_SPEC_ROWS: a forward under the threshold copies nothing ahead (and so claims nothing); at the
    # threshold it does
    sizes = (640, 640, 640, 640)
    _, bad_lo, lo = run_seq(1, sizes, (255,), 20, 0.75, 2, 2, True, True, 0.0, forwards=2, between=0, spec_rows=256)
    _, bad_hi, hi = run_seq(1, sizes, (256,), 20, 0.75, 2, 2, True, True, 0.0, forwards=2, between=0, spec_rows=256)
    check("K31_PROMPT_SPEC_ROWS=256: no copies ahead in forwards of 255 rows, copies ahead at 256 rows",
          not bad_lo and not bad_hi and lo.get("ahead", 0) == 0 and lo.get("used", 0) == 0 and hi.get("ahead", 0) > 0
          and hi.get("used", 0) > 0,
          f"(255 rows: {lo.get('ahead', 0)} copied ahead, {lo.get('used', 0)} used; 256 rows: {hi.get('ahead', 0)} "
          f"copied ahead, {hi.get('used', 0)} used)")


def part5():
    """K31_PROMPT_TICK_LATE: where the residency's in-forward tick falls in a layer's expert step (host order).
    Off: before the plan's first staging launch, as before. On: after the last MegaMoE call and the cut picks' mean
    (the tail), once per eligible layer; the same credit either way; never in a layer that is not a tick layer."""
    import contextlib
    torch.cuda.current_stream = lambda device=None: base._MAIN
    torch.cuda.Stream = lambda device=None, **kw: base._Stream()
    torch.cuda.Event = lambda **kw: base._Event()
    torch.cuda.stream = lambda s_: contextlib.nullcontext()
    torch.cuda.synchronize = lambda device=None: None
    rng = random.Random(4)
    ok, detail = True, []
    for late in (0, 1):
        for idx, want_tick in ((1, False), (5, True), (6, False), (9, True)):
            ev = []

            class Res:
                accepted = True
                eager_tokens = 0

                def tick(self, n):
                    ev.append("tick")
            res = Res()
            case = base.make_case(rng, 64, 50, 200)
            ids, w, row_map, bank_map = case
            G = int((bank_map >= 0).sum())

            def stager(g):
                def f(bank_ptr, lst):
                    ev.append("stage")
                return f

            def orig(mod, x, w_, ids_, shared):
                ev.append("moe")
                return torch.zeros(x.shape[0], 4, dtype=torch.float64)

            def cut_mean(y, served, tid, tw, table, band=None):
                ev.append("mean")
            st = types.SimpleNamespace(idx=idx, mod=None, row_map=row_map, bank_map=bank_map, band_map=None,
                                       band_ok=True, bank_n=G, bank=list(range(G)), strong=1,
                                       stagers=[stager(g) for g in range(pp.GROUPS)], bank_ptr=4096, bank_cpu=None,
                                       layout={"record_bytes": 9289728}, cut_mean=object())
            base.setflag("K31_PROMPT_CELANE", 0)
            base.setflag("K31_PROMPT_SPEC", 0)
            base.setflag("K31_PROMPT_AHEAD", 0)
            base.setflag("K31_PROMPT_TICK_LATE", late)
            pp._LANE_ON[0] = False
            celane._LANE.clear()
            pp.forward_prompt(None, st, torch.zeros(64, 4), w, ids, None, orig=orig, res=[res], cut_mean=cut_mean,
                              first_layer=1)
            n_tick = ev.count("tick")
            good = n_tick == (1 if want_tick else 0) and res.eager_tokens == (max(1, round(64 * 4 / 92))
                                                                              if (idx - 1) % 4 == 0 else 0)
            if want_tick:
                good &= (ev[-1] == "tick" and ev.index("tick") > max(i for i, e in enumerate(ev) if e != "tick")) \
                    if late else ev[0] == "tick"
            ok &= good
            if not good:
                detail.append(f"late {late} layer {idx}: {ev[:3]}..{ev[-3:]} credit {res.eager_tokens}")
    base.setflag("K31_PROMPT_TICK_LATE")
    check("the in-forward tick: before the plan (off) or after the layer's last kernel is queued (on), once per tick "
          "layer, the same credit", ok, "(" + "; ".join(detail) + ")" if detail else "")


def main():
    torch.set_num_threads(1)
    rng = random.Random(21)
    part1(rng)
    part2()
    part3()
    part4()
    part5()
    print(f"CELANE POOL CPU: {'PASS' if not fails else f'FAIL ({len(fails)} checks, first: ' + '; '.join(fails[:3]) + ')'}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
