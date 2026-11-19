"""The copy-engine lane's bookkeeping (k31_engine/celane.py, K31_PROMPT_CELANE), CPU only: no kernel runs here.

1. take / parts / runs against brute-force definitions on random inputs; host_kept on CPU tensors.
2. prompt_path.forward_prompt end to end on CPU (streams and events replaced by no-ops, so everything runs in host
   order): fake stagers write the bank slot they stage into a fake row table, a fake lane copies slots into fake
   scratch buffers and lands them into the rows, a fake MegaMoE snapshots the 128 shared rows and its ids at every
   call. With the lane on (shares 0.1-1 including 0.625 and 0.75, 1-3 buffers, rounds and half-rounds, the expand
   beside the stager or after it; set through the runtime flags, as the server's control file sets them) every call
   must see the same rows and ids as with the lane off, and every staged pick's row must hold that expert's bank slot.
   The fake lane also asserts the host order the real one relies on: a unit is issued into a buffer only after the
   unit that last used the buffer has landed; units land in order, once. The same on the GPU test's own case (seed 3,
   1,024 rows, a bank of 520, bank modes all and valve). Switching the lane off at runtime releases it; a scratch
   that does not fit falls back to the stager, one attempt per setting.
3. Why test_celane.py plans with integer amplitudes: the valve's kept order on the GPU test's case under other
   summation orders (CUDA's index_add_ adds in no fixed order): it changes with the study's amplitudes, and must not
   change with integer ones.
4. The real lane (celane.Lane, both round loops) on simulated streams: copies, expands, stager launches and MegaMoE
   calls are queued per stream with their event records and waits, and run in random legal orders (12 seeds), so a
   missing wait or a race shows up. Every call must read rows that hold its picks' records, and the rows it reads
   must be those of the lane-off reference (shares 0.125-1 including 0.625 and 0.75, 1-2 copy streams, 1-2 buffers,
   XSIDE on and off, bank modes all and valve, rounds and half-rounds). Rows a call does not read may depend on the
   order in half-rounds (the stage-ahead stream fills the other half beside the call), never in rounds.
Run in the engine image, CPU only (from the repository root):
    docker run --rm --network none --cpuset-cpus 58 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD/engine":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_celane_cpu.py
Exit 0 and "CELANE CPU: PASS" = all cases agree.
"""
import contextlib
import os
import random
import sys
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import celane  # noqa: E402
from k31_engine import flags  # noqa: E402
from k31_engine import prompt_path as pp  # noqa: E402

E, K, R = 896, 16, 128
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


# ------------------------------------------------------------------------------------------- 1. pure helpers
def part1(rng):
    ok = all(celane.take(s, m) == max(0, min(m, int(s * m + 0.5))) and 0 <= celane.take(s, m) <= m
             for s in (0.05, 0.1, 0.125, 0.25, 1 / 3, 0.5, 0.75, 0.999, 1.0) for m in range(0, 129))
    ok &= all(celane.take(s, m) <= celane.take(s, R) for s in (0.1, 0.25, 0.5, 1.0) for m in range(129))
    check("take: floor(share x m + 1/2) in [0, m], never above the share of a full round", ok)
    ok = True
    for n in range(0, 70):
        for lanes in (1, 2, 3, 4):
            p = celane.parts(n, lanes)
            ok &= len(p) == lanes and p[0][0] == 0 and p[-1][1] == n
            ok &= all(p[i][1] == p[i + 1][0] for i in range(lanes - 1))
            ok &= max(b - a for a, b in p) - min(b - a for a, b in p) <= 1
    check("parts: contiguous cover, sizes within one", ok)
    ok = True
    for _ in range(2000):
        n = rng.randrange(0, 40)
        kind = rng.random()
        if kind < 0.3:
            s0 = rng.randrange(0, 500)
            slots = list(range(s0, s0 + n))
        elif kind < 0.6:
            slots = sorted(rng.sample(range(600), n))
        else:
            slots = rng.sample(range(600), n)
        Rb, d0 = rng.choice((16, 4096, 9289728)), rng.choice((0, 3, 1 << 20))
        tr = celane.runs(slots, Rb, d0)
        ok &= len(tr) % 3 == 0
        got = []                                          # expand the triples back to (dst, src) per record
        for i in range(0, len(tr), 3):
            d, s, b = tr[i:i + 3]
            ok &= b > 0 and b % Rb == 0 and s % Rb == 0 and (d - d0) % Rb == 0
            got += [((d - d0) // Rb + k, s // Rb + k) for k in range(b // Rb)]
        ok &= got == [(i, s) for i, s in enumerate(slots)]
        brk = sum(1 for i in range(1, n) if slots[i] != slots[i - 1] + 1)
        ok &= len(tr) // 3 == (brk + 1 if n else 0)       # one copy per maximal run, no more
    check("runs: every record once, at its place, one copy per run of consecutive slots", ok)
    kept = torch.tensor([5, 9, 2, -1, -1], dtype=torch.long)
    ok = celane.host_kept(torch.tensor(3), kept) == [5, 9, 2] and celane.host_kept(torch.tensor(0), kept) == []
    check("host_kept: the first n_keep slots", ok)


# ------------------------------------------------------------------------------------------- 2. the rounds
class _Stream:
    device = torch.device("cpu")

    def wait_stream(self, s):
        pass

    def wait_event(self, e):
        pass


class _Event:
    def record(self, s=None):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


_MAIN = _Stream()
torch.cuda.current_stream = lambda device=None: _MAIN
torch.cuda.Stream = lambda device=None, **kw: _Stream()
torch.cuda.Event = lambda **kw: _Event()
torch.cuda.stream = lambda s: contextlib.nullcontext()
torch.cuda.synchronize = lambda device=None: None


class FakeLane:
    """celane.Lane's interface on fake rows: issue() fills scratch buffer u % bufs at once, land() writes it into the
    rows. Asserts the host order the real lane's events rely on."""

    def __init__(self, share, bufs, rows, src):
        self.share, self.bufs, self.rows, self.src = share, bufs, rows, src
        self.lanes, self.xblocks = 1, 608
        self.buf = [[] for _ in range(bufs)]
        self.owner = [None] * bufs
        self.pending = {}
        self.landed = []
        self.records = 0

    def begin(self):
        assert not self.pending, "a layer began with units in flight"

    def issue(self, u, src, slots):
        assert src == self.src, "copies must read the layer's bank"
        assert u not in self.pending, f"unit {u} issued twice"
        assert all(s >= 0 for s in slots)
        b = u % self.bufs
        if slots:
            assert self.owner[b] is None, f"unit {u} issued into buffer {b} before unit {self.owner[b]} landed"
            self.owner[b] = u
            self.buf[b] = list(slots)
        self.pending[u] = len(slots)
        self.records += len(slots)

    def land(self, u, mod, row0):
        assert u in self.pending, f"unit {u} landed without being issued"
        assert not self.landed or u == self.landed[-1] + 1, "units must land in order"
        n = self.pending.pop(u)
        self.landed.append(u)
        b = u % self.bufs
        if n:
            assert self.owner[b] == u, f"buffer {b} holds unit {self.owner[b]}, not {u}"
            assert 0 <= row0 and row0 + n <= R
            self.rows[row0:row0 + n] = torch.tensor(self.buf[b][:n], dtype=self.rows.dtype)
            self.owner[b] = None
        return n


def make_case(rng, T, H, G):
    """Experts: H native (absolute rows 128 + i), G in the bank (slots in a shuffled order), the rest nowhere."""
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
    ids = torch.tensor([rng.sample(range(E), K) for _ in range(T)], dtype=torch.int32)
    w = torch.tensor([sorted((rng.random() for _ in range(K)), reverse=True) for _ in range(T)])
    return ids, w, row_map, bank_map


def run_forward(case, lane_cfg, ahead, xside, inject=True):
    """One forward_prompt with fakes -> (calls: [(rows seen, ids)], y, lane records, stager records). With inject
    False the flags ask for the lane but none is put in place: celane.get() builds one (or fails to)."""
    ids, w, row_map, bank_map = case
    T = ids.shape[0]
    rows = torch.full((R,), -1, dtype=torch.int64)
    staged = [0]
    G = int((bank_map >= 0).sum())

    def stager(g):
        def f(bank_ptr, lst):
            assert bank_ptr == 4096 and lst.dtype == torch.int32 and lst.numel() == pp.SLOTS
            for k in range(pp.SLOTS):
                if int(lst[k]) >= 0:
                    rows[pp.SLOTS * g + k] = int(lst[k])
                    staged[0] += 1
        return f

    calls = []

    def orig(mod, x, w_, ids_, shared):
        snap = rows.clone()
        sel = (ids_ >= 0) & (ids_ < R)
        exp = bank_map[ids.long()].long()
        assert torch.equal(snap[ids_.long().clamp(0, R - 1)][sel], exp[sel]), "a staged pick's row holds another record"
        nat = (ids_ >= R)
        assert torch.equal(ids_[nat], row_map[ids.long()][nat]), "a native pick's row moved"
        calls.append((snap, ids_.clone()))
        return torch.full((x.shape[0], 4), float(snap.sum()), dtype=torch.float64) + ids_.double().sum(1, keepdim=True)

    st = types.SimpleNamespace(idx=1, mod=None, row_map=row_map, bank_map=bank_map, band_map=None, band_ok=True,
                               bank_n=G, bank=list(range(G)), strong=1, stagers=[stager(g) for g in range(pp.GROUPS)],
                               bank_ptr=4096, bank_cpu=None, layout={"record_bytes": 9289728},
                               cut_mean=None)
    setflag("K31_PROMPT_AHEAD", int(ahead))
    setflag("K31_PROMPT_CELANE_XSIDE", int(xside))
    celane._XS["stream"] = None
    lane = None
    if lane_cfg is not None:
        share, bufs = lane_cfg
        lane = FakeLane(share, bufs, rows, 4096)
        setflag("K31_PROMPT_CELANE", share)
        setflag("K31_PROMPT_CELANE_LANES", 1)
        setflag("K31_PROMPT_CELANE_BUFS", bufs)
        if inject:
            celane._LANE[:] = [lane]                  # celane.get() returns it (same share, streams, buffers)
        else:
            lane = None
            celane._LANE.clear()
    else:
        setflag("K31_PROMPT_CELANE", 0)
    x = torch.zeros(T, 4)
    y = pp.forward_prompt(None, st, x, w, ids, None, orig=orig, res=None)
    if lane is not None:
        assert not lane.pending, "units issued but never landed"
    return calls, y, (lane.records if lane else 0), staged[0]


def gpu_case():
    """The case of test_celane.py part 3, generated the same way: 1,024 rows, 105 native rows, a bank of 520 records."""
    E_, T, H, G = 896, 1024, 105, 520
    rng = random.Random(3)
    perm = list(range(E_))
    rng.shuffle(perm)
    row_map = torch.full((E_,), -1, dtype=torch.int32)
    bank_map = torch.full((E_,), -1, dtype=torch.int32)
    order = list(range(G))
    rng.shuffle(order)
    for i, e in enumerate(perm[:H]):
        row_map[e] = R + i
    for j, e in enumerate(perm[H:H + G]):
        bank_map[e] = order[j]
    ids = torch.tensor([rng.sample(range(E_), K) for _ in range(T)], dtype=torch.int32)
    w = torch.tensor([sorted((rng.random() for _ in range(K)), reverse=True) for _ in range(T)])
    return ids, w, row_map, bank_map


def compare(case, mode, ahead, shares, bufs_list, label):
    """The lane on (every share x buffers x expand placement) against the lane off on one case -> cases run."""
    T = case[0].shape[0]
    G = int((case[3] >= 0).sum())
    ref_calls, ref_y, _, ref_staged = run_forward(case, None, ahead, False)
    n_kept = ref_staged
    n = 0
    for share in shares:
        for bufs in bufs_list:
            for xside in (False, True):                  # XSIDE has no effect in half-rounds: it must change nothing
                calls, y, lane_n, st_n = run_forward(case, (share, bufs), ahead, xside)
                same = len(calls) == len(ref_calls) and all(
                    torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(calls, ref_calls))
                unit = pp.HALF if ahead and -(-n_kept // R) > 1 else R
                want = sum(celane.take(share, max(0, min(unit, n_kept - u * unit)))
                           for u in range(max(1, -(-n_kept // unit))))
                ok = same and torch.equal(y, ref_y) and lane_n + st_n == n_kept and lane_n == want
                n += 1
                if not ok:
                    check(f"rounds{label} {mode} T {T} G {G} ahead {ahead} share {share} bufs {bufs} xside {xside}",
                          False, f"(calls {len(calls)} vs {len(ref_calls)}, same rows {same}, lane {lane_n} (want "
                          f"{want}) + stager {st_n} vs {n_kept} records)")
    print(f"  {label.strip() or 'random'} {mode} T {T} G {G} ahead {ahead}: {n_kept} records, {len(ref_calls)} MegaMoE "
          f"calls", flush=True)
    return n


def part2(rng):
    n_cases = 0
    for mode in ("all", "valve"):
        setflag("K31_PROMPT_BANK", mode)
        for T, H, G in ((40, 100, 0), (12, 100, 60), (64, 60, 128), (64, 50, 200), (120, 40, 300), (160, 20, 517),
                        (200, 0, 640)):
            case = make_case(rng, T, H, G)
            for ahead in (False, True):
                n_cases += compare(case, mode, ahead, (0.1, 0.125, 0.25, 0.5, 0.625, 0.75, 1.0), (1, 2, 3), "")
    check(f"forward_prompt with the lane = without it, call by call ({n_cases} cases)", not any(
        f.startswith("rounds ") for f in fails))
    # the case of the GPU test (test_celane.py part 3), with its integer amplitudes and its settings
    keep_amp = pp.AMP
    pp.AMP = tuple(float(round(a * 1000)) for a in pp.AMP)
    pp._C.clear()
    case = gpu_case()
    n_cases = 0
    for mode in ("all", "valve"):
        setflag("K31_PROMPT_BANK", mode)
        for ahead in (False, True):
            n_cases += compare(case, mode, ahead, (0.125, 0.25, 0.5, 0.625, 0.75, 1.0), (1, 2), " gpu-case")
    check(f"the GPU test's case (1,024 rows, a bank of 520): the lane = without it, call by call ({n_cases} cases)",
          not any(f.startswith("rounds gpu-case") for f in fails))
    pp.AMP = keep_amp
    pp._C.clear()
    # the copy streams' parts: the real lane splits a unit's records with parts(); the scratch offsets are a * R
    setflag("K31_PROMPT_BANK")
    # switched off at runtime: the next chunk releases the lane
    case = make_case(rng, 64, 50, 200)
    run_forward(case, (0.25, 1), False, False)
    held = bool(celane._LANE)
    run_forward(case, None, False, False)
    check("the lane switched off at runtime is released at the next prompt chunk", held and not celane._LANE)
    # no room for the scratch: the stager stages everything, one attempt per setting, a new setting tries again
    tries = []

    class NoRoom:
        def __init__(self, *a, **k):
            tries.append(k.get("bufs"))
            raise torch.cuda.OutOfMemoryError("test: no room for the scratch")

    real = celane.Lane
    celane.Lane = NoRoom
    try:
        ref_calls, ref_y, _, _ = run_forward(case, None, False, False)
        c1, y1, _, s1 = run_forward(case, (0.25, 1), False, False, inject=False)
        c2, y2, _, _ = run_forward(case, (0.25, 1), False, False, inject=False)
        run_forward(case, (0.25, 2), False, False, inject=False)
        same = all(len(c) == len(ref_calls) and all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                                                    for a, b in zip(c, ref_calls)) for c in (c1, c2))
        ok = same and torch.equal(y1, ref_y) and torch.equal(y2, ref_y) and not celane._LANE and tries == [1, 2]
        check("no room for the lane's scratch: the stager stages every record, one attempt per setting", ok,
              f"(attempts {tries})")
    finally:
        celane.Lane = real
        celane._FAILED[0] = None


def part3():
    """Why the GPU test plans with integer amplitudes: CUDA's index_add_ sums the pick amplitudes in no fixed order; the
    study's amplitudes are multiples of 0.001, so records tie exactly and their float32 sums differ in the last bit
    with the order. The same picks in other row orders model that here (CPU sums in index order)."""
    ids, w, row_map, bank_map = gpu_case()
    G = int((bank_map >= 0).sum())

    def kept_of(perm):
        kept, n_keep, *_ = pp.plan_rounds(ids[perm], w[perm], row_map, bank_map, None, G, 3, mode="valve")
        return kept[:int(n_keep)].tolist()

    perms = [torch.randperm(ids.shape[0], generator=torch.Generator().manual_seed(s)) for s in range(10)]
    ident = torch.arange(ids.shape[0])
    keep_amp = pp.AMP
    base = kept_of(ident)
    flips = sum(kept_of(p) != base for p in perms)
    pp.AMP = tuple(float(round(a * 1000)) for a in pp.AMP)
    pp._C.clear()
    base = kept_of(ident)
    flips_int = sum(kept_of(p) != base for p in perms)
    pp.AMP = keep_amp
    pp._C.clear()
    print(f"  valve plan of the GPU test's case, other summation orders: kept order differs in {flips} of 10 with the "
          f"study's amplitudes, in {flips_int} of 10 with integer amplitudes", flush=True)
    check("valve plan with integer amplitudes: the same kept order in every summation order", flips_int == 0)


# ------------------------------------------------------------------------------------------- 4. stream orders
SIM: list = [None]
SIM_REC = 80                                              # a tiny record: the simulated scratch holds slot numbers
SIM_LAYOUT = {"record_bytes": SIM_REC, "native_bytes": {"w13": 64, "w2": 32, "w13s": 16, "w2s": 16}}
SIM_BANK = 1 << 20                                        # the simulated bank's base address


class SimStream:
    """A CUDA stream, simulated: its work, event records and event waits are queued and run later, in an order the
    simulator draws among the legal ones."""
    device = torch.device("cpu")

    def __init__(self, sim):
        self.sim, self.q = sim, []
        sim.streams.append(self)

    def ready(self):
        kind, x = self.q[0]
        return kind != "wait" or x.done

    def step(self):
        kind, x = self.q.pop(0)
        if kind == "work":
            x()
        elif kind == "record":
            x.done = True

    def push(self, fn):
        self.q.append(("work", fn))

    def wait_event(self, ev):
        if ev.recorded:                                   # a wait on an event never recorded is a no-op, as in CUDA
            self.q.append(("wait", ev))

    def wait_stream(self, other):
        ev = SimEvent()
        ev.record(other)
        self.wait_event(ev)

    def synchronize(self):
        self.sim.drain()


class SimEvent:
    def __init__(self, **kw):
        self.recorded = self.done = False

    def record(self, stream=None):
        self.recorded, self.done = True, False
        (stream if stream is not None else SIM[0].cur).q.append(("record", self))

    def synchronize(self):
        SIM[0].drain()

    def query(self):
        return self.done


class Sim:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.streams = []
        self.main = SimStream(self)
        self.cur = self.main

    def drain(self):
        """Run every queued item: at each step one stream whose next item may run, drawn at random."""
        while True:
            ready = [s for s in self.streams if s.q and s.ready()]
            if not ready:
                assert not any(s.q for s in self.streams), "the simulated streams deadlocked"
                return
            self.rng.choice(ready).step()


@contextlib.contextmanager
def _sim_stream(s):
    prev = SIM[0].cur
    SIM[0].cur = s
    try:
        yield
    finally:
        SIM[0].cur = prev


class SimMod:
    """celane's extension on the simulated streams: a copy writes slot numbers into the scratch (keyed by address), an
    expand writes them into the rows; both are work on the current stream."""

    def __init__(self, rows, scratch):
        self.rows, self.scratch = rows, scratch

    def copy(self, dst, src, triples):
        assert src == SIM_BANK, "copies must read the layer's bank"
        items = []
        for i in range(0, len(triples), 3):
            d, s_, nb = triples[i:i + 3]
            items += [(dst + d + k * SIM_REC, s_ // SIM_REC + k) for k in range(nb // SIM_REC)]
        SIM[0].cur.push(lambda: self.scratch.update(items))
        return len(triples) // 3

    def expand(self, src, rec_bytes, n, bases, strides, row0, segs, lut, blocks):
        assert rec_bytes == SIM_REC and 0 <= row0 and row0 + n <= R
        addrs = [src + i * SIM_REC for i in range(n)]

        def work():
            for i, a in enumerate(addrs):
                self.rows[row0 + i] = self.scratch.get(a, -2)          # -2: a scratch record never copied

        SIM[0].cur.push(work)


def sim_forward(case, seed, share, lanes, bufs, ahead, xside):
    """forward_prompt on simulated streams with the real lane (celane.Lane) -> [(rows, rows read, read list, ids)]."""
    ids, w, row_map, bank_map = case
    T = ids.shape[0]
    G = int((bank_map >= 0).sum())
    sim = Sim(seed)
    SIM[0] = sim
    torch.cuda.current_stream = lambda device=None: sim.cur
    torch.cuda.Stream = lambda device=None, **kw: SimStream(sim)
    torch.cuda.Event = lambda **kw: SimEvent()
    torch.cuda.stream = _sim_stream
    torch.cuda.synchronize = lambda device=None: sim.drain()
    pp._AH["stream"], celane._XS["stream"] = None, None
    celane._LANE.clear()
    pp._LANE_ON[0] = False
    rows = torch.full((R,), -1, dtype=torch.int64)
    scratch = {}
    celane._MOD[:] = [SimMod(rows, scratch)]
    calls = []

    def stager(g):
        def f(bank_ptr, lst):
            vals = lst.tolist()

            def work():
                for k, v in enumerate(vals):
                    if v >= 0:
                        rows[pp.SLOTS * g + k] = v
            SIM[0].cur.push(work)
        return f

    def orig(mod, x, w_, ids_, shared):
        ids_c = ids_.clone()
        sel = torch.unique(ids_c[(ids_c >= 0) & (ids_c < R)].long())
        want = bank_map[ids.long()].long()
        staged = (ids_c >= 0) & (ids_c < R)

        def work():
            assert torch.equal(rows[ids_c.long().clamp(0, R - 1)][staged], want[staged]), \
                "a MegaMoE call read a row that does not hold its pick's record"
            calls.append((rows.clone(), rows[sel].clone(), sel, ids_c))
        SIM[0].cur.push(work)
        return torch.zeros(x.shape[0], 4, dtype=torch.float64) + ids_c.double().sum(1, keepdim=True)

    mod = types.SimpleNamespace(**{n: torch.zeros(R, 16, dtype=torch.uint8) for n in
                                   ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")})
    st = types.SimpleNamespace(idx=1, mod=mod, row_map=row_map, bank_map=bank_map, band_map=None, band_ok=True,
                               bank_n=G, bank=list(range(G)), strong=1, stagers=[stager(g) for g in range(pp.GROUPS)],
                               bank_ptr=SIM_BANK, bank_cpu=None, layout=SIM_LAYOUT, cut_mean=None)
    setflag("K31_PROMPT_AHEAD", int(ahead))
    setflag("K31_PROMPT_CELANE_XSIDE", int(xside))
    setflag("K31_PROMPT_CELANE", share)
    setflag("K31_PROMPT_CELANE_LANES", lanes)
    setflag("K31_PROMPT_CELANE_BUFS", bufs)
    y = pp.forward_prompt(mod, st, torch.zeros(T, 4), w, ids, None, orig=orig, res=None)
    sim.drain()
    return calls, y


def part4():
    """The real lane on simulated streams: queued work runs in random legal orders, so a missing wait or a race
    against the stage-ahead stream shows up. Every call must read rows that hold its picks' records, and the rows it
    reads must be those of the lane-off reference; the rows a call does not read may differ with the order (the
    stage-ahead stream fills the other half while a call runs), which is why the GPU test fingerprints read rows."""
    case = gpu_case()
    seeds = range(12)
    n, bad, other_half = 0, [], {}
    for mode in ("all", "valve"):
        setflag("K31_PROMPT_BANK", mode)
        for ahead in (False, True):
            ref, ref_y = sim_forward(case, 0, 0, 1, 1, ahead, False)
            for seed in seeds:
                cfgs = [(sh, ln, bf, xs) for sh in (0.125, 0.25, 0.5, 0.625, 0.75, 1.0) for ln in (1, 2)
                        for bf in (1, 2) for xs in (False, True)]
                for cfg in [(0, 1, 1, False)] + cfgs:
                    n += 1
                    try:
                        calls, y = sim_forward(case, seed, cfg[0], cfg[1], cfg[2], ahead, cfg[3])
                    except AssertionError as e:
                        bad.append(f"{mode} ahead {ahead} seed {seed} share {cfg[0]} lanes {cfg[1]} bufs {cfg[2]} "
                                   f"xside {cfg[3]}: {e}")
                        continue
                    ok = len(calls) == len(ref) and torch.equal(y, ref_y) and all(
                        torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
                        for a, b in zip(calls, ref))
                    if not ok:
                        bad.append(f"{mode} ahead {ahead} seed {seed} share {cfg[0]} lanes {cfg[1]} bufs {cfg[2]} "
                                   f"xside {cfg[3]}")
                    if cfg[0] == 0:
                        key = (mode, ahead)
                        other_half[key] = other_half.get(key, 0) + (not all(
                            torch.equal(a[0], b[0]) for a, b in zip(calls, ref)))
    for (mode, ahead), k in other_half.items():
        print(f"  without the lane, bank {mode}, {'half-rounds' if ahead else 'rounds'}: all 128 rows at a call "
              f"differ from another order's in {k} of {len(seeds)} orders", flush=True)
    check(f"streams in random legal orders: every call reads its records, the same rows with the lane as without "
          f"({n} runs)", not bad, f"({len(bad)} failing: {'; '.join(bad[:6])})" if bad else "")
    check("rounds: the rows a call does not read do not depend on the order either",
          other_half[("all", False)] == 0 and other_half[("valve", False)] == 0)


def main():
    torch.set_num_threads(1)
    rng = random.Random(18)
    part1(rng)
    part2(rng)
    part3()
    part4()
    print(f"CELANE CPU: {'PASS' if not fails else f'FAIL ({len(fails)} checks, first: ' + '; '.join(fails[:3]) + ')'}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
