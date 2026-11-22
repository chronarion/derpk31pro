"""K31_PROMPT_FIRST_CHUNK (k31_engine/sched.py) against vLLM's own split rule, no GPU.

Runs in the vLLM image: the wrapper is put around the real Scheduler._mamba_block_aligned_split, called with a
stand-in scheduler object that holds only the fields the rule reads (the values of a derpk31pro server: 16,384-token
hybrid blocks, 64-token prefix unit, internal KDA checkpoints declared with alignment 16).
  python3 tests/test_sched_cap.py
"""
import os
import random
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31_engine import sched  # noqa: E402

from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402

ORIG = Scheduler._mamba_block_aligned_split
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def scheduler(biggest=8192, unit=64, block=16384, threshold=0):
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=block), hash_block_size=unit, block_size=block,
        mamba_partial_cache_hit=True, use_eagle_block_drop=False, num_hash_lookahead_tokens=0,
        mamba_has_prefill_checkpoint_blocks=True, mamba_prefill_checkpoint_alignment=16,
        max_num_scheduled_tokens=biggest, mamba_shared_prefix_checkpoint=False,
        scheduler_config=types.SimpleNamespace(long_prefill_token_threshold=threshold))


def request(n, rid="r"):
    return types.SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=n, num_tokens=n, shared_prefix_boundary=0,
                                 request_id=rid)


def read(split, s, n, hit=0, budgets=None, rid="r"):
    """The chunks [(start, rows)] of a prompt of n tokens with a prefix hit: the first call as a waiting request (it
    has computed nothing; the call carries the hit), the rest as a running one."""
    req = request(n, rid)
    out, start, k = [], hit, 0
    while start < n:
        offer = min(n - start, s.max_num_scheduled_tokens, budgets[k % len(budgets)] if budgets else 1 << 30)
        if req.num_computed_tokens == 0:
            got = split(s, req, offer, hit, 0)
        else:
            got = split(s, req, offer)
        assert 0 < got <= offer, (start, offer, got)
        out.append((start, got))
        start += got
        req.num_computed_tokens = start
        k += 1
        assert k < 10000
    return out


def sizes(chunks):
    return [c[1] for c in chunks]


def expect(n, hit, biggest, cap, rows, unit=64, block=16384, least=0):
    """An independent model: a chunk ends at the earliest of its largest size (the cap while fewer than `rows` rows
    are read, for a request with more than `least` rows of its own), the next block boundary and the prompt's last
    hash boundary."""
    tail = (n - 1) // unit * unit
    out, start = [], hit
    caps, bounds = (cap, rows) if isinstance(cap, tuple) else ((cap,), (rows,))
    while start < n:
        size = biggest
        if n - hit > least:
            for c, r in zip(caps, bounds):
                if c > 0 and start - hit < r:
                    size = min(biggest, c)
                    break
        end = min(start + size, n)
        for stop in ((start // block + 1) * block, tail):
            if start < stop < end:
                end = stop
        out.append(end - start)
        start = end
    return out


# 1. the documented sequences
s = scheduler()
off = sizes(read(ORIG, s, 40000))
v1 = sizes(read(sched.wrap(ORIG, 4096, 4096), scheduler(), 40000))
v2 = sizes(read(sched.wrap(ORIG, 4096, 8192), scheduler(), 40000))
check("a 40,000-token prompt, 8,192-row chunks: off / first chunk capped / two chunks capped",
      off == [8192, 8192, 8192, 8192, 7168, 64] and v1 == [4096, 8192, 4096, 8192, 8192, 7168, 64]
      and v2 == [4096, 4096, 8192, 8192, 8192, 7168, 64], f"({off} / {v1} / {v2})")

# 2. against the independent model: prompt lengths, hits, both variants, both prefix units
bad, n_cases = [], 0
rng = random.Random(7)
for unit in (64, 16):
    for cap, rows in ((0, 0), (4096, 4096), (4096, 8192), (2048, 4096), (1024, 1024)):
        for _ in range(300):
            n = rng.choice([rng.randint(1, 600), rng.randint(600, 9000), rng.randint(9000, 70000)])
            hit = rng.choice([0, 0, rng.randrange(0, n, unit) if n > unit else 0])
            hit = min(hit, (n - 1) // unit * unit)
            sc = scheduler(unit=unit)
            split = sched.wrap(ORIG, cap, rows) if cap else ORIG
            got = sizes(read(split, sc, n, hit))
            want = expect(n, hit, 8192, cap, rows, unit)
            n_cases += 1
            if got != want:
                bad.append(f"n {n} hit {hit} unit {unit} cap {cap}/{rows}: {got[:8]} want {want[:8]}")
check(f"chunk sizes equal an independent model of the rule with the cap ({n_cases} prompts)", not bad,
      f"({len(bad)} differ: {'; '.join(bad[:3])})" if bad else "")

# 2b. K31_PROMPT_FIRST_MIN: a prompt that fits one chunk is read in one; a longer one starts with the short chunk
bad, n_min = [], 0
for cap, rows, least in ((1024, 1024, 4096), (2048, 2048, 8192), (1024, 2048, 4096)):
    for _ in range(300):
        n = rng.choice([rng.randint(1, 4200), rng.randint(4000, 9000), rng.randint(9000, 70000)])
        hit = rng.choice([0, 0, rng.randrange(0, n, 64) if n > 64 else 0])
        hit = min(hit, (n - 1) // 64 * 64)
        got = sizes(read(sched.wrap(ORIG, cap, rows, least), scheduler(), n, hit))
        want = expect(n, hit, 8192, cap, rows, least=least)
        n_min += 1
        if got != want:
            bad.append(f"n {n} hit {hit} cap {cap}/{rows} min {least}: {got[:6]} want {want[:6]}")
a_ = sizes(read(sched.wrap(ORIG, 1024, 1024, 4096), scheduler(), 4000))
b_ = sizes(read(sched.wrap(ORIG, 1024, 1024, 4096), scheduler(), 6000))
c_ = sizes(read(ORIG, scheduler(biggest=4096), 6000))
check(f"the cap only for requests with more rows than K31_PROMPT_FIRST_MIN ({n_min} prompts); a 4,000-token prompt "
      f"in one chunk, a 6,000-token one in as many chunks as with 4,096-row chunks",
      not bad and a_ == [3968, 32] and b_ == [1024, 4928, 48] and len(b_) == len(c_),
      f"({a_} / {b_} / {c_}; {len(bad)} differ: {'; '.join(bad[:2])})")

# 2c. several steps: the documented ramp, and lists against the independent model
ramp = ((1024, 3072, 4096), (1024, 4096, 8192))
a_ = sizes(read(sched.wrap(ORIG, *ramp, 4096), scheduler(), 28672))
b_ = sizes(read(sched.wrap(ORIG, 4096, 8192), scheduler(), 28672))
c_ = sizes(read(sched.wrap(ORIG, *ramp, 4096), scheduler(), 4000))
d_ = sizes(read(sched.wrap(ORIG, (1024, 4096), (1024, 8192)), scheduler(), 28672))
bad, n_ramp = [], 0
for caps, bounds, least in (ramp + (4096,), ramp + (0,), ((1024, 4096), (1024, 8192), 0), ((40, 4056), (40, 4096), 256),
                            ((2048, 2048, 4096), (2048, 4096, 16384), 0)):
    for _ in range(300):
        n = rng.choice([rng.randint(1, 4200), rng.randint(4000, 9000), rng.randint(9000, 70000)])
        hit = rng.choice([0, 0, rng.randrange(0, n, 64) if n > 64 else 0])
        hit = min(hit, (n - 1) // 64 * 64)
        got = sizes(read(sched.wrap(ORIG, caps, bounds, least), scheduler(), n, hit))
        want = expect(n, hit, 8192, caps, bounds, least=least)
        n_ramp += 1
        if got != want:
            bad.append(f"n {n} hit {hit} steps {caps}/{bounds} min {least}: {got[:7]} want {want[:7]}")
check(f"several steps ({n_ramp} prompts equal the model): a 28,672-token prompt as 1,024 / 3,072 / 4,096 and then the "
      f"two-chunk cap's boundaries; a 4,000-token prompt in one chunk",
      not bad and a_ == [1024, 3072, 4096, 8192, 8192, 4032, 64] and b_ == [4096, 4096, 8192, 8192, 4032, 64]
      and a_[3:] == b_[2:] and c_ == [3968, 32] and d_ == [1024, 4096, 4096, 7168, 8192, 4032, 64],
      f"({a_} / {b_} / {c_} / {d_}; {len(bad)} differ: {'; '.join(bad[:2])})")

# 3. with a shared budget (other requests take rows): every chunk is what the rule itself gives for some offer, the
#    prompt is read to its end, no chunk under the cap's reach is larger than the cap
bad = []
for trial in range(200):
    n = rng.randint(100, 60000)
    budgets = [rng.choice([8192, 8187, 8192 - 15, 4096, 5000, 2048]) for _ in range(5)]
    sc = scheduler()
    chunks = read(sched.wrap(ORIG, 4096, 4096), sc, n, 0, budgets)
    if sum(sizes(chunks)) != n:
        bad.append(f"n {n}: rows {sum(sizes(chunks))}")
    for start, got in chunks:
        if start < 4096 and got > 4096:
            bad.append(f"n {n}: a chunk of {got} rows at {start}")
check("with other requests taking rows of the step: the prompt is read to its end, capped chunks stay under the cap",
      not bad, f"({bad[:3]})" if bad else "")

# 4. the largest chunk is a whole block: the cap stands aside (the rule would give a shortened chunk no rows)
sc = scheduler(biggest=16384)
logged = []
sched._LOG = lambda m: logged.append(m)
got = sizes(read(sched.wrap(ORIG, 4096, 4096), sc, 40000))
ref = sizes(read(ORIG, scheduler(biggest=16384), 40000))
check("16,384-row chunks: the cap is off, the chunks are the rule's own, said once",
      got == ref and len(logged) == 1 and "off" in logged[0], f"({got} / {ref}; {len(logged)} log lines)")
sc = scheduler(biggest=16384, threshold=4096)            # a prefill threshold below the block: chunks may end inside
got = sizes(read(sched.wrap(ORIG, 2048, 2048), sc, 20000))
check("16,384-row chunks with a long-prefill threshold under the block: the cap applies", got[0] == 2048,
      f"({got[:4]})")

# 5. a preempted request starts over
sc = scheduler()
split = sched.wrap(ORIG, 4096, 4096)
req = request(30000)
a = split(sc, req, 8192, 0, 0)
req.num_computed_tokens = a
b = split(sc, req, 8192)
req.num_computed_tokens = 0                              # preempted: waiting again, with a cache hit of 4,096 now
c = split(sc, req, 8192, 4096, 0)
req.num_computed_tokens = 4096 + c
d = split(sc, req, 8192)
check("a preempted request's first chunk is capped again", (a, b, c, d) == (4096, 8192, 4096, 8192),
      f"({a}, {b}, {c}, {d})")

# 6. the remembered requests are bounded
sc = scheduler()
split = sched.wrap(ORIG, 4096, 4096)
for i in range(3000):
    split(sc, request(5000, f"q{i}"), 4096, 0, 0)
check("the scheduler remembers at most 1,024 requests", len(sc._k31_first) <= sched.KEEP, f"({len(sc._k31_first)})")

# 7. settings and the patch
check("settings: off without the variable, rows default to the cap",
      sched.settings({}) == (0, 0, 0) and sched.settings({"K31_PROMPT_FIRST_CHUNK": "4096"}) == (4096, 4096, 0)
      and sched.settings({"K31_PROMPT_FIRST_CHUNK": "4096", "K31_PROMPT_FIRST_ROWS": "8192"}) == (4096, 8192, 0)
      and sched.settings({"K31_PROMPT_FIRST_CHUNK": "1024", "K31_PROMPT_FIRST_MIN": "4096"}) == (1024, 1024, 4096)
      and sched.settings({"K31_PROMPT_FIRST_ROWS": "8192"}) == (0, 0, 0))
L = {"K31_PROMPT_FIRST_CHUNK": "1024,3072,4096", "K31_PROMPT_FIRST_ROWS": "1024,4096,8192", "K31_PROMPT_FIRST_MIN": "4096"}
raised = []
for env in ({"K31_PROMPT_FIRST_CHUNK": "1024,4096", "K31_PROMPT_FIRST_ROWS": "8192"},
            {"K31_PROMPT_FIRST_CHUNK": "1024,4096", "K31_PROMPT_FIRST_ROWS": "4096,4096"},
            {"K31_PROMPT_FIRST_CHUNK": "1024,x"}):
    try:
        sched.settings(env)
        raised.append(False)
    except ValueError:
        raised.append(True)
check("settings, several steps: two lists of one length; without the rows list each step holds for one chunk; lists "
      "that do not fit raise; a zero in the list is off",
      sched.settings(L) == ((1024, 3072, 4096), (1024, 4096, 8192), 4096)
      and sched.settings({"K31_PROMPT_FIRST_CHUNK": "1024,4096"}) == ((1024, 4096), (1024, 5120), 0)
      and sched.settings({"K31_PROMPT_FIRST_CHUNK": "[1024, 4096]", "K31_PROMPT_FIRST_ROWS": "[1024, 8192]"})
      == ((1024, 4096), (1024, 8192), 0)
      and sched.settings({"K31_PROMPT_FIRST_CHUNK": "1024,0"}) == (0, 0, 0) and all(raised), f"({raised})")
mod = types.SimpleNamespace(Scheduler=type("Scheduler", (), {"_mamba_block_aligned_split": ORIG}))
os.environ.pop("K31_PROMPT_FIRST_CHUNK", None)
sched.patch(mod)
untouched = mod.Scheduler._mamba_block_aligned_split is ORIG
os.environ["K31_PROMPT_FIRST_CHUNK"] = "4096"
sched.patch(mod)
once = mod.Scheduler._mamba_block_aligned_split
sched.patch(mod)
check("patch: nothing without the setting, one wrapper with it", untouched and once is not ORIG
      and mod.Scheduler._mamba_block_aligned_split is once)

# 8. the values as runtime flags: a live reader switches the cap between calls
state = {"v": (4096, 4096, 0)}
split = sched.wrap(ORIG, 0, 0, 0, live=lambda: state["v"])
a_ = sizes(read(split, scheduler(), 40000))
state["v"] = (0, 0, 0)
b_ = sizes(read(split, scheduler(), 40000, rid="r2"))
state["v"] = (40, 40, 256)
c_ = sizes(read(split, scheduler(), 1024, rid="r3"))
d_ = sizes(read(split, scheduler(), 200, rid="r4"))
check("runtime values: cap on, off, and a 40-row scout chunk for prompts of more than 256 rows",
      a_[:2] == [4096, 8192] and b_[0] == 8192 and c_ == [40, 920, 64] and d_ == [192, 8],
      f"({a_[:3]} / {b_[:2]} / {c_} / {d_})")

# 8b. K31_PROMPT_FIRST_FREE: the steps after the first only where they cost no forward
two = ((4096, 4096), (4096, 8192))
on = sched.wrap(ORIG, *two, 0, free=lambda: True)
a_ = sizes(read(on, scheduler(), 12288, rid="f1"))           # two capped chunks would cost a forward: one
b_ = sizes(read(on, scheduler(), 28672, rid="f2"))           # the same count either way: two
c_ = sizes(read(on, scheduler(), 13000, rid="f3"))
d_ = sizes(read(sched.wrap(ORIG, *two, 0), scheduler(), 12288, rid="f4"))       # without the option: both steps
bad, worse, kept, n_free = [], 0, 0, 0
for _ in range(1500):
    n = rng.choice([rng.randint(1, 9000), rng.randint(8000, 17000), rng.randint(9000, 70000)])
    hit = rng.choice([0, 0, rng.randrange(0, n, 64) if n > 64 else 0])
    hit = min(hit, (n - 1) // 64 * 64)
    got = sizes(read(sched.wrap(ORIG, *two, 0, free=lambda: True), scheduler(), n, hit))
    one = sizes(read(sched.wrap(ORIG, 4096, 4096), scheduler(), n, hit))
    both = sizes(read(sched.wrap(ORIG, *two, 0), scheduler(), n, hit))
    n_free += 1
    worse += len(got) > len(one)
    kept += got == both
    if got != (both if len(both) <= len(one) else one) or sum(got) != n - hit:
        bad.append(f"n {n} hit {hit}: {got[:6]} one {one[:6]} both {both[:6]}")
    if sched.plan(n, hit, 8192, *two, 64, 16384) != both or sched.plan(n, hit, 8192, 4096, 4096, 64, 16384) != one:
        bad.append(f"n {n} hit {hit}: the count ahead differs from the rule")
check(f"steps only where free ({n_free} prompts): never a forward more than the first step alone, both steps "
      f"wherever they cost none ({kept} prompts), the count ahead equals the rule's chunks",
      not bad and worse == 0 and a_ == [4096, 8128, 64] and b_ == [4096, 4096, 8192, 8192, 4032, 64]
      and c_ == [4096, 4096, 4800, 8] and d_ == [4096, 4096, 4032, 64],
      f"({a_} / {b_} / {c_} / {d_}; {worse} worse; {len(bad)} differ: {'; '.join(bad[:2])})")

# 9. runtime values that do not fit: the last good ones stay, said once, and the prompt is read to its end
said = []
sched._LOG = lambda m: said.append(m)
state = {"env": {"K31_PROMPT_FIRST_CHUNK": "1024,3072,4096", "K31_PROMPT_FIRST_ROWS": "1024,4096,8192"}}
split = sched.wrap(ORIG, 4096, 4096, 0, live=lambda: sched.settings(state["env"]))
a_ = sizes(read(split, scheduler(), 28672, rid="s1"))
state["env"] = {"K31_PROMPT_FIRST_CHUNK": "1024,4096", "K31_PROMPT_FIRST_ROWS": "8192"}
b_ = sizes(read(split, scheduler(), 28672, rid="s2"))
state["env"] = {"K31_PROMPT_FIRST_CHUNK": "0"}
c_ = sizes(read(split, scheduler(), 28672, rid="s3"))
check("runtime lists: the ramp applies; values that do not fit keep the last good ones (said once); 0 switches off",
      a_ == [1024, 3072, 4096, 8192, 8192, 4032, 64] and b_ == a_ and len(said) == 1 and "ignored" in said[0]
      and c_[0] == 8192, f"({a_} / {b_} / {c_[:3]}; {len(said)} lines)")

print(f"SCHED CAP: {'PASS' if not fails else f'FAIL ({len(fails)} checks: ' + '; '.join(fails[:3]) + ')'}")
sys.exit(0 if not fails else 1)
