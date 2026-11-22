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
    while start < n:
        size = biggest if (cap <= 0 or start - hit >= rows or n - hit <= least) else min(biggest, cap)
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

print(f"SCHED CAP: {'PASS' if not fails else f'FAIL ({len(fails)} checks: ' + '; '.join(fails[:3]) + ')'}")
sys.exit(0 if not fails else 1)
