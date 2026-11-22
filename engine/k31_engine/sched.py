"""K31_PROMPT_FIRST_CHUNK=N (start-up setting; unset or 0 = off): a request's first prompt chunks have at most N rows.

Why: a prompt's first chunk is read on the expert tiers the previous request left. The residency and the bank follow a
prompt at its chunk boundaries (a layer's scores see a chunk only after the layer has run it, and what they promote
lands in later chunks), and most of what a prompt gains from that it gains at the first boundary. With prompt chunks
larger than 4,096 rows (--max-num-batched-tokens 8192) more rows would be read before it. This cap keeps the first
boundary where it is while later chunks take the larger size.

K31_PROMPT_FIRST_ROWS=M (default N): the cap holds while the request has read fewer than M rows of its own (rows after
its prefix-cache hit). M = N caps the first chunk only; M = 2N keeps two chunks of N rows before the larger ones.
K31_PROMPT_FIRST_MIN=L (default 0): only requests with more than L rows of their own to read are capped. With L = the
largest chunk a shorter first chunk costs no extra forward: a prompt that fits one chunk is still read in one.
Several steps: both values may be comma lists of the same length, K31_PROMPT_FIRST_CHUNK=N1,N2,... and
K31_PROMPT_FIRST_ROWS=M1,M2,... (rising): chunks have at most N1 rows while the request has read fewer than M1 rows
of its own, then at most N2 while fewer than M2, and so on; after the last step the full size. Without the rows
list each step holds for one chunk (M = N1, N1 + N2, ...). 1024,3072,4096 with 1024,4096,8192 reads a long prompt as
1,024 / 3,072 / 4,096 rows and then in full chunks: a short first chunk, and the boundaries of 4,096-row chunks
from row 4,096 on.

Where: vLLM's scheduler clips every prefill chunk to the positions it may end at (Scheduler._mamba_block_aligned_split:
the hybrid block boundary, the prompt's last hash boundary, shared-prefix junctions). The wrapper shortens the chunk
offered to that rule, so the rule's own stops apply to the shorter chunk as to any other. It stands aside (the chunk
goes through unchanged) when
- chunks may not end inside a block (the largest chunk is at least one hybrid block: a shortened chunk would be
  cut back to the block's start and the request would never run), or
- the rule gives the shortened chunk no rows where it gives the full one some.
A request's own start is taken while it waits (it has computed nothing then, and the call carries its cache hit), so
a preempted request starts over.

The wrapper is installed when K31_PROMPT_FIRST_CHUNK is set at start-up; the three values are then runtime flags too
(flags.py: the control file wins over the environment; K31_PROMPT_FIRST_CHUNK=0 there switches the cap off).

A cap of 40 rows or fewer makes the first chunk a step that replays a captured graph (the decode path): a scout
that scores the prompt's first rows and earns promotion credit ~0.2 s before the main forward starts.
"""
import os
import sys

from . import flags

_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
KEEP = 1024                                          # requests remembered (the oldest half goes when full)


def _ints(v) -> tuple:
    """A setting's value as numbers: 4096, "4096", "1024,3072,4096" or a list."""
    if v is None or v == "":
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return tuple(int(x) for x in str(v).replace("[", "").replace("]", "").split(",") if x.strip())


def settings(env=None):
    """(cap, rows, least) from the environment: (0, 0, 0) = off. One step: two numbers. Several steps: two tuples
    of the same length (cap[i] holds while the request has read fewer than rows[i] rows of its own). Raises
    ValueError for lists that do not fit each other."""
    env = os.environ if env is None else env
    caps = _ints(env.get("K31_PROMPT_FIRST_CHUNK", "0"))
    rows = _ints(env.get("K31_PROMPT_FIRST_ROWS", ""))
    least = int(env.get("K31_PROMPT_FIRST_MIN", "0") or 0)
    if not caps or min(caps) <= 0:
        return (0, 0, 0)
    if not rows or rows == (0,):                     # each step holds for one chunk
        rows = tuple(sum(caps[:i + 1]) for i in range(len(caps)))
    if len(rows) != len(caps) or any(b <= a for a, b in zip(rows, rows[1:])) or rows[0] <= 0:
        raise ValueError(f"K31_PROMPT_FIRST_CHUNK={caps} and K31_PROMPT_FIRST_ROWS={rows}: one rising row count "
                         f"per chunk size is needed")
    if len(caps) == 1:
        return (caps[0], rows[0], least)
    return (caps, rows, least)


def capped(start: int, first: int, num_new_tokens: int, cap, rows) -> int:
    """The chunk to offer: at most cap[i] rows while the request has read fewer than rows[i] rows of its own (the
    first step whose row count is above the rows read); the full chunk after the last step. cap and rows are two
    numbers (one step) or two tuples."""
    caps = cap if isinstance(cap, tuple) else (cap,)
    bounds = rows if isinstance(rows, tuple) else (rows,)
    done = start - first
    for c, r in zip(caps, bounds):
        if c > 0 and done < r:
            return min(num_new_tokens, c)
    return num_new_tokens


def wrap(orig, cap, rows, least: int = 0, live=None):
    """The scheduler's split rule with the cap (see the module's text); cap and rows as `settings` gives them. live:
    a function -> (cap, rows, least) read at every call (the runtime flags), else the three values given here."""
    good = [(cap, rows, least)]                      # the last values that read well
    said = [None]

    def split(self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0):
        if live is not None:
            try:
                good[0] = live()
            except Exception as ex:                  # noqa: BLE001  (runtime values that do not fit: the last good
                msg = f"{type(ex).__name__}: {ex}"   # ones stay; the scheduler must not stop)
                if said[0] != msg:
                    said[0] = msg
                    _LOG(f"K31_PROMPT_FIRST_CHUNK: runtime values ignored ({msg})")
        cap, rows, least = good[0]
        start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        seen = self.__dict__.setdefault("_k31_first", {})
        rid = request.request_id
        if request.num_computed_tokens == 0 or rid not in seen:      # waiting (or first seen): its start is here
            if rid not in seen and len(seen) >= KEEP:
                for k in list(seen)[:KEEP // 2]:
                    del seen[k]
            seen[rid] = start
        n = num_new_tokens
        if max(request.num_prompt_tokens, request.num_tokens - 1) - seen[rid] > least:   # enough rows of its own
            n = capped(start, seen[rid], num_new_tokens, cap, rows)
        if n < num_new_tokens:
            biggest = self.max_num_scheduled_tokens
            thr = self.scheduler_config.long_prefill_token_threshold
            if thr > 0:
                biggest = min(biggest, thr)
            if self.cache_config.block_size <= biggest:              # chunks must end on block boundaries here
                if not self.__dict__.get("_k31_first_off"):
                    self.__dict__["_k31_first_off"] = True
                    _LOG(f"K31_PROMPT_FIRST_CHUNK: off, the largest chunk ({biggest}) is not smaller than a hybrid "
                         f"block ({self.cache_config.block_size})")
                n = num_new_tokens
        if n < num_new_tokens:
            got = orig(self, request, n, num_new_local_computed_tokens, num_external_computed_tokens)
            if got > 0:
                return got
        return orig(self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens)
    return split


def patch(module) -> None:
    """vllm.v1.core.sched.scheduler: wrap Scheduler._mamba_block_aligned_split when the setting is on."""
    cap, rows, least = settings()
    cls = getattr(module, "Scheduler", None)
    if not cap or cls is None or getattr(cls, "_k31_first_chunk", False):
        return
    names = ("K31_PROMPT_FIRST_CHUNK", "K31_PROMPT_FIRST_ROWS", "K31_PROMPT_FIRST_MIN")
    cls._mamba_block_aligned_split = wrap(cls._mamba_block_aligned_split, cap, rows, least,
                                          live=lambda: settings({k: flags.get(k, "") for k in names}))
    cls._k31_first_chunk = True
    steps = list(zip(cap, rows)) if isinstance(cap, tuple) else [(cap, rows)]
    _LOG("K31_PROMPT_FIRST_CHUNK: a request's prompt chunks have at most "
         + ", then ".join(f"{c} rows while it has read fewer than {r} rows of its own" for c, r in steps)
         + (f" (requests with more than {least} rows to read)" if least else ""))
