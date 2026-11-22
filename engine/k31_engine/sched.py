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

Where: vLLM's scheduler clips every prefill chunk to the positions it may end at (Scheduler._mamba_block_aligned_split:
the hybrid block boundary, the prompt's last hash boundary, shared-prefix junctions). The wrapper shortens the chunk
offered to that rule, so the rule's own stops apply to the shorter chunk as to any other. It stands aside (the chunk
goes through unchanged) when
- chunks may not end inside a block (the largest chunk is at least one hybrid block: a shortened chunk would be
  cut back to the block's start and the request would never run), or
- the rule gives the shortened chunk no rows where it gives the full one some.
A request's own start is taken while it waits (it has computed nothing then, and the call carries its cache hit), so
a preempted request starts over.
"""
import os
import sys

_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
KEEP = 1024                                          # requests remembered (the oldest half goes when full)


def settings(env=None):
    """(cap, rows, least) from the environment: (0, 0, 0) = off."""
    env = os.environ if env is None else env
    cap = int(env.get("K31_PROMPT_FIRST_CHUNK", "0") or 0)
    rows = int(env.get("K31_PROMPT_FIRST_ROWS", "0") or 0) or cap
    least = int(env.get("K31_PROMPT_FIRST_MIN", "0") or 0)
    return (cap, rows, least) if cap > 0 else (0, 0, 0)


def capped(start: int, first: int, num_new_tokens: int, cap: int, rows: int) -> int:
    """The chunk to offer: at most `cap` rows while the request has read fewer than `rows` of its own."""
    if cap <= 0 or start - first >= rows:
        return num_new_tokens
    return min(num_new_tokens, cap)


def wrap(orig, cap: int, rows: int, least: int = 0):
    """The scheduler's split rule with the cap (see the module's text)."""
    def split(self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0):
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
    if cap <= 0 or cls is None or getattr(cls, "_k31_first_chunk", False):
        return
    cls._mamba_block_aligned_split = wrap(cls._mamba_block_aligned_split, cap, rows, least)
    cls._k31_first_chunk = True
    _LOG(f"K31_PROMPT_FIRST_CHUNK={cap}: a request's prompt chunks have at most {cap} rows while it has read fewer "
         f"than {rows} rows of its own" + (f" (requests with more than {least} rows to read)" if least else ""))
