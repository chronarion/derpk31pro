"""Forced decoding and exact KL for the KLD harness (vLLM V2 runner logits processor; `--logits-processors
k31_engine.force:ForceTokens`).

The KLD protocol: brandonmusic's cuts, domains and metrics, each cut scored as the engine serves it: a
short prompt prefilled, the rest teacher-forced one decode step at a time, so the expert cache adapts at decode speed.
A request with `vllm_xargs = {"k31_force": [ids...], "k31_start": P}` gets token ids[i] sampled at sequence position
P + i: every other logit is set to -inf. The server's log-probs are the raw ones (vLLM's default `raw_logprobs`, taken
before logits processors), so each forced position returns the engine's own next-token distribution.

Exact KL (brandonmusic's metric is the full vocabulary; a top-64 estimate measured 2.5x low on this set). With
`"k31_kl_hidden": path` instead of `k31_kl_ref`, the reference is its final hidden states (runner `--dump-hidden`, bf16
bits [N, T, H]; 15 GB for all 512 cuts) and its logits are rebuilt here with the target's BF16 LM head in fp32. With
`"k31_kl_ref": path, "k31_kl_index": i, "k31_kl_out": path` the processor loads row block i of the reference's full
log-probabilities (runner `--dump-logits`: bf16 bits [N, T, V]) to the GPU when the request arrives, computes
KL(reference || engine) over the full vocabulary from the raw logits at every forced row (row at input position p =
reference row p, which predicts token p + 1), and saves the per-position values (float32 [T], NaN where not computed)
to k31_kl_out near the request's end (rewritten on each of its last steps).

Long sequences: a request that also carries `"k31_kl_rows": [A, B]` takes the row-range path at the end of this file
(only rows [A, B) are read and scored; the reference block stays in pinned host memory, the forced rows sit on the GPU).
On that path `"k31_kl_rev": 1` also computes the reverse divergence KL(engine || reference) of the same rows from the
same two distributions and saves it beside k31_kl_out (FILE.rev.npy); the forward values are computed as without it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


def _ref_logits(h: torch.Tensor, V: int, chunk: int = 16384) -> torch.Tensor:
    """Reference logits [rows, V] in fp32 from its final hidden rows (bf16) and the target's BF16 LM head (the same
    checkpoint tensor the runner scored with), upcast chunk by chunk."""
    from k31_engine.hook import LM_HEAD
    W = LM_HEAD[0]
    if W is None or W.shape[0] < V:
        raise RuntimeError("k31_kl_hidden needs the target's LM head (hook.LM_HEAD)")
    hf = h.float()
    out = torch.empty((h.shape[0], V), dtype=torch.float32, device=h.device)
    for c in range(0, V, chunk):
        out[:, c:c + chunk] = hf @ W[c:min(V, c + chunk)].float().T
    return out


INSTANCE: list = [None]       # the live ForceTokens (hook.K31_FORCE_DRAFTS reads its forced-token table)
# `"k31_kl_next": j` (harness --prefetch): while this request decodes, a thread reads reference block j into one of two
# pinned buffers; the request that names block j then copies it to the GPU from there (~13 ms) instead of reading the
# file and copying from pageable memory inside add_request (~0.3-0.5 s of a stalled engine step per cut)
_PF: dict = {"pool": None, "bufs": [None, None], "turn": 0, "fut": None, "key": None}


def _pf_read(path: str, i: int, b: int):
    src = np.load(path, mmap_mode="r")[i]
    buf = _PF["bufs"][b]
    if buf is None or buf.numel() < src.nbytes:
        buf = _PF["bufs"][b] = torch.empty(src.nbytes, dtype=torch.uint8, pin_memory=True)
    dst = buf[:src.nbytes].numpy().view(src.dtype).reshape(src.shape)
    np.copyto(dst, src)
    return b, src.shape, src.dtype


def _pf_start(path: str, i: int) -> None:
    if _PF["pool"] is None:
        from concurrent.futures import ThreadPoolExecutor
        _PF["pool"] = ThreadPoolExecutor(1)
    b = _PF["turn"]
    _PF["turn"] ^= 1
    _PF["key"] = (path, int(i))
    _PF["fut"] = _PF["pool"].submit(_pf_read, path, int(i), b)


def _pf_take(path: str, i: int, dev):
    """The prefetched block (path, i) on the GPU, or None if it was not prefetched."""
    if _PF["fut"] is None or _PF["key"] != (path, int(i)):
        return None
    b, shape, dtype = _PF["fut"].result()
    _PF["fut"] = _PF["key"] = None
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    out = torch.empty(n, dtype=torch.uint8, device=dev)
    out.copy_(_PF["bufs"][b][:n], non_blocking=True)     # the engine's stream, ordered as the pageable copy was; the
    torch.cuda.current_stream(dev).synchronize()         # buffer is refilled two requests later
    return out.view(torch.int16).view(*shape) if np.dtype(dtype).itemsize == 2 else out.view(*shape)


class ForceTokens(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        INSTANCE[0] = self
        self.dev = req_states.device
        self.max_len = vllm_config.model_config.max_model_len
        # forced token for the row whose INPUT position is p (it samples sequence position p + 1), or -1
        self.tok = torch.full((req_states.max_num_reqs, self.max_len), -1, dtype=torch.int32, device=self.dev)
        self.pending = {}
        self.kl = {}             # slot -> {"ref": bf16 [T, V] on the GPU, "out": float32 [T], "path": str, "end": int}

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        x = (sampling_params.extra_args or {}).get("k31_force")
        if x is None:
            return
        if not isinstance(x, list) or "k31_start" not in sampling_params.extra_args:
            raise ValueError("k31_force needs a list of token ids and k31_start (the prompt length)")
        rows = sampling_params.extra_args.get("k31_kl_rows")
        if rows is not None and (not isinstance(rows, (list, tuple)) or len(rows) != 2 or
                                 not all(isinstance(v, int) for v in rows) or rows[0] < 0 or rows[1] <= rows[0]):
            raise ValueError("k31_kl_rows must be [A, B] with 0 <= A < B (absolute rows to score)")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        xa = sampling_params.extra_args or {}
        x = xa.get("k31_force")
        self.pending[req_idx] = None if x is None else (int(xa["k31_start"]), x)
        self.kl.pop(req_idx, None)
        if x is not None and xa.get("k31_kl_rows") is not None and (xa.get("k31_kl_ref") or xa.get("k31_kl_hidden")):
            self.kl[req_idx] = _rows_entry(xa, len(x), self.dev)        # long sequences: the row-range path (below)
            return True
        if x is not None and (xa.get("k31_kl_ref") or xa.get("k31_kl_hidden")):
            hidden = bool(xa.get("k31_kl_hidden"))
            path = xa["k31_kl_hidden"] if hidden else xa["k31_kl_ref"]
            got = _pf_take(path, int(xa["k31_kl_index"]), self.dev)
            if got is not None:
                ref = got.view(torch.bfloat16)
            else:
                src = np.load(path, mmap_mode="r")[int(xa["k31_kl_index"])]
                ref = torch.from_numpy(np.array(src).view(np.int16)).to(self.dev).view(torch.bfloat16)
            if int(xa.get("k31_kl_next", -1)) >= 0:
                _pf_start(path, int(xa["k31_kl_next"]))
            T = ref.shape[0]
            # out[T] is a dummy entry that absorbs the rows of other requests (no duplicate indices in index_put_)
            self.kl[req_idx] = {"ref": ref, "hidden": hidden, "out": torch.full((T + 1,), float("nan"), device=self.dev),
                                "path": xa["k31_kl_out"], "end": int(xa["k31_start"]) + len(x)}
        return x is not None

    def apply_staged_writes(self) -> None:
        for slot, v in self.pending.items():
            self.tok[slot].fill_(-1)
            if v is not None:
                start, ids = v
                n = min(len(ids), self.max_len - start + 1)
                self.tok[slot, start - 1:start - 1 + n] = torch.tensor(ids[:n], dtype=torch.int32).to(self.dev)
        self.pending.clear()

    def apply(self, logits: torch.Tensor, ctx) -> torch.Tensor:
        slots = ctx.expanded_idx_mapping.long()
        pos = ctx.pos.long().clamp(0, self.max_len - 1)
        if self.kl:                                  # exact KL from the raw logits, before the forcing below
            lp = torch.log_softmax(logits.float(), dim=-1)
            for slot, k in self.kl.items():
                if "lo" in k:                            # the row-range path: its forced rows [lo, hi) on the GPU
                    _rows_apply(k, slot, slots, pos, lp)
                    continue
                ref = k["ref"]
                T = ref.shape[0]
                if not k["hidden"] and ref.shape[1] != lp.shape[1]:
                    raise RuntimeError(f"k31_kl_ref has {ref.shape[1]} columns, the logits {lp.shape[1]}")
                mine = (slots == slot) & (pos < T)
                if k["hidden"]:                  # reference logits rebuilt from its final hidden rows, fp32, chunked
                    r = _ref_logits(ref[pos.clamp(max=T - 1)], lp.shape[1])
                else:
                    r = ref[pos.clamp(max=T - 1)].float()               # [rows, V] log-probs of the reference
                r = r - torch.logsumexp(r, -1, keepdim=True)             # renormalize (bf16 storage / raw logits)
                p0 = r.exp()
                v = torch.where(p0 > 0, p0 * (r - lp), 0.0).sum(-1)
                k["out"].index_put_((torch.where(mine, pos, T),), v)
            for b, slot in enumerate(ctx.idx_mapping_np.tolist()):   # save on the request's last steps
                k = self.kl.get(slot)
                if k is not None and ctx.seq_lens_upper_bound_np[b] >= k["end"] - 4:
                    np.save(k["path"], k["out"][:-1].cpu().numpy())
                    if k.get("rev") is not None:
                        np.save(k["path"][:-4] + ".rev.npy", k["rev"][:-1].cpu().numpy())
        f = self.tok[slots, pos]
        rows = (f >= 0).unsqueeze(1)
        idx = f.clamp_min(0).long().unsqueeze(1)
        keep = logits.gather(1, idx)
        logits.masked_fill_(rows, float("-inf"))
        logits.scatter_(1, idx, torch.where(rows, torch.zeros_like(keep), keep))
        return logits


# ------------------------------------------------------------------------------------------- prompt rows (prefill)
# KLD must hold on the prompt rows too. The forced rows above are decode rows; a request's prompt rows go
# through the prompt path in chunks and only the last one is sampled. vLLM's runner calls the prompt-logprob worker
# after every forward with all of the forward's hidden rows, whether or not a request asked for prompt log-probs: the
# wrapper below computes, for every prompt row of a request with an exact-KL reference, the full-vocabulary logits
# from that row and KL(reference || engine), into the same per-position output as the decode rows (input position p
# against reference row p). The prompt's last row is the first sampled one and stays with `apply`.
PROMPT_KL_CHUNK = 512


def _prompt_kl(logits_fn, hidden_states, input_batch, prompt_lens) -> None:
    ft = INSTANCE[0]
    if ft is None or not ft.kl:
        return
    for i, slot in enumerate(input_batch.idx_mapping_np.tolist()):
        k = ft.kl.get(slot)
        if k is None:
            continue
        if "lo" in k:                                    # the row-range path: prompt rows from pinned host memory
            _rows_prompt(k, logits_fn, hidden_states, input_batch, i, int(prompt_lens[slot]), ft.dev)
            continue
        prompt_len = int(prompt_lens[slot])
        chunk_start = int(input_batch.num_computed_prefill_tokens_np[i])
        chunk_end = chunk_start + int(input_batch.num_scheduled_tokens[i])
        if chunk_start >= prompt_len:
            continue                                     # a decode step: `apply` scores its rows
        ref = k["ref"]
        T = ref.shape[0]
        lo, hi = chunk_start, min(chunk_end, prompt_len - 1, T)
        base = int(input_batch.query_start_loc_np[i]) - chunk_start
        for a in range(lo, hi, PROMPT_KL_CHUNK):
            b = min(a + PROMPT_KL_CHUNK, hi)
            lp = torch.log_softmax(logits_fn(hidden_states[base + a:base + b]).float(), dim=-1)
            if k["hidden"]:
                r = _ref_logits(ref[a:b], lp.shape[1])
            else:
                r = ref[a:b].float()
            r = r - torch.logsumexp(r, -1, keepdim=True)
            p0 = r.exp()
            k["out"][a:b] = torch.where(p0 > 0, p0 * (r - lp), 0.0).sum(-1)
            del lp, r, p0


def patch_prompt_kl(module) -> None:
    """Wrap PromptLogprobsWorker.compute_prompt_logprobs (vllm.v1.worker.gpu.sample.prompt_logprob)."""
    cls = getattr(module, "PromptLogprobsWorker", None)
    if cls is None or getattr(cls, "_k31_prompt_kl", False):
        return
    orig = cls.compute_prompt_logprobs

    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens,
                                prompt_lens):
        _prompt_kl(logits_fn, hidden_states, input_batch, prompt_lens)
        return orig(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens, prompt_lens)
    cls.compute_prompt_logprobs = compute_prompt_logprobs
    cls._k31_prompt_kl = True


# ------------------------------------------------------------------------------------ long sequences (row ranges)
# A request whose extra args carry "k31_kl_rows": [A, B] (absolute rows to score; harness/kld_long.py sends it) takes
# this path instead of the one above, which holds the whole [T, 7168] reference block on the GPU (0.47 GB at 32K rows,
# 1.9 GB at 128K, 15 GB at 1M) and indexes it by absolute position:
# - only rows [A, B) of the block are read, into pinned host memory (`k31_kl_next` reads the next block's rows the same
#   way on a thread while this request runs);
# - the rows the decode steps score (from the prompt's last row on) are copied to the GPU when the request arrives;
#   prompt rows go to the GPU per prompt chunk, PROMPT_KL_CHUNK rows at a time, and the host block is released after
#   the prompt's last chunk;
# - the reference may hold a row range of every sequence (runner --dump-rows: FILE.json's row_start / row_end) and may
#   be several files in order (comma-separated: k31_kl_index counts across them, e.g. the runner's two halves);
# - rows outside [A, B) (or outside what the reference holds) are not scored: NaN in the output, which has one entry
#   per sequence position (end = k31_start + len(k31_force)).
# The KL arithmetic and every call's shape are the path above's (a prompt sub-chunk the range cuts is computed whole,
# see _rows_prompt; `apply` gathers one reference row per logits row either way), so the same rows give bit-equal
# values on the GPU too (engine/tests/test_force_rows.py).
_PARTS: dict = {}                # (reference spec, mtimes) -> [(path, sequences, row_start, row_end)]
_RF: dict = {"pool": None, "fut": None, "key": None}      # the prefetched block: future of _rows_read, its key


def _ref_parts(spec: str):
    stamp = tuple(os.stat(p).st_mtime_ns for p in spec.split(","))     # a rewritten reference is read again
    parts = _PARTS.get((spec, stamp))
    if parts is None:
        parts = []
        for path in spec.split(","):
            mm = np.load(path, mmap_mode="r")
            r0, r1 = 0, int(mm.shape[1])
            if os.path.exists(path + ".json"):
                with open(path + ".json") as f:
                    d = json.load(f)
                r0, r1 = int(d.get("row_start", 0)), int(d.get("row_end", mm.shape[1]))
                if r1 - r0 != mm.shape[1]:
                    raise RuntimeError(f"{path}.json: rows [{r0}, {r1}), the file holds {mm.shape[1]} per sequence")
            parts.append((path, int(mm.shape[0]), r0, r1))
        _PARTS[(spec, stamp)] = parts
    return parts


def _rows_read(spec: str, i: int, A: int, B: int):
    """Rows [A, B) of reference block i (absolute rows; clipped to the rows the file holds) as an int16 tensor (the
    bf16 bits) in pinned host memory when CUDA is up; returns (rows, lo, hi) with rows[j] = absolute row lo + j."""
    j = i
    for path, n, r0, r1 in _ref_parts(spec):
        if j < n:
            break
        j -= n
    else:
        raise IndexError(f"reference block {i}: {spec} holds {sum(p[1] for p in _ref_parts(spec))}")
    lo, hi = max(A, r0), min(B, r1)
    if hi <= lo:
        return None, lo, lo
    src = np.load(path, mmap_mode="r")[j, lo - r0:hi - r0]
    if src.dtype.itemsize != 2:
        raise RuntimeError(f"{path}: {src.dtype}, expected bf16 bits (uint16 / int16)")
    host = torch.empty(src.shape, dtype=torch.int16, pin_memory=torch.cuda.is_available())
    np.copyto(host.numpy(), src.view(np.int16))
    return host, lo, hi


def _rows_take(spec: str, i: int, A: int, B: int):
    key = (spec, int(i), int(A), int(B))
    if _RF["fut"] is not None and _RF["key"] == key:
        got = _RF["fut"].result()
        _RF["fut"] = _RF["key"] = None
        return got
    return _rows_read(spec, int(i), int(A), int(B))


def _rows_start(spec: str, i: int, A: int, B: int) -> None:
    if _RF["pool"] is None:
        from concurrent.futures import ThreadPoolExecutor
        _RF["pool"] = ThreadPoolExecutor(1)
    _RF["key"] = (spec, int(i), int(A), int(B))
    _RF["fut"] = _RF["pool"].submit(_rows_read, spec, int(i), int(A), int(B))


def _rows_entry(xa: dict, n_force: int, dev) -> dict:
    hidden = bool(xa.get("k31_kl_hidden"))
    spec = xa["k31_kl_hidden"] if hidden else xa["k31_kl_ref"]
    start = int(xa["k31_start"])
    end = start + n_force
    A, B = (int(v) for v in xa["k31_kl_rows"])
    A, B = max(A, 0), min(B, end)
    host, lo, hi = _rows_take(spec, int(xa["k31_kl_index"]), A, B)
    if int(xa.get("k31_kl_next", -1)) >= 0:
        _rows_start(spec, int(xa["k31_kl_next"]), A, B)
    dlo = max(lo, start - 1)                     # forced rows: the prompt's last row is the first sampled one
    ref = host[dlo - lo:hi - lo].to(dev).view(torch.bfloat16) if hi > dlo else None
    pend = min(hi, start - 1)                    # prompt rows [lo, pend) stay in host memory until the prompt is done
    return {"ref": ref, "lo": dlo, "hi": max(hi, dlo), "host": host if pend > lo else None, "host_lo": lo,
            "host_hi": pend, "hidden": hidden, "out": torch.full((end + 1,), float("nan"), device=dev),
            "rev": torch.full((end + 1,), float("nan"), device=dev) if xa.get("k31_kl_rev") else None,
            "path": xa["k31_kl_out"], "end": end}


def _rows_apply(k: dict, slot: int, slots: torch.Tensor, pos: torch.Tensor, lp: torch.Tensor) -> None:
    """`apply`'s exact KL for a row-range request: rows at input positions [lo, hi) of this slot."""
    ref = k["ref"]
    if ref is None:
        return
    lo, R, T = k["lo"], ref.shape[0], k["out"].shape[0] - 1       # out[T]: the dummy entry other rows land in
    if not k["hidden"] and ref.shape[1] != lp.shape[1]:
        raise RuntimeError(f"k31_kl_ref has {ref.shape[1]} columns, the logits {lp.shape[1]}")
    mine = (slots == slot) & (pos >= lo) & (pos < k["hi"])
    at = (pos - lo).clamp(0, R - 1)
    if k["hidden"]:
        r = _ref_logits(ref[at], lp.shape[1])
    else:
        r = ref[at].float()
    r = r - torch.logsumexp(r, -1, keepdim=True)
    p0 = r.exp()
    v = torch.where(p0 > 0, p0 * (r - lp), 0.0).sum(-1)
    k["out"].index_put_((torch.where(mine, pos, T),), v)
    if k.get("rev") is not None:                 # KL(engine || reference) of the same rows
        q0 = lp.exp()
        vr = torch.where(q0 > 0, q0 * (lp - r), 0.0).sum(-1)
        k["rev"].index_put_((torch.where(mine, pos, T),), vr)


def _rows_prompt(k: dict, logits_fn, hidden_states, input_batch, i: int, prompt_len: int, dev) -> None:
    """`_prompt_kl` for a row-range request: this chunk's prompt rows in [host_lo, host_hi), reference rows copied from
    pinned host memory. The sub-chunks are always the path above's (PROMPT_KL_CHUNK rows from the chunk's start, up to
    the prompt's last row), wherever the range starts or ends: on the GPU a row's logits, log-softmax and KL sum depend
    on the row count of the call that computes them (cuBLAS and the reduction kernels choose kernels by shape), so a
    row computed in a sub-chunk cut at the range's edge would round differently. A sub-chunk the range cuts is computed
    whole (reference rows outside the range zero) and only its rows inside the range are kept."""
    chunk_start = int(input_batch.num_computed_prefill_tokens_np[i])
    chunk_end = chunk_start + int(input_batch.num_scheduled_tokens[i])
    if chunk_start >= prompt_len:
        return                                       # a decode step: `apply` scores its rows
    host = k["host"]
    if host is not None:
        hlo, hhi = k["host_lo"], k["host_hi"]
        end = min(chunk_end, prompt_len - 1)
        base = int(input_batch.query_start_loc_np[i]) - chunk_start
        for a in range(chunk_start, end, PROMPT_KL_CHUNK):
            b = min(a + PROMPT_KL_CHUNK, end)
            r0, r1 = max(a, hlo), min(b, hhi)        # this sub-chunk's rows inside the range
            if r1 <= r0:
                continue
            if r0 == a and r1 == b:
                rows = host[a - hlo:b - hlo].to(dev, non_blocking=True)
            else:
                rows = torch.zeros((b - a,) + tuple(host.shape[1:]), dtype=host.dtype, device=dev)
                rows[r0 - a:r1 - a].copy_(host[r0 - hlo:r1 - hlo], non_blocking=True)
            rows = rows.view(torch.bfloat16)
            lp = torch.log_softmax(logits_fn(hidden_states[base + a:base + b]).float(), dim=-1)
            if k["hidden"]:
                r = _ref_logits(rows, lp.shape[1])
            else:
                r = rows.float()
            r = r - torch.logsumexp(r, -1, keepdim=True)
            p0 = r.exp()
            v = torch.where(p0 > 0, p0 * (r - lp), 0.0).sum(-1)
            k["out"][r0:r1] = v[r0 - a:r1 - a]
            del p0, v
            if k.get("rev") is not None:             # after p0 is freed: the same peak memory as without it
                q0 = lp.exp()
                k["rev"][r0:r1] = torch.where(q0 > 0, q0 * (lp - r), 0.0).sum(-1)[r0 - a:r1 - a]
                del q0
            del lp, r, rows
    if chunk_end >= prompt_len:
        k["host"] = None                             # the prompt is done: release the pinned block
