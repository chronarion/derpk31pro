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
"""
from __future__ import annotations

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

    def add_request(self, req_idx: int, sampling_params) -> bool:
        xa = sampling_params.extra_args or {}
        x = xa.get("k31_force")
        self.pending[req_idx] = None if x is None else (int(xa["k31_start"]), x)
        self.kl.pop(req_idx, None)
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
