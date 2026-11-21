"""derpk31pro engine v0: Kimi K3.1 Pro's routed experts in vLLM's MegaMoE (kimi-k31 image), one GB300.

vLLM loads a checkpoint without routed experts (store/kstrip). This hook
  1. builds every K3.1 MoE layer's `DeepGemmMegaMoEExperts` with only its own rows (the stock module would allocate
     896 rows per layer, 1.4 TB, and DeepGEMM takes at most 512 rows per rank): S resident native experts, plus in
     mode "staged" SLOTS (16) staging rows;
  2. after the model has loaded (GPUModelRunner.load_model), reads each layer's resident experts from Moonshot's
     checkpoint (K31_NATIVE, the official shards) and writes them into the rows through the module's own kernel-layout
     loader (`_load_weight`); in mode "staged" also loads the layer's Grace bank (2-bit records, store/build_store.py,
     from K31_HI2) into pinned host memory; then checks the first MoE layer against float32 reference formulas;
  3. per MoE call maps the router's picks to rows (CUDA-graph safe: fixed shapes, device data):
       hbm:     resident expert -> its row; anything else is cut (id -1: the kernel skips the pick);
       staged:  resident -> its row; a miss of rank < K31_STRONG (by routing weight within its token) whose expert has
                a record in the bank -> one of the SLOTS staging rows, which the stager fills from the record just
                before the call (2-bit, the depth-2 code map); everything else cut. At most SLOTS distinct staged experts
                per call (strongest first by their largest weight); an eager call (a prompt chunk) with more stages them
                in rounds of SLOTS, one MegaMoE call each, and sums the outputs.
Sets per layer: K31_ROWMAP (JSON {"layers": {"<layer>": {"hbm": [...], "bank": [...]}}}) or, without it, the usage
ranking K31_RANK (JSON {"layers": {"<layer>": [expert, ...] most used first}}): the first K31_HBM experts resident, the
next K31_BANK in the bank. One ACTIVE line when bound; a mismatch stops the engine with one line.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import mmap
import os
import re
import sys
import time

import numpy as np
import torch

E, TOPK, LATENT, INTER = 896, 16, 3584, 3072
TARGET = "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe"
RUNNERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")
PREFIX = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.block_sparse_moe\.experts$")
MODE = os.environ.get("K31_MODE", "hbm")
ROWMAP = os.environ.get("K31_ROWMAP", "")
RANK = os.environ.get("K31_RANK", "")
HBM = int(os.environ.get("K31_HBM", "115"))
# K31_HBM_MAP=FILE (JSON {"layers": {"<layer>": rows}} or {"<layer>": rows}): resident rows per MoE layer (by the
# layer's sensitivity) instead of K31_HBM everywhere; the band and the bank then start after each
# layer's own resident cut of K31_RANK (the RTX sidecar takes the same map: --hbm-map)
HBM_MAP_PATH = os.environ.get("K31_HBM_MAP", "")
_HBM_MAP: dict = {}


def _hbm_of(layer: int) -> int:
    if HBM_MAP_PATH and not _HBM_MAP:
        raw = json.load(open(HBM_MAP_PATH))
        _HBM_MAP.update({int(k): int(v) for k, v in raw.get("layers", raw).items()})
    return _HBM_MAP.get(int(layer), HBM)


# K31_BAND_MAP / K31_BANK_MAP: RTX band records and Grace bank records per layer, same file form; a
# layer's band is the next n_L experts of its rank list after its rows, its bank the next g_L after those (the RTX
# sidecar takes the band map too: --band-map). Unset: K31_BAND_N and K31_BANK for every layer.
BAND_MAP_PATH = os.environ.get("K31_BAND_MAP", "")
BANK_MAP_PATH = os.environ.get("K31_BANK_MAP", "")
_LMAPS: dict = {}


def _layer_count(path: str, layer: int, default: int) -> int:
    if not path:
        return default
    if path not in _LMAPS:
        raw = json.load(open(path))
        _LMAPS[path] = {int(k): int(v) for k, v in raw.get("layers", raw).items()}
    return _LMAPS[path].get(int(layer), default)
BANK = int(os.environ.get("K31_BANK", "300"))
STRONG = int(os.environ.get("K31_STRONG", "3"))
HI2 = os.environ.get("K31_HI2", "/hi2")
NATIVE = os.environ.get("K31_NATIVE", "/native")
STAGER_SMS = int(os.environ.get("K31_STAGER_SMS", "24"))
# K31_STAGER_SMS_EAGER: SMs per stager launch on prompt chunks (the eager path stages every bank candidate in rounds,
# with nothing else running beside it; 24 SMs read ~306 GB/s of records, the C2C link carries ~450)
STAGER_SMS_EAGER = int(os.environ.get("K31_STAGER_SMS_EAGER", "0")) or STAGER_SMS
# K31_PROMPT_PATH=1: prompt chunks through prompt_path.py (every bank pick on prompt rows, the residency
# ticking inside the forward, one host sync per layer); 0 (default until its checks pass) = the eager branch below
from . import flags as _flags        # runtime flags (flags.py): the prompt path's settings may change between steps
# K31_PROMPT_SELFCHECK=1: the first two prompt chunks of each layer also go through the older branch on the same
# inputs (mode t1 and K31_PROMPT_TICK_LAYERS=0 serve the same picks: the outputs must agree to the order of additions)
_SELFCHECK: dict = {}
# K31_BAND_HINT=1 (librtxband_fused_v5.so): the call's band plan right after top-k on the band's stream,
# with the served flags and a hint that sends the RTX for the call's records while the latent down still runs
BAND_HINT = os.environ.get("K31_BAND_HINT", "0") == "1"
_IN_EXEC: list = [False]          # inside the runner's execute_model (a serving step), set by its wrapper


def _prompt_ticks() -> bool:
    return _flags.on("K31_PROMPT_PATH") and _flags.num("K31_PROMPT_TICK_LAYERS", 4) > 0
PROMOTE = float(os.environ.get("K31_PROMOTE", "0"))        # residency.py: swaps per layer per token (0 = static)
HALFLIFE = float(os.environ.get("K31_HALFLIFE", "32"))     # tokens, for the residency scores
LAM = 2.0 ** (-1.0 / HALFLIFE)
PROMOTE_WORKERS = int(os.environ.get("K31_PROMOTE_WORKERS", "6"))
BANK_PROMOTE = float(os.environ.get("K31_BANK_PROMOTE", "0"))  # residency.py: bank refills per layer per token
BANK_EXCL = os.environ.get("K31_BANK_EXCL", "0") == "1"       # the Grace bank holds no expert that has an HBM row
DG_SMS = int(os.environ.get("K31_DG_SMS", "0"))            # 0 = DeepGEMM's default (all SMs)
_RES = []                                                   # the Residency (mode staged)
_LAST_MOE = [-1]                                            # index of the last MoE layer (window test join)
_FIRST_MOE = [-1]                                           # index of the first MoE layer (K31_RES_STREAM eager join)
LM_HEAD = [None]                                            # the target model's LM head weight (force.py)
_STEPS = [0]                                                # execute_model calls (residency.py logs the rate)
SELFCHECK = os.environ.get("K31_SELFCHECK", "1") == "1"
WINDOW_TEST = os.environ.get("K31_WINDOW_TEST", "0") == "1"   # capture test: a side-stream window per MoE layer
_WIN = {"side": None, "ev": None, "buf": None}
SLOTS = 16                  # staging slots per call
GROUPS = 7 if os.environ.get("K31_DEFER", "0") == "1" else 8   # slot groups shared by all layers (vmm.py): layer L
                                                    # stages into group L % GROUPS (with deferral the 8th group is the deferral rows)
STAGE_ROWS = SLOTS * GROUPS  # 128 shared staging rows, first in every layer's tensors
# Deferral (engine/DEFERRAL.md v1): a layer's misses of rank DEFER_FROM..DEFER_TO-1 (0-based, by weight) with a bank
# record, the DEFER_SLOTS strongest by weight^2, are staged in a side-stream window after its MegaMoE call into 16
# shared deferral rows and computed as extra rows of the next layer's call; their share lands after that layer's MoE.
DEFER = os.environ.get("K31_DEFER", "0") == "1"
DEFER_FROM = int(os.environ.get("K31_DEFER_FROM", os.environ.get("K31_STRONG", "3")))
DEFER_TO = int(os.environ.get("K31_DEFER_TO", "12"))
DEFER_SLOTS = min(16, int(os.environ.get("K31_DEFER_SLOTS", "8")))
DEFER_BASE = STAGE_ROWS
FUSED_DEFER = os.environ.get("K31_DEFER_FUSED", "1") == "1"   # Triton plan + completion kernels (defer.py)
# where a late share lands: "layer" = after the next layer's MoE (pseudo rows ride in its MegaMoE call); "mlp" = before
# the next layer's MoE input mix (a small MegaMoE call on the pseudo rows right after its attention)
DEFER_LAND = os.environ.get("K31_DEFER_LAND", "layer")
# shared rows (staging, then deferral; resident rows follow). The VMM mapping needs whole 2 MiB granules for every
# kernel tensor (the smallest row is 344,064 B = 21 x 2^14): the shared row count must be a multiple of 128
DEFER_ROWS = SLOTS
# stage 1 ring: the last RING own rows of each layer hold 2-bit records the lane fetched; they stay mapped
# (on time for later tokens) until a later fetch evicts them (stale rows first, then the lowest residency score)
# block ends (layers 11, 23, ..., 83) defer too (sensitivity map 11-17: their late share costs 0.15 of a skip, the
# cheapest place to defer): the share lands in the stored AttnRes block state the next layer wrote, not in the new
# block's running sum; layers listed in K31_DEFER_SKIP_LAYERS never defer (90, 91: late costs as much as a skip)
DEFER_BLOCKEND = os.environ.get("K31_DEFER_BLOCKEND", "1") == "1"
DEFER_SKIP = {int(x) for x in os.environ.get("K31_DEFER_SKIP_LAYERS", "90,91").split(",") if x.strip()}
_AR = {"res": None}             # the AttnRes stored block states of the current forward ([T, blocks, H])
# per-layer deferral slots (JSON {"<layer>": D}, e.g. /engine/profiles/defer-slots-map.json from the sensitivity map:
# slots in proportion to what a late share saves over a skip in that layer, same total as D = 4 everywhere)
DEFER_SLOTS_MAP = os.environ.get("K31_DEFER_SLOTS_MAP", "")
# cut picks served by their expert's mean output (K31_CUT_MEAN = a directory of runner `--save-means` tables
# means-L{i}.pt, [896, 3584]): a pick with no row that is not deferred adds weight x mean to the call's routed latent sum
# (sensitivity study 11-17: halves the cost of skipping, in-sample)
CUT_MEAN = os.environ.get("K31_CUT_MEAN", "")
# per-layer rank bands (sensitivity study layer-opt.json, {"bands": {"<layer>": [dfrom, dto]}}: on time below dfrom,
# late in [dfrom, dto), skipped from dto; held out: half the KL of uniform bands at the same budget). On time here is
# the serial stager, which serves ranks < 4 at most (unitk's SP): ranks the band puts on time beyond that go late.
LAYER_BANDS = os.environ.get("K31_LAYER_BANDS", "")
# K31_RES_MAXRANK=R: a token's picks of weight rank >= R (0-based) are cut even when their expert has an HBM row (fewer
# distinct experts per MegaMoE call: 2.62 us each); with K31_CUT_MEAN they get the mean substitution like any cut.
# K31_RES_MAXRANK_MAP: JSON {"caps": {"<layer>": R}} per MoE layer (others: K31_RES_MAXRANK). Default 16 = off.
RES_MAXRANK = int(os.environ.get("K31_RES_MAXRANK", "16"))
RES_MAXRANK_MAP = os.environ.get("K31_RES_MAXRANK_MAP", "")
# K31_STAGE_ALL=1: every pick of an expert staged for the call uses its staging row, at any rank and in any row (unitk
# ALLROWS; the same in the torch rounds of calls past 16 rows), not only the candidate picks of rank < strong that
# staged it: the same records, the same distinct experts in MegaMoE. Not with K31_DEFER=1 (a deferred pick of a staged
# expert would be served twice). Default 0: candidate picks only.
STAGE_ALL = os.environ.get("K31_STAGE_ALL", "0") == "1"
# K31_POLICY=FILE (policy.py): per-layer, per-rank expert-miss policies (hit / band / stage / defer / mean / skip),
# compiled at bind into each layer's resident cap, staging ranks, deferral range, band on/off + cap and mean on/off;
# it takes precedence over K31_STRONG, K31_LAYER_BANDS, K31_RES_MAXRANK(_MAP) and K31_DEFER_TO for those values
POLICY = os.environ.get("K31_POLICY", "")
BAND_SERVED = os.environ.get("K31_BAND_SERVED", "0") == "1"
# K31_LATE=1: the late tier on the Grace pool (latetier.py): a decode call's bank picks that nothing serves
# on time are computed by the pool's cores and land half a layer late (one up GEMV per layer beside the next attention)
LATE = os.environ.get("K31_LATE", "0") not in ("", "0")
_LT: list = [None]                  # latetier.Tier once bound
# K31_PRESTAGE=1 (prestage.py): decode pre-staging -- a side stream predicts each MoE layer's picks from its inputs
# (the MoE input without the layer's attention output) at the end of _pre_attn_norm and stages their records while
# the attention runs; the MoE call maps its real picks onto them and stages only the misses (unitk PRE + FILL)
PRESTAGE = os.environ.get("K31_PRESTAGE", "0") == "1"
# K31_STAGE_ROWS=c0,c1,... (decode, mode staged): row p of a request's verify block (p = 0 the bonus row, then the
# drafts) stages its picks of rank < min(strong, c_p) only; later rows take the last value. A draft row is used only
# when the drafts before it were accepted (per-position acceptance ~0.69 / 0.43 / 0.24 / 0.14 at 4 drafts), so its
# strong misses buy less KLD per staged record; with K31_STAGE_ALL a staged expert still serves its picks in every row
STAGE_ROWCAPS = [int(x) for x in os.environ.get("K31_STAGE_ROWS", "").split(",") if x.strip()]
_RCAP: list = [None]                # int32 [16] per row of a decode call (bind)
# K31_STAGE_SP=8 (unitk.SP; needs K31_DECODE_SPLIT=1): a policy may stage ranks 0-7 instead of 0-3.
# K31_STAGE_CAP=n (1..16; a policy's `stage_cap` per layer wins): a decode call stages at most n records, the candidates
# with the largest routing weight first (unitk's order), so a layer's staging time per call is bounded whatever ranks
# its policy stages. Default: the 16 slots of the call.
STAGE_SP = int(os.environ.get("K31_STAGE_SP", "4") or 4)
ONE_ROWS = 16 if STAGE_SP == 4 else 8                # the largest call the one-round plan takes (unitk)
STAGE_CAP = int(os.environ.get("K31_STAGE_CAP", "0") or 0) or None
# K31_STAGE_PRIOR=p0,p1,... (as K31_BAND_PRIOR): the experts a decode call stages are ordered by their largest routing
# weight x the prior chance that the pick's row is accepted (row p of a verify block: p0 the bonus row, then the
# drafts; later rows take the last value) instead of by weight alone. It decides which experts keep a slot when a
# call has more candidates than K31_STAGE_CAP (or than its 16 slots); calls under the cap stage what they staged.
STAGE_PRIOR = [float(x) for x in os.environ.get("K31_STAGE_PRIOR", "").split(",") if x.strip()]
_RPRI: list = [None]                # float32 [16] per row of a decode call (bind)
# K31_DECODE_SPLIT=1: a decode step of more than 8 rows (several requests' verify blocks in one batch) runs each MoE
# layer as one call per verify block instead of one call for the batch, so every request gets what a request alone
# gets: the RTX band (it takes calls of up to 8 rows), the call's 16 staging slots, the per-row staging caps, the
# resident caps and the accepted-row residency scores. Without it a batch of 9-16 rows loses the band and shares one
# call's slots, and a batch of more than 16 rows (4 requests x (1 + 4 drafts) = 20) takes the one-round fallback
# below: no band, 16 slots for the whole batch, and residency scores from every row with the decay of a prompt chunk.
# Applies to calls captured into CUDA graphs and to eager decode-only steps of up to K31_DECODE_SPLIT_MAX rows
# (default 40); prompt chunks keep the prompt path. Not with K31_DEFER, K31_PRESTAGE, K31_MEGAMOE_NOPREP, K31_LATE.
# K31_DECODE_SPLIT_ROWS=n (1..8, default 8 = the band's call size): the largest batch that stays one call. With few
# drafts several requests fit in 8 rows (4 requests x (1 + 1 draft)) and share one call's band experts, staging
# slots and K31_STAGE_CAP; n = the rows of one verify block gives every request its own call there too.
DECODE_SPLIT = os.environ.get("K31_DECODE_SPLIT", "0") == "1"
SPLIT_ROWS = int(os.environ.get("K31_DECODE_SPLIT_ROWS", "8") or 8)   # the largest call that is not split
SPLIT_MAX = int(os.environ.get("K31_DECODE_SPLIT_MAX", "40"))
PEND_ROWS = 64                      # rows of the accepted-row buffers with the split (residency.py)
_SPLIT: dict = {"block": SPLIT_ROWS, "rows": SPLIT_ROWS, "P": 1, "sub": False, "row0": 0, "decode": False,
                "seen": set()}


def split_block(P: int, rows: int = SPLIT_ROWS) -> int:
    """Rows per call of a split batch: as many whole verify blocks (P rows each) as fit in `rows`, else `rows`."""
    return P * (rows // P) if P <= rows else rows


def split_blocks(T: int, block: int, rows: int = SPLIT_ROWS) -> list:
    """The row ranges [(a, b), ...] of a decode batch of T rows as calls of `block` rows (a whole number of verify
    blocks, at most `rows`); a tail of up to `rows` rows (the last blocks plus a graph's padding rows) is one call."""
    out, a = [], 0
    while T - a > rows:
        out.append((a, a + block))
        a += block
    if T > a:
        out.append((a, T))
    return out


class _PadRows:
    """MegaMoE skips the rows the forward context marks as padding: a call on rows [a, b) of the batch needs the
    mask's rows (as prompt_path._PaddingSlice; a no-op without a forward context or a mask)."""

    def __init__(self, a, b):
        self.a, self.b, self.ctx, self.p = a, b, None, None

    def __enter__(self):
        try:
            from vllm.forward_context import get_forward_context, is_forward_context_available
            if is_forward_context_available():
                ctx = get_forward_context()
                p = getattr(ctx, "is_padding", None)
                if p is not None and p.shape[0] > self.b - self.a:
                    self.ctx, self.p = ctx, p
                    ctx.is_padding = p[self.a:self.b]
        except ImportError:
            pass
        return self

    def __exit__(self, *exc):
        if self.ctx is not None:
            self.ctx.is_padding = self.p


def _split_wanted(hidden_states) -> bool:
    """K31_DECODE_SPLIT: this MoE call is a decode batch to run block by block (not already one of its blocks; a
    call inside a graph capture, or an eager step in which no request brings more rows than a verify block)."""
    return DECODE_SPLIT and hidden_states.dim() == 2 and not _SPLIT["sub"] and \
        _SPLIT["rows"] < hidden_states.shape[0] <= SPLIT_MAX and \
        (_SPLIT["decode"] or torch.cuda.is_current_stream_capturing() or _breakable_capture())


def _split_call(one, hidden_states, idx: int = -1):
    """K31_DECODE_SPLIT: one MoE layer over a decode batch as one call per verify block (split_blocks). `one(rows)`
    is the whole decode-size path (router, band, plan, staging, MegaMoE, finish, up-projection) on those rows; the
    plan reads the block's first row from _SPLIT for the accepted-row buffers. A call's output is copied out before
    the next call runs (kernels may return buffers they reuse)."""
    T = hidden_states.shape[0]
    blocks = split_blocks(T, _SPLIT["block"], _SPLIT["rows"])
    if T not in _SPLIT["seen"]:
        _SPLIT["seen"].add(T)
        _LOG(f"decode split: a batch of {T} rows as calls of {[b - a for a, b in blocks]} rows (first seen in layer "
             f"{idx}, capturing {torch.cuda.is_current_stream_capturing()})")
    out = None
    _SPLIT["sub"] = True
    try:
        for a, b in blocks:
            _SPLIT["row0"] = a
            with _PadRows(a, b):
                y = one(hidden_states[a:b])
            if out is None:
                out = torch.empty((T,) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device)
            out[a:b].copy_(y)
    finally:
        _SPLIT["sub"], _SPLIT["row0"] = False, 0
    return out


# K31_DECODE_JOINT=1|2 (with K31_DECODE_SPLIT=1): a decode batch that the split runs as one MoE call per verify block
# shares the work that does not depend on the request. Every request keeps its own plan (unitk with its row caps,
# stage cap, slot group and accepted-row offsets), its own RTX band call and its own staged records, as in the split:
# the same picks are computed, staged, sent to the band or given the mean.
#   1: the dense parts of a MoE layer (router gate + top-k, latent down-projection, shared experts, RMSNorm +
#      up-projection) run once per "dense chunk" (consecutive verify blocks of at most K31_DECODE_JOINT_DENSE rows in
#      all, default 16 = the largest call of the decode-size kernels) instead of once per block; the MegaMoE section
#      (plan, band send, stager, MegaMoE, band finish / mean) still runs per block. (The int8 GEMV takes 8 rows per
#      launch: a 10-row chunk costs it two launches, as two 5-row blocks do; the router gate, top-k, the shared
#      experts and the fused up-projection take a chunk in one. K31_DECODE_JOINT_DENSE=40 makes the batch one chunk:
#      20 rows are then 3 int8 launches instead of 4, and the up-projection leaves its fused 16-row kernel.)
#   2: as 1, and one MegaMoE call per layer for the whole batch: block j's plan addresses slot group (L + j) % 8, the
#      blocks' records are staged one after another, their band calls are sent ahead (the mailbox holds 4) and
#      finished after the one MegaMoE call, so a resident expert that several requests pick is read once per step and
#      MegaMoE's fixed cost is paid once. Layers or batches it does not cover (more blocks than slot groups, more
#      band calls than mailbox slots, a band without the fused v4 finish) take level 1.
# K31_DECODE_JOINT_CHECK=n: the first n eager decode batches per layer run the split, level 1 and level 2 on the same
# rows and log whether the outputs are bit-equal (start-up dummy runs; the batch then continues with the joint output).
DECODE_JOINT = int(os.environ.get("K31_DECODE_JOINT", "0") or 0)
JOINT_DENSE_ROWS = int(os.environ.get("K31_DECODE_JOINT_DENSE", "16") or 16)
JOINT_CHECK = int(os.environ.get("K31_DECODE_JOINT_CHECK", "0") or 0)
# K31_DECODE_JOINT_STAT=1 (a counting run, not a speed run: ~15 small launches per layer): level 2 counts, per joint
# call, the records staged, the records another block of the call stages too, and the distinct rows MegaMoE reads
# against the sum of the blocks' own; logged every 2,000 steps.
JOINT_STAT = os.environ.get("K31_DECODE_JOINT_STAT", "0") == "1"
BAND_NSLOT = 4                      # band_mailbox.h NSLOT: calls the mailbox holds before one is finished
_JT: dict = {"on": False, "r0": 0, "msum": None, "checks": {}, "seen": set(), "level": {}}


def dense_chunks(blocks: list, rows: int = JOINT_DENSE_ROWS) -> list:
    """Consecutive verify blocks grouped into dense chunks of at most `rows` rows: [(first row, end row, [blocks])].
    The fewest chunks a greedy fill needs, then as even as whole blocks allow (20 rows of 5-row blocks at 16:
    10 + 10, not 15 + 5). A block larger than `rows` is a chunk by itself."""
    if not blocks:
        return []
    n, used = 1, 0
    for a, b in blocks:                                  # the fewest chunks
        if used and used + (b - a) > rows:
            n, used = n + 1, 0
        used += b - a
    out, cur, left = [], [], n
    total = blocks[-1][1] - blocks[0][0]
    target = total / n
    for i, (a, b) in enumerate(blocks):
        size = (cur[-1][1] - cur[0][0]) if cur else 0
        rest = len(blocks) - i                           # blocks not placed yet, this one included
        if cur and (size + (b - a) > rows or (left > 1 and size >= target - 1e-9) or rest < left):
            out.append((cur[0][0], cur[-1][1], cur))
            cur, left = [], left - 1
        cur.append((a, b))
    out.append((cur[0][0], cur[-1][1], cur))
    return out


def _joint_level(st, blocks: list) -> int:
    """The joint level this layer runs a batch of these blocks at (see K31_DECODE_JOINT)."""
    if DECODE_JOINT < 2 or getattr(st, "jstagers", None) is None or len(blocks) > GROUPS or \
            max(b - a for a, b in blocks) > ONE_ROWS:
        return min(DECODE_JOINT, 1)
    band_on = st.band_map is not None and getattr(st, "band_ok", True)
    if band_on and not (TAIL1 and BAND_FIN_SIDE and BAND_SIDE and not BAND_HINT and _BAND.get("fused") is not None
                        and getattr(st, "jtags", None) is not None and len(blocks) <= BAND_NSLOT
                        and max(b - a for a, b in blocks) <= 8):
        return 1
    return 2


def _joint_stat(rows: list) -> None:
    """K31_DECODE_JOINT_STAT: [calls, blocks, records staged, records staged by an earlier block too, rows read by the
    blocks one by one, distinct rows of the call] += this call's (device counters, no host sync)."""
    dev = rows[0][0].device
    if _JT.get("stat") is None:
        _JT["stat"] = torch.zeros(6, dtype=torch.int64, device=dev)
    c = _JT["stat"]
    sl = torch.stack([sg for _r, sg in rows]).long()                     # [B, SLOTS] bank indices, -1 = none
    live = sl >= 0
    B = sl.shape[0]
    flat = sl.reshape(-1)
    same = (flat[:, None] == flat[None, :]) & live.reshape(-1)[:, None] & live.reshape(-1)[None, :]
    blk = torch.arange(B, device=dev).repeat_interleave(sl.shape[1])
    dup = (same & (blk[None, :] < blk[:, None])).any(1).sum()            # staged by an earlier block as well
    own = torch.zeros(B, 2048, dtype=torch.int64, device=dev)
    for j, (r, _sg) in enumerate(rows):
        ok = r >= 0
        own[j].scatter_(0, r.long().clamp_min(0).reshape(-1), ok.reshape(-1).long())
    # (staged rows of different blocks are different rows, so only resident rows can coincide)
    c += torch.stack([torch.ones((), dtype=torch.int64, device=dev), torch.full((), B, dtype=torch.int64, device=dev),
                      live.sum(), dup, (own > 0).sum(), (own.sum(0) > 0).sum()])


def _joint_moe(self, st, chunks: list, fronts: list, lds: list):
    """K31_DECODE_JOINT=2: the MegaMoE section of one layer for a decode batch, one MegaMoE call for all its rows.
    fronts[c] = (latent rows, weights, ids, ...) of dense chunk c (the latent rows may be pending on the aux stream:
    lds holds their events). Per verify block: unitk's plan into its own slot group, its band call (sent ahead,
    its tag kept for the finish), its records staged; then one MegaMoE call; then per block the band finish (with the
    cut picks' mean sum and the RMSNorm statistic, K31_TAIL1) or the mean substitution. Returns the routed latent sum
    [T, LATENT]; the band finishes may be pending on the fin stream (_FIN, joined by the routed output transform)."""
    from . import unitk
    from . import defer as _dm
    if st.row_map is None:
        raise RuntimeError(f"K31 layer {st.idx}: experts not bound (load_model hook did not run)")
    if _RES and st.idx == _FIRST_MOE[0]:                 # K31_RES_STREAM (as _forward): an eager step waits for the
        _RES[0].join_eager()                             # tick's GPU work here
    main = torch.cuda.current_stream()
    band_on = st.band_map is not None and getattr(st, "band_ok", True)
    tail = band_on and getattr(st, "msum", None) is not None
    gb, fz = _BAND.get("gb"), _BAND.get("fused")
    base_g = st.idx % GROUPS
    blocks = []                                          # (a, b, chunk index, first row of the chunk)
    for c, (ca, _cb, blks) in enumerate(chunks):
        blocks += [(a, b, c, ca) for a, b in blks]
    band_ev = None
    if band_on:                                          # the sends, each block's rows and work list, on the band stream
        if _BSIDE["stream"] is None:
            _BSIDE["stream"] = torch.cuda.Stream(device=main.device)
        bs = _BSIDE["stream"]
        bs.wait_stream(main)
        for ev in lds:                                   # the send quantizes the latent rows
            bs.wait_event(ev)
        with torch.cuda.stream(bs):
            for j, (a, b, c, ca) in enumerate(blocks):
                x, w, ids = fronts[c][:3]
                _band_send(st, x[a - ca:b - ca], ids[a - ca:b - ca], w[a - ca:b - ca],
                           served=st.jserved[j] if BAND_SERVED else None)
                st.jtags[j:j + 1].copy_(gb.st[1:2])       # the call's tag (0: nothing sent) for its finish
            band_ev = torch.cuda.Event()
            band_ev.record(bs)
    rows, served = [], []
    for j, (a, b, c, ca) in enumerate(blocks):           # the plans: block j stages into slot group (L + j) % GROUPS
        _x, w, ids = fronts[c][:3]
        r, stage = unitk.plan(ids[a - ca:b - ca], w[a - ca:b - ca], st.row_map, st.bank_map, st.strong,
                              SLOTS * ((base_g + j) % GROUPS), score=st.score, lam=LAM, stats=st.stats,
                              pend=getattr(st, "pend", None), resmax=st.resmax, rowcap=_RCAP[0], cap=st.stage_cap,
                              row0=a, rowprior=_RPRI[0])
        rows.append((r, stage))
        served.append(st.jserved[j] if (band_on and BAND_SERVED) else None)
    fs = None
    if tail:
        if _FIN["stream"] is None:
            _FIN["stream"] = torch.cuda.Stream(device=main.device)
        fs = _FIN["stream"]
        if st.cut_mean is not None:                      # the mean sums beside the stagers and MegaMoE
            fs.wait_stream(main)                         # the plans' row ids
            with torch.cuda.stream(fs):
                fs.wait_event(band_ev)                   # the sends' served flags
                for j, (a, b, c, ca) in enumerate(blocks):
                    _x, w, ids = fronts[c][:3]
                    sv = served[j]
                    _dm.cut_mean_pre(_JT["msum"][a:b], rows[j][0], ids[a - ca:b - ca], w[a - ca:b - ca], st.cut_mean,
                                     band=st.band_map if sv is None else None, served=sv)
    if JOINT_STAT:
        _joint_stat(rows)
    for j in range(len(blocks)):
        st.jstagers[(base_g + j) % GROUPS](st.bank_ptr, rows[j][1])
    for ev in lds:                                       # prep and MegaMoE read the latent rows
        main.wait_event(ev)
    one = len(fronts) == 1
    x = fronts[0][0] if one else torch.cat([f[0] for f in fronts])
    w = fronts[0][1] if one else torch.cat([f[1] for f in fronts])
    ids = fronts[0][2]
    rall = rows[0][0] if len(rows) == 1 else torch.cat([r for r, _s in rows])
    out = _orig["forward"](self, x, w, rall.to(ids.dtype), None)
    if tail and out.dim() == 2 and out.is_contiguous() and out.dtype == torch.bfloat16:
        if st.up_eps is None:
            moe = _MOE.get(st.idx)
            rot = getattr(moe, "routed_output_transform", None) if moe is not None else None
            st.up_eps = float(getattr(getattr(rot, "norm", None), "variance_epsilon", 1e-6))
        fs.wait_stream(main)                             # after MegaMoE
        with torch.cuda.stream(fs):
            fs.wait_event(band_ev)
            for j, (a, b, c, ca) in enumerate(blocks):
                gb.st[1:2].copy_(st.jtags[j:j + 1])       # this block's call
                _band_finish(st, out[a:b], msum=_JT["msum"][a:b] if st.cut_mean is not None else None,
                             rstd=st.jrstd[a:], eps=st.up_eps)
            fev = torch.cuda.Event()
            fev.record(fs)
        _FIN["ev"] = fev                                 # joined in the routed output transform (_fin_join)
        _FIN["keep"] = (rows, fronts, x, w, rall)        # referenced until the join
        _JT["rstd"] = st.jrstd
        return out
    assert not band_on, "K31_DECODE_JOINT=2: a band layer without the fused tail finish must take level 1"
    if st.cut_mean is not None:                          # one launch for the batch (the kernel works row by row)
        _dm.cut_mean(out, rall, ids if one else torch.cat([f[2] for f in fronts]), w, st.cut_mean)
    _JT["rstd"] = None
    return out
# K31_W8I_DRAFT=1: the drafter's linears and the FP8 head copy on the int8 GEMV (w8i.py) instead of W8A16 / Marlin
W8I_DRAFT = os.environ.get("K31_W8I_DRAFT", "0") == "1"
# K31_FORCE_DRAFTS=1: in forced decoding (force.py, the KLD harness) the
# drafts of a forced request are the next reference tokens up to an accepted count drawn per step from the bench's
# per-position acceptance (K31_FORCE_ACCEPT, P(draft i accepted); pooled DRAFTS-4 bench windows 11-18: 2.54 tokens
# per step), then a wrong token: verify forwards keep their 5 rows and the engine advances ~2.5 tokens per step, as
# when it serves, so the expert cache adapts per token at the serving rate (the drafter's own guesses of reference
# text accepted ~1.2-1.9 per step: the cache adapted ~2x more per token than in serving). The drafter still runs.
FORCE_DRAFTS = os.environ.get("K31_FORCE_DRAFTS", "0") == "1"
FORCE_ACCEPT = [float(v) for v in os.environ.get("K31_FORCE_ACCEPT", "0.665,0.432,0.266,0.180").split(",") if v]
FORCE_SEED = int(os.environ.get("K31_FORCE_SEED", "1234"))
# K31_FORCE_REJECT=drafter: the first emulated rejection is the drafter's own draft when it differs
# from the reference (else token + 1) and the later drafts stay the drafter's, so rejected rows route like plausible
# text, as in serving ("next", the default: token + 1 at every rejected position: junk rows that took band-cap places
# and staging slots from the real rows of their call)
FORCE_REJECT = os.environ.get("K31_FORCE_REJECT", "next")
_FD: dict = {"gen": None, "p": None, "steps": 0, "rows": 0}


def _force_drafts(idx_mapping, seq_lens, draft_tokens, cu_num_logits) -> None:
    """Rewrite the forced requests' drafts in the runner's per-slot draft table before the verify inputs are built
    (vllm's combine_sampled_and_draft_tokens): draft j of batch row b sits at input position F + j, F = seq_len -
    num_logits (the last sampled token), and must be the token at that sequence position, force.tok[slot, F + j - 1];
    the first emulated rejection gets (token + 1) % V; slots without a forced token keep the drafter's drafts."""
    from . import force
    ft = force.INSTANCE[0]
    S = draft_tokens.shape[-1]
    if ft is None or S == 0:
        return
    nr = idx_mapping.shape[0]
    dev = draft_tokens.device
    if _FD["gen"] is None:
        _FD["gen"] = torch.Generator(device=dev).manual_seed(FORCE_SEED)
        p = (FORCE_ACCEPT + [0.0] * S)[:S]
        _FD["p"] = torch.tensor(p, dtype=torch.float32, device=dev)
    slots = idx_mapping.long()
    nl = (cu_num_logits[1:nr + 1] - cu_num_logits[:nr]).long()
    nd = nl - 1
    F = seq_lens[:nr].long() - nl
    j = torch.arange(1, S + 1, device=dev)
    pos = (F[:, None] + j[None, :] - 1).clamp(0, ft.tok.shape[1] - 1)
    forced = ft.tok[slots[:, None], pos].to(draft_tokens.dtype)
    u = torch.rand(nr, generator=_FD["gen"], device=dev)
    n = (u[:, None] < _FD["p"][None, :]).sum(1)
    cur = draft_tokens[slots]
    if FORCE_REJECT == "drafter":
        rej = torch.where(cur != forced, cur, (forced + 1) % 163840)
        want = torch.where(j[None, :] <= n[:, None], forced, rej)
        lim = torch.minimum(n + 1, nd)
    else:
        want = torch.where(j[None, :] <= n[:, None], forced, (forced + 1) % 163840)
        lim = nd
    use = (forced >= 0) & (j[None, :] <= lim[:, None]) & (F[:, None] >= 0)
    draft_tokens[slots] = torch.where(use, want, cur)


def _patch_combine(runner_module) -> None:
    """K31_FORCE_DRAFTS: wrap the runner module's combine_sampled_and_draft_tokens (verify inputs from the per-slot
    draft table)."""
    orig = getattr(runner_module, "combine_sampled_and_draft_tokens", None)
    if orig is None or getattr(orig, "_k31", False):
        _LOG(f"forced drafts: {runner_module.__name__} has no combine_sampled_and_draft_tokens; not installed")
        return

    def combine(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len, draft_tokens,
                cu_num_logits, num_logits, num_new_sampled_tokens=1):
        if num_new_sampled_tokens == 1 and draft_tokens.shape[-1] > 0:
            _force_drafts(idx_mapping, seq_lens, draft_tokens, cu_num_logits)
        return orig(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len, draft_tokens,
                    cu_num_logits, num_logits, num_new_sampled_tokens)
    combine._k31 = True
    runner_module.combine_sampled_and_draft_tokens = combine
    _LOG(f"forced drafts: reference tokens as drafts, accepted per step with P(i) = {FORCE_ACCEPT} "
         f"({1 + sum(FORCE_ACCEPT):.2f} tokens per step at full drafts), seed {FORCE_SEED}")
# K31_TAIL1=1 (needs K31_BAND_FIN_SIDE=1 and librtxband_fused_v4.so): the cut picks' mean sum is
# computed into a float32 buffer beside the stager and MegaMoE (defer.cut_mean_pre on the fin stream right after
# unitk), and one kernel after MegaMoE (the v4 band finish) adds the band rows and that sum in one rounding and writes
# the routed up-projection's RMSNorm statistic, which the up GEMM reads (w8a16 NORM 2): the cut-mean, rms_norm and add
# kernels leave the chain
TAIL1 = os.environ.get("K31_TAIL1", "0") == "1"
_FUSED_LIB = os.environ.get("K31_BAND_FUSED_LIB", "")
_POLICY: dict = {}
# RTX band (band.py): picks without a resident row whose expert has a record in the band sidecar's store
# (rtxband/code/band/sidecar.py) are computed on the RTX PRO 6000 from 2-bit records, on time. K31_BAND = the mailbox
# segment (/dev/shm/...); K31_BAND_N = records per layer: the experts ranked K31_HBM .. K31_HBM + N - 1 by K31_RANK
# (the bank's static start follows them: native rows, band and bank are disjoint). The sidecar writes
# <segment>.map.npy (int32 [93, 896] store slots, -1 = none) and <segment>.ready once its store is loaded.
BAND = os.environ.get("K31_BAND", "")
BAND_N = int(os.environ.get("K31_BAND_N", "0")) if BAND else 0
BAND_LIB = os.environ.get("K31_BAND_LIB", "/engine/band/librtxband_gb300.so")
BAND_TIMEOUT_US = float(os.environ.get("K31_BAND_TIMEOUT_US", "60"))      # the finish's spin bound (decode calls)
BAND_EAGER_TIMEOUT_US = float(os.environ.get("K31_BAND_EAGER_TIMEOUT_US", "5000"))
# K31_BAND_BULK=1: prompt chunks of more than K31_BAND_BULK_MIN rows send their band picks in one
# request per MoE layer (band.Bulk; the sidecar parks its service kernel while a prompt is read) instead of decode
# calls of 8 rows; K31_BAND_BULK_MIN stays above the largest captured graph size, so a bulk step never replays one
BAND_BULK = os.environ.get("K31_BAND_BULK", "0") == "1"
BAND_BULK_MIN = int(os.environ.get("K31_BAND_BULK_MIN", "48"))
_BULK: list = [None]
BAND_WAIT_S = float(os.environ.get("K31_BAND_WAIT_S", "1200"))            # bind waits this long for the sidecar
BAND_FUSED = os.environ.get("K31_BAND_FUSED", "1") == "1"   # 2 launches per call (rtxband_fused.cu) instead of 6
# K31_BAND_SIDE=1: the band send (plan + MXFP8 rows + mailbox release) on its own side stream forked after the router,
# joined before the finish: unitk, the stager and MegaMoE no longer wait for it (~4 us per layer off the chain). Its own
# stream, not the shared experts' (the send must not queue behind them: the RTX needs its rows early).
BAND_SIDE = os.environ.get("K31_BAND_SIDE", "0") == "1"
_BSIDE: dict = {"stream": None}
# K31_SHARED_SIDE=1: the shared experts (they read only the MoE input) run on a side stream forked at the top of
# KimiMoE.forward and joined just before the routed up-projection adds their output, so they overlap the router,
# unitk, the band send, staging and MegaMoE instead of following them (the 11-17 decode trace has them after MegaMoE
# on the critical path, ~40 us per layer). The same fork / join pattern as the deferral window (captured as a graph
# branch).
# K31_SHARED_SIDE=2: fork after the router GEMV / latent down / top-k instead of at the top of KimiMoE.forward (forked
# at the top, the side gate_up, the router gate and the latent down are launched together and need ~48 us of
# whole-GPU time between them; the chain's router finished last and nothing was left to hide under the stager). Forked
# after the router, the side GEMMs meet only single-CTA kernels and the 24-CTA stager; give them one-wave configs (or
# CUTLASS: fp8c) so they leave SM slots to the chain.
SHARED_SIDE_MODE = int(os.environ.get("K31_SHARED_SIDE", "0") or 0)
SHARED_SIDE = SHARED_SIDE_MODE > 0
# K31_SHARED_EARLY=1: the shared experts fork right after the router GEMV,
# before top-k, so their gate_up can end before MegaMoE takes the SMs (decode-size calls on the deferred-join path)
SHARED_EARLY = os.environ.get("K31_SHARED_EARLY", "0") == "1"
# K31_SHARED_SIDE=3: as 2, and the router's latent down-projection (vLLM runs it on its aux stream and
# joins it before _maybe_overlap_router_and_down_proj returns) is joined only where its output is read: the band send
# (its own branch) and prep / MegaMoE. unitk and the stager read top-k's outputs only, so they no longer wait for it.
_LD = {"ev": None}
_LD_SEEN = {"nojoin": 0, "vllm": 0}
# K31_MEGAMOE_NOPREP=1 (noprep.py, needs K31_SHARED_SIDE=3): decode-size calls feed MegaMoE without vLLM's prep kernel:
# the latent down's epilogue writes the MXFP8 rows, the plan kernel (top-k + unitk) the row ids and weights
NOPREP = os.environ.get("K31_MEGAMOE_NOPREP", "0") == "1"


def _nop():
    from . import noprep
    return noprep


def _breakable_capture() -> bool:
    """vLLM's breakable CUDA graph capture is active: it runs the latent down serially then (a fork and its join must
    stay in one graph segment), and so does K31_SHARED_SIDE=3."""
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        return bool(BreakableCUDAGraphCapture.is_active())
    except Exception:                                    # noqa: BLE001
        return False


def _ld_wait(stream) -> None:
    """`stream` waits for the pending latent down-projection (K31_SHARED_SIDE=3); it stays pending for main."""
    if _LD["ev"] is not None:
        stream.wait_event(_LD["ev"])


def _ld_join() -> None:
    """The current stream joins the pending latent down-projection, if any (before anything reads its output)."""
    if _LD["ev"] is not None:
        torch.cuda.current_stream().wait_event(_LD["ev"])
        _LD["ev"] = None
# K31_BAND_FIN_SIDE=1: the band finish (a one-CTA bounded wait on the RTX's result + the add into the routed sum) and
# the mean substitution run on their own branch forked after MegaMoE, joined at the routed output transform: the shared
# experts after MegaMoE (or their tail) overlap it instead of following it (7.6 us x 92)
BAND_FIN_SIDE = os.environ.get("K31_BAND_FIN_SIDE", "0") == "1"
_FIN: dict = {"stream": None, "ev": None}
# K31_GATE_SIDE=1: the DSA layers' output gate (g_proj, x -> 96 x 128) runs on the side stream forked at the top of
# KimiK3DSAAttention.forward and joined before the indexer's score (the layer's first graph break), so it overlaps the
# attention front; vLLM's MLA wrapper, which calls g_proj after the attention (21.8 us FP8 + 4.9 us quant per DSA layer
# on the critical path in the 11-17 trace), then gets the precomputed gate. Only the unpacked wrapper path (FP8 /
# W8A16 linears); vLLM's own packed BF16 path already overlaps it.
# K31_GATE_SIDE=2: fork g_proj right after q_b (so q_b no longer shares the GPU with it: 28-30 us
# against 11 alone) and join where the MLA wrapper multiplies by the gate, after the indexer and the attention; only in
# decode-size calls outside a breakable (piecewise) capture: the decode graphs are FULL, with no break in between
GATE_SIDE_MODE = int(os.environ.get("K31_GATE_SIDE", "0") or 0)
GATE_SIDE = GATE_SIDE_MODE > 0
_GATE: dict = {"ev": None, "late": False}
# K31_ROUTER_GEMV=1: the MoE gate (BF16 weights [896, 7168], float32 logits) for calls of <= 16 rows by w8a16's
# split-K kernel (linear16) instead of GateLinear's cuteDSL ll_bf16_gemm (7.3 us at 3 rows = 1.8 TB/s, 11-17 trace)
ROUTER_GEMV = os.environ.get("K31_ROUTER_GEMV", "0") == "1"
# K31_DRAFT_W8=1: the DSpark drafter's own linears matching K31_DRAFT_W8_LAYERS (default its MLPs) get FP8 W8A16 copies
# at load (the BF16 weights stay for any other reader) and run on w8a16's split-K kernel for calls of <= 16 rows. The
# 11-17 trace has the drafter's BF16 GEMMs at ~1.0 ms per step (5.8 TB/s: efficient, but 2 bytes per weight). The
# target's arithmetic is unchanged; only the drafts (acceptance) can move.
DRAFT_W8 = os.environ.get("K31_DRAFT_W8", "0") == "1"
# K31_FUSE_UP=1: KimiRoutedOutputTransform's RMSNorm + up-projection (W8A16) + shared-expert residual add in one launch
# (w8a16.linear_norm_res) for calls of <= 16 rows: drops the rms_norm (4.4 us) and add (1.5 us) kernels per layer
FUSE_UP = os.environ.get("K31_FUSE_UP", "0") == "1"
DRAFT_W8_LAYERS = [p for p in os.environ.get("K31_DRAFT_W8_LAYERS", "mlp.gate_up_proj,mlp.down_proj").split(",") if p]
# K31_DRAFT_W8_HEAD=1: the drafter's base logits (the target's shared BF16 lm_head, 2.35 GB, 405 us per step in the
# 11-17 trace) from an FP8 W8A16 copy of the head (+1.2 GB); the target's own logits keep the BF16 head
DRAFT_W8_HEAD = os.environ.get("K31_DRAFT_W8_HEAD", "0") == "1"
# K31_HEAD_W8=1: the target's logits too from the FP8 copy (415 us BF16 per step; changes the served logits, so the
# KLD harness sees it: its references are rebuilt with the BF16 weight, LM_HEAD, which stays)
HEAD_W8 = os.environ.get("K31_HEAD_W8", "0") == "1"
# K31_DSA_FUSED=1: the DSA layers' small decode kernels (index key insert, index query quant + weight fold + BF16 copy,
# MLA cache insert; q_pe copy + FP8 query quant + head pad; sigmoid gate) in three launches, bit-exact against vLLM's
# path (dsa_fused.py; parts K31_DSA_FUSED_PARTS, rows K31_DSA_FUSED_ROWS, in-engine check K31_DSA_FUSED_CHECK)
DSA_FUSED = os.environ.get("K31_DSA_FUSED", "0") == "1"
# K31_DRAFT_TOPP=1 (drafttopp.py): each request's own top-k / top-p also truncate the DSpark drafter's proposal (the
# drafts and the cached draft logits the verifier reads; exact: the verifier reads the q the drafts came from).
# K31_DRAFT_ARGMAX=1: the drafts of sampled requests are the drafter's argmax (a point-mass q). One of the two.
DRAFT_TOPP = os.environ.get("K31_DRAFT_TOPP", "0") == "1"
# K31_DRAFT_FULL_KV=1: the drafter's sliding-window attention layers are given to vLLM's KV manager as full attention.
# A sliding-window group turns off vLLM's fine-grained (64-token) prefix-cache hits, so every request re-reads its
# whole prompt (the chat template's shared prefix included) and prompts over 40 tokens leave the CUDA graphs for the
# eager prompt path: ~1 s added to every first token with the final DSpark drafter. Attention still applies the
# layer's window; the drafter's KV is kept for the whole context (~2.5 GiB more at 262,144 tokens).
DRAFT_FULL_KV = os.environ.get("K31_DRAFT_FULL_KV", "0") == "1"
DRAFT_ARGMAX = os.environ.get("K31_DRAFT_ARGMAX", "0") == "1"
_DRAFT_HEAD = [False]
_SIDE: dict = {"stream": None}
_BAND: dict = {"gb": None, "stats": None, "eager_call": None, "eager_args": None}
ATTN_RES_BLOCK = 12
RING = int(os.environ.get("K31_RING", "0"))
assert 0 <= RING <= SLOTS
SHARED = STAGE_ROWS + (DEFER_ROWS if DEFER else 0)
assert SHARED % 128 == 0, "shared rows must fill whole 2 MiB granules"
WINDOW = WINDOW_TEST or DEFER
_DEF = {"pending": None, "late": None, "stager": None, "count": None}
_MOE: dict = {}             # layer idx -> vLLM KimiMoE module (completion of late shares)
_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
_flags._log[0] = _LOG
_orig: dict = {}
LAYERS: dict = {}           # layer idx -> LayerState
_rowmap_cache: dict = {}
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
# the depth-2 code map as values (a staged record): magnitude codes 0-3 -> 0.75, 4-7 -> 3.0, in units of the block scale
FP4_TRICK = torch.tensor([.75] * 4 + [3.0] * 4 + [-.75] * 4 + [-3.0] * 4, dtype=torch.float32)


class LayerState:
    def __init__(self, idx: int, hbm: list, bank: list):
        self.idx = idx
        self.hbm = [int(e) for e in hbm]
        self.bank = [int(e) for e in bank] if MODE == "staged" else []
        both = self.hbm + self.bank
        if len(set(self.hbm)) != len(self.hbm) or len(set(self.bank)) != len(self.bank) or \
                not all(0 <= e < E for e in both) or (PROMOTE <= 0 and len(set(both)) != len(both)):
            raise RuntimeError(f"K31 rowmap layer {idx}: resident and bank experts must be distinct ids in 0..{E - 1}"
                               f" (they may overlap only with promotion on)")
        self.S = len(self.hbm)
        self.slots = SHARED if MODE == "staged" else 0          # shared staging rows come first
        self.slot_base = SLOTS * (idx % GROUPS) if MODE == "staged" else 0
        self.row = [-1] * E
        for r, e in enumerate(self.hbm):
            self.row[e] = self.slots + r
        self.bank_idx = [-1] * E
        for j, e in enumerate(self.bank):
            self.bank_idx[e] = j
        self.row_map = self.bank_map = None      # device int32 [E]: resident row / bank index, or -1
        self.mod = self.stager = self.bank_t = self.bank_buf = None
        self.bank_ptr = 0
        self.arange = None
        self.score = self.stats = None               # residency inputs (residency.py), written by unitk
        self.stagers = None                          # eager rounds: one stager per slot group
        self.layout = None
        self.defer_ok = False
        self.defer_slots = DEFER_SLOTS
        self.cut_mean = None
        self.up_eps = None                 # K31_TAIL1: the routed output transform's RMSNorm eps (first call)
        self.strong, self.defer_from, self.defer_to = STRONG, DEFER_FROM, DEFER_TO
        self.resmax = RES_MAXRANK
        self.stage_cap = STAGE_CAP                   # K31_STAGE_CAP / the policy's stage_cap: records per decode call
        self.band = []                               # experts with a record in the RTX band (static start)
        self.band_map = self.band_call = self.band_args = self.band_stats = None
        self.ring_ok = False
        self.ring_base = -1
        self.ring_owner = self.ring_stager = None


def _rowmap() -> dict:
    if not _rowmap_cache:
        if ROWMAP:
            raw = json.load(open(ROWMAP))
            layers = raw.get("layers", raw)
            _rowmap_cache.update({int(k): {"hbm": v["hbm"], "bank": v.get("bank", [])} for k, v in layers.items()})
            src = ROWMAP
        elif RANK:
            raw = json.load(open(RANK))
            # with promotion on, the bank holds the first BANK by usage including the resident ones (an evicted
            # expert stays stageable); otherwise the next BANK after the resident ones
            # K31_BANK_EXCL=1: the bank never holds an expert that has an HBM row (the next BANK after the resident
            # ones even with promotion on; promotions swap: the promoted expert's slot takes the evicted one's record)
            b0 = HBM if (BANK_EXCL or PROMOTE <= 0) else 0
            if BAND_N and b0 != HBM:
                raise RuntimeError("K31: the RTX band needs an exclusive bank (K31_BANK_EXCL=1)")
            b0 += BAND_N
            for k, v in raw["layers"].items():
                h = _hbm_of(int(k))                       # K31_HBM_MAP: this layer's resident rows
                nb = _layer_count(BAND_MAP_PATH, int(k), BAND_N) if BAND_N else 0   # K31_BAND_MAP
                g = _layer_count(BANK_MAP_PATH, int(k), BANK)                       # K31_BANK_MAP
                c0 = (h if (BANK_EXCL or PROMOTE <= 0) else 0) + nb
                if BANK_EXCL and h + nb + g > len(v):
                    raise RuntimeError(f"K31: layer {k}: {h} rows + {nb} band + {g} bank records exceed its {len(v)} "
                                       f"ranked experts (K31_HBM_MAP / K31_BAND_MAP / K31_BANK_MAP)")
                _rowmap_cache[int(k)] = {"hbm": sorted(int(e) for e in v[:h]),
                                         "band": [int(e) for e in v[h:h + nb]],
                                         "bank": sorted(int(e) for e in v[c0:c0 + g])}
            src = f"{RANK} (first {HBM if not HBM_MAP_PATH else HBM_MAP_PATH} by usage resident" + \
                (f", next {BAND_N if not BAND_MAP_PATH else BAND_MAP_PATH} in the RTX band" if BAND_N else "") \
                + (f", next {BANK if not BANK_MAP_PATH else BANK_MAP_PATH} in the bank)" if MODE == "staged" else ")")
        else:
            raise RuntimeError("K31: set K31_ROWMAP or K31_RANK")
        _LOG(f"rowmap {src}: {len(_rowmap_cache)} layers")
    return _rowmap_cache


# --------------------------------------------------------------------------------------- DeepGemmMegaMoEExperts
def _init(self, vllm_config, *, num_experts, num_local_experts, experts_start_idx, **kw):
    m = PREFIX.search(kw.get("prefix", ""))
    idx = int(m.group(1)) if m else -1
    if m is None or num_experts != E or idx not in _rowmap():
        _orig["init"](self, vllm_config, num_experts=num_experts, num_local_experts=num_local_experts,
                      experts_start_idx=experts_start_idx, **kw)
        self._k31 = None
        return
    if kw.get("top_k") != TOPK or kw.get("hidden_size") != LATENT or kw.get("intermediate_size") != INTER:
        raise RuntimeError(f"K31: layer {idx} is not Kimi K3.1 Pro's latent MoE (top_k {kw.get('top_k')}, hidden "
                           f"{kw.get('hidden_size')}, intermediate {kw.get('intermediate_size')})")
    st = LayerState(idx, _rowmap()[idx]["hbm"], _rowmap()[idx]["bank"])
    st.band = list(_rowmap()[idx].get("band", []))
    # staged: a 1-row stand-in now; _bind replaces the four kernel tensors by shared-slot virtual ranges (vmm.py)
    rows = 1 if MODE == "staged" else st.S
    kw["num_logical_experts"] = E
    from . import rowslice                       # K31_MEGAMOE_ROWS: the symmetric buffer sized for that many rows
    kw = rowslice.megamoe_kw(vllm_config, kw)
    _orig["init"](self, vllm_config, num_experts=rows, num_local_experts=rows, experts_start_idx=0, **kw)
    self._k31 = st
    st.mod = self
    LAYERS[idx] = st
    if len(LAYERS) == 1:
        _LOG(f"layer {idx} ({kw.get('prefix')}): mode {MODE}, {st.S} resident rows"
             + (f" after {st.slots} shared staging rows ({SLOTS} per call), {len(st.bank)} experts in the bank, "
                f"strong ranks < {STRONG}"
                if MODE == "staged" else "") + f", the rest cut")


def _plan(st: LayerState, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """-> (row ids for the call [T, K], stage [SLOTS] bank indices or -1, staged-pick mask, slot of each pick, order)
    Fixed shapes only. Candidates: misses of rank < STRONG with a bank record; at most SLOTS distinct experts."""
    T, K = topk_ids.shape
    dev = topk_ids.device
    valid = topk_ids >= 0
    ids = topk_ids.long().clamp_min(0)
    native = st.row_map[ids]
    bank = st.bank_map[ids]
    w = topk_weights.float()
    order = w.argsort(dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, st.arange[:K].expand(T, K))
    cand = valid & (native < 0) & (bank >= 0) & (rank < st.strong)
    score = torch.zeros(E, dtype=torch.float32, device=dev)
    score.scatter_reduce_(0, ids.reshape(-1), torch.where(cand, w + 1.0, 0.0).reshape(-1), "amax")
    return native, bank, cand, score, valid


def _defer_plan(st, topk_ids, topk_weights):
    """This call's deferred picks: rank in [DEFER_FROM, DEFER_TO), no resident row, a bank record; the DEFER_SLOTS
    strongest distinct experts by summed weight^2. Returns (stage list int32 [16], pseudo-row ids int32 [T, K],
    pseudo-row weights [T, K], number deferred)."""
    T, K = topk_ids.shape
    if T <= 8 and FUSED_DEFER:                         # one Triton launch (defer.py), same result as below
        from . import defer
        stage, pids, pw = defer.plan(topk_ids, topk_weights, st.row_map, st.bank_map, st.defer_from, st.defer_to,
                                     st.defer_slots, DEFER_BASE, _DEF["count"])
        return stage, pids, pw, None
    valid = topk_ids >= 0
    e = topk_ids.long().clamp_min(0)
    native = st.row_map[e]
    bank = st.bank_map[e]
    wf = topk_weights.float()
    order = wf.argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, st.arange[:K].expand(T, K))
    cand = valid & (native < 0) & (bank >= 0) & (rank >= st.defer_from) & (rank < st.defer_to)
    energy = torch.zeros(E, dtype=torch.float32, device=topk_ids.device)
    energy.index_add_(0, e.reshape(-1), torch.where(cand, wf * wf, 0.0).reshape(-1))
    v, top = energy.topk(SLOTS)
    ok = (v > 0) & (st.arange[:SLOTS] < st.defer_slots)
    slot_of = torch.full((E,), -1, dtype=torch.long, device=topk_ids.device)
    slot_of.scatter_(0, top, torch.where(ok, st.arange[:SLOTS], -1))
    ds = slot_of[e]
    deferred = cand & (ds >= 0)
    stage = torch.where(ok, st.bank_map[top], -1).to(torch.int32)
    pids = torch.where(deferred, DEFER_BASE + ds, -1).to(torch.int32)
    pw = torch.where(deferred, topk_weights, torch.zeros_like(topk_weights))
    return stage, pids, pw, deferred.sum()


class _Padding2:
    """MegaMoE skips rows the forward context marks as padding (VLLM_MOE_SKIP_PADDING): with 2T rows (T real, then T
    pseudo rows carrying the previous layer's deferred picks) the mask is repeated for the pseudo rows."""

    def __init__(self, T):
        self.T, self.ctx, self.p = T, None, None

    def __enter__(self):
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if is_forward_context_available():
            ctx = get_forward_context()
            p = getattr(ctx, "is_padding", None)
            if p is not None:
                self.ctx, self.p = ctx, p
                q = p[:self.T]
                ctx.is_padding = torch.cat([q, q])
        return self

    def __exit__(self, *a):
        if self.ctx is not None:
            self.ctx.is_padding = self.p


def _forward_defer(self, st, hidden_states, topk_weights, topk_ids, ids, shared_experts, main, band_on=False):
    T = topk_ids.shape[0]
    pend = _DEF["pending"]
    _DEF["pending"] = None
    if DEFER_LAND == "layer" and pend is not None and pend["idx"] == st.idx - 1 and pend["T"] == T:
        hs2 = torch.cat([hidden_states, pend["x"]])
        w2 = torch.cat([topk_weights, pend["w"]])
        ids2 = torch.cat([ids.to(topk_ids.dtype), pend["ids"].to(topk_ids.dtype)])
        with _Padding2(T):
            out2 = _orig["forward"](self, hs2, w2, ids2, shared_experts)
        out = out2[:T]
        _DEF["late"] = (pend["idx"], pend["u"], out2[T:])
    else:
        out = _orig["forward"](self, hidden_states, topk_weights, ids.to(topk_ids.dtype), shared_experts)
    if band_on:                           # on time: part of u_now, before any late completion is taken from it
        _band_finish(st, out)
    bmap = st.band_map if band_on else None
    if st.ring_ok and T <= 8 and st.score is not None:     # (the bind-time self-check runs before residency exists)
        from . import defer
        late = st.defer_ok
        src, ring, pids, pw = defer.plan_ring(topk_ids, topk_weights, st.row_map, st.bank_map, st.ring_owner, st.score,
                                              st.defer_from, st.defer_to, st.defer_slots, DEFER_BASE, st.ring_base,
                                              RING, late,
                                              _DEF["count"])
        if st.cut_mean is not None:       # before u_now is saved: the completion adds the late share on what was used
            defer.cut_mean(out, ids, topk_ids, topk_weights, st.cut_mean, pids, band=bmap)
        if _WIN["side"] is None:
            _WIN["side"] = torch.cuda.Stream(device=main.device)
        side = _WIN["side"]
        side.wait_stream(main)
        with torch.cuda.stream(side):                     # the window: the lane's records into ring rows (+ late rows)
            st.ring_stager(st.bank_ptr, src, ring, late)
            ev = torch.cuda.Event()
            ev.record(side)
        _WIN["ev"] = ev
        # src / ring are read by the side stream: keep them referenced until the next layer has joined the window
        _DEF["keep"] = (src, ring)
        if late:
            _DEF["pending"] = {"idx": st.idx, "T": T, "x": hidden_states.clone(), "w": pw, "ids": pids,
                               "u": out.clone(), "stage": src}
    elif st.defer_ok:
        stage, pids, pw, n = _defer_plan(st, topk_ids, topk_weights)
        if st.cut_mean is not None:       # before u_now is saved: the completion adds the late share on what was used
            from . import defer as _dm
            _dm.cut_mean(out, ids, topk_ids, topk_weights, st.cut_mean, pids, band=bmap)
        if n is not None and _DEF["count"] is not None:
            _DEF["count"].add_(n)
        if _WIN["side"] is None:
            _WIN["side"] = torch.cuda.Stream(device=main.device)
        side = _WIN["side"]
        side.wait_stream(main)
        with torch.cuda.stream(side):                     # the window: stage the deferred records
            _DEF["stager"](st.bank_ptr, stage)
            ev = torch.cuda.Event()
            ev.record(side)
        _WIN["ev"] = ev                                   # joined before the next attention's graph break
        # the stage list is read by the side stream: it must stay referenced (not returned to the main stream's
        # allocator pool) until the next layer has joined the window
        _DEF["pending"] = {"idx": st.idx, "T": T, "x": hidden_states.clone(), "w": pw, "ids": pids,
                           "u": out.clone(), "stage": stage}
    elif st.cut_mean is not None:         # a layer that does not defer: every cut pick takes its mean
        from . import defer as _dm
        _dm.cut_mean(out, ids, topk_ids, topk_weights, st.cut_mean, band=bmap)
    return out


def _forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
    st = getattr(self, "_k31", None)
    if st is None:
        _ld_join()
        return _orig["forward"](self, hidden_states, topk_weights, topk_ids, shared_experts)
    if st.row_map is None:
        raise RuntimeError(f"K31 layer {st.idx}: experts not bound (load_model hook did not run)")
    if _RES and st.idx == _FIRST_MOE[0]:                 # K31_RES_STREAM: an eager path with no graph replay since the
        _RES[0].join_eager()                             # tick waits for its GPU work here (no-op while capturing)
    T0, _K0 = topk_ids.shape
    sub = _SPLIT["sub"]                                  # K31_DECODE_SPLIT: one verify block of a larger decode batch
    if not (MODE != "hbm" and T0 <= ONE_ROWS and (torch.cuda.is_current_stream_capturing() or sub
                                                  or T0 * st.strong <= SLOTS)):
        _ld_join()                                       # every path but the one-round one reads hidden_states first
    if MODE == "hbm":
        rows = st.row_map[topk_ids.clamp_min(0).long()]
        ids = torch.where(topk_ids >= 0, rows, -1).to(topk_ids.dtype)
        return _orig["forward"](self, hidden_states, topk_weights, ids, shared_experts)
    T, K = topk_ids.shape
    capturing = torch.cuda.is_current_stream_capturing()
    if T <= ONE_ROWS and (capturing or sub or T * st.strong <= SLOTS):   # one round: the fused planner (unitk.py)
        from . import unitk
        main = torch.cuda.current_stream()
        if WINDOW_TEST and _WIN["ev"] is not None:       # join the previous layer's window before this MegaMoE call
            main.wait_event(_WIN["ev"])
            _WIN["ev"] = None
        band_on = st.band_map is not None and T <= 8 and getattr(st, "band_ok", True)
        band_ev = hint_ev = None
        if band_on and BAND_SIDE:                        # the send beside unitk / staging / MegaMoE
            if _BSIDE["stream"] is None:
                _BSIDE["stream"] = torch.cuda.Stream(device=main.device)
            bs = _BSIDE["stream"]
            bs.wait_stream(main)
            fz = _BAND.get("fused")
            if BAND_HINT and fz is not None and fz.has_hint:   # the plan + hint before the latent down's wait
                with torch.cuda.stream(bs):
                    fz.hint(topk_ids.int().contiguous(), topk_weights.float().contiguous(), st.row_map, st.band_map,
                            st.band_stats, st.idx, cap=getattr(st, "band_cap", None),
                            served=getattr(st, "band_served", None))
                    hint_ev = torch.cuda.Event()
                    hint_ev.record(bs)
            _ld_wait(bs)                                 # the send quantizes the latent rows (K31_SHARED_SIDE=3)
            with torch.cuda.stream(bs):
                _band_send(st, hidden_states, topk_ids, topk_weights, hinted=hint_ev is not None)
                band_ev = torch.cuda.Event()
                band_ev.record(bs)
        elif band_on:                                    # the RTX starts on the band's picks now
            _ld_join()
            _band_send(st, hidden_states, topk_ids, topk_weights)
        nop = _nop().take(st, T) if NOPREP else None     # K31_MEGAMOE_NOPREP: this call began at its router
        pre = None
        if PRESTAGE:                                     # K31_PRESTAGE: this layer's records staged ahead (side stream)
            from . import prestage
            pre = prestage.take(st, T)
        if nop is not None:                              # the plan kernel's rows (it wrote MegaMoE's routing inputs)
            ids, stage = _nop().plan(st, nop, topk_ids, topk_weights, LAM)
        else:
            ids, stage = unitk.plan(topk_ids, topk_weights, st.row_map, st.bank_map, st.strong, st.slot_base,
                                    score=st.score, lam=LAM, stats=st.stats, pend=getattr(st, "pend", None),
                                    resmax=st.resmax, pre=pre["pre"] if pre is not None else None,
                                    fill=pre is not None,   # with pre: `stage` = the misses (the late list)
                                    rowcap=_RCAP[0], cap=st.stage_cap if pre is None else None,
                                    row0=_SPLIT["row0"] if sub else 0, rowprior=_RPRI[0])
        if _LT[0] is not None:                           # K31_LATE: this call's late job, on the late stream
            _LT[0].post(st, hidden_states, topk_ids, topk_weights, ids, main, _LD["ev"], _breakable_capture())
        tail = TAIL1 and BAND_FIN_SIDE and band_on and not DEFER and _BAND.get("fused") is not None and T <= 8 \
            and getattr(st, "msum", None) is not None
        if tail:                                         # the mean sum beside the stager and MegaMoE
            if _FIN["stream"] is None:
                _FIN["stream"] = torch.cuda.Stream(device=main.device)
            fs = _FIN["stream"]
            if st.cut_mean is not None:
                fs.wait_stream(main)                     # unitk's row ids
                with torch.cuda.stream(fs):
                    sv = _served(st, band_on)
                    if sv is not None and (hint_ev or band_ev) is not None:
                        fs.wait_event(hint_ev or band_ev)   # the served flags (the hint's, else the send's)
                    from . import defer as _dm
                    _dm.cut_mean_pre(st.msum[:T], ids, topk_ids, topk_weights, st.cut_mean,
                                     band=st.band_map if sv is None else None, served=sv)
        st.stager(st.bank_ptr, stage)
        if pre is not None:
            main.wait_event(pre["copy"])                 # the records staged ahead are in their rows
        if PRESTAGE and st.idx == _LAST_MOE[0]:
            prestage.join_stale()                        # nothing of this forward is left on the side stream
        _ld_join()                                       # prep and MegaMoE read the latent rows
        if DEFER:
            if band_ev is not None:                      # the deferral path finishes the band itself: join first
                main.wait_event(band_ev)
            return _forward_defer(self, st, hidden_states, topk_weights, topk_ids, ids, shared_experts, main, band_on)
        out = _nop().megamoe(self, nop, hidden_states, shared_experts) if nop is not None else None   # without prep
        if out is None:
            out = _orig["forward"](self, hidden_states, topk_weights, ids.to(topk_ids.dtype), shared_experts)
        if tail and out.dim() == 2 and out.is_contiguous():
            fs = _FIN["stream"]
            fs.wait_stream(main)                         # after MegaMoE
            if st.up_eps is None:
                moe = _MOE.get(st.idx)
                rot = getattr(moe, "routed_output_transform", None) if moe is not None else None
                st.up_eps = float(getattr(getattr(rot, "norm", None), "variance_epsilon", 1e-6))
            with torch.cuda.stream(fs):
                if band_ev is not None:
                    fs.wait_event(band_ev)
                _band_finish(st, out, msum=st.msum[:T] if st.cut_mean is not None else None,
                             rstd=st.rstd[_JT["r0"]:] if _JT["on"] else st.rstd,   # (joint: the row in its dense chunk)
                             eps=st.up_eps)
                fev = torch.cuda.Event()
                fev.record(fs)
            _FIN["ev"] = fev                             # joined in the routed output transform (_fin_join)
            _FIN["keep"] = (ids, topk_ids, topk_weights)
            _FIN["rstd"] = st.rstd                       # its up GEMM reads the statistic (NORM 2)
            return out
        if BAND_FIN_SIDE and (band_on or st.cut_mean is not None) and out.dim() == 2:
            if _FIN["stream"] is None:
                _FIN["stream"] = torch.cuda.Stream(device=main.device)
            fs = _FIN["stream"]
            fs.wait_stream(main)                         # after MegaMoE
            with torch.cuda.stream(fs):
                if band_ev is not None:
                    fs.wait_event(band_ev)
                if band_on:
                    _band_finish(st, out)
                if st.cut_mean is not None:
                    from . import defer as _dm
                    _dm.cut_mean(out, ids, topk_ids, topk_weights, st.cut_mean, band=st.band_map if band_on else None,
                                 served=_served(st, band_on))
                fev = torch.cuda.Event()
                fev.record(fs)
            _FIN["ev"] = fev                             # joined in the routed output transform (_fin_join)
            _FIN["keep"] = (ids, topk_ids, topk_weights) # referenced until the join (their memory is main's)
            return out
        if band_on:
            if band_ev is not None:
                main.wait_event(band_ev)
            _band_finish(st, out)
        if st.cut_mean is not None:
            from . import defer as _dm
            _dm.cut_mean(out, ids, topk_ids, topk_weights, st.cut_mean, band=st.band_map if band_on else None,
                         served=_served(st, band_on))
        if WINDOW_TEST:                                  # fork: a tiny kernel on a side stream after this MegaMoE call
            if _WIN["side"] is None:
                _WIN["side"] = torch.cuda.Stream(device=main.device)
            side = _WIN["side"]
            side.wait_stream(main)
            with torch.cuda.stream(side):
                _WIN["buf"].add_(1)
                ev = torch.cuda.Event()
                ev.record(side)
            if st.idx == _LAST_MOE[0]:                   # the last MoE layer joins at once (no later layer here)
                main.wait_event(ev)
            else:
                _WIN["ev"] = ev
        return out
    picks = topk_ids.long().clamp_min(0)
    native, bank, cand, score, valid = _plan(st, topk_ids, topk_weights)
    if st.score is not None:                                   # residency scores, as unitk keeps them
        st.score.mul_(LAM ** T)
        st.score.index_add_(0, picks.reshape(-1), torch.where(valid, topk_weights.float() ** 2, 0.0).reshape(-1))
        if _RES and _RES[0].accepted and not capturing and st.idx == min(LAYERS) and not _prompt_ticks():
            _RES[0].eager_tokens += T        # prompt rows (all real tokens) earn promotion credit like accepted ones
    nat_ids = torch.where(valid & (native >= 0), native, -1)
    if capturing:                     # a graph-sized call past T > 16: one round in the layer's slot group
        v, e = score.topk(SLOTS)
        ok = v > 0
        slot_of = torch.full((E,), -1, dtype=torch.long, device=picks.device)
        slot_of.scatter_(0, e, torch.where(ok, st.arange[:SLOTS], -1))
        st.stager(st.bank_ptr, torch.where(ok, st.bank_map[e], -1).to(torch.int32))
        slot = slot_of[picks]
        use = (valid & (native < 0)) if STAGE_ALL else cand       # K31_STAGE_ALL: every pick of a staged expert
        ids = torch.where(use & (slot >= 0), st.slot_base + slot, nat_ids).to(topk_ids.dtype)
        out = _orig["forward"](self, hidden_states, topk_weights, ids, shared_experts)
        if st.cut_mean is not None:
            from . import defer as _dm
            _dm.cut_mean(out, ids, topk_ids, topk_weights, st.cut_mean)
        return out
    if _flags.on("K31_PROMPT_PATH"):  # a prompt chunk: prompt_path.py (every bank pick, ticks inside the forward)
        from . import prompt_path
        from . import defer as _dm
        bulk = _BULK[0]
        bulk = bulk if bulk is not None and bulk.mode and T <= bulk.max_rows else None
        y = prompt_path.forward_prompt(self, st, hidden_states, topk_weights, topk_ids, shared_experts,
                                       orig=_orig["forward"], res=_RES if _IN_EXEC[0] else None,
                                       band_eager=_band_eager,
                                       cut_mean=_dm.cut_mean, band_bulk=bulk, first_layer=min(LAYERS))
        if _flags.on("K31_PROMPT_SELFCHECK") and _SELFCHECK.get(st.idx, 0) < 2:   # the same inputs, older branch
            _SELFCHECK[st.idx] = _SELFCHECK.get(st.idx, 0) + 1
            y_old = _eager_legacy(self, st, hidden_states, topk_weights, topk_ids, shared_experts, picks, native,
                                  cand, score, valid, nat_ids)
            rel = float((y.float() - y_old.float()).norm() / y_old.float().norm().clamp_min(1e-30))
            _LOG(f"prompt path selfcheck layer {st.idx} call {_SELFCHECK[st.idx]} ({T} rows): relative L2 {rel:.2e}")
        return y
    return _eager_legacy(self, st, hidden_states, topk_weights, topk_ids, shared_experts, picks, native, cand, score,
                         valid, nat_ids)


def _eager_legacy(self, st, hidden_states, topk_weights, topk_ids, shared_experts, picks, native, cand, score, valid,
                  nat_ids):
    """The eager branch before prompt_path.py (K31_PROMPT_PATH=0, and the selfcheck's reference)."""
    # eager (a prompt chunk, K31_PROMPT_PATH=0): every candidate is staged, in rounds of all SHARED slot rows (the
    # layers run one after another here, so a layer may use every group), with one MegaMoE call per round and the
    # outputs summed
    n_cand = int((score > 0).sum())
    rounds = max(1, -(-n_cand // STAGE_ROWS))
    top_v, top_e = score.topk(min(E, STAGE_ROWS * rounds))
    if st.stagers is None:
        from .stager import Stager
        st.stagers = [Stager(st.mod, SLOTS * g, st.layout, sms=STAGER_SMS_EAGER) for g in range(GROUPS)]
    ar = torch.arange(STAGE_ROWS, dtype=torch.long, device=picks.device)
    y = None
    served = valid & (native >= 0)
    use = (valid & (native < 0)) if STAGE_ALL else cand           # K31_STAGE_ALL: every pick of a staged expert
    for r in range(rounds):
        v, e = top_v[r * STAGE_ROWS:(r + 1) * STAGE_ROWS], top_e[r * STAGE_ROWS:(r + 1) * STAGE_ROWS]
        ok = v > 0
        slot_of = torch.full((E,), -1, dtype=torch.long, device=picks.device)
        slot_of.scatter_(0, e, torch.where(ok, ar[:e.numel()], -1))
        stage = torch.full((STAGE_ROWS,), -1, dtype=torch.int32, device=picks.device)
        stage[:e.numel()] = torch.where(ok, st.bank_map[e], -1).to(torch.int32)
        for g in range(min(GROUPS, -(-e.numel() // SLOTS))):
            st.stagers[g](st.bank_ptr, stage[SLOTS * g:SLOTS * (g + 1)].contiguous())
        slot = slot_of[picks]
        ids = torch.where(use & (slot >= 0), slot, nat_ids if r == 0 else -1).to(topk_ids.dtype)
        out = _orig["forward"](self, hidden_states, topk_weights, ids, shared_experts if r == 0 else None)
        y = out if y is None else y + out
        served = served | (use & (slot >= 0))
    band_eager = st.band_map is not None and getattr(st, "band_ok", True)
    if band_eager:
        _band_eager(st, hidden_states, topk_ids, topk_weights, y)
    if st.cut_mean is not None:
        from . import defer as _dm
        _dm.cut_mean(y, torch.where(served, 0, -1).to(torch.int32), topk_ids, topk_weights, st.cut_mean,
                     band=st.band_map if band_eager else None)
    return y


# ------------------------------------------------------------------------------------------------- binding
def _write_expert(mod, r: int, t: dict, dev) -> None:
    """One expert's six checkpoint tensors into kernel row r, as vLLM's weight_loader would write them."""
    for shard in ("w1", "w2", "w3"):
        packed, scale = t[shard]
        pre = "w2" if shard == "w2" else "w13"
        mod._load_weight(getattr(mod, f"{pre}_weight").data[r], packed.to(dev), shard, False)
        mod._load_weight(getattr(mod, f"{pre}_weight_scale").data[r], scale.to(dev), shard, True)


def _cut_mean_table(idx: int) -> torch.Tensor:
    """Layer idx's expert means [896, 3584]: K31_CUT_MEAN is a .safetensors file (tensor "L{idx}", as published) or a
    directory of the runner's means-L{idx}.pt files."""
    if CUT_MEAN.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(CUT_MEAN, framework="pt", device="cpu") as f:
            return f.get_tensor(f"L{idx}")
    return torch.load(os.path.join(CUT_MEAN, f"means-L{idx}.pt"), map_location="cpu", weights_only=False)["mean"]


def _load_bank(st: LayerState, layout: dict) -> int:
    """The layer's bank records from L{idx}.hi2 into pinned host memory (anonymous mmap registered with CUDA: torch's
    pinned allocator would round each layer up to a power of two). Returns bytes read."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    R = layout["record_bytes"]
    G = len(st.bank)
    st.bank_buf = mmap.mmap(-1, max(G, 1) * R)
    t = torch.frombuffer(st.bank_buf, dtype=torch.uint8).view(max(G, 1), R)
    fd = os.open(os.path.join(HI2, f"L{st.idx:02d}.hi2"), os.O_RDONLY)
    try:
        j = 0
        while j < G:                                   # runs of consecutive expert ids: one read each
            k = j
            while k + 1 < G and st.bank[k + 1] == st.bank[k] + 1:
                k += 1
            view, pos = memoryview(t[j:k + 1].numpy()).cast("B"), st.bank[j] * R
            while view:
                n = os.preadv(fd, [view], pos)
                if n <= 0:
                    raise IOError(f"short read in L{st.idx:02d}.hi2 at {pos}")
                view, pos = view[n:], pos + n
            j = k + 1
    finally:
        os.close(fd)
    if int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0x3)) != 0:
        raise RuntimeError(f"K31: could not pin layer {st.idx}'s bank ({t.numel() / 2**30:.1f} GiB)")
    st.bank_cpu = t                                 # residency.py copies hi2 records from it by DMA
    st.bank_t = get_accelerator_view_from_cpu_tensor(t)
    st.bank_ptr = st.bank_t.data_ptr()
    return G * R


def _dequant(packed: torch.Tensor, scale: torch.Tensor, table: torch.Tensor = FP4) -> torch.Tensor:
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    vals = table.to(packed.device)[codes]
    s = torch.exp2(scale.float() - 127)
    return (vals.view(*scale.shape, 32) * s.unsqueeze(-1)).view(vals.shape)


def _mxfp8(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 E4M3 with a power-of-two scale per 32 values along the last dimension (smallest scale that does not
    clip), dequantized: what MegaMoE does to its inputs and to the L1 output it feeds L2."""
    xf = x.float().reshape(-1, 32)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return ((xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)


@torch.no_grad()
def _selfcheck(st: LayerState, tensors: dict, cfg, dev, bank_tensors: dict) -> None:
    """Layer st.idx as it will run (the row map, the stager, the kernel) against float32 formulas on the same
    experts' official weights, for 4 tokens: 13 picks on resident experts, and in mode staged the 3 strongest on bank
    experts (reference: their 2-bit code map)."""
    mod = st.mod
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 4
    x = (torch.randn((n, LATENT), generator=g) * 0.5).to(dev, torch.bfloat16)
    res = torch.tensor(sorted(tensors), dtype=torch.long)
    staged_pool = torch.tensor(sorted(bank_tensors), dtype=torch.long) if bank_tensors else None
    ids, ws = [], []
    ns = int(getattr(st, "strong", STRONG))          # this layer's staged ranks (K31_LAYER_BANDS / K31_POLICY)
    for _ in range(n):
        if staged_pool is not None:
            a = staged_pool[torch.randperm(len(staged_pool), generator=g)[:ns]]
            b = res[torch.randperm(len(res), generator=g)[:TOPK - ns]]
            w = torch.cat((torch.rand(ns, generator=g) + 1.0, torch.rand(TOPK - ns, generator=g) * 0.9 + 0.05))
            ids.append(torch.cat((a, b))); ws.append(w)
        else:
            ids.append(res[torch.randperm(len(res), generator=g)[:TOPK]]); ws.append(torch.rand(TOPK, generator=g) + 0.1)
    ids, w = torch.stack(ids), torch.stack(ws)
    w = (w / w.sum(-1, keepdim=True)).float()
    beta, lbeta = float(cfg.activation_situ_beta), float(cfg.activation_situ_linear_beta)
    want = {k: torch.zeros((n, LATENT), dtype=torch.float32, device=dev) for k in ("exact", "pre")}
    xf, xq = x.float(), _mxfp8(x.float())
    for t in range(n):
        for k in range(TOPK):
            e = int(ids[t, k])
            src, table = (tensors[e], FP4) if e in tensors else (bank_tensors[e], FP4_TRICK)
            w1, w2, w3 = (_dequant(src[s][0].to(dev), src[s][1].to(dev), table) for s in ("w1", "w2", "w3"))
            for name, xin in (("exact", xf[t]), ("pre", xq[t])):
                gate, up = xin @ w1.T, xin @ w3.T
                a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (lbeta * torch.tanh(up / lbeta) if lbeta else up)
                want[name][t] += (w[t, k] * (a @ w2.T)) if name == "exact" else (_mxfp8(w[t, k] * a) @ w2.T)
    cap = st.resmax                              # a resident cap (K31_POLICY hit ranks) gives picks past it the mean:
    st.resmax = TOPK                             # the check compares every pick exactly, so it runs uncapped
    rcap, _RCAP[0] = _RCAP[0], None              # and without per-row staging caps (K31_STAGE_ROWS)
    scap, st.stage_cap = st.stage_cap, None      # or a cap on the records a call stages (K31_STAGE_CAP / stage_cap)
    try:
        got = mod(x, w.to(dev), ids.to(dev, torch.int32)).float()
    finally:
        st.resmax = cap
        _RCAP[0] = rcap
        st.stage_cap = scap
    rel = {k: ((got - v).norm() / v.norm()).item() for k, v in want.items()}
    _LOG(f"selfcheck layer {st.idx}: kernel vs float32 formulas (4 tokens x 16 picks"
         + (f", {ns} per token staged from 2-bit records" if staged_pool is not None else "")
         + f"), relative difference: exact {rel['exact']:.2e}, with the kernel's MXFP8 steps {rel['pre']:.2e}")
    # measured 11-17 on layer 1 (hbm): exact 3.5e-2, pre 1.1e-2: the kernel applies the routing weight before it
    # quantizes the intermediate; the rest is its fast-math SiTU and scale rounding. A wrong layout gives ~1, one
    # wrong pick of the 16 ~0.25
    if not rel["pre"] < 3e-2:
        raise RuntimeError(f"K31: selfcheck failed on layer {st.idx} ({rel})")


def _bind(runner) -> None:
    if not LAYERS:
        raise RuntimeError("K31: no MoE layer was built through the hook (wrong --moe-backend or model?)")
    from .ckpt import Experts
    dev = runner.device
    cfg = runner.vllm_config.model_config.hf_text_config
    if (cfg.num_experts, cfg.num_experts_per_token, cfg.routed_expert_hidden_size, cfg.moe_intermediate_size) != \
            (E, TOPK, LATENT, INTER):
        raise RuntimeError("K31: the served config is not Kimi K3.1 Pro's MoE")
    if STAGE_ALL and DEFER:
        raise RuntimeError("K31_STAGE_ALL=1 needs K31_DEFER=0 (a deferred pick of a staged expert would count twice)")
    if STAGE_ROWCAPS and MODE == "staged":
        P = _rows_per_request(runner)
        rc = [STAGE_ROWCAPS[min(t % P, len(STAGE_ROWCAPS) - 1)] for t in range(16)]
        _RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev)
        _LOG(f"K31_STAGE_ROWS: row p of a {P}-row verify block stages ranks < min(strong, {rc[:P]}[p])")
    if STAGE_PRIOR and MODE == "staged":
        if not all(0.0 <= v < float("inf") for v in STAGE_PRIOR):
            raise RuntimeError(f"K31_STAGE_PRIOR: row priors {STAGE_PRIOR} must be finite values >= 0")
        P = _rows_per_request(runner)
        rp = [STAGE_PRIOR[min(t % P, len(STAGE_PRIOR) - 1)] for t in range(16)]
        _RPRI[0] = torch.tensor(rp, dtype=torch.float32, device=dev)
        _LOG(f"K31_STAGE_PRIOR: staging slots by largest weight x the row's prior; rows of a {P}-row verify block: "
             f"{', '.join(f'{v:g}' for v in rp[:P])}")
    if STAGE_SP != 4 and not DECODE_SPLIT:
        raise RuntimeError("K31_STAGE_SP=8 needs K31_DECODE_SPLIT=1 (the wider plan takes calls of up to 8 rows)")
    if STAGE_CAP is not None and not 1 <= STAGE_CAP <= SLOTS:
        raise RuntimeError(f"K31_STAGE_CAP={STAGE_CAP} outside 1..{SLOTS}")
    if DECODE_SPLIT:
        off = [n for n, v in (("K31_DEFER", DEFER), ("K31_PRESTAGE", PRESTAGE), ("K31_MEGAMOE_NOPREP", NOPREP),
                              ("K31_LATE", LATE), ("K31_WINDOW_TEST", WINDOW_TEST)) if v]
        if MODE != "staged" or off or not 1 <= SPLIT_ROWS <= 8 or not SPLIT_ROWS < SPLIT_MAX <= PEND_ROWS:
            raise RuntimeError(f"K31_DECODE_SPLIT=1 needs K31_MODE=staged, K31_DECODE_SPLIT_ROWS in 1..8, "
                               f"K31_DECODE_SPLIT_MAX in {SPLIT_ROWS + 1}..{PEND_ROWS} and none of: "
                               + (", ".join(off) or "(none set)"))
        P = _rows_per_request(runner)
        _SPLIT["P"] = P
        _SPLIT["rows"] = max(SPLIT_ROWS, min(P, 8))      # one request's verify block is never split (up to 8 rows)
        _SPLIT["block"] = split_block(P, _SPLIT["rows"])
        _LOG(f"K31_DECODE_SPLIT: decode batches of {_SPLIT['rows'] + 1}..{SPLIT_MAX} rows run every MoE layer as "
             f"calls of {_SPLIT['block']} rows ({P} rows per request); e.g. {4 * P} rows -> "
             f"{[b - a for a, b in split_blocks(4 * P, _SPLIT['block'], _SPLIT['rows'])]}")
    if PRESTAGE and (MODE != "staged" or DEFER or NOPREP or LATE):
        raise RuntimeError("K31_PRESTAGE=1 needs K31_MODE=staged and K31_DEFER=0, K31_MEGAMOE_NOPREP=0, K31_LATE=0")
    if PRESTAGE:
        from . import prestage
        prestage.prepare(dev, int(getattr(cfg, "hidden_size", 7168)))
        _LOG(f"K31_PRESTAGE: decode pre-staging on a side stream (candidates of predicted rank < strong + "
             f"{prestage.EXTRA}, at most {SLOTS - prestage.RESERVE} slots ahead, the misses staged at the MoE call)")
    if STAGE_ALL:
        _LOG("K31_STAGE_ALL: every pick of a staged expert uses its staging row (any rank, any row of the call)")
    if os.environ.get("K31_BAND_PRIOR", "").strip():
        _BAND["rows_per_req"] = _rows_per_request(runner)  # K31_BAND_PRIOR: rows of one request's verify block
        if not BAND_N:
            _LOG("K31_BAND_PRIOR: no RTX band in this profile (K31_BAND_N=0): no effect")
    layout = None
    if MODE == "staged":
        from .stager import Stager
        layout = json.load(open(os.path.join(HI2, "layout.json")))
        if layout.get("format") != "hi2-k1":
            raise RuntimeError(f"K31: {HI2} is not a hi2-k1 record store")
    ck = Experts(NATIVE)
    t0, nbytes, bank_bytes = time.time(), 0, 0
    first = min(LAYERS)
    _LAST_MOE[0] = max(LAYERS)
    _FIRST_MOE[0] = first
    if WINDOW_TEST:
        _WIN["buf"] = torch.zeros(1, dtype=torch.int64, device=dev)
    shared = None
    for idx, st in sorted(LAYERS.items()):
        keep, bank_keep = {}, {}
        if MODE == "staged":                 # the four kernel tensors as [shared slots | own resident rows] ranges
            from .vmm import SharedSlots
            names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
            with torch.cuda.device(dev):
                if shared is None:
                    shared = {n: SharedSlots(getattr(st.mod, n), SHARED) for n in names}
                for n in names:
                    setattr(st.mod, n, torch.nn.Parameter(shared[n].tensor(st.S), requires_grad=False))
            st.mod.num_experts = st.mod.num_local_experts = st.slots + st.S
            st.mod.experts_end_idx = st.mod.experts_start_idx + st.slots + st.S
            torch.cuda.empty_cache()
        for b0 in range(0, st.S, Experts.BATCH):
            batch = st.hbm[b0:b0 + Experts.BATCH]
            tensors = ck.read_layer(idx, batch)
            nbytes += ck.last_bytes
            for r, e in enumerate(batch, start=b0):
                _write_expert(st.mod, st.slots + r, tensors[e], dev)
            if idx == first and SELFCHECK and b0 == 0:   # copies of a few experts for the self-check
                keep = {e: {s: (p.clone(), q.clone()) for s, (p, q) in tensors[e].items()} for e in batch[:24]}
        st.row_map = torch.tensor(st.row, dtype=torch.int32).to(dev)
        st.bank_map = torch.tensor(st.bank_idx, dtype=torch.int32).to(dev)
        st.arange = torch.arange(max(SLOTS, TOPK), dtype=torch.long, device=dev)
        if MODE == "staged":
            bank_bytes += _load_bank(st, layout)
            st.stager = Stager(st.mod, st.slot_base, layout, sms=STAGER_SMS)
            if PRESTAGE:                                 # the side stream's plan outputs (prestage.fork)
                st.pre_bufs = (torch.empty((16, TOPK), dtype=torch.int32, device=dev),
                               torch.full((SLOTS,), -1, dtype=torch.int32, device=dev))
                if prestage.SMS > 0:                     # K31_PRESTAGE_SMS: a narrower stager for the copies ahead
                    st.pre_stager = Stager(st.mod, st.slot_base, layout, sms=prestage.SMS)
            # no deferral out of the last MoE layer or a block's last layer (its share would cross a stored block)
            st.defer_ok = DEFER and idx != max(LAYERS) and idx not in DEFER_SKIP and \
                (DEFER_BLOCKEND or (idx + 1) % ATTN_RES_BLOCK != 0)
            if CUT_MEAN:
                st.cut_mean = _cut_mean_table(idx).to(dev, torch.bfloat16).contiguous()
            if RES_MAXRANK_MAP:
                st.resmax = int(json.load(open(RES_MAXRANK_MAP))["caps"].get(str(idx), RES_MAXRANK))
            if LAYER_BANDS:
                dfrom, dto = json.load(open(LAYER_BANDS))["bands"].get(str(idx), [STRONG, DEFER_TO])
                st.strong = min(int(dfrom), 4)                   # unitk stages at most ranks 0-3 on time
                st.defer_from, st.defer_to = st.strong, max(st.strong, int(dto))
                if st.defer_to <= st.defer_from:
                    st.defer_ok = False                          # this layer has no late band
            if POLICY:
                _apply_policy(st, idx)
            if DEFER and DEFER_SLOTS_MAP:
                st.defer_slots = int(json.load(open(DEFER_SLOTS_MAP)).get(str(idx), DEFER_SLOTS))
                st.defer_ok = st.defer_ok and st.defer_slots > 0
            if DEFER and _DEF["stager"] is None:          # the deferral rows are shared: one stager serves all layers
                _DEF["stager"] = Stager(st.mod, DEFER_BASE, layout, sms=STAGER_SMS)
                _DEF["count"] = torch.zeros((), dtype=torch.int64, device=dev)
            if DEFER and RING > 0 and idx != max(LAYERS):  # ring rows: every MoE layer fetches (late only if defer_ok)
                from .stager import RingStager
                assert st.S > RING, f"K31 layer {idx}: {st.S} rows cannot hold a ring of {RING}"
                st.ring_ok = True
                st.ring_base = st.slots + st.S - RING
                st.ring_owner = torch.tensor(st.hbm[st.S - RING:], dtype=torch.int32, device=dev)
                st.ring_stager = RingStager(st.mod, DEFER_BASE, st.ring_base, layout, sms=STAGER_SMS)
            st.layout = layout
            if idx == first and SELFCHECK and st.bank:
                some = st.bank[:12]
                t = ck.read_layer(idx, some)
                bank_keep = {e: {s: (p.clone(), q.clone()) for s, (p, q) in t[e].items()} for e in some}
        if idx == first and SELFCHECK:
            torch.cuda.synchronize(dev)
            _selfcheck(st, keep, cfg, dev, bank_keep)
        if idx % 10 == 0:
            _LOG(f"layer {idx} bound, {time.time() - t0:.0f} s, {nbytes / 1e9:.0f} GB read"
                 + (f", banks {bank_bytes / 1e9:.0f} GB" if bank_bytes else ""))
    ck.close()
    torch.cuda.synchronize(dev)
    if DG_SMS:                  # MegaMoE (persistent, grid-synchronized) on fewer SMs: side kernels keep the rest
        from vllm.utils.deep_gemm import _import_deep_gemm
        _import_deep_gemm().set_num_sms(DG_SMS)
        _LOG(f"DeepGEMM kernels on {DG_SMS} SMs (side work: merges, staging)")
    if MODE == "staged" and os.environ.get("K31_POOL", "0") == "1":     # the Grace pool (cpupool.py), off by default
        from . import cpupool
        cpupool.from_env([(i, s.bank_cpu, len(s.bank)) for i, s in sorted(LAYERS.items())], log=_LOG)
    if BAND_N:
        _bind_band(dev)
    if DECODE_JOINT:
        _bind_joint(dev, layout)
    if MODE == "staged":
        from .residency import Residency
        _RES.append(Residency(LAYERS, NATIVE, PROMOTE, PROMOTE_WORKERS, dev, _write_expert, HI2, layout,
                              bank_promote=BANK_PROMOTE))
        n_lo = sum(os.path.exists(os.path.join(HI2, f"L{i:02d}.lo2")) for i in LAYERS)
        _LOG(f"residency: promotions {'by merge of hi2 + lo2 records' if n_lo == len(LAYERS) else f'by merge in {n_lo} of {len(LAYERS)} layers, else from the official shards'}")
    rows = sum(st.S for st in LAYERS.values())
    if MODE == "staged":                         # one symmetric buffer for every layer's expert count (_get_symm_buffer)
        _SYMM["emax"] = max(st.slots + st.S for st in LAYERS.values())
    _LOG(f"ACTIVE mode {MODE}: {len(LAYERS)} MoE layers, {rows} resident expert rows in HBM "
         f"({rows * 17.55e6 / 1e9:.0f} GB)"
         + (f", {SHARED} staging rows shared by all layers ({SLOTS} per call), banks {bank_bytes / 1e9:.0f} GB "
            f"pinned (strong ranks < {STRONG}), promotion {PROMOTE:g} and bank refill {BANK_PROMOTE:g} per layer-token "
            f"(half-life {HALFLIFE:g} tokens)"
            + (f"; deferral: misses of rank {DEFER_FROM}..{DEFER_TO - 1} (0-based), up to {DEFER_SLOTS} experts per "
               f"layer-step, staged in a side window"
               + (" and computed after the next layer's attention (lands before its MoE)" if DEFER_LAND == "mlp"
                  else " and computed by the next layer's call (lands one layer late)")
               if DEFER else "")
            if MODE == "staged" else "")
         + f", the rest cut; bound in {time.time() - t0:.0f} s ({nbytes / 1e9:.0f} GB read from {NATIVE})")
    if LATE:                                     # after the band and the residency (the plan reads their pooled maps)
        from . import latetier
        _LT[0] = latetier.bind(LAYERS, dev, cfg, runner.model, max(LAYERS), DEFER, FUSE_UP, MODE)
    if NOPREP:                                   # eligible layers, and the first one checked against vLLM's kernels
        if SHARED_SIDE_MODE != 3:
            _LOG("noprep: K31_MEGAMOE_NOPREP needs K31_SHARED_SIDE=3 (the hook computes the latent down there): off")
        else:
            _nop().bind(runner.model, dev, _LOG)


def _bind_joint(dev, layout) -> None:
    """K31_DECODE_JOINT: level 2's per-layer state: a stager per slot group (block j of a batch stages into group
    (L + j) % GROUPS), and for band layers one served-flag buffer and one tag per block and the RMSNorm statistic of
    the batch's rows; the cut picks' mean sums of a batch share one buffer (a layer's finishes read it before the
    next layer's plans write it: both run on the fin stream)."""
    if not DECODE_SPLIT or MODE != "staged" or not SHARED_SIDE:
        raise RuntimeError("K31_DECODE_JOINT needs K31_DECODE_SPLIT=1, K31_MODE=staged and K31_SHARED_SIDE >= 1")
    if DECODE_JOINT not in (1, 2) or not 1 <= JOINT_DENSE_ROWS <= SPLIT_MAX:
        raise RuntimeError(f"K31_DECODE_JOINT={DECODE_JOINT} (1 or 2), K31_DECODE_JOINT_DENSE={JOINT_DENSE_ROWS} "
                           f"(1..{SPLIT_MAX})")
    nb = len(split_blocks(SPLIT_MAX, _SPLIT["block"], _SPLIT["rows"]))
    if DECODE_JOINT >= 2:
        from .stager import Stager
        band = 0
        for _idx, st in sorted(LAYERS.items()):
            st.jstagers = [Stager(st.mod, SLOTS * g, layout, sms=STAGER_SMS) for g in range(GROUPS)]
            if st.band_map is not None:
                st.jserved = torch.zeros((BAND_NSLOT, 16 * 16), dtype=torch.int32, device=dev)
                st.jtags = torch.zeros(BAND_NSLOT, dtype=torch.int64, device=dev)
                st.jrstd = torch.zeros(PEND_ROWS, dtype=torch.float32, device=dev)
                band += 1
        if band and TAIL1:
            _JT["msum"] = torch.zeros(SPLIT_MAX, LATENT, dtype=torch.float32, device=dev)
    e4 = split_blocks(4 * _SPLIT["P"], _SPLIT["block"], _SPLIT["rows"])
    _LOG(f"K31_DECODE_JOINT={DECODE_JOINT}: decode batches of {_SPLIT['rows'] + 1}..{SPLIT_MAX} rows share the dense "
         f"parts of a MoE layer per chunk of <= {JOINT_DENSE_ROWS} rows"
         + (f" and run one MegaMoE call per layer (up to {min(nb, GROUPS)} blocks, {BAND_NSLOT} with the RTX band; "
            f"block j stages into slot group (layer + j) % {GROUPS})" if DECODE_JOINT >= 2 else "")
         + f"; e.g. {4 * _SPLIT['P']} rows -> dense chunks of {[cb - ca for ca, cb, _ in dense_chunks(e4)]} rows"
         + (f"; check of the first {JOINT_CHECK} eager batches per layer" if JOINT_CHECK else ""))


def _rows_per_request(runner) -> int:
    """Rows of one request's verify block in a decode call: 1 + the draft tokens (vLLM's uniform decode query length;
    1 without speculative decoding)."""
    vc = getattr(runner, "vllm_config", None)
    try:
        sc = getattr(vc, "speculative_config", None)
        if sc is not None and getattr(sc, "enable_adaptive_verification", False):
            _LOG("K31_BAND_PRIOR: adaptive verification trims verify blocks; the row priors of calls with more than "
                 "one request are approximate")
        n = getattr(vc, "uniform_decode_query_len", None)
        if n is None:
            n = 1 + int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc is not None else 1
        return max(1, int(n))
    except Exception:                                    # noqa: BLE001
        return 1


def _bind_prior() -> None:
    """K31_BAND_PRIOR (B1): in calls captured into CUDA graphs (decode verify blocks), the fused send and hint order
    the cap by weight x each row's prior chance of being accepted; row t of a call is row t % P of a request's verify
    block (P = 1 + drafts; exact at one request per call, and the band only takes calls of <= 8 rows). Eager calls
    (a prompt's last chunk of <= 8 rows) keep today's plan."""
    from . import band
    fz = _BAND.get("fused")
    if fz is None:
        raise RuntimeError("K31_BAND_PRIOR needs the fused band send (K31_BAND_FUSED=1)")
    P = int(_BAND.get("rows_per_req") or 1)
    rows = band.prior_rows(band.BAND_PRIOR, P, band.MAXT)
    fz.set_prior(rows)
    _BAND["prior_rows"] = rows
    _LOG(f"band cap order (K31_BAND_PRIOR): largest weight x the row's prior in decode graphs; {P} rows per request, "
         f"rows 0-7 of a call: {', '.join(f'{v:g}' for v in rows)} (calls under their cap send what they sent "
         f"without it)")


def _bind_band(dev) -> None:
    """The RTX band: wait for the sidecar's ready file, check its store map against this engine's band sets, map the
    mailbox into this process (ATS) and give every layer its work-list buffer and launch arguments."""
    from . import band
    t0 = time.time()
    while not os.path.exists(BAND + ".ready"):
        if time.time() - t0 > BAND_WAIT_S:
            raise RuntimeError(f"K31 band: no {BAND}.ready after {BAND_WAIT_S:.0f} s (is the sidecar running?)")
        time.sleep(1.0)
    m = np.load(BAND + ".map.npy")
    gb = band.Gb(BAND_LIB, BAND, dev, BAND_TIMEOUT_US)
    if not gb.running():
        raise RuntimeError("K31 band: the service kernel is not running")
    _BAND["gb"] = gb
    if BAND_FUSED:
        # K31_BAND_FUSED_LIB: another build of the fused kernels in the same directory (librtxband_fused_v2.so:
        # band finish loads all of a thread's result rows before it adds them, one host round trip instead of ~5)
        _BAND["fused"] = band.Fused(os.path.join(os.path.dirname(BAND_LIB),
                                                 os.environ.get("K31_BAND_FUSED_LIB", "librtxband_fused.so")), gb)
    if band.BAND_PRIOR:
        _bind_prior()
    _BAND["stats"] = torch.zeros((max(LAYERS) + 1, 3), dtype=torch.int32, device=dev)
    if BAND_BULK:
        if not os.path.exists(BAND + ".bulk"):
            raise RuntimeError(f"K31_BAND_BULK=1: no {BAND}.bulk (the sidecar needs K31_BAND_BULK=1 too)")
        _BULK[0] = band.Bulk(os.path.join(os.path.dirname(BAND_LIB), "libbandbulk.so"),
                             os.path.join(os.path.dirname(BAND_LIB), "libbandhost.so"), BAND + ".bulk", dev)
        _LOG(f"band bulk requests: {BAND}.bulk, up to {_BULK[0].max_rows} rows, for prompt steps of more than "
             f"{BAND_BULK_MIN} rows")
    _BAND["eager_call"] = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
    _BAND["eager_args"] = gb.args(_BAND["eager_call"], BAND_EAGER_TIMEOUT_US)
    n = 0
    for idx, st in sorted(LAYERS.items()):
        have = set(np.nonzero(m[idx] >= 0)[0].tolist())
        if have != set(st.band):
            raise RuntimeError(f"K31 band: layer {idx}: the sidecar's store holds {len(have)} experts, this engine "
                               f"expects {len(st.band)} others (same K31_RANK, K31_HBM, K31_BAND_N on both sides?)")
        st.band_map = torch.tensor(m[idx], dtype=torch.int32, device=dev)
        # K31_BAND_SERVED=1 (librtxband_fused_v3.so): the fused send writes which picks it serves; the cut-mean then
        # gives the mean to band picks past the cap (1.7-2.6% of picks were plain skips)
        # (v5 = v4 + the plan / hint kernel of K31_BAND_HINT)
        if (BAND_SERVED and not any(v in _FUSED_LIB for v in ("v3", "v4", "v5"))) or \
                (TAIL1 and not any(v in _FUSED_LIB for v in ("v4", "v5"))):
            raise RuntimeError("K31_BAND_SERVED=1 needs K31_BAND_FUSED_LIB=librtxband_fused_v3/v4/v5.so (older builds "
                               "never write the flags: the mean would be added to picks the band serves); K31_TAIL1=1 "
                               "needs librtxband_fused_v4/v5.so")
        st.band_served = torch.zeros(16 * 16, dtype=torch.int32, device=dev) if BAND_SERVED else None
        st.msum = torch.zeros(8, 3584, dtype=torch.float32, device=dev) if TAIL1 else None
        st.rstd = torch.zeros(16, dtype=torch.float32, device=dev) if TAIL1 else None
        st.band_call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
        st.band_args = gb.args(st.band_call)
        st.band_stats = _BAND["stats"][idx]
        n += len(have)
    nper = sorted(len(st.band) for st in LAYERS.values())
    _LOG(f"RTX band: {n} records over {len(LAYERS)} layers "
         f"({nper[0] if nper[0] == nper[-1] else f'{nper[0]}-{nper[-1]}'} per layer) behind {BAND}, finish bound "
         f"{BAND_TIMEOUT_US:g} us; first call seq {int(gb.st[0]) + 1}")


def _apply_policy(st, idx: int) -> None:
    """K31_POLICY: this layer's compiled policy into its LayerState (policy.py); a policy the kernels cannot run, or one
    that needs a switch that is off (deferral, mean tables), stops the bind with the reason."""
    from . import policy
    if "params" not in _POLICY:
        _POLICY["params"] = policy.load(POLICY)
        _LOG(f"policy {POLICY}:\n" + policy.describe(_POLICY["params"]))
    p = _POLICY["params"][idx]
    st.policy = p
    st.resmax = p.resmax
    st.strong = p.strong
    st.band_ok = p.band
    st.band_cap = p.band_cap
    if getattr(p, "stage_cap", None) is not None:
        st.stage_cap = p.stage_cap
    if p.defer is not None:
        if not DEFER:
            raise RuntimeError(f"K31 policy: layer {idx} defers ranks {p.defer} but K31_DEFER is off")
        st.defer_from, st.defer_to = p.defer
    else:
        st.defer_ok = False
    if getattr(p, "late", None) is not None and not LATE:
        raise RuntimeError(f"K31 policy: layer {idx} sends ranks {p.late} late but K31_LATE is off")
    if p.mean and st.cut_mean is None:
        raise RuntimeError(f"K31 policy: layer {idx} uses the mean substitute but K31_CUT_MEAN gives no tables")
    if not p.mean:
        st.cut_mean = None


def _served(st, band_on: bool):
    """The send's per-pick served flags for the cut-mean (K31_BAND_SERVED, fused send only), else None."""
    if not (band_on and BAND_SERVED and _BAND.get("fused") is not None):
        return None
    return getattr(st, "band_served", None)


def _band_send(st, x, topk_ids, topk_weights, hinted: bool = False, served=False) -> None:
    """Before the call's staging and MegaMoE: its band work list and MXFP8 rows into the mailbox. hinted: the plan /
    hint kernel has already written the served flags and stats (K31_BAND_HINT). served: another buffer for the
    call's served flags than the layer's (K31_DECODE_JOINT=2: one per block; None = no flags)."""
    from . import band
    gb = _BAND["gb"]
    fz = _BAND.get("fused")
    if fz is not None:
        sv = getattr(st, "band_served", None) if served is False else served
        fz.qsend((topk_ids if topk_ids.dtype == torch.int32 else topk_ids.int()).contiguous(),
                 (topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()).contiguous(),
                 st.row_map, st.band_map, x.contiguous(), None if hinted else st.band_stats, st.idx,
                 cap=getattr(st, "band_cap", None), served=None if hinted else sv)
        return
    band.plan(topk_ids, topk_weights, st.row_map, st.band_map, st.band_call, st.band_stats)
    band.quant(x.contiguous(), gb.xq, gb.xs)
    gb.send(st.band_args)


def _band_finish(st, out, msum=None, rstd=None, eps=0.0) -> None:
    """After MegaMoE: the band's result rows (if they came in time) into the routed latent sum out [T, 3584]; with
    msum / rstd (K31_TAIL1, v4): the cut picks' mean sum too, and the RMSNorm statistic of the result."""
    gb = _BAND["gb"]
    fz = _BAND.get("fused")
    if fz is not None and out.is_contiguous() and out.dtype == torch.bfloat16:
        fz.finish(out, msum=msum, rstd=rstd, eps=eps)
        return
    assert msum is None and rstd is None
    T = out.shape[0]
    gb.dst[:T].zero_()
    gb.finish(st.band_args)
    out.add_(gb.dst[:T])


def _band_eager(st, x, topk_ids, topk_weights, y) -> None:
    """A prompt chunk: the band's picks in calls of 8 rows and 16 experts each, one after another, into y."""
    from . import band
    gb = _BAND["gb"]
    call, a = _BAND["eager_call"], _BAND["eager_args"]
    T = x.shape[0]
    for r0 in range(0, T, band.MAXT):
        r1 = min(T, r0 + band.MAXT)
        xc = x[r0:r1].contiguous()
        band.quant(xc, gb.xq, gb.xs)
        for base in range(0, 16 * band.MAXT, band.MAXE):
            tmp = st.band_stats if base == 0 else torch.zeros(3, dtype=torch.int32, device=x.device)
            band.plan(topk_ids[r0:r1].contiguous(), topk_weights[r0:r1].contiguous(), st.row_map, st.band_map,
                      call, tmp, base)
            if base:                      # picks the first call counted as past its 16 experts, served by this one
                st.band_stats[0] += tmp[0]
                st.band_stats[1] -= tmp[0]
            if int(call[0]) == 0:
                break
            gb.dst[:r1 - r0].zero_()
            gb.send(a)
            gb.finish(a)
            y[r0:r1].add_(gb.dst[:r1 - r0])


# ------------------------------------------------------------------------------------------------- install
# One MegaMoE symmetric buffer for every layer. vLLM keys its cached buffer by the module's expert count; with per-layer
# rows (K31_HBM_MAP) each distinct count would get its own 156-320 MiB buffer (R1, 11-18: ~60 of them, out of memory
# at the first 512-row forward). The buffer for the largest count is allocated once and every other count gets a
# DeepGEMM SymmBuffer view on the same memory (`base=`). Safe across counts: the kernel's persistent state (grid and
# barrier counters, per-expert send / receive counts, ring signals) is the fixed-layout MegaMoESignals struct at the
# buffer's start, and each launch zeroes the counters it used before it exits ("clean blocks" in
# sm100_fp8_fp4_mega_moe.cuh); the count-dependent regions after it are written before they are read in every launch.
_SYMM: dict = {"emax": 0, "views": {}}


def _get_symm_buffer(self):
    E, emax = int(self.num_experts), int(_SYMM["emax"])
    if emax <= 0 or E >= emax or self.weight_format.uses_nvfp4_api:
        return _orig["symm"](self)
    key = (E, self.max_num_tokens, self.top_k, self.hidden_size, self.intermediate_size, self.num_shared_experts,
           self.weight_format.mma_type, self.activation.name, torch.accelerator.current_device_index())
    sb = _SYMM["views"].get(key)
    if sb is None:
        self.num_experts = emax
        try:
            base = _orig["symm"](self)                     # vLLM's cached buffer for the largest count
        finally:
            self.num_experts = E
        from vllm.utils.deep_gemm import _import_deep_gemm
        mega = importlib.import_module(_import_deep_gemm().__name__ + ".mega")
        sb = mega.SymmBuffer(base.group, E, self.max_num_tokens, self.top_k, self.hidden_size,
                             self.intermediate_size, self.num_shared_experts,
                             mma_type=self.weight_format.mma_type, activation=self.activation.name, base=base)
        _SYMM["views"][key] = sb
        if len(_SYMM["views"]) == 1:
            _LOG(f"MegaMoE symmetric buffer: one {base.buffer.nbytes / 2**20:.0f} MiB buffer (sized for {emax} experts) "
                 f"shared by every layer's expert count")
    return sb


def _patch_moe(module) -> None:
    cls = module.DeepGemmMegaMoEExperts
    if getattr(cls, "_k31", False):
        return
    _orig["init"], _orig["forward"] = cls.__init__, cls.forward
    from . import rowslice                       # K31_MEGAMOE_ROWS: calls of more rows in row slices (else unchanged)
    _orig["forward"] = rowslice.megamoe_forward(_orig["forward"])
    cls.__init__, cls.forward = _init, _forward
    if hasattr(cls, "get_symm_buffer"):
        _orig["symm"] = cls.get_symm_buffer
        cls.get_symm_buffer = _get_symm_buffer
    cls._k31 = True


MEM_REPORT = os.environ.get("K31_MEM_REPORT", "0") == "1"
# K31_MEM_PEAK=<rows> at start-up: hooks for the memory peaks of steps of >= <rows> tokens (mempeak.py; the threshold
# is then a runtime flag)
MEM_PEAK = os.environ.get("K31_MEM_PEAK", "0") not in ("", "0")


def _mem_report(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    alloc, res = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    used = total - free
    g = 2 ** 30
    _LOG(f"memory {tag}: used {used / g:.2f} GiB of {total / g:.2f} (free {free / g:.2f}); torch allocated "
         f"{alloc / g:.2f}, reserved {res / g:.2f} (cached {(res - alloc) / g:.2f}); outside torch {(used - res) / g:.2f}")


def _patch_runner(module) -> None:
    cls = module.GPUModelRunner
    if getattr(cls, "_k31", False):
        return
    if DRAFT_FULL_KV:
        orig_spec = cls.get_kv_cache_spec

        def get_kv_cache_spec(self):
            from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, replace_as
            specs = orig_spec(self)
            names = set(getattr(getattr(self, "speculator", None), "draft_attn_layer_names", ()) or ()) & specs.keys()
            n = 0
            for name in names:
                if isinstance(specs[name], SlidingWindowSpec):
                    specs[name] = replace_as(specs[name], FullAttentionSpec, drop=("extra_retained_tokens",))
                    n += 1
            _LOG(f"K31_DRAFT_FULL_KV: {n} of the drafter's {len(names)} attention layers given to the KV manager as "
                 f"full attention (fine-grained prefix-cache hits stay on)")
            return specs
        cls.get_kv_cache_spec = get_kv_cache_spec
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        _bind(self)
        if DSA_FUSED:                            # per-instance forwards on the target's DSA layers (dsa_fused.py)
            from . import dsa_fused
            dsa_fused.bind(self.model, _LOG)
        if FORCE_DRAFTS:                         # the module the live runner class comes from (vLLM's GPU runner v2)
            import importlib
            _patch_combine(importlib.import_module(type(self).__module__))
        spec = getattr(self, "speculator", None)
        if (DRAFT_W8 or DRAFT_W8_HEAD) and spec is not None and isinstance(getattr(spec, "model", None),
                                                                           torch.nn.Module):
            _draft_w8(spec.model)
        if DRAFT_TOPP or DRAFT_ARGMAX:
            if DRAFT_TOPP and DRAFT_ARGMAX:
                raise RuntimeError("K31: K31_DRAFT_TOPP=1 and K31_DRAFT_ARGMAX=1 exclude each other; set one")
            from . import drafttopp
            drafttopp.bind(self, argmax=DRAFT_ARGMAX, forced=FORCE_DRAFTS, log=_LOG)
        for name, m in self.model.named_modules():   # the target's BF16 LM head, for force.py's hidden-state references
            if type(m).__name__ == "ParallelLMHead" and "draft" not in name and "mtp" not in name:
                LM_HEAD[0] = m.weight
                _LOG(f"LM head for exact-KL references: {name} {tuple(m.weight.shape)} {m.weight.dtype}")
                if HEAD_W8:
                    _head_w8(m)
                break
        if MEM_PEAK:                             # K31_MEM_PEAK: per-step memory peaks of prompt chunks (mempeak.py)
            from . import mempeak
            mempeak.install(self.model)
        if MEM_REPORT:
            _mem_report("after load_model")
        return out

    cls.load_model = load_model
    # K31_MEM_REPORT=1: GB300 memory at start-up milestones (after load_model, after the KV cache, after graph
    # capture): free (cudaMemGetInfo), torch allocated / reserved, and the rest (CUDA context, modules, graph pools,
    # allocations outside torch's caching allocator such as cuBLAS / DeepGEMM workspaces and registered buffers)
    if MEM_REPORT:
        for name in ("initialize_kv_cache", "capture_model"):
            f = getattr(cls, name, None)
            if f is None or getattr(f, "_k31_mem", False):
                continue

            def wrap(self, *a, _f=f, _n=name, **kw):
                out = _f(self, *a, **kw)
                _mem_report(f"after {_n}")
                return out
            wrap._k31_mem = True
            setattr(cls, name, wrap)
    # residency between steps (never waits for the GPU): after execute_model, which runs once per step in every mode
    # (with speculative decoding the V2 runner does the whole step there and rarely calls sample_tokens)
    orig_exec = cls.execute_model

    every = max(1, int(os.environ.get("K31_TICK_EVERY", "1")))   # tick every n-th step (credit keeps the rate)
    pend = [0]

    rewarm = os.environ.get("K31_REWARM", "0") == "1" and RING > 0 and DEFER
    eager = [False]

    def execute_model(self, *args, **kwargs):
        so = args[0] if args else kwargs.get("scheduler_output")
        ntok = getattr(so, "total_num_scheduled_tokens", 1) or 1
        if rewarm and _RES and eager[0] and ntok <= 16:          # the first decode step after prompt rows
            t0 = time.time()
            n = _RES[0].rewarm()
            eager[0] = False
            if _RES[0].n_rewarm <= 3 or _RES[0].n_rewarm % 50 == 0:
                _LOG(f"re-warm {_RES[0].n_rewarm}: {n} ring records fetched ({1e3 * (time.time() - t0):.0f} ms host)")
        if ntok > 16:
            eager[0] = True
        _flags.refresh()                                         # runtime flags (flags.py) for this step
        if _BULK[0] is not None:                                 # park the band's service kernel for prompt steps
            _BULK[0].set_mode(ntok > BAND_BULK_MIN and _flags.on("K31_BAND_BULK", "1"))
        if MEM_PEAK:
            from . import mempeak
            mempeak.step_begin(ntok)
        if DECODE_SPLIT:                                         # a decode-only step: no request brings more rows
            nst = getattr(so, "num_scheduled_tokens", None)      # than a verify block (eager calls split only then)
            try:
                per = list(nst.values()) if isinstance(nst, dict) else [int(v) for v in nst]
            except TypeError:
                per = []
            _SPLIT["decode"] = bool(per) and ntok <= SPLIT_MAX and max(per) <= _SPLIT["P"]
        _IN_EXEC[0] = True                                       # the prompt path may tick the residency (serving
        try:                                                     # steps only: never the start-up's profile run,
            out = orig_exec(self, *args, **kwargs)               # whose promotions would land during graph capture)
        finally:
            _IN_EXEC[0] = False
            _SPLIT["decode"] = False
        if MEM_PEAK:
            mempeak.step_end()
        _STEPS[0] += 1
        if JOINT_STAT and _STEPS[0] % 2000 == 0 and _JT.get("stat") is not None:
            n = [int(v) for v in _JT["stat"].tolist()]
            _LOG(f"decode joint stat after {_STEPS[0]} steps: {n[0]} joint calls of {n[1] / max(1, n[0]):.2f} blocks; "
                 f"records staged {n[2]} ({n[2] / max(1, n[1]):.2f} per block), {n[3]} of them "
                 f"({100.0 * n[3] / max(1, n[2]):.1f}%) staged by another block of the call too; MegaMoE rows read "
                 f"block by block {n[4]} ({n[4] / max(1, n[1]):.1f} per block), distinct per call {n[5]} "
                 f"({100.0 * n[5] / max(1, n[4]):.1f}%)")
        if _LT[0] is not None:
            _LT[0].tick()                                        # K31_LATE: its counters every K31_LATE_LOG_S
        if _RES:
            pend[0] += ntok                                              # promotion credit is per token
            if _STEPS[0] % every == 0:
                _RES[0].tick(pend[0])
                pend[0] = 0
        return out

    cls.execute_model = execute_model
    if os.environ.get("K31_ACCEPTED", "0") == "1" and hasattr(cls, "sample"):
        # residency scores and credit from accepted rows: after the step's (rejection) sampling, commit the rows whose
        # input was a real token (row t of request r with t - start[r] < num_sampled[r])
        orig_sample = cls.sample

        def sample(self, hidden_states, input_batch, grammar_output):
            out = orig_sample(self, hidden_states, input_batch, grammar_output)
            if _RES and input_batch.num_reqs > 0 and torch.is_tensor(out[1]) and out[1].is_cuda:
                _RES[0].commit(input_batch.query_start_loc, out[1], input_batch.num_reqs)
            return out

        cls.sample = sample
        _LOG("residency scores and promotion credit from accepted rows (K31_ACCEPTED)")
    cls._k31 = True


def _patch_mlopart(module) -> None:
    """DeepGEMM's locality partitioning starts an MPS daemon inside the container at the first MegaMoE call (seen in
    every test server's shutdown log, 11-17); it changes how kernels share the GPU and buys nothing at one rank and
    T <= 64. K31_MLOPART=1 keeps it."""
    if os.environ.get("K31_MLOPART", "0") != "1":
        module.is_available = lambda device_uuid: False


def _join_window(cls, name) -> None:
    """Join the pending side-stream window before an attention op that breaks vLLM's CUDA graph capture (KDA core,
    MLA attention): the fork after a MegaMoE call and this join then sit in the same graph segment."""
    orig = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        ev = _WIN["ev"]
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
            _WIN["ev"] = None
        return orig(self, *args, **kwargs)
    setattr(cls, name, wrapped)


def _patch_kda(module) -> None:
    if WINDOW:
        _join_window(module.KimiK3DeltaAttention, "_forward")


def _patch_mla(module) -> None:
    if WINDOW:
        _join_window(module.MultiHeadLatentAttention, "_attention")


def _patch_dsa(module) -> None:
    if WINDOW:                      # an MLA layer's first graph break is its DSA indexer's sparse_attn_indexer op
        _join_window(module.KimiK3DSAIndexer, "score")
    if GATE_SIDE and not getattr(module.KimiK3DSAAttention, "_k31_gate", False):
        _gate_side(module)
    from . import rowslice          # K31_DSA_ROW_SLICE (runtime flag): a prompt chunk's sparse attention in row slices
    rowslice.install_dsa()


def _gate_join() -> None:
    ev = _GATE["ev"]
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
        _GATE["ev"] = None


def _gate_side(module) -> None:
    """K31_GATE_SIDE: fork g_proj at the top of the DSA attention, join before the indexer's score."""
    cls = module.KimiK3DSAAttention
    orig_fwd = cls.forward

    def gp_forward(gp, x):
        pre = gp._k31_pre
        if pre is not None and pre.shape[0] == x.shape[0]:
            gp._k31_pre = None
            _gate_join()                     # mode 1: a no-op (score joined); mode 2: the join, after the attention
            return pre, None
        return gp._k31_orig_fwd(x)

    def qb_forward(qb, attn, x):
        out = qb._k31_orig_fwd(x)
        hx = getattr(attn, "_k31_gate_x", None)
        if hx is not None:                   # K31_GATE_SIDE=2: q_b is queued; g_proj forks behind it
            attn._k31_gate_x = None
            main = torch.cuda.current_stream()
            side = _SIDE["stream"]
            side.wait_stream(main)
            with torch.cuda.stream(side):
                gate = attn.g_proj._k31_orig_fwd(hx)[0]
                ev = torch.cuda.Event()
                ev.record(side)
            _GATE["ev"] = ev
            attn.g_proj._k31_pre = gate
        return out

    def forward(self, positions, hidden_states):
        gp = getattr(self, "g_proj", None)
        if gp is None or getattr(self, "_packed", None) is not None or hidden_states.dim() != 2 or \
                hidden_states.shape[0] > 16:
            return orig_fwd(self, positions, hidden_states)
        if not hasattr(gp, "_k31_orig_fwd"):
            gp._k31_orig_fwd = gp.forward
            gp._k31_pre = None
            gp.forward = lambda x, _gp=gp: gp_forward(_gp, x)
        main = torch.cuda.current_stream()
        if _SIDE["stream"] is None:
            _SIDE["stream"] = torch.cuda.Stream(device=main.device)
        side = _SIDE["stream"]
        _gate_join()
        qb = getattr(self, "q_b_proj", None)
        if GATE_SIDE_MODE == 2 and qb is not None and not _breakable_capture():
            if not hasattr(qb, "_k31_orig_fwd"):
                qb._k31_orig_fwd = qb.forward
                qb.forward = lambda x, _qb=qb, _a=self: qb_forward(_qb, _a, x)
            self._k31_gate_x = hidden_states         # q_b's wrapper forks g_proj on it
            _GATE["late"] = True
            try:
                return orig_fwd(self, positions, hidden_states)
            finally:
                self._k31_gate_x = None
                _GATE["late"] = False
                gp._k31_pre = None
                _gate_join()
        side.wait_stream(main)                       # hidden_states is ready
        with torch.cuda.stream(side):
            gate = gp._k31_orig_fwd(hidden_states)[0]
            ev = torch.cuda.Event()
            ev.record(side)
        _GATE["ev"] = ev
        gp._k31_pre = gate
        try:
            return orig_fwd(self, positions, hidden_states)
        finally:
            gp._k31_pre = None
            _gate_join()
    cls.forward = forward
    orig_score = module.KimiK3DSAIndexer.score

    def score(self, *args, **kwargs):
        if not _GATE["late"]:                        # mode 1: the fork's join sits in the same graph segment as the fork
            _gate_join()
        return orig_score(self, *args, **kwargs)
    module.KimiK3DSAIndexer.score = score
    cls._k31_gate = True
    _LOG("DSA output gate: g_proj on a side stream beside the attention front (K31_GATE_SIDE)")


def _late_delta(L, u_now, u_late):
    """Layer L's late share in hidden space: up_L(norm_L(u_now + u_late) - norm_L(u_now))."""
    tr = _MOE[L].routed_output_transform
    if FUSED_DEFER:
        from . import defer
        diff = defer.diff(u_now, u_late, tr.norm.weight, tr.norm.variance_epsilon)
    else:
        g = tr.norm.weight.float()
        eps = tr.norm.variance_epsilon

        def rmsn(v):
            return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * g
        un = u_now.float()
        diff = (rmsn(un + u_late.float()) - rmsn(un)).to(u_now.dtype)
    delta, _ = tr.up_proj(diff)
    return delta


def _fin_join() -> None:
    ev = _FIN.get("ev")
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
        _FIN["ev"] = None
        _FIN["keep"] = None


def _fin_take() -> None:
    """K31_LATE's routed output transform: join the finish branch and drop the statistic (it runs vLLM's own steps)."""
    _FIN.pop("rstd", None)
    _fin_join()


def _patch_model(module) -> None:
    """KimiMoE.forward: register the module by layer; after it, complete the previous layer's late share exactly,
    delta = up_L(norm_L(u_now + u_late) - norm_L(u_now)) (difference in fp32, then layer L's up-projection), and add
    it to this layer's MoE output (it lands one layer late)."""
    if PRESTAGE and not getattr(module.KimiDecoderLayer, "_k31_prestage", False):
        from . import prestage
        dlp = module.KimiDecoderLayer
        orig_pre = dlp._pre_attn_norm
        orig_post_ps = dlp._post_attn_norm
        ar_fn, fgt_fn = module.attn_res, module.fused_grouped_topk

        def _pre_attn_norm_ps(self, hidden_states, residual, prefix_sum):
            """K31_PRESTAGE: the layer's inputs are final here; fork the side stream's prediction and staging."""
            out = orig_pre(self, hidden_states, residual, prefix_sum)
            mlp = getattr(self, "mlp", None)
            st = getattr(getattr(mlp, "experts", None), "_k31", None)
            if st is not None and getattr(self, "use_attn_res", False) and getattr(mlp, "gate", None) is not None \
                    and not _breakable_capture():        # a fork and its join must stay in one graph segment
                hs, ps, res = out
                prestage.fork(self, st, mlp, ps, res, hs.shape[0], ar_fn, fgt_fn, SLOTS, rowcap=_RCAP[0])
            return out

        def _post_attn_norm_ps(self, hidden_states, residual, prefix_sum):
            """K31_PRESTAGE: the side stream has read the prefix sum this call adds the attention output to in place."""
            st = getattr(getattr(getattr(self, "mlp", None), "experts", None), "_k31", None)
            if st is not None:
                prestage.wait_read(st.idx)
            return orig_post_ps(self, hidden_states, residual, prefix_sum)
        dlp._pre_attn_norm = _pre_attn_norm_ps
        dlp._post_attn_norm = _post_attn_norm_ps
        dlp._k31_prestage = True
        _LOG("K31_PRESTAGE: decoder layers fork the pre-gate after _pre_attn_norm")
    if (FUSE_UP or BAND_FIN_SIDE or LATE) and not getattr(module.KimiRoutedOutputTransform, "_k31", False):
        rot = module.KimiRoutedOutputTransform
        orig_rot = rot.forward

        def rot_forward(self, hidden_states, residual=None):
            if _LT[0] is not None:                       # K31_LATE: a call whose layer sent a late job keeps u and y
                out = _LT[0].rot(self, hidden_states, residual, _fin_take)
                if out is not None:
                    return out
            rstd = _FIN.pop("rstd", None)                # K31_TAIL1: the finish wrote the statistic of these rows
            _fin_join()
            w8 = getattr(self.up_proj, "_k31_w8", None) if (FUSE_UP or rstd is not None) else None
            if w8 is not None and w8[1] is None:         # a Marlin weight (K31_W8_MARLIN): not the split-K layout
                w8 = None
            norm = self.norm
            if w8 is not None and residual is not None and norm is not None and hidden_states.dim() == 2 and \
                    hidden_states.shape[0] <= 16 and hidden_states.dtype == torch.bfloat16 and \
                    residual.dtype == torch.bfloat16 and residual.is_contiguous() and \
                    getattr(self.up_proj, "bias", None) is None and getattr(norm, "weight", None) is not None:
                from . import w8a16
                return w8a16.linear_norm_res(hidden_states.contiguous(), w8[0], w8[1], norm.weight.data,
                                             norm.variance_epsilon, residual, rstd=rstd)
            return orig_rot(self, hidden_states, residual)
        rot.forward = rot_forward
        rot._k31 = True
        if FUSE_UP or BAND_FIN_SIDE:
            _LOG("routed output transform: RMSNorm + W8A16 up-projection + residual add fused (K31_FUSE_UP)")
    if LATE and not getattr(module.KimiDecoderLayer, "_k31_late", False):
        dlate = module.KimiDecoderLayer
        orig_pan = dlate._post_attn_norm

        def _post_attn_norm_late(self, hidden_states, residual, prefix_sum):
            """K31_LATE: the previous layer's late share (landed beside this layer's attention) goes into the attention
            output, or into the stored block state the previous layer ended, before this layer's MoE input mix."""
            lt = _LT[0]
            if lt is not None and lt.pend is not None:
                hidden_states = lt.join(self.layer_idx, hidden_states, residual)
            return orig_pan(self, hidden_states, residual, prefix_sum)
        dlate._post_attn_norm = _post_attn_norm_late
        dlate._k31_late = True
    if not DEFER and not SHARED_SIDE and not DECODE_SPLIT:
        return
    cls = module.KimiMoE
    orig = cls.forward

    fgt = module.fused_grouped_topk

    def _router_down_nojoin(moe, h, fork_shared=None):
        """vLLM's _maybe_overlap_router_and_down_proj for a decode-size MegaMoE call, without its join: the latent
        down-projection on vLLM's aux stream, the router GEMV and grouped top-k on this one; the aux stream's event is
        left in _LD for _forward to join where the latent rows are read (the band send's branch, prep / MegaMoE)."""
        main = torch.cuda.current_stream()
        # (a block of a split batch takes its own events: the module's pair would be recorded once per block)
        ev0, ev1 = (torch.cuda.Event(), torch.cuda.Event()) if (_SPLIT["sub"] or _JT["on"]) \
            else moe._down_proj_events
        ev0.record(main)                             # h is ready
        # router GEMV and top-k first, as vLLM orders them: the graph launches its parallel branches in capture order,
        # and a latent down captured first takes the SMs and delays the router (arm C / D 11-18: router + top-k
        # 11.5 -> 21.3 us, which cancelled the deferred join's 11 us)
        logits, _ = moe.gate(h)
        if fork_shared is not None:                  # K31_SHARED_EARLY: the shared experts start here, before top-k
            fork_shared()
        nop = NOPREP and _nop().begin(moe, h, MODE == "staged" and not DEFER, SLOTS)
        r = _nop().route(moe, h, logits, LAM) if nop else None     # top-k + unitk + MegaMoE's routing inputs
        if r is not None:
            w, ids = r
        else:
            w, ids = fgt(hidden_states=h, gating_output=logits, topk=moe.experts.top_k,
                         renormalize=moe.moe_renormalize, e_score_correction_bias=moe.gate.e_score_correction_bias.data,
                         num_expert_group=moe.num_expert_group, topk_group=moe.topk_group,
                         scoring_func=moe.moe_router_activation_func, routed_scaling_factor=moe.routed_scaling_factor)
        with torch.cuda.stream(moe._down_proj_stream):
            moe._down_proj_stream.wait_event(ev0)
            routed_h = _nop().latent_down(moe, h) if nop else None   # the same rows + MegaMoE's MXFP8 copy
            if routed_h is None:
                routed_h, _ = moe.routed_expert_down_proj(h)
            ev1.record(moe._down_proj_stream)
        _LD["ev"] = ev1
        return routed_h, w, ids

    def moe_front(self, h):
        """The dense front of KimiMoE.forward's MegaMoE branch for the rows h: router gate + top-k, the latent
        down-projection (pending on the aux stream in _LD on the deferred-join path) and the shared experts forked
        onto the side stream. -> (latent rows, weights, ids, {"out": shared output, "ev": its event})."""
        main = torch.cuda.current_stream()
        if _SIDE["stream"] is None:
            _SIDE["stream"] = torch.cuda.Stream(device=main.device)
        side = _SIDE["stream"]
        box = {}

        def fork():                                  # the shared experts on the side stream, from h
            side.wait_stream(main)
            with torch.cuda.stream(side):
                box["out"] = self.shared_experts(h)
                box["ev"] = torch.cuda.Event()
                box["ev"].record(side)
        # (a dense chunk of a joint decode batch may hold more than 16 rows: K31_DECODE_JOINT_DENSE)
        if SHARED_SIDE_MODE == 3 and h.shape[0] <= (max(16, JOINT_DENSE_ROWS) if _JT["on"] else 16) \
                and getattr(self, "_down_proj_stream", None) is not None \
                and self.routed_expert_down_proj is not None and getattr(self, "use_mega_moe", False) \
                and not _breakable_capture():
            routed_h, router_output, topk_ids = _router_down_nojoin(self, h, fork if SHARED_EARLY else None)
            _LD_SEEN["nojoin"] += 1
        elif SHARED_SIDE_MODE >= 2:                  # after the router GEMV, the latent down and top-k
            routed_h, router_output, topk_ids = self._maybe_overlap_router_and_down_proj(h)
            if SHARED_SIDE_MODE == 3:
                _LD_SEEN["vllm"] += 1
                if _LD_SEEN["vllm"] in (1, 1000):
                    _LOG(f"latent down: vLLM's joined path for a call of {h.shape[0]} rows (aux stream "
                         f"{getattr(self, '_down_proj_stream', None) is not None}, breakable capture "
                         f"{_breakable_capture()}; deferred-join calls so far {_LD_SEEN['nojoin']})")
        if SHARED_SIDE_MODE == 3 and _LD_SEEN["nojoin"] in (1, 92, 10000):
            _LOG(f"latent down: deferred join (K31_SHARED_SIDE=3), {_LD_SEEN['nojoin']} calls, capturing "
                 f"{torch.cuda.is_current_stream_capturing()}")
        if "out" not in box:                         # h is ready; the fork after the router, latent down and top-k
            fork()
        if SHARED_SIDE_MODE < 2:
            routed_h, router_output, topk_ids = self._maybe_overlap_router_and_down_proj(h)
        return routed_h, router_output, topk_ids, box

    def moe_shared_side(self, hidden_states):
        """KimiMoE.forward's MegaMoE branch with the shared experts forked onto a side stream."""
        num_tokens, hidden_size = hidden_states.shape
        h = hidden_states.view(-1, hidden_size)
        main = torch.cuda.current_stream()
        routed_h, router_output, topk_ids, box = moe_front(self, h)
        final = self.experts(routed_h, router_output, topk_ids)
        main.wait_event(box["ev"])                   # the join: the shared output is complete
        final = self.routed_output_transform(final, residual=box["out"])
        return final.view(num_tokens, hidden_size)

    def joint_ok(self, st) -> bool:
        return bool(DECODE_JOINT) and SHARED_SIDE and getattr(self, "use_mega_moe", False) and \
            self.shared_experts is not None and self.routed_output_transform is not None

    def joint_forward(self, st, hidden_states, level=None):
        """K31_DECODE_JOINT: a decode batch of several verify blocks through one MoE layer (see the flag's note)."""
        T, hidden_size = hidden_states.shape
        h = hidden_states.view(-1, hidden_size)
        blocks = split_blocks(T, _SPLIT["block"], _SPLIT["rows"])
        chunks = dense_chunks(blocks, JOINT_DENSE_ROWS)
        lv = _joint_level(st, blocks)
        lv = lv if level is None else min(lv, level)
        if (T, lv) not in _JT["seen"]:
            _JT["seen"].add((T, lv))
            _LOG(f"decode joint: a batch of {T} rows at level {lv}: MoE calls of {[b - a for a, b in blocks]} rows"
                 f"{' as one MegaMoE call' if lv == 2 else ''}, dense chunks of {[cb - ca for ca, cb, _ in chunks]} "
                 f"rows (first seen in layer {st.idx}, capturing {torch.cuda.is_current_stream_capturing()})")
        _JT["level"][lv] = _JT["level"].get(lv, 0) + 1
        main = torch.cuda.current_stream()
        outs = []
        _JT["on"] = True
        try:
            if lv >= 2:
                fronts, lds = [], []
                for ca, cb, _blks in chunks:
                    fronts.append(moe_front(self, h[ca:cb]))
                    if _LD["ev"] is not None:        # this chunk's latent down, pending on the aux stream
                        lds.append(_LD["ev"])
                        _LD["ev"] = None
                routed = _joint_moe(self.experts, st, chunks, fronts, lds)
                rstd = _JT.pop("rstd", None)
                for (ca, cb, _blks), f in zip(chunks, fronts):
                    main.wait_event(f[3]["ev"])      # the join: this chunk's shared output is complete
                    if rstd is not None:
                        _FIN["rstd"] = rstd[ca:cb]   # the finishes wrote the statistic of these rows (K31_TAIL1)
                    outs.append(self.routed_output_transform(routed[ca:cb], residual=f[3]["out"]))
            else:
                for ca, cb, blks in chunks:
                    routed_h, w, ids, box = moe_front(self, h[ca:cb])
                    ys = []
                    _SPLIT["sub"] = True
                    try:
                        for a, b in blks:
                            _SPLIT["row0"], _JT["r0"] = a, a - ca
                            with _PadRows(a, b):
                                y = self.experts(routed_h[a - ca:b - ca], w[a - ca:b - ca], ids[a - ca:b - ca])
                            if _FIN.get("ev") is not None:   # the block's finish branch, as the split joins it
                                main.wait_event(_FIN["ev"])  # before its routed output transform
                                _FIN["ev"] = None
                                _FIN["keep"] = None
                            ys.append(y)
                    finally:
                        _SPLIT["sub"], _SPLIT["row0"], _JT["r0"] = False, 0, 0
                    if len({int(y.data_ptr()) for y in ys}) != len(ys):
                        raise RuntimeError("K31_DECODE_JOINT: MegaMoE returned one buffer for two calls of a layer")
                    final = ys[0] if len(ys) == 1 else torch.cat(ys)
                    main.wait_event(box["ev"])       # the join: this chunk's shared output is complete
                    outs.append(self.routed_output_transform(final, residual=box["out"]))
        finally:
            _JT["on"], _JT["r0"] = False, 0
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        return out.view(T, hidden_size)

    def joint_check(self, st, hidden_states):
        """K31_DECODE_JOINT_CHECK: the split twice, then level 1 and the configured level on the same rows."""
        _JT["checks"][st.idx] = _JT["checks"].get(st.idx, 0) + 1
        split = lambda: _split_call(lambda rows: one_forward(self, st, rows), hidden_states, st.idx)   # noqa: E731
        r1 = split().clone()
        r2 = split().clone()
        y1 = joint_forward(self, st, hidden_states, level=1).clone()
        y = joint_forward(self, st, hidden_states)
        torch.cuda.synchronize()

        def cmp(u, v):
            if torch.equal(u, v):
                return "bit-equal"
            d = (u.float() - v.float())
            return f"DIFFERS (max abs {float(d.abs().max()):.3e}, relative L2 " \
                   f"{float(d.norm() / v.float().norm().clamp_min(1e-30)):.3e}, " \
                   f"{int((u != v).any(dim=-1).sum())} of {u.shape[0]} rows)"
        _LOG(f"decode joint check layer {st.idx} ({hidden_states.shape[0]} rows, call {_JT['checks'][st.idx]}): "
             f"split twice {cmp(r2, r1)}; level 1 vs split {cmp(y1, r1)}; level "
             f"{_joint_level(st, split_blocks(hidden_states.shape[0], _SPLIT['block'], _SPLIT['rows']))} vs level 1 "
             f"{cmp(y, y1)}")
        return y

    def forward(self, hidden_states):
        st = getattr(getattr(self, "experts", None), "_k31", None)
        if st is not None:
            _MOE[st.idx] = self
        if st is not None and _split_wanted(hidden_states):      # K31_DECODE_SPLIT: one call per verify block
            if joint_ok(self, st):                               # K31_DECODE_JOINT: with the shared work done once
                if JOINT_CHECK > _JT["checks"].get(st.idx, 0) and not torch.cuda.is_current_stream_capturing():
                    return joint_check(self, st, hidden_states)
                return joint_forward(self, st, hidden_states)
            return _split_call(lambda rows: one_forward(self, st, rows), hidden_states, st.idx)
        return one_forward(self, st, hidden_states)

    def one_forward(self, st, hidden_states):
        if SHARED_SIDE and st is not None and getattr(self, "use_mega_moe", False) and \
                self.shared_experts is not None and self.routed_output_transform is not None:
            out = moe_shared_side(self, hidden_states)
        else:
            out = orig(self, hidden_states)
        if not DEFER:
            return out
        late = _DEF["late"]
        if st is not None and late is not None and late[0] == st.idx - 1:
            _DEF["late"] = None
            L, u_now, u_late = late
            d = _late_delta(L, u_now, u_late).view_as(out)
            if (L + 1) % ATTN_RES_BLOCK == 0 and _AR["res"] is not None:
                # L ended block b = (L + 1) / 12, which this layer stored: complete the stored state in place (every
                # later layer and the output read it through their mixes); this layer's own inputs read it before
                _AR["res"][:, (L + 1) // ATTN_RES_BLOCK, :].add_(d)
            else:
                out = out + d
        return out
    cls.forward = forward
    dl0 = module.KimiDecoderLayer
    orig_fwd = dl0.forward

    def layer_forward(self, positions, hidden_states, residual, prefix_sum=None, **kwargs):
        _AR["res"] = residual if residual is not None and residual.dim() == 3 else None
        return orig_fwd(self, positions, hidden_states, residual, prefix_sum, **kwargs)
    dl0.forward = layer_forward
    if DEFER_LAND != "mlp":
        return
    dl = module.KimiDecoderLayer
    orig_post = dl._post_attn_norm

    def _post_attn_norm(self, hidden_states, residual, prefix_sum):
        """Half-layer landing: the previous layer's deferred experts, computed now by a small MegaMoE call on the
        pseudo rows (the window has been joined at this layer's attention), land in this layer's attention delta, so
        its MoE input mix already reads them."""
        pend = _DEF["pending"]
        mlp = getattr(self, "mlp", None)
        ex = getattr(mlp, "experts", None)
        st = getattr(ex, "_k31", None)
        if pend is not None and st is not None and pend["idx"] == st.idx - 1 \
                and pend["T"] == hidden_states.shape[0]:
            _DEF["pending"] = None
            u_late = _orig["forward"](ex, pend["x"], pend["w"], pend["ids"].to(torch.int32), None)
            d = _late_delta(pend["idx"], pend["u"], u_late).view_as(hidden_states)
            L = pend["idx"]
            if (L + 1) % ATTN_RES_BLOCK == 0 and residual is not None and residual.dim() == 3:
                residual[:, (L + 1) // ATTN_RES_BLOCK, :].add_(d)     # the stored block state this layer wrote
            else:
                hidden_states = hidden_states + d
        return orig_post(self, hidden_states, residual, prefix_sum)
    dl._post_attn_norm = _post_attn_norm


def _patch_fp8_online(module) -> None:
    if os.environ.get("K31_DENSE_FP6", "0") == "1" or os.environ.get("K31_DENSE_FMT"):   # Humming dense (fp6.py)
        from . import fp6
        fp6.install(_LOG)


def _head_w8(head) -> bool:
    """An FP8 W8A16 copy of an LM head (BF16 [V, H]) and the class-level apply that uses it for calls of <= 16 rows:
    every call with K31_HEAD_W8, else only inside the drafter's compute_draft_logits (_DRAFT_HEAD). The BF16 weight
    stays (force.py's KL references and longer calls read it)."""
    from . import w8a16
    if head is None:
        return False
    if getattr(head, "_k31_w8h", None) is not None:
        return True
    hw = getattr(head, "weight", None)
    if hw is None or hw.dtype != torch.bfloat16 or hw.dim() != 2 or hw.shape[1] % w8a16.GK:
        return False
    from . import w8i
    i8 = W8I_DRAFT and w8i.supported(*hw.shape)
    if not i8 and not w8a16.faster_than_bf16(*hw.shape):
        _LOG(f"LM head {tuple(hw.shape)}: kept BF16 (W8A16 not measured faster)")
        return False
    if i8:                                           # K31_W8I_DRAFT: the int8 GEMV (199 us at 4 rows, 5.9 TB/s)
        q, sc = w8i.I8W(hw.data), None
        nb = q.nbytes
    else:
        q, sc = w8a16.quantize(hw.data)
        nb = q.numel() + 2 * sc.numel()
        q, sc = w8a16.maybe_marlin(q, sc)            # K31_W8_MARLIN: Marlin's layout where it measured faster
    head._k31_w8h = (q, sc, hw.shape[0])
    hcls = type(head.quant_method)
    if not getattr(hcls, "_k31_w8h", False):
        orig_h = hcls.apply

        def head_apply(self, layer, x, bias=None, _orig=orig_h):
            w8 = getattr(layer, "_k31_w8h", None)
            if (HEAD_W8 or _DRAFT_HEAD[0]) and w8 is not None and x.dim() == 2 and x.shape[0] <= 16 and \
                    x.dtype == torch.bfloat16:
                from . import w8a16 as _w
                y = _w.linear(x.contiguous(), w8[0], w8[1])
                return y if bias is None else y + bias
            return _orig(self, layer, x, bias)
        hcls.apply = head_apply
        hcls._k31_w8h = True
    torch.cuda.empty_cache()
    _LOG(f"LM head {tuple(hw.shape)}: {'int8' if i8 else 'FP8 W8A16'} copy ({nb / 1e9:.2f} GB"
         f"{', Marlin' if sc is None and not i8 else ''}) for "
         f"{'every call' if HEAD_W8 else 'the drafter'}")
    return True


def _draft_w8(model) -> None:
    """K31_DRAFT_W8: FP8 W8A16 copies of the drafter's matching BF16 linears, and their apply on w8a16 (<= 16 rows)."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from . import w8a16
    n, nbytes = 0, 0
    classes = set()
    kept = []
    for name, sub in (model.named_modules() if DRAFT_W8 else ()):
        if not isinstance(sub, LinearBase) or "kv_b_proj" in name or \
                not ("*" in DRAFT_W8_LAYERS or any(p in name for p in DRAFT_W8_LAYERS)):
            continue
        w = getattr(sub, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % w8a16.GK or \
                not isinstance(sub.quant_method, UnquantizedLinearMethod):
            continue
        from . import w8i
        if W8I_DRAFT and w8i.supported(*w.shape):    # K31_W8I_DRAFT: the int8 GEMV (faster than BF16 on every shape)
            q, sc = w8i.I8W(w.data), None
            nbytes += q.nbytes
        else:
            if not w8a16.faster_than_bf16(*w.shape):  # only where the benchmark measured W8A16 faster than BF16
                kept.append(f"{name.split('.')[-1]} {tuple(w.shape)}")
                continue
            q, sc = w8a16.quantize(w.data)
            nbytes += q.numel() + 2 * sc.numel()
            q, sc = w8a16.maybe_marlin(q, sc)        # K31_W8_MARLIN: Marlin's layout where it measured faster
        sub._k31_w8d = (q, sc, w.shape[0])
        classes.add(type(sub.quant_method))
        n += 1
    for cls in classes:                     # UnquantizedLinearMethod and any subclass with its own apply
        if getattr(cls, "_k31_w8d", False):
            continue
        orig_apply = cls.apply

        def apply(self, layer, x, bias=None, _orig=orig_apply):
            w8 = getattr(layer, "_k31_w8d", None)
            if w8 is not None and x.dim() == 2 and x.shape[0] <= 16 and x.dtype == torch.bfloat16:
                from . import w8a16 as _w
                y = _w.linear(x.contiguous(), w8[0], w8[1])
                return y if bias is None else y + bias
            return _orig(self, layer, x, bias)
        cls.apply = apply
        cls._k31_w8d = True
    if DRAFT_W8_HEAD and hasattr(model, "compute_draft_logits"):
        head = getattr(model, "lm_head", None)
        if _head_w8(head):
            hw = head.weight
            orig_cdl = model.compute_draft_logits

            def compute_draft_logits(hidden_states, _orig=orig_cdl):
                _DRAFT_HEAD[0] = True                 # only the drafter's call takes the FP8 copy of the shared head
                try:
                    return _orig(hidden_states)
                finally:
                    _DRAFT_HEAD[0] = False
            model.compute_draft_logits = compute_draft_logits
            _LOG(f"drafter head: base logits from an FP8 W8A16 copy of the shared lm_head {tuple(hw.shape)}")
    torch.cuda.empty_cache()
    _LOG(f"drafter W8A16: {n} linears ({nbytes / 1e9:.2f} GB of FP8 copies) matching {DRAFT_W8_LAYERS}, classes "
         f"{sorted(c.__name__ for c in classes)}; kept BF16 (not measured faster): {sorted(set(kept))}")


def _patch_gate(module) -> None:
    if not ROUTER_GEMV:
        return
    cls = module.GateLinear
    if getattr(cls, "_k31", False):
        return
    orig = cls.forward

    def forward(self, x):
        w = getattr(self, "weight", None)
        if x.dim() == 2 and x.shape[0] <= 16 and x.dtype == torch.bfloat16 and w is not None and \
                w.dtype == torch.bfloat16 and getattr(self, "is_unquantized", False) and \
                self.out_dtype == torch.float32 and w.shape[1] % 128 == 0 and getattr(self, "bias", None) is None:
            from . import w8a16
            y = w8a16.linear16(x.contiguous(), w)
            return (y, None) if getattr(self, "return_bias", True) else y
        return orig(self, x)
    cls.forward = forward
    cls._k31 = True
    _LOG("router gate: decode-size calls by w8a16.linear16 (split-K, float32 logits)")


PATCHES = {TARGET: _patch_moe, "vllm.third_party.deep_gemm.utils.mlopart": _patch_mlopart,
           "vllm.model_executor.layers.fused_moe.router.gate_linear": _patch_gate,
           "vllm.model_executor.layers.quantization.online.fp8": _patch_fp8_online,
           "vllm.models.kimi_k3.nvidia.model": _patch_model,
           "vllm.models.kimi_k3.nvidia.kda": _patch_kda, "vllm.models.kimi_k3.nvidia.mla": _patch_mla,
           "vllm.models.kimi_k3.nvidia.dsa": _patch_dsa,
           "vllm.v1.worker.gpu.sample.prompt_logprob": lambda m: __import__(
               "k31_engine.force", fromlist=["patch_prompt_kl"]).patch_prompt_kl(m),
           **({"vllm.v1.worker.gpu.spec_decode.dspark.speculator": lambda m: __import__(
               "k31_engine.drafttopp", fromlist=["patch"]).patch(m)} if DRAFT_TOPP or DRAFT_ARGMAX else {}),
           **{r: _patch_runner for r in RUNNERS}}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def wrapped(module, _exec=exec_module, _name=name):
            _exec(module)
            PATCHES[_name](module)

        spec.loader.exec_module = wrapped
        return spec


def install() -> None:
    if MODE not in ("hbm", "staged"):
        raise RuntimeError(f"K31: mode {MODE!r} is not built (hbm, staged)")
    if os.environ.get("K31_TIMING", "0") == "1":     # GPU time of every CUDA graph replay, logged every 30 s
        from . import timing
        timing.install()
    for name, fn in PATCHES.items():
        if name in sys.modules:
            fn(sys.modules[name])
    sys.meta_path.insert(0, _Finder())
