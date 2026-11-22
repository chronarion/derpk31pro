"""The prompt-chunk path of a MoE layer (eager calls of more than 16 rows: prompt chunks), K31_PROMPT_PATH=1 (default).

What a prompt row gets, against the decode path's table (today's prompt rows were served 2.3-3x
worse than the same rows decoded, because the chunk used the decode table and the residency could not move during it):
  1. Native rows: every pick with a row, as in decode.
  2. Bank (K31_PROMPT_BANK): `all` (default) stages every pick with a record in the Grace bank, any rank: the layer's
     bank is streamed once per chunk; `valve` keeps only the records with the largest summed pick amplitude in the chunk
     (at most ceil(K31_PROMPT_VALVE x the layer's records)); `t1` is the decode table (picks of rank < st.strong).
     K31_PROMPT_RANKS (default 16) limits the picks that count to rank < r.
  3. The RTX band: today's 8-row calls (hook._band_eager), or one bulk request per layer and chunk when it exists.
  4. The mean for the rest (hook's cut-mean tables).
  5. The residency ticks inside the forward, every K31_PROMPT_TICK_LAYERS MoE layers (default 4; 0 = once per step, as
     before), each tick with that share of the chunk's promotion credit, so promotions, band feed and bank refills run
     during a long prompt and later chunks see an adapted cache.
A record's place in the staging order is its shared row (place // 128 = round, place % 128 = row): no per-chunk top-k,
one host sync per layer (the number of rounds). MegaMoE runs in row slices of K31_PROMPT_SUB rows when set (0 = the
whole chunk). K31_PROMPT_AHEAD=1 stages the next half-round on its own stream while MegaMoE reads the other half.
K31_PROMPT_CELANE=<share> lands the last share of every round's (half-round's) records through the copy engines and an
HBM-side expand instead of the stager (celane.py; same rows, same bytes, same MegaMoE calls). K31_PROMPT_SPEC=1 lets
those copies run ahead of the plan: the next layer's first lane records cross the link during this layer's tail and
the next layer's attention and dense work (celane.Pool; bit-identical output). K31_PROMPT_TICK_LATE=1 runs the
residency's in-forward tick after a layer's rounds are queued instead of before its plan. K31_PROMPT_PROF=1 logs each
prompt step's phases on the GPU's clock (pprof.py).
Reference model of the plan: tests/prompt_plan_model.py; tests/test_prompt_path.py compares plan_rounds with it.
"""
import math
import os

import torch

E, K = 896, 16
SLOTS, GROUPS = 16, 8
STAGE_ROWS = SLOTS * GROUPS
from . import flags                       # every setting is read per call (runtime flags, flags.py)
from . import pprof                       # K31_PROMPT_PROF: the step's phases on the GPU's clock (off: one dict read)
_LANE_ON = [False]                        # the copy-engine lane (K31_PROMPT_CELANE) holds its scratch
N_MOE = 92
# the study's pick amplitude by rank (sensitivity/data/model2.json "a"): what a cut pick of that rank costs, relative
AMP = (0.484, 0.119, 0.097, 0.083, 0.044, 0.040, 0.037, 0.034, 0.023, 0.022, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017)
_C: dict = {}


def _consts(dev):
    if dev not in _C:
        _C[dev] = (torch.tensor(AMP, dtype=torch.float32, device=dev), torch.arange(E, dtype=torch.long, device=dev))
    return _C[dev]


def plan_rounds(topk_ids, topk_weights, row_map, bank_map, band_map, n_bank: int, strong: int,
                mode: str = None, valve: float = None, ranks: int = None, shared: int = STAGE_ROWS):
    """-> (kept [n_bank] long: bank slots in staging order, -1 past the kept ones; n_keep device scalar;
           place [T, K] long: the pick's place in the staging order or -1; nat_ids [T, K] long: native row or -1;
           band [T, K] bool: picks the band serves; cut [T, K] bool: picks nobody computes)
    Fixed shapes, no host sync. Matches prompt_plan_model.plan() (mode "t1" = ranks=strong + mode "all" there)."""
    mode = flags.get("K31_PROMPT_BANK", "all") if mode is None else mode       # t1 | all | valve
    valve = flags.num("K31_PROMPT_VALVE", 0.5) if valve is None else valve
    ranks = flags.num("K31_PROMPT_RANKS", 16) if ranks is None else ranks
    T, Kk = topk_ids.shape
    dev = topk_ids.device
    amp, ar = _consts(dev)
    valid = topk_ids >= 0
    ids = topk_ids.long().clamp_min(0)
    native = row_map[ids].long()
    bank = bank_map[ids].long()
    is_row = valid & (native >= 0)
    if band_map is not None:
        is_band = valid & ~is_row & (band_map[ids] >= 0)
    else:
        is_band = torch.zeros_like(valid)
    order_w = topk_weights.float().argsort(dim=1, descending=True, stable=True)
    rank = torch.empty_like(order_w)
    rank.scatter_(1, order_w, ar[:Kk].expand(T, Kk))
    lim = strong if mode == "t1" else ranks
    cand = valid & ~is_row & ~is_band & (bank >= 0) & (rank < lim)
    G = max(int(n_bank), 1)
    mass = torch.zeros(G, dtype=torch.float32, device=dev)
    mass.index_add_(0, bank.clamp_min(0).reshape(-1), torch.where(cand, amp[rank.clamp_max(15)], 0.0).reshape(-1))
    hits = torch.zeros(G, dtype=torch.int32, device=dev)
    hits.index_add_(0, bank.clamp_min(0).reshape(-1), cand.to(torch.int32).reshape(-1))
    hit = hits > 0
    if mode == "valve":
        # heaviest records first, ties by lower slot (stable sort of the negated mass); records without a pick last
        order = torch.argsort(torch.where(hit, -mass, torch.full_like(mass, float("inf"))), stable=True)
        cap = int(math.ceil(valve * G))
    else:
        order = torch.argsort((~hit).to(torch.uint8), stable=True)      # hit records first, in slot order
        cap = G
    if flags.on("K31_PROMPT_SPEC"):                      # the same value without a scalar copied from the host: with
        n_keep = hit.sum().clamp(max=cap)                # copies ahead in flight that copy waits behind them
    else:
        n_keep = torch.minimum(hit.sum(), torch.tensor(cap, device=dev))
    pos = ar[:G]
    kept = torch.where(pos < n_keep, order, torch.full_like(order, -1))
    place_of = torch.full((G,), -1, dtype=torch.long, device=dev)
    place_of.scatter_(0, order, torch.where(pos < n_keep, pos, torch.full_like(pos, -1)))
    place = torch.where(cand, place_of[bank.clamp_min(0)], torch.full_like(bank, -1))
    nat_ids = torch.where(is_row, native, torch.full_like(native, -1))
    cut = valid & ~is_row & ~is_band & (place < 0)
    return kept, n_keep, place, nat_ids, is_band, cut


class _PaddingSlice:
    """MegaMoE skips rows the forward context marks as padding; a row slice needs the mask's slice (as hook._Padding2
    repeats it for the deferral's pseudo rows)."""

    def __init__(self, a, b):
        self.a, self.b, self.ctx, self.p = a, b, None, None

    def __enter__(self):
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if is_forward_context_available():
            ctx = get_forward_context()
            p = getattr(ctx, "is_padding", None)
            if p is not None and p.shape[0] > self.b - self.a:
                self.ctx, self.p = ctx, p
                ctx.is_padding = p[self.a:self.b]
        return self

    def __exit__(self, *exc):
        if self.ctx is not None:
            self.ctx.is_padding = self.p


def _megamoe(orig, mod, x, w, ids, shared_experts, dtype):
    """One round's MegaMoE over the chunk, in row slices of SUB (the slices' outputs are disjoint rows)."""
    T = x.shape[0]
    SUB = flags.num("K31_PROMPT_SUB", 0)
    if SUB <= 0 or T <= SUB:
        return orig(mod, x, w, ids.to(dtype), shared_experts)
    outs = []
    for a in range(0, T, SUB):
        b = min(T, a + SUB)
        with _PaddingSlice(a, b):
            outs.append(orig(mod, x[a:b], w[a:b], ids[a:b].to(dtype).contiguous(), shared_experts))
    return torch.cat(outs)


def _stage_round(st, kept, r, shared=STAGE_ROWS):
    """Round r's records into the shared rows: one stager launch per slot group, as today (an empty list returns
    before the kernel touches host memory)."""
    lst = kept[r * shared:(r + 1) * shared]
    if lst.numel() < shared:
        lst = torch.cat([lst, torch.full((shared - lst.numel(),), -1, dtype=lst.dtype, device=lst.device)])
    lst = lst.to(torch.int32)
    for g in range(GROUPS):
        st.stagers[g](st.bank_ptr, lst[SLOTS * g:SLOTS * (g + 1)].contiguous())


def _tick(st, T, res, first_layer: int):
    """The residency's tick inside a prompt forward: every TICK_LAYERS MoE layers, each with its share of the chunk's
    promotion credit (residency.py caps a tick's credit by that tick's tokens, so the credit must come in shares)."""
    TICK_LAYERS = flags.num("K31_PROMPT_TICK_LAYERS", 4)
    if not res or TICK_LAYERS <= 0 or not res[0].accepted:
        return
    if (st.idx - first_layer) % TICK_LAYERS:
        return
    r = res[0]
    r.eager_tokens += max(1, round(T * TICK_LAYERS / N_MOE))
    if st.idx != first_layer:            # the first group's share is picked up by this tick or by the next one
        r.tick(0)
        if hasattr(r, "join_eager"):     # K31_RES_STREAM: the tick's GPU work (row-map writes, merges) is on its own
            r.join_eager()               # stream; this layer's plan must read the row map after it


def forward_prompt(mod, st, hidden_states, topk_weights, topk_ids, shared_experts, native=None, bank=None,
                   valid=None, nat_ids=None, orig=None, res=None, band_eager=None, cut_mean=None, band_bulk=None,
                   first_layer: int = 1, resv=None, next_st=None):
    """A prompt chunk (T > 16 rows, not capturing) through one MoE layer -> the routed latent sum [T, 3584].
    resv: the residency list as the hook holds it (also when `res` is None because ticks are not allowed in this
    forward), next_st: the next MoE layer's state; both only for K31_PROMPT_SPEC (celane.Pool)."""
    T, Kk = topk_ids.shape
    dev = topk_ids.device
    prof = pprof.on()
    if prof:
        pprof.mark(st.idx, 0)
    # K31_PROMPT_TICK_LATE=1: the tick's host work (10-12 ms: landed promotions merged, maps written, refills and
    # write-backs issued) runs at the end of this call, once the layer's rounds and tail are queued, while the GPU
    # and the link work through them, instead of before the plan with the main stream idle. Its GPU work follows
    # this layer's kernels in stream order, so the whole expert step reads the maps as the plan read them; the next
    # layer sees the tick's changes.
    late = flags.on("K31_PROMPT_TICK_LATE")
    if not late:
        _tick(st, T, res, first_layer)
    if prof:
        pprof.mark(st.idx, 1)
    n_bank = int(getattr(st, "bank_n", 0) or len(st.bank))            # record slots of this layer's bank
    # K31_STAGER_SMS_EAGER (a runtime flag here): thread blocks per stager launch on prompt chunks; unset or 0 = the
    # decode setting (K31_STAGER_SMS, 24). The stagers are rebuilt when it changes (they hold no GPU memory).
    sms = int(flags.num("K31_STAGER_SMS_EAGER", 0)) or int(os.environ.get("K31_STAGER_SMS", "24"))
    if st.stagers is None or getattr(st.stagers[0], "sms", sms) != sms:
        from .stager import Stager
        st.stagers = [Stager(st.mod, SLOTS * g, st.layout, sms=sms) for g in range(GROUPS)]
    kept, n_keep, place, nat, is_band, cut = plan_rounds(topk_ids, topk_weights, st.row_map, st.bank_map,
                                                         st.band_map if getattr(st, "band_ok", True) else None,
                                                         n_bank, st.strong)
    if prof and pprof.tiers_on():                                    # K31_PROMPT_PROF_TIERS: picks by serving tier
        pprof.tiers(topk_weights, nat, place, is_band, cut)
    use_band = st.band_map is not None and getattr(st, "band_ok", True)
    if use_band and band_bulk is not None:                           # the RTX starts on the chunk's band picks now
        band_bulk.send(st, hidden_states, topk_ids, topk_weights)
    share = min(1.0, flags.num("K31_PROMPT_CELANE", 0.0))            # the copy-engine lane (celane.py); 0 = off
    lane = None
    if share > 0:
        from . import celane
        lane = celane.get(st, topk_ids.device, share)                # None: its scratch did not fit (logged)
    if lane is not None:                                             # the lane addresses its copies from the host:
        kept_h = celane.host_kept(n_keep, kept)                      # the one host sync brings the kept list too
        n_kept = len(kept_h)
        _LANE_ON[0] = True
    else:
        n_kept = int(n_keep)                                         # the layer's one host sync
        if _LANE_ON[0]:                                              # the lane was switched off at runtime:
            from . import celane                                     # its scratch goes back to torch's cache
            celane.release()
            _LANE_ON[0] = False
    rounds = max(1, -(-n_kept // STAGE_ROWS))
    if prof:
        pprof.mark(st.idx, 2)
        pprof.count(rounds, n_kept)
    rnd = torch.div(place, STAGE_ROWS, rounding_mode="floor")
    pos = place - rnd * STAGE_ROWS
    y = None
    ahead = flags.on("K31_PROMPT_AHEAD") and rounds > 1
    if lane is not None:
        spec = flags.on("K31_PROMPT_SPEC") and not flags.on("K31_PROMPT_AHEAD")
        celane.set_mode(lane, spec)
        if spec:                                         # the scratch as a pool with copies ahead of the plan
            y = celane.rounds_pool(orig, mod, st, hidden_states, topk_weights, shared_experts, kept, kept_h, place,
                                   nat, n_kept, topk_ids.dtype, lane, resv=resv)
            # the next layer's first units ahead of its plan (before a late tick: the copies start as slots free
            # up; what the tick changes in the bank is caught at the claim). Only when this forward has enough rows
            # to pick most of a layer's records (K31_PROMPT_SPEC_ROWS, default 256): a short forward needs a third
            # of them, and what it does need would queue behind the copies made ahead.
            if T >= flags.num("K31_PROMPT_SPEC_ROWS", 256):
                celane.speculate(lane, next_st, resv)
        else:
            y = celane.rounds(orig, mod, st, hidden_states, topk_weights, shared_experts, kept, kept_h, place, nat,
                              n_kept, topk_ids.dtype, share, ahead=ahead, lane=lane)
    elif ahead:
        y = _rounds_ahead(orig, mod, st, hidden_states, topk_weights, shared_experts, kept, place, nat, n_kept,
                          topk_ids.dtype)
    else:
        for r in range(rounds):
            _stage_round(st, kept, r)
            ids = torch.where((place >= 0) & (rnd == r), pos, nat if r == 0 else torch.full_like(nat, -1))
            out = _megamoe(orig, mod, hidden_states, topk_weights, ids, shared_experts if r == 0 else None,
                           topk_ids.dtype)
            y = out if y is None else y + out           # out of place: a round's output may be MegaMoE's own buffer
            if prof:
                pprof.round_mark(st.idx, r)
    if prof:
        pprof.mark(st.idx, 3)
    if use_band:
        if band_bulk is not None:
            band_bulk.finish(st, y)
        elif band_eager is not None:
            band_eager(st, hidden_states, topk_ids, topk_weights, y)
    if st.cut_mean is not None and cut_mean is not None:
        served = ~cut
        cut_mean(y, torch.where(served, 0, -1).to(torch.int32), topk_ids, topk_weights, st.cut_mean,
                 band=st.band_map if use_band else None)
    if prof:
        pprof.mark(st.idx, 4)
    if late:                                             # after every kernel of this layer's expert step is queued:
        _tick(st, T, res, first_layer)                   # they all read the maps the plan read (band map, row map)
    return y


# ------------------------------------------------------------------------------------------- second step: stage ahead
_AH: dict = {"stream": None}
HALF = STAGE_ROWS // 2


def _rounds_ahead(orig, mod, st, x, w, shared_experts, kept, place, nat, n_kept, dtype):
    """The shared rows as two halves of 64 (slot groups 0-3 and 4-7). Half-round h stages records [64 h, 64 h + 64)
    into half h % 2 on the stager's own stream while MegaMoE reads the other half:
        stage(h + 1) may start when MegaMoE(h - 1) has ended (it overwrites that call's rows);
        MegaMoE(h) may start when stage(h) has ended.
    The stager streams continuously (~309 GB/s) and the GPU's other SMs compute beside it instead of waiting.
    A pick's row: place % 128 is already (half, slot) because half h holds places [64 h, 64 h + 64)."""
    main = torch.cuda.current_stream()
    if _AH["stream"] is None:
        _AH["stream"] = torch.cuda.Stream(device=main.device)
    side = _AH["stream"]
    halves = max(1, -(-n_kept // HALF))
    hr = torch.div(place, HALF, rounding_mode="floor")
    row = place - torch.div(place, STAGE_ROWS, rounding_mode="floor") * STAGE_ROWS
    moe_done = [None] * (halves + 1)
    staged = [None] * (halves + 1)

    def stage(h):
        if h >= halves:
            return
        side.wait_stream(main) if h == 0 else None
        if h >= 2 and moe_done[h - 2] is not None:
            side.wait_event(moe_done[h - 2])                 # the call that last read this half
        with torch.cuda.stream(side):
            lst = kept[h * HALF:(h + 1) * HALF]
            if lst.numel() < HALF:
                lst = torch.cat([lst, torch.full((HALF - lst.numel(),), -1, dtype=lst.dtype, device=lst.device)])
            lst = lst.to(torch.int32)
            g0 = (h % 2) * (GROUPS // 2)
            for g in range(GROUPS // 2):
                st.stagers[g0 + g](st.bank_ptr, lst[SLOTS * g:SLOTS * (g + 1)].contiguous())
            ev = torch.cuda.Event()
            ev.record(side)
        staged[h] = ev

    stage(0)
    stage(1)
    y = None
    for h in range(halves):
        main.wait_event(staged[h])
        ids = torch.where((place >= 0) & (hr == h), row, nat if h == 0 else torch.full_like(nat, -1))
        out = _megamoe(orig, mod, x, w, ids, shared_experts if h == 0 else None, dtype)
        ev = torch.cuda.Event()
        ev.record(main)
        moe_done[h] = ev
        stage(h + 2)                                         # refill the half this call has just read
        y = out if y is None else y + out
    return y
