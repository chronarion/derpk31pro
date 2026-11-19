"""Universes of the runner and their expert policies: which version of each routed pick a universe computes.

A pick is one token's choice of one expert; its rank is its position among the token's 16 picks by routing weight
(0 = strongest). Treatments: NATIVE (the official MXFP4 expert), PLANES (the expert at `depth` bit planes, see
formats.py), CUT (the pick is dropped), DEFER (computed exactly, but its contribution reaches the residual stream
`late` layers later: derpseek's `deferred-fast`, "a missed expert one sublayer late instead of cutting it"), WDEFER
(the wavefront deferred: computed at `depth` planes now, and the correction (official minus `depth`-plane output)
arrives `late` layers later, when the refinement planes would have arrived).

Expert policies:
  native   every pick NATIVE
  planes   every pick PLANES
  hotcut   a serving policy simulated causally inside each window: per layer a hot set that follows
           the text (the `hot` experts with the largest decaying sum of squared routing weights over the window's
           earlier tokens, half-life `halflife` tokens; ties, such as experts not used yet at a window's start, go to
           the lower expert id, so a window starts from an arbitrary set), then a warm band (the next `warm`). A pick
           of a hot expert is NATIVE, of a warm expert `warm_mode`. Any other pick is a miss: `strong_mode` if its
           rank < `strong`, otherwise `miss`. Modes: "planes" (the base: derpseek's wavefront), "wdefer" (the base now,
           the correction `late` layers later), and for weak misses also "cut" and "defer". Promotions are free here
           (an upper bound on what a swap budget allows). `mid` > `strong` adds a middle band of ranks
           (strong <= rank < mid) with its own `mid_mode`.
Modifiers for every policy (derpseek's lossy profiles):
  drop     the `drop` weakest ranks are always CUT, resident or not (derpseek `lossy-fast`'s singleton drop-2)
  cut_max  at most this many CUT misses per token, the weakest first; misses past the cap are read at PLANES
           (derpseek's cut cap; drop picks do not count against it)
  renorm   rescale the surviving picks of a token to the original weight sum after cuts
  cover    (hotcut) only the top `cover` experts by score are in memory; misses outside are cut (
           the GB300 alone holds ~630 of 896 per layer, ~740 with the RTX)
  link_max (hotcut) at most this many link reads per token, strongest first; the rest are cut (the lane's budget)
"""
import json
import math
import os
from dataclasses import asdict, dataclass

import torch

NATIVE, PLANES, CUT, DEFER, WDEFER, PDEFER = 0, 1, 2, 3, 4, 5        # PDEFER: the 2-bit output, landing `late` later
NTREAT = 6
MODES = {"planes": PLANES, "cut": CUT, "defer": DEFER, "wdefer": WDEFER, "pdefer": PDEFER}


@dataclass
class Universe:
    name: str = "native"
    experts: str = "native"     # native | planes | hotcut | static (a fixed set per layer from `rank`)
    depth: int = 2              # bit planes for PLANES picks
    recon: str = "centroid"     # centroid | onesfill | trick (the engine's stager: ones-fill, power-of-two gain)
    act: str = "bf16"           # routed experts' input activations: bf16 | mxfp8
    dense: str = "bf16"         # non-expert matrices (not the router): bf16 | mxfp8 | fp6 | fp8b128 (vLLM fp8_per_block) | fp8b128w (its weights, BF16 inputs) | int8g128 (W8A16) | f32 | bf16noise (noise-floor tests)
    hot: int = 125
    warm: int = 0
    strong: int = 5
    halflife: float = 128.0
    miss: str = "cut"           # a weak miss: cut | defer | planes | wdefer
    strong_mode: str = "planes"  # a strong miss: planes | wdefer
    warm_mode: str = "planes"    # a warm-band pick: planes | wdefer
    late: int = 1               # layers until a deferred contribution or correction arrives
    land: str = "layer"         # where it lands: "layer" = after the arrival layer; "mlp" = between the arrival
                                # layer's attention and its MoE (late=1: one sublayer late, derpseek's deferred-fast)
    cover: int = 0              # > 0: only the top `cover` experts by score are in memory (HBM + Grace); a miss on
                                # any other expert is CUT whatever its rank (it is only on NVMe). 0 = all in memory
    link_max: int = 16          # at most this many misses per token read over the link, strongest first; the rest
                                # are CUT (the lane's budget)
    rank: str = ""              # experts "static": JSON {"layers": {"<layer>": [expert, ...] most used first}}; the
                                # first `hot` of a layer are its fixed set (engine v0-hbm's rowmap)
    promote: float = -1         # hotcut: < 0 = the hot set is the ideal top-`hot` at every token (instant
                                # promotion); >= 0 = at most this many swaps per layer per token (a credit), starting
                                # from the static set of `rank` (the engine loads experts at a limited rate); `cover`
                                # then counts static places (the bank is a static set)
    delay: int = 0              # budgeted promotion: tokens between a swap's decision and its first use
    dense_keep: str = ""        # with `dense`: comma-separated substrings of module names kept in BF16; with dense
                                # "map": ordered "substring:fmt" pairs, "*" for the rest (stream._dense_map_copy)
    reference: bool = False     # universe 0 only: use it as the reference instead of BF16 native (e.g. K3.1's official
                                # W4A8 arithmetic: MXFP4 experts on MXFP8 inputs, model card "quantization-aware")
    mid: int = 0                # misses with strong <= rank < mid use `mid_mode` (a middle band of ranks)
    mid_mode: str = "defer"
    drop: int = 0
    cut_max: int = 16
    renorm: bool = False
    # deferral map (layer reordering sensitivity): with experts "native", picks of rank >= defer_from are DEFERred
    # (computed on this layer's input, landing `late` layers later at `land`) and picks of rank >= skip_from are CUT,
    # only in the MoE layers listed in `layers` ("a-b,c,d-e"; empty = every MoE layer)
    defer_from: int = -1
    skip_from: int = -1
    layers: str = ""
    # policy ladder (hotcut / static): per token, the misses that are in memory (`cover`), strongest first: the first
    # `ontime_max` are 2-bit on time, the next `defer_max` arrive late as `defer_mode` (pdefer = 2-bit, defer =
    # native), the rest are cut (-1 = off)
    ontime_max: int = -1
    defer_max: int = -1
    defer_mode: str = "pdefer"
    cut_sub: str = ""           # "mean": a cut pick adds its routing weight x its expert's mean output (a table from
                                # a pass run with --save-means, loaded with --means) instead of nothing
    # HBM tier design (11-17): per-layer hot counts (JSON {"<layer>": H}, else `hot`), the hot set's format (native
    # rows or 2-bit resident records), and an adaptive warm band under budgeted promotion (a second budgeted set of
    # hot + warm experts with its own promotion rate; the band = that set minus the hot set; `warm_mode` treats it)
    hot_layers: str = ""
    hot_mode: str = "native"
    warm_promote: float = -1.0
    # adaptive bank: with budgeted promotion and `cover` > 0, the experts with a record in memory are a
    # budgeted set of `cover` that starts as the static top-`cover` and swaps at most `cover_promote` per layer-token
    # (NVMe refills; usable after `delay` tokens), scored as the hot set; < 0 = the static set. `cover_layers`: JSON
    # {"<layer>": cover} per layer
    cover_promote: float = -1.0
    cover_layers: str = ""
    cover_excl: bool = False    # the bank's `cover` records are for experts outside the hot set (a set of hot + cover;
                                # an evicted expert's 2-bit record is written back from its HBM row) instead of
                                # including the resident ones as today

    def describe(self) -> str:
        if self.experts == "native":
            e = "native"
        elif self.experts == "planes":
            e = f"planes{self.depth}-{self.recon}"
        else:
            e = (f"hot{self.hot}+warm{self.warm} ({self.warm_mode}), misses rank<{self.strong} {self.strong_mode}, "
                 f"weaker {self.miss} (planes{self.depth}-{self.recon}), "
                 + (f"static set from {os.path.basename(self.rank)}" if self.experts == "static" else f"hl{self.halflife:g}"))
            if self.mid > self.strong:
                e += f", ranks {self.strong}-{self.mid - 1} {self.mid_mode}"
            if "defer" in (self.miss + self.strong_mode + self.warm_mode + (self.mid_mode if self.mid else "")
                           + (self.defer_mode if self.defer_max > 0 else "")):
                e += f", late by {self.late}" + (" (lands before the MoE)" if self.land == "mlp" else "")
        mods = []
        if self.cover:
            mods.append(f"{self.cover} experts in memory" + (f" per layer from {os.path.basename(self.cover_layers)}"
                                                              if self.cover_layers else "")
                        + (" besides the hot set" if self.cover_excl else "")
                        + (f", adaptive bank ({self.cover_promote:g} refills per layer-token)"
                           if self.cover_promote >= 0 and self.promote >= 0 else ""))
        if self.promote >= 0:
            mods.append(f"at most {self.promote:g} promotions per layer-token from the static set"
                        + (f", usable after {self.delay}" if self.delay else ""))
        if self.link_max < 16:
            mods.append(f"at most {self.link_max} link reads per token")
        if self.drop:
            mods.append(f"drop the {self.drop} weakest")
        if self.cut_max < 16:
            mods.append(f"at most {self.cut_max} cuts per token")
        if self.renorm:
            mods.append("renorm")
        if self.cut_sub:
            mods.append(f"cut picks -> {self.cut_sub} output")
        if self.hot_layers:
            mods.append(f"hot counts per layer from {os.path.basename(self.hot_layers)}")
        if self.hot_mode != "native":
            mods.append(f"hot set as {self.hot_mode}")
        if self.warm and self.promote >= 0:
            mods.append(f"adaptive warm band {self.warm} ({self.warm_mode}, promotion "
                        f"{self.promote if self.warm_promote < 0 else self.warm_promote:g})")
        if self.ontime_max >= 0 or self.defer_max >= 0:
            mods.append(f"per token {max(self.ontime_max, 0)} on time (2-bit), {max(self.defer_max, 0)} late by "
                        f"{self.late} ({self.defer_mode}), the rest cut")
        if self.defer_from >= 0:
            mods.append(f"ranks >= {self.defer_from} deferred by {self.late}" + (" (land before the MoE)" if self.land == "mlp" else ""))
        if self.skip_from >= 0:
            mods.append(f"ranks >= {self.skip_from} skipped")
        if self.layers and (self.defer_from >= 0 or self.skip_from >= 0):
            mods.append(f"in layers {self.layers}")
        keep = f" (BF16: {self.dense_keep})" if self.dense_keep else ""
        return f"{self.name}: experts {e}{', ' + ', '.join(mods) if mods else ''}; act {self.act}, dense {self.dense}{keep}"


def load_universes(path: str | None) -> list[Universe]:
    """A JSON list of universe objects; the first must be the reference (native, bf16, bf16, nothing dropped) and is
    added if absent."""
    us = [Universe(**u) for u in json.load(open(path))] if path else []
    if not us or not (us[0].reference or (us[0].experts, us[0].act, us[0].dense, us[0].drop) == ("native", "bf16", "bf16", 0)):
        us.insert(0, Universe())
    assert not any(u.reference for u in us[1:]), "only universe 0 can be the reference"
    names = [u.name for u in us]
    assert len(set(names)) == len(names), f"universe names must be unique: {names}"
    for u in us:
        assert u.miss in MODES and u.mid_mode in MODES, u
        assert u.strong_mode in ("planes", "wdefer") and u.warm_mode in ("planes", "wdefer"), u
        assert u.experts in ("native", "planes", "hotcut", "static") and u.late >= 1 and u.land in ("layer", "mlp"), u
        assert u.experts != "static" or u.rank, f"{u.name}: static experts need a rank file"
        assert u.promote < 0 or (u.experts == "hotcut" and u.rank), \
            f"{u.name}: budgeted promotion is for hotcut with a rank file (its start set)"
        assert u.hot_mode in ("native", "planes"), f"{u.name}: hot_mode is native or planes"
        assert u.cover_promote < 0 or (u.promote >= 0 and (u.cover > 0 or u.cover_layers)), \
            f"{u.name}: an adaptive bank (cover_promote) needs budgeted promotion and a cover"
    return us


def hot_scores(ids: torch.Tensor, w: torch.Tensor, num_experts: int, halflife: float) -> torch.Tensor:
    """ids, w [B, T, K] -> score [B, T, E] float64: for token t, sum over u < t of lam^(t-1-u) * w_u,e^2."""
    B, T, _ = ids.shape
    lam = 2.0 ** (-1.0 / halflife)
    x = torch.zeros(B, T, num_experts, dtype=torch.float64, device=ids.device)
    x.scatter_add_(2, ids.long(), w.double() ** 2)
    t = torch.arange(T, dtype=torch.float64, device=ids.device)
    cum = (x * (lam ** -t)[None, :, None]).cumsum(1)          # sum_{u<=t} lam^-u x_u (<= 2^(T/halflife): fine)
    score = torch.zeros_like(cum)
    score[:, 1:] = cum[:, :-1] * (lam ** (t[1:] - 1))[None, :, None]
    return score


_RANKS: dict = {}


def static_place(path: str, layer: int, num_experts: int, device) -> torch.Tensor:
    """[num_experts] long: each expert's place in the layer's usage ranking (0 = most used)."""
    if path not in _RANKS:
        raw = json.load(open(path))
        _RANKS[path] = {int(k): v for k, v in raw.get("layers", raw).items()}
    order = torch.tensor(_RANKS[path][layer], dtype=torch.long)
    if sorted(order.tolist()) != list(range(num_experts)):
        raise ValueError(f"{path} layer {layer} is not a ranking of {num_experts} experts")
    place = torch.empty(num_experts, dtype=torch.long)
    place[order] = torch.arange(num_experts)
    return place.to(device)


def budgeted_hot(ids, w, num_experts, H, P, delay, halflife, start_place) -> torch.Tensor:
    """ids, w [B, T, K] -> hot [B, T, K]: whether each pick's expert is in a set of H that starts as the static top-H
    (start_place < H) and changes by at most P swaps per token (fractional P: a credit), each swap taking the
    strongest-scored expert of the ideal top-H that is outside for the weakest-scored member, if it scores more.
    Scores as hot_scores (decayed squared weights of earlier tokens); a swap decided before token t is usable from
    token t + delay. Sequential over tokens, vectorized over the batch (runner/tools/churn.py simulates the same)."""
    B, T, K = ids.shape
    dev = ids.device
    E = num_experts
    if dev.type == "cuda" and delay <= 8 and os.environ.get("K31_BHOT_TORCH", "0") != "1":
        from ._bhot import budgeted_hot_triton           # one launch per call, bit-equal to the loop below
        return budgeted_hot_triton(ids, w, num_experts, H, P, delay, halflife, start_place)
    lam = 2.0 ** (-1.0 / halflife)
    in_set = (start_place < H)[None, :].expand(B, E).clone()
    usable = in_set.clone()
    score = torch.zeros((B, E), dtype=torch.float64, device=dev)
    credit = torch.zeros(B, dtype=torch.float64, device=dev)
    pmax = max(1, math.ceil(P)) if P > 0 else 0
    j = torch.arange(max(pmax, 1), device=dev)
    queue = []
    hot = torch.zeros((B, T, K), dtype=torch.bool, device=dev)
    ids = ids.long()
    for t in range(T):
        if pmax:
            credit += P
            n = credit.floor().clamp(max=pmax)
            credit -= n
            ideal = torch.zeros_like(in_set)
            ideal.scatter_(1, score.topk(H, dim=1).indices, True)
            cv, ci = torch.where(ideal & ~in_set, score, -1.0).topk(pmax, dim=1)
            mv, mi = torch.where(in_set, score, float("inf")).topk(pmax, dim=1, largest=False)
            do = (cv > mv) & (cv >= 0) & (j[None, :pmax] < n[:, None])
            in_set.scatter_(1, ci, in_set.gather(1, ci) | do)
            in_set.scatter_(1, mi, in_set.gather(1, mi) & ~do)
            if delay:
                queue.append(in_set.clone())
                if len(queue) > delay:
                    usable = queue.pop(0)
            else:
                usable = in_set
        hot[:, t] = usable.gather(1, ids[:, t])
        score *= lam
        score.scatter_add_(1, ids[:, t], w[:, t].double() ** 2)
    return hot


def _in_layers(spec: str, layer) -> bool:
    if not spec:
        return True
    if layer is None:
        raise ValueError("a layer-restricted universe needs the layer index")
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        if int(lo) <= layer <= int(hi or lo):
            return True
    return False


_HOTL: dict = {}


def _per_layer(path: str, layer, default: int) -> int:
    if not path or layer is None:
        return default
    if path not in _HOTL:
        _HOTL[path] = {int(k): int(v) for k, v in json.load(open(path)).items()}
    return _HOTL[path].get(int(layer), default)


def hot_count(u: "Universe", layer) -> int:
    return _per_layer(u.hot_layers, layer, u.hot)


def cover_count(u: "Universe", layer) -> int:
    return _per_layer(u.cover_layers, layer, u.cover)


def treatments(u: Universe, ids: torch.Tensor, w: torch.Tensor, num_experts: int, layer: int = None) -> torch.Tensor:
    """ids, w [B, T, K] sorted by weight (descending) -> uint8 [B, T, K] of NATIVE / PLANES / CUT / DEFER."""
    B, T, K = ids.shape
    dev = ids.device
    rank = torch.arange(K, device=dev).view(1, 1, K).expand(B, T, K)
    if u.experts == "native":
        out = torch.full((B, T, K), NATIVE, dtype=torch.uint8, device=dev)
        if (u.defer_from >= 0 or u.skip_from >= 0) and _in_layers(u.layers, layer):
            if u.defer_from >= 0:
                out[rank >= u.defer_from] = DEFER
            if u.skip_from >= 0:
                out[rank >= u.skip_from] = CUT
    elif u.experts == "planes":
        out = torch.full((B, T, K), PLANES, dtype=torch.uint8, device=dev)
    else:
        budget = u.experts == "hotcut" and u.promote >= 0
        if u.experts == "static" or budget:                                  # a fixed set: the layer's usage ranking
            if layer is None:
                raise ValueError("static experts and budgeted promotion need the layer index")
            place = static_place(u.rank, layer, num_experts, dev)
            pick = place[ids.long()]
        else:
            score = hot_scores(ids, w, num_experts, u.halflife)
            order = score.argsort(dim=2, descending=True, stable=True)      # ties (no score yet): lower ids first
            place = torch.empty_like(order)
            place.scatter_(2, order, torch.arange(num_experts, device=dev).expand_as(order))
            pick = place.gather(2, ids.long())                               # each pick's expert's place in the set
        H = hot_count(u, layer)
        if budget:
            hot = budgeted_hot(ids, w, num_experts, H, u.promote, u.delay, u.halflife, place)
            if u.warm > 0:                   # the adaptive warm band: a budgeted set of H + warm, minus the hot set
                pw = u.promote if u.warm_promote < 0 else u.warm_promote
                warm = budgeted_hot(ids, w, num_experts, H + u.warm, pw, u.delay, u.halflife, place) & ~hot
            else:
                warm = torch.zeros_like(hot)
        else:
            hot = pick < H
            warm = (pick < H + u.warm) & ~hot
        out = torch.full((B, T, K), MODES[u.miss], dtype=torch.uint8, device=dev)
        if u.mid > u.strong:
            out[(rank >= u.strong) & (rank < u.mid)] = MODES[u.mid_mode]
        out[rank < u.strong] = MODES[u.strong_mode]
        out[warm] = MODES[u.warm_mode]
        out[hot] = PLANES if u.hot_mode == "planes" else NATIVE
        if u.cut_max < K:                    # cap the cut misses per token: the weakest stay cut, the rest PLANES
            cand = out == CUT
            from_weak = cand.flip(-1).cumsum(-1).flip(-1)                    # candidates at this rank or weaker
            out[cand & (from_weak > u.cut_max)] = PLANES
        miss = ~hot & ~warm
        C = cover_count(u, layer)
        CS = C + H if (C and u.cover_excl) else C
        if C and budget and u.cover_promote >= 0:    # the adaptive bank: a budgeted set of C refilled from NVMe
            banked = budgeted_hot(ids, w, num_experts, CS, u.cover_promote, u.delay, u.halflife, place)
        else:
            banked = pick < CS if C else None
        if banked is not None:               # not in memory at all: nothing to read, whatever the rank
            out[miss & ~banked] = CUT
        if u.ontime_max >= 0 or u.defer_max >= 0:     # the window's budget per token, strongest first
            cand = miss & banked if banked is not None else miss.clone()   # every miss with a record in memory
            cum = cand.cumsum(-1)
            on_n, df_n = max(u.ontime_max, 0), max(u.defer_max, 0)
            out[cand & (cum <= on_n)] = PLANES
            out[cand & (cum > on_n) & (cum <= on_n + df_n)] = MODES[u.defer_mode]
            out[cand & (cum > on_n + df_n)] = CUT
        if u.link_max < K:                   # the lane's budget: the strongest link reads survive
            link = miss & (out != CUT)
            out[link & (link.cumsum(-1) > u.link_max)] = CUT
    if u.drop:
        out[rank >= K - u.drop] = CUT
    return out


def as_dicts(us: list[Universe]) -> list[dict]:
    return [asdict(u) for u in us]
