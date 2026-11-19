"""Expert-miss policies (K31_POLICY): what the engine does with each routed pick, per MoE layer and rank.

A pick (layer L, rank r = its weight rank among the token's 16, 0 = strongest) is served by the first treatment of the
fixed menu that its policy allows and that is available for its expert:

  hit    the expert has a native HBM row (exact)
  band   the expert has a record in the RTX band: 2-bit, computed on the RTX, on time
  stage  the expert has a Grace bank record: copied into an HBM staging row, 2-bit, on time (~33 us per record on
         the critical path)
  late   the expert has a bank record: computed by the Grace pool, half a layer late, on top of the mean it gets on
         time (latetier.py; needs K31_LATE=1); `late_slots` = records per call (0-8)
  defer  the expert has a bank record: computed one layer late (the deferral window; needs K31_DEFER=1)
  mean   cut, replaced by weight x the expert's mean output (K31_CUT_MEAN tables)
  (skip) cut: always the last resort

The order is fixed (it is also the order of cost on this engine: band experts never have bank records, and an on-time
treatment always beats a late one). A policy only says which treatments each rank may use.

File (JSON):
  {
    "version": 1,
    "env": {"K31_BAND_FEED": "0.25", ...},            optional: global knobs, exported by k31.sh unless already set
    "policies": {                                      named policies; each field is optional:
      "standard": {"hit": "0-15",                       ranks that may use a native row (default all)
                   "band": "0-15", "band_cap": 16,      ranks the RTX may serve (default all) / experts per call
                   "stage": "0-2",                      ranks staged from the bank when missing (default none)
                   "late": "0-15", "late_slots": 5,     ranks the Grace pool may serve late / records per call
                   "defer": "3-11",                     ranks deferred one layer (default none)
                   "mean": "0-15"}                      ranks given the mean substitute when cut (default none)
    },
    "default": "standard",                             policy of every layer not named below
    "layers": {"12-35": "cheap", "48-59": "sensitive"},
    "overrides": {"92": {"stage": "0-3"}},             per-layer fields on top of the assigned policy
    "table": {"7": [["hit", "band", "stage", "mean"], ... 16 lists]}   per-layer, per-rank lookup table (wins)
  }
Rank sets: "0-15", "0-2,5", "none", [0, 1, 2], or a single int.

Canonical form: mask[layer][rank], a bitmask over the menu (HIT | BAND | STAGE | DEFER | MEAN). Every way of writing a
policy (named, overridden, table) compiles to it, and `compile_layer` turns one layer's 16 masks into the parameters
today's kernels take. Version 1 kernels need: hit and stage as rank prefixes (resident cap, `strong` <= 4), defer one
contiguous range, band and mean all-or-none per layer; anything else is refused with the reason. Lookup-table policies
with arbitrary masks need kernels that read the masks (unitk: hit / stage, the band plan: band, cut_mean: mean, the
deferral plan: defer); the file format and the canonical form stay as they are.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

HIT, BAND, STAGE, DEFER, MEAN, LATE = 1, 2, 4, 8, 16, 32
NAMES = {"hit": HIT, "band": BAND, "stage": STAGE, "defer": DEFER, "mean": MEAN, "late": LATE}
ORDER = ("hit", "band", "stage", "late", "defer", "mean")
LATE_MAX = 8                # records per late job (cpustage_seg2.h CPUSEG2_PMAX)
K = 16                      # picks per token
LAYERS = range(1, 93)       # MoE layers of K3.1 (model layer numbers)
STAGE_MAX = 4               # unitk considers ranks < 4 for staging (SP)
DEFAULTS = {"hit": "0-15", "band": "0-15", "stage": "none", "defer": "none", "mean": "none", "late": "none"}


class PolicyError(ValueError):
    pass


def parse_ranks(v) -> set:
    """'0-15', '0-2,5', 'none' / '' / None, [0, 1], 3 -> a set of ranks in 0..15."""
    if v is None:
        return set()
    if isinstance(v, bool):
        return set(range(K)) if v else set()
    if isinstance(v, int):
        v = [v]
    if isinstance(v, (list, tuple)):
        out = set(int(x) for x in v)
    else:
        s = str(v).strip().lower()
        if s in ("", "none", "-"):
            return set()
        if s in ("all", "*"):
            return set(range(K))
        out = set()
        for part in s.split(","):
            part = part.strip()
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if b < a:
                    raise PolicyError(f"rank range {part!r} runs backwards")
                out.update(range(a, b + 1))
            elif re.fullmatch(r"\d+", part):
                out.add(int(part))
            else:
                raise PolicyError(f"cannot read rank set {v!r}")
    bad = [r for r in out if not 0 <= r < K]
    if bad:
        raise PolicyError(f"ranks {sorted(bad)} outside 0..{K - 1}")
    return out


def parse_layers(spec: str) -> list:
    """'1-11', '12', '1-11,48-59', '*' -> model layer numbers (1..92)."""
    s = str(spec).strip()
    if s in ("*", "all"):
        return list(LAYERS)
    out = []
    for part in s.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.fullmatch(r"\d+", part):
            out.append(int(part))
        else:
            raise PolicyError(f"cannot read layer set {spec!r}")
    bad = [L for L in out if L not in LAYERS]
    if bad:
        raise PolicyError(f"layers {bad} outside {LAYERS.start}..{LAYERS.stop - 1}")
    return out


def masks_of(fields: dict) -> list:
    """A policy's fields -> 16 bitmasks (rank 0 first)."""
    unknown = set(fields) - set(NAMES) - {"band_cap", "note", "late_slots"}
    if unknown:
        raise PolicyError(f"unknown policy fields {sorted(unknown)} (menu: {', '.join(ORDER)}, band_cap, late_slots)")
    m = [0] * K
    for name in ORDER:
        for r in parse_ranks(fields.get(name, DEFAULTS[name])):
            m[r] |= NAMES[name]
    return m


@dataclass
class LayerParams:
    """One layer's compiled policy: what the v1 kernels take."""
    resmax: int = K             # resident picks of rank >= resmax are cut
    band: bool = True           # the RTX band serves this layer's band experts
    band_cap: int = 16          # at most this many distinct band experts per call
    strong: int = 0             # misses of rank < strong with a bank record are staged
    defer: tuple | None = None  # (from, to): misses of these ranks with a bank record are deferred
    mean: bool = False          # cut picks get the mean substitute
    late: tuple | None = None   # (from, to): bank misses of these ranks may be computed late by the Grace pool
    late_n: int = 0             # records per late job (the pool's job size for this layer)
    policy: str = ""
    masks: list = field(default_factory=lambda: [0] * K)


def _prefix(ranks: set, what: str, layer: int) -> int:
    n = len(ranks)
    if ranks != set(range(n)):
        raise PolicyError(f"layer {layer}: '{what}' must be a rank prefix 0..n-1 on the v1 kernels, got {sorted(ranks)}")
    return n


def compile_layer(layer: int, masks: list, band_cap: int = 16, name: str = "", late_slots=None) -> LayerParams:
    """16 bitmasks -> LayerParams, or PolicyError if the v1 kernels cannot express them."""
    sets = {n: {r for r in range(K) if masks[r] & bit} for n, bit in NAMES.items()}
    p = LayerParams(policy=name, masks=list(masks), band_cap=int(band_cap))
    p.resmax = _prefix(sets["hit"], "hit", layer)
    if sets["band"] and sets["band"] != set(range(K)):
        raise PolicyError(f"layer {layer}: 'band' must be all ranks or none on the v1 kernels, got {sorted(sets['band'])}")
    p.band = bool(sets["band"])
    if not 1 <= p.band_cap <= 16:
        raise PolicyError(f"layer {layer}: band_cap {band_cap} outside 1..16")
    p.strong = _prefix(sets["stage"], "stage", layer)
    if p.strong > STAGE_MAX:
        raise PolicyError(f"layer {layer}: 'stage' reaches rank {p.strong - 1}; unitk stages ranks < {STAGE_MAX}")
    d = sorted(sets["defer"])
    if d:
        if d != list(range(d[0], d[-1] + 1)):
            raise PolicyError(f"layer {layer}: 'defer' must be one contiguous rank range on the v1 kernels, got {d}")
        lo = max(d[0], p.strong)                    # a staged rank is on time: it never also defers
        p.defer = (lo, d[-1] + 1) if lo <= d[-1] else None
    if sets["mean"] and sets["mean"] != set(range(K)):
        raise PolicyError(f"layer {layer}: 'mean' must be all ranks or none on the v1 kernels, got {sorted(sets['mean'])}")
    p.mean = bool(sets["mean"])
    lr = sorted(sets["late"])
    if lr:
        if lr != list(range(lr[0], lr[-1] + 1)):
            raise PolicyError(f"layer {layer}: 'late' must be one contiguous rank range, got {lr}")
        if p.defer is not None:
            raise PolicyError(f"layer {layer}: 'late' and 'defer' exclude each other (K31_LATE / K31_DEFER)")
        n = int(late_slots if late_slots is not None else os.environ.get("K31_LATE_SLOTS", "5") or 5)
        if not 0 <= n <= LATE_MAX:
            raise PolicyError(f"layer {layer}: late_slots {n} outside 0..{LATE_MAX}")
        p.late, p.late_n = (lr[0], lr[-1] + 1), n
    return p


def load(path: str) -> dict:
    """K31_POLICY file -> {layer: LayerParams} for layers 1..92."""
    with open(path) as f:
        cfg = json.load(f)
    if int(cfg.get("version", 1)) != 1:
        raise PolicyError(f"{path}: version {cfg.get('version')} (this engine reads 1)")
    pols = cfg.get("policies", {})
    for nm, fl in pols.items():
        masks_of(fl)                                 # validate every named policy once, even unused ones
    default = cfg.get("default")
    if default is not None and default not in pols:
        raise PolicyError(f"default policy {default!r} is not defined")
    assign = {L: default for L in LAYERS}
    for spec, nm in cfg.get("layers", {}).items():
        if nm not in pols:
            raise PolicyError(f"layers {spec!r}: policy {nm!r} is not defined")
        for L in parse_layers(spec):
            assign[L] = nm
    over = {}
    for spec, fl in cfg.get("overrides", {}).items():
        for L in parse_layers(spec):
            over.setdefault(L, {}).update(fl)
    table = {}
    for spec, rows in cfg.get("table", {}).items():
        if not isinstance(rows, list) or len(rows) != K:
            raise PolicyError(f"table {spec!r}: needs {K} entries (rank 0 first)")
        m = []
        for r, row in enumerate(rows):
            bits = 0
            for t in (row if isinstance(row, list) else [row]):
                t = str(t).lower()
                if t == "skip":
                    continue
                if t not in NAMES:
                    raise PolicyError(f"table {spec!r} rank {r}: unknown treatment {t!r}")
                bits |= NAMES[t]
            m.append(bits)
        for L in parse_layers(spec):
            table[L] = m
    out = {}
    for L in LAYERS:
        nm = assign[L]
        fields = dict(pols[nm]) if nm is not None else {}
        fields.update(over.get(L, {}))
        cap = int(fields.get("band_cap", os.environ.get("K31_BAND_MAXE", "16") or 16))
        masks = table.get(L) if L in table else masks_of({k: v for k, v in fields.items()
                                                          if k not in ("band_cap", "late_slots")})
        label = ("table" if L in table else (nm or "(defaults)")) + ("+override" if L in over else "")
        out[L] = compile_layer(L, masks, cap, label, fields.get("late_slots"))
    return out


def env_section(path: str) -> dict:
    """The file's optional "env" object (global knobs), as strings."""
    with open(path) as f:
        cfg = json.load(f)
    return {str(k): str(v) for k, v in cfg.get("env", {}).items()}


def describe(params: dict) -> str:
    """One line per run of consecutive layers with the same compiled parameters."""
    def key(p):
        return (p.policy, p.resmax, p.band, p.band_cap, p.strong, p.defer, p.mean, p.late, p.late_n)
    lines, run = [], None
    for L in LAYERS:
        p = params[L]
        if run and key(params[run[1]]) == key(p) and run[1] == L - 1:
            run[1] = L
        else:
            if run:
                lines.append(run)
            run = [L, L]
    lines.append(run)
    out = []
    for a, b in lines:
        p = params[a]
        out.append(f"layers {a}-{b} [{p.policy}]: hit < {p.resmax}, band {'on' if p.band else 'off'} (cap {p.band_cap}), "
                   f"stage < {p.strong}, defer {p.defer or '-'}, mean {'on' if p.mean else 'off'}"
                   + (f", late {p.late[0]}..{p.late[1] - 1} x{p.late_n}" if p.late else ""))
    return "\n".join(out)


if __name__ == "__main__":                            # python3 -m k31_engine.policy FILE: check and print a policy
    import sys
    try:
        prm = load(sys.argv[1])
    except PolicyError as e:
        print(f"POLICY ERROR: {e}")
        sys.exit(1)
    print(describe(prm))
