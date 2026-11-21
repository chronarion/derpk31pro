# Expert-miss policies (`K31_POLICY`)

One file says what the engine does with every routed pick it cannot serve from a native HBM row, layer by layer and
rank by rank, and can carry the global knobs (tier sizes, residency rates) that go with it. Code:
`k31_engine/policy.py` (parser, compiler, checker), applied at bind by `hook._apply_policy`. Check a file without a
GPU:

    python3 -m k31_engine.policy /engine/profiles/policy-study.json

## The treatments

A pick is (layer L, rank r), r = its place among the token's 16 picks by routing weight (0 = strongest). It is served
by the first treatment of this fixed menu that its policy allows **and** that is available for its expert:

| treatment | when available | what it costs (11-18 measurements) |
|---|---|---|
| `hit` | the expert has a native HBM row | exact; 2.6 us per expert inside MegaMoE |
| `band` | the expert has a record in the RTX band | 2-bit on time: 0.07-0.15 of a skip in KL; ~0 GB300 time while the RTX keeps up |
| `stage` | the expert has a Grace bank record | 2-bit on time; ~33 us per record on the critical path (C2C) |
| `late` | the expert has a bank record and the late tier is on (`K31_LATE=1`) | 2-bit, computed by the Grace pool, half a layer late on top of the mean it got on time: ~0.14 of a skip for rank 0, 0.2-0.3 for ranks 4-15 [model]; no GB300 time per record, one up GEMV per layer beside the next attention; `late_slots` = records per call (0-8) |
| `defer` | the expert has a bank record and deferral is on | exact one layer late: 0.16-0.37 of a skip; completion kernels every layer |
| `mean` | the layer has a mean table (`K31_CUT_MEAN`) | cut, plus weight x the expert's mean output: ~0.5 of a skip for weak picks, nothing for strong |
| skip | always | cut |

The order is fixed: band experts never have bank records (the band and the bank are disjoint), and an on-time
treatment always beats a late one. A policy only decides which ranks may use each treatment.

Two switches change what a treatment covers, not which ranks may use it:
- `K31_STAGE_ALL=1`: the `stage` ranks decide which experts a call stages; once staged, every pick of that expert in
  the call uses its record, at any rank and in any row (off: only the picks of the `stage` ranks). No extra records,
  no extra MegaMoE experts. Not with `K31_DEFER=1`.
- `K31_STAGE_CAP=n` (or a policy's `stage_cap`, per layer): a decode call stages at most n records, the candidates
  with the largest routing weight first, so the staging time per call is bounded whatever ranks `stage` names.
- `K31_STAGE_ROWS=c0,c1,...`: row p of a request's verify block stages its ranks < c_p only.
- `K31_STAGE_PRIOR=p0,p1,...`: the staged experts are ordered by weight x the prior chance of their row being
  accepted (as `K31_BAND_PRIOR`), which decides who keeps a slot under `stage_cap`.
- `K31_DECODE_SPLIT=1` (the profiles' default): a decode step that carries several requests runs each MoE layer as
  one call per request's verify block, so the per-call limits above and below (the band's 8 rows and experts per
  call, the 16 staging slots, `stage_cap`, the row caps) apply per request, not per step, and the residency scores
  come from accepted rows at any number of requests. Without it a step of 9-16 rows has no band and one call's
  slots, and a step of more than 16 rows takes a one-round fallback (no band, no row caps, scores from every row).
- `K31_DECODE_JOINT=1|2` (off by default; with `K31_DECODE_SPLIT=1`): the requests of a decode step share the work
  that does not depend on the request, and every request keeps its own plan, staged records and band call (the same
  picks are computed, staged, sent to the band or given the mean as with the split alone). 1: the dense parts of a
  MoE layer (router and top-k, latent down-projection, shared experts, norm and up-projection) run once per chunk
  of whole verify blocks (`K31_DECODE_JOINT_DENSE` rows at most, default 16). 2: also one MegaMoE call per layer for
  all requests: request j's records go to slot group (layer + j) mod 8, the band calls are sent ahead and finished
  after the call, and a resident expert that several requests pick is read once. `K31_DECODE_JOINT_CHECK=n` runs
  the first n start-up batches of every layer through the split and both levels and logs whether the outputs are
  bit-equal.
- `K31_BAND_PRIOR=p0,p1,...`: a call with more band experts than `band_cap` keeps those with the largest weight x
  their row's prior chance of being accepted (`p_i` for row i of a request's verify block: row 0 always, draft row i
  when drafts 1..i are accepted; e.g. the measured acceptance `1,0.64,0.36,0.19,0.09` at 4 drafts) instead of the
  largest weight. Decode calls in CUDA graphs only; calls under the cap are unchanged.

## The file

```json
{
  "version": 1,
  "env": {"K31_BAND_FEED": "0.25", "BAND_READAHEAD": "8"},
  "policies": {
    "cheap":     {"hit": "0-15", "band": "0-15", "stage": "0",   "mean": "0-15"},
    "sensitive": {"hit": "0-15", "band": "0-15", "stage": "0-2", "mean": "0-15", "band_cap": 12},
    "tail":      {"stage": "0-3", "mean": "all"}
  },
  "default": "cheap",
  "layers": {"48-59": "sensitive", "72-83": "sensitive", "90-92": "tail"},
  "overrides": {"92": {"band_cap": 16}},
  "table": {"7": [["hit", "band", "stage", "mean"], ["hit", "band", "mean"], ...16 entries...]}
}
```

- **Fields of a policy**: `hit`, `band`, `stage`, `late`, `defer`, `mean` take rank sets (`"0-15"`, `"0-2,5"`,
  `"none"`, `"all"`, `[0, 1]`, an int); `band_cap` is the most distinct band experts per call (1-16; default
  `K31_BAND_MAXE`); `late_slots` the records per late job (0-8; default `K31_LATE_SLOTS`); `stage_cap` the most
  records staged per decode call (1-16; default `K31_STAGE_CAP`, else 16). Defaults: `hit` and `band`
  all ranks, `stage`, `late`, `defer` and `mean` none.
- **`layers`** assigns policies to model layers (`"1-11"`, `"12"`, `"1-11,48-59"`, `"*"`); unnamed layers take
  `default`.
- **`overrides`** change single fields of the assigned policy for some layers.
- **`table`** (a lookup-table policy) gives a layer its 16 per-rank treatment lists directly; it wins over the rest.
- **`env`** holds global knobs; `k31.sh up` exports each one unless it is already set (an arm's own variables win;
  the profile's defaults fill what neither gives). Knobs that belong here: `K31_HBM`, `K31_BAND_N`, `K31_BANK`,
  `K31_PROMOTE`, `K31_BANK_PROMOTE`, `K31_BAND_FEED`, `K31_HALFLIFE`, `K31_ACCEPTED`, `K31_CUT_MEAN`,
  `K31_BAND_MAXE`, `K31_STAGE_ALL`, `K31_BAND_PRIOR`, `BAND_READAHEAD`. Speed plumbing (dense formats, side streams,
  the MegaMoE fork, PDL) stays outside.

`K31_POLICY` takes precedence over `K31_STRONG`, `K31_LAYER_BANDS`, `K31_RES_MAXRANK(_MAP)` and `K31_DEFER_TO` for the
values it sets. A policy that needs a switch that is off (a `defer` range without `K31_DEFER=1`, `mean` without
`K31_CUT_MEAN`) stops the bind with the reason.

## How a file runs

Every form compiles to one canonical table: `mask[layer][rank]`, a bitmask over the menu. `compile_layer` turns a
layer's 16 masks into what the kernels take today:

| kernel parameter | from the masks | v1 constraint |
|---|---|---|
| resident cap `resmax` (unitk) | ranks allowed `hit` | a prefix 0..n-1 |
| `band_ok`, `band_cap` (band send, residency feed) | ranks allowed `band` | all or none |
| `strong` (unitk staging) | ranks allowed `stage` | a prefix, at most ranks 0-3 |
| `late`, `late_n` (late plan, `latetier.py`) | ranks allowed `late`, `late_slots` | one contiguous range; not with `defer` |
| `defer_from`, `defer_to` (deferral plan) | ranks allowed `defer` | one contiguous range; staged ranks never also defer |
| `cut_mean` (mean tables loaded or not) | ranks allowed `mean` | all or none |

Anything else is refused with the reason (tests/test_policy.py checks every case).

## Lookup-table policies (next)

The file format and the canonical table already hold arbitrary per-(layer, rank) tables. To run them, the kernels
read the masks instead of the scalars: unitk takes a 16-entry `hit` / `stage` mask per layer, the band plan a `band`
mask, cut_mean a `mean` mask, the deferral plan a `defer` mask. Nothing in a file written today changes. Natural
extensions of the table: a routing-weight axis (bucketed weights per rank) and a per-expert axis.

## Examples (`profiles/`)

| file | what |
|---|---|
| `policy-s1.json` | bd-all-c8-s1's miss handling (rank 0 staged in layers 12-35, ranks 0-2 elsewhere) |
| `policy-wide.json` | rank 0 staged in layers 1-47, 60-71, 84-89 (staging time per layer over the study's sensitivity) |
| `policy-feed.json` | the score-fed band: rank 0 staged everywhere, at most 8 band experts per call, feed 0.25 |
| `policy-study.json` | named policies by the sensitivity study's blocks (cheap / standard / sensitive / tail) |
