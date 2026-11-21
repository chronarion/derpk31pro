# On-policy tail benchmark (`tail-v1`)

Fixed-text KLD reads a fixed corpus through the engine, one conversation at a time, and compares every next-token
distribution with the official model's. It cannot see three things: what happens on text the engine writes itself,
what happens when several requests share the engine, and errors that are rare but confident. An engine that scored
0.06 on fixed text fell into repetition loops under load. This benchmark covers that side: the served engine writes
answers under its real serving condition, and the official model then judges every token it wrote.

## What is measured

A configuration answers the frozen prompt set once per in-flight count (temperature 1, top-p 0.95, a fixed seed per
prompt, at most 1,024 tokens per answer). For every generated token `x`, with `q` the served engine's own probability
of it and `p` the official model's probability of it given the same text so far:

| name | definition | faithful engine | needs the engine's log-probs |
|---|---|---|---|
| **gap** | mean of `log q(x) - log p(x)`: a one-sample estimate of the reverse KL of the served engine against the official model | ~0.003 | yes |
| **excess surprise** | mean of `-log p(x)` minus the official model's own expected `-log p` when it samples from its nucleus at the same position | ~0.003 | no |
| **out of nucleus** | `x` is outside the official model's top-p 0.95 nucleus: the official sampler could not have written it | ~1.5 per 1,000 | no |
| **tail curve** | `p(x)` < 1%, < 0.1% ("near-impossible"), < 0.01%, per 1,000 tokens, each beside its **official floor** | = floor | no |
| **confident errors** | `q(x)` >= 50% and `p(x)` < 1% | 0 | yes |

The **official floor** of a threshold is the rate at which the official model itself, sampling from its nucleus at
the same positions, writes a token under that threshold (the sum of `p(v) / Z` over nucleus tokens with `p(v)` below
it). It is computed exactly from the reference pass's full distributions. The reported tail is the observed rate and
the **excess** = observed - floor. The "faithful engine" column is the measurement's own floor: the official model in
one runner layout sampled, in another layout scored.

Headline, per in-flight count: **gap**, **near-impossible tokens per 1,000** (p < 0.1%), **confident errors per
1,000**. Everything else is in the score file.

## How to read it

- **The gap is the score to rank configurations by.** Every token contributes; on the frozen set it is good to about
  +-7%. It is the reverse direction of fixed-text KLD (mass the engine invents, not mass it fails to cover), on the
  engine's own text, under load.
- **Near-impossible tokens are the damage count.** It needs nothing from the engine but its tokens, so a wrong
  self-report cannot hide it. Its floor is close to zero (a token under 0.1% is inside a 95% nucleus only on very
  flat distributions: 0.002 per 1,000 on reasoning text, up to ~0.8 on chat), so it does not read as a multiple of
  the original; read the excess. It is a small count: a healthy configuration produces ~1 per 1,000, so one capture
  is good to about +-25% and separates "twice as many", not 10%.
- **Confident errors mark loop entries.** In the 13 repetition loops captured from a damaged configuration, the
  tokens around a loop's start held 6 times the usual rate of confident errors, and 9 of 13 loops had at least one;
  plain near-impossible tokens were only 1.5 times as frequent there.
- **None of these is a loop rate.** They are per-token rates on the first 1,024 tokens. About one loop was seen per
  1,300 near-impossible tokens, some loops are the official model's own behaviour, and loops started anywhere from
  token 127 to token 26,212. Loops and answers that hit the token cap are counted from long answers (below).
- **An engine that is sharper than the original** (lower effective temperature) scores well on every count. Its
  excess surprise turns negative and its fixed-text KLD rises. Sampling settings are recorded in every capture, and
  scores with different settings are not compared.
- **The in-flight count is part of the number.** Capture at 1 and at the highest count the configuration serves.
- Intervals are 95% bootstrap intervals over prompts (resampled within each kind of prompt). Two configurations are
  compared on the same prompts and seeds with `tailscore.py compare` (paired over prompts). Their texts diverge after
  the first different token, so pairing removes the prompt mix, not the sampling noise: for the gap it narrows the
  interval by ~40%, for the counts by ~10%.

## The frozen set

`tools/onpolicy/sets/tail-v1.json`: 120 prompts with one seed each.

| kind | prompts | source |
|---|---|---|
| gpqa | 48 | GPQA Diamond, the first 48 records, as lil-bench's `gpqa-diamond` profile sends them |
| math, code, chat | 12 each | Nemotron-SFT Math v4, Competitive-Programming v2, Instruction-Following-Chat v3 |
| agentic | 12 | Nemotron-SFT Agentic v2 (search): system prompt, tools, first user turn |
| multilingual | 12 | Nemotron-SFT Multilingual v2 (Japanese STEM) |
| science, safety | 6 each | Nemotron-SFT Science v2, Safety v2 |

The non-GPQA prompts are conversation openings taken from the end of the fixed-text KLD protocol's source files, rows
no fixed-text window uses, at most 3,000 characters. The manifest holds, per prompt, its id, seed and the sha256 of
its messages and tools, plus the sha256 of every source file; the prompt text is rebuilt locally and verified
(GPQA's text is not republished). GPQA Diamond is also a task benchmark here: nothing is trained on these prompts
and this score is not accuracy, but a configuration should not be chosen on the GPQA row alone. Limits: one
language besides English, agentic prompts stop at the first step, no long contexts.

## Running it

```bash
# once: rebuild the prompt file from the manifest (fails unless all 120 hashes match)
python3 tools/onpolicy/prompts.py tail-v1.jsonl --recipe tools/onpolicy/sets/tail-v1.json \
  --gpqa gpqa_diamond.jsonl --gpqa-template gpqa_template.json --src SOURCE_DIR

# on the serving machine: capture a configuration at 1 and 4 requests in flight (~40 minutes)
TAG=myconfig tools/onpolicy/tail_capture.sh default captures/ tail-v1.jsonl "1 4" [VAR=value ...]

# where the official model runs: score each capture (reference pass ~20 minutes on one GPU)
GPU=<device> tools/onpolicy/tail_score.sh captures/myconfig-c4.jsonl scores/

# compare two configurations on the same prompts and seeds
python3 tools/onpolicy/tailscore.py compare scores/a-c4.json scores/b-c4.json
```

`tail_score.sh` builds the set (`build_sets.py`), runs the reference runner over it, computes the exact floors
(`tailfloor.py`) and writes `NAME.json` / `NAME.txt` (`tailscore.py score`): the table above for all prompts, per
kind and per position in the answer, the answers that stopped, hit the cap or looped, and for each count how many
tokens a +-10% interval would need. `tailscore.py selftest` checks the definitions on a synthetic model.

## Beside it

- **Loops and cap hits**: answers captured with a long cap (`capture.sh`, 32,768 tokens) at the served in-flight
  count; `capture.py` flags repetition loops. A release reports loops and cap hits per 100 answers.
- **Fixed-text KLD** stays the measure of mass the engine fails to cover, and is also read with several sequences in
  flight (`kld.py --conc`).
- **Task benchmarks** stay the measure of whether answers are right.
