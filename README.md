# derpk31pro

Kimi K3.1 Pro on a single DGX Station GB300, as close to the official model as GPQA Diamond can tell: 186.3 of 198
answered right (average of three runs; Moonshot reports 94.2%, 186.5 of 198), at about 76 tokens per second for one
conversation and 111 for four at once. A `fast` profile trades some of that closeness for over 100 tokens per second.

## What is derpk31pro

Kimi K3.1 Pro is a very large mixture-of-experts model: the official checkpoint is 1.56 TB, and every token is
computed by 16 of 896 experts in each of its 92 expert layers. A DGX Station GB300 has 252 GB of GPU memory, so most
of the model cannot sit on the GPU.

derpk31pro serves the model from one DGX Station anyway, by keeping each expert where it is cheapest to use:

- **GPU memory** holds about 9,400 experts in their official form: the ones the conversation uses most. Which ones
  changes as the conversation goes; experts move in from the drive in the background.
- **The RTX PRO 6000** (the Station's second card) holds about 11,000 more experts in a compact 2-bit form and computes
  them at the same time as the GB300 computes its own.
- **The CPU's memory** holds about 42,000 experts in 2-bit form. When a token needs one of them, the GB300 copies it
  in for that step.
- **Everything else is a trade we measure.** Not every pick matters equally: some layers barely notice when an
  expert is approximated, others carry the error forward, and a token's strongest picks carry most of its weight.
  derpk31pro's philosophy is to profile the impact of every shortcut, layer by layer and pick by pick, and to write
  the result down as a policy: which picks are computed on the RTX, which are copied in from CPU memory, and which get
  the expert's average output instead. Policies sit at different points between speed and closeness; several are
  included, and you can write your own (see [Policies](#policies)).

Everything is fed from one 4 TB NVMe drive. The engine runs inside vLLM's `kimi-k31` image and is built around
MegaMoE, the expert kernel from DeepGEMM, which computes a whole expert layer in one launch.

## How fast and how close

| profile | decode, 1 / 2 / 4 conversations at once | reading a prompt (32K / 128K tokens) | GPQA Diamond (198 questions) |
|---|---|---|---|
| `default` (GB300 + RTX PRO 6000) | 76.2 / 95.5 / 111.0 tokens/s | 2,117 / 2,163 tokens/s | 186.3 (runs: 186, 185, 188) |
| `nortx` (GB300 alone) | 75.9 / 90.2 / 108.4 tokens/s | 2,232 / 2,228 tokens/s | 187.0 (runs: 188, 186) |
| `fast` (GB300 + RTX PRO 6000) | 103.4 / not measured / not measured | 2,138 / 2,164 tokens/s | not measured |
| `fast-nortx` (GB300 alone) | 88.3 / not measured / not measured | 2,246 / 2,215 tokens/s | not measured |

Every profile serves up to four conversations of up to 262,144 tokens each (a 260,000-token prompt reads at about
2,000 tokens/s).

| how close (lower is closer) | writing its own answers: distance from the official model, 1 / 4 conversations at once | near-impossible tokens per 1,000 (4 at once) | confident mistakes per 1,000 (4 at once) | long answers stuck in a loop (4 at once) | KLD on fixed text, 512 samples (1 at a time) |
|---|---|---|---|---|---|
| `default` | 0.082 / 0.094 | 1.24 | 0.66 | 0 of 72 | measurement in progress, expected below `fast`'s 0.0619 |
| `nortx` | 0.111 / 0.125 | 2.17 | 1.10 | 0 of 72 | measurement in progress, expected below `fast-nortx`'s 0.0756 |
| `fast` | 0.101 / 0.105 (science questions only) | 1.90 (science questions only) | not measured | not measured | 0.0619 |
| `fast-nortx` | 0.118 / 0.121 (science questions only) | 2.39 (science questions only) | not measured | not measured | 0.0756 |
| an early build of `fast` | 0.300 (4 at once) | 10.8 | 3.84 | 6 of 64 | 0.0619 |
| the official model, for scale | 0.003 | under 0.01 on science questions, up to 0.8 on chat | 0 | | 0 |

- **Speed:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) (lil-bench) 1.7.1.
  Decode is its decode cell (a chat prompt, generated tokens counted over 30 s) with 1, 2 or 4 conversations at once,
  the mean of four runs; reading a prompt is the standard profile's prompt length divided by the time to the first
  token (32,030 and 127,363 tokens). Sampling is the model card's: temperature 1, top-p 0.95, which is also the
  engine's default. The DSpark drafter proposes 4 tokens per step; how many it gets right depends on the text it
  happens to sample, so one run can land 4% either side of the average.
- **GPQA Diamond:** lil-bench's `--test-profile gpqa-diamond`, all 198 questions, at the model card's settings
  (reasoning effort max, temperature 1, top-p 0.95), four questions at a time. One run can land about 3.5 questions
  either side of a configuration's true score, so the table shows the average of three runs (two for `nortx`) and
  the runs themselves.
  No answer in any run ran out of room.
- **Writing its own answers:** the engine answers 120 fixed prompts, alone and with four requests running at once, and
  the official model then judges every token it wrote. This catches rare, confident mistakes and damage that only
  appears under load, which KLD on fixed text does not see. The loop count comes from 72 longer answers (up to 32,768
  tokens). The `fast` rows were measured on 48 science questions only, which read closer than the full set (the early
  build: 0.22 on them against 0.30 on all 120). See [Measuring what the engine writes](#measuring-what-the-engine-writes).
- **KLD on fixed text** (lower is closer): how far the next-token probabilities are from the official model's on text
  the official model would write, read one conversation at a time. See [Measuring KLD](#measuring-kld).
- **The early build** is `fast` before the decode split (see [Serving several conversations](#serving-several-conversations)):
  it read well on fixed text, one conversation at a time, and fell apart with four. It is in the table because it is why
  the second measurement exists.

## What you need

- A DGX Station GB300 (GB300 + Grace, 512 GB of CPU memory). The RTX PRO 6000 Blackwell is optional: the `nortx`
  profile runs on the GB300 alone, slower and less close to the official model (below).
- One fast 4 TB NVMe drive for the model data (~3.1 TB: the official checkpoint, a stripped copy, and the 2-bit
  records). Its read speed is part of the engine's speed: while decoding, the engine reads 9.4 to 10.2 GB/s from it
  (about 400 experts a second moving into GPU memory, and nearly as many 2-bit copies refreshed in CPU memory), and we
  have seen it keep the drive fully busy at 13.7 GB/s. Every number here was measured on a **WD_BLACK SN8100 4 TB**
  (PCIe Gen5 x4, 14 GB/s rated read). A Gen4 drive (about 7 GB/s) still works, but the experts follow the
  conversation more slowly, so expect lower speed and higher KLD; we have not measured how much.
- Docker with the NVIDIA Container Toolkit (CDI device names), and the `vllm/vllm-openai:kimi-k31` image.
- The official checkpoint (`moonshotai/Kimi-K3.1-Pro`) and the DSpark drafter (`Inferact/Kimi-K3.1-Pro-DSpark`).

## Getting started

```bash
cp config.example.env config.env     # then set the paths in it
./build.sh                           # builds the engine's and the sidecar's GPU libraries inside the vLLM image
docker build -t derpk31pro-runner tools/runner/docker
engine/store/run_kstrip.sh           # the stripped checkpoint (everything but the experts), ~107 GB
engine/store/run_build_store.sh      # the 2-bit records of every expert, ~1.4 TB
engine/k31.sh up                     # starts the server on 127.0.0.1:8113 (OpenAI-compatible API)
```

Without an RTX PRO 6000, start the `nortx` profile instead: `engine/k31.sh up nortx`. For more speed and less
closeness, start `fast` (or `fast-nortx`): the earlier settings, which stage fewer picks. The first start takes several
minutes: it reads the resident experts and up to 418 GB of 2-bit records into memory and compiles kernels.
`engine/k31.sh down` stops it; `engine/k31.sh logs` shows the log.

<details>
<summary>The average-output tables</summary>

Skipped picks are replaced by each expert's average output, from tables measured on held-out text and published on
Hugging Face ([chronarion/derpk31pro-means](https://huggingface.co/chronarion/derpk31pro-means), 591 MB each).
`default` and `nortx` use the refitted table (`k31-fmeans.safetensors`: each expert's average refitted on the picks
that actually get it, weighted by how much each pick contributes to the layer's output; about 12% lower KLD in our
reference runner); `fast` and `fast-nortx` use the first one (`k31-expert-means.safetensors`):

```bash
hf download chronarion/derpk31pro-means k31-fmeans.safetensors k31-expert-means.safetensors --local-dir /models/k31-means
```

and set `MEANS_DIR=/models/k31-means` in `config.env`. Without them, start with `K31_CUT_MEAN=` (empty): skipped
picks then add nothing, which raises KLD. The first table can be rebuilt with the reference runner's means pass
(`tools/runner`, `--save-means`).
</details>

## How it works

Each of the model's 92 expert layers picks 16 of its 896 experts for every token, ranked by weight. Where each pick
is computed depends on where that expert currently is. The default profile places the 82,432 experts like this:

| Where | Experts | Per layer | Form | How a pick of it is computed |
|---|---:|---:|---|---|
| GPU memory | 9,390 (11%) | 46 to 378 | Moonshot's own MXFP4 | exactly as shipped, in one MegaMoE launch with the layer's other picks |
| RTX PRO 6000 | 10,856 (13%) | 48 to 216 | 2-bit copies | on the RTX, while the GB300 runs MegaMoE; the GB300 adds the results in the kernel that finishes the layer |
| CPU memory | 41,661 (51%) | 200 to 640 | 2-bit copies (387 GB) | copied into one of 128 shared staging rows on the GB300 and computed in the same MegaMoE launch |
| Nowhere in memory | 20,525 (25%) | 0 to 602 | | skipped: the expert's average output, scaled by the pick's weight, stands in for it |

Without the RTX (`nortx`), the band's experts move to CPU memory: 9,390 in GPU memory, 45,000 in CPU memory (418 GB),
28,042 nowhere.

Layers whose output matters more get more GPU memory (`engine/profiles/hbm-map-256k-final.json`, `band-map.json`,
`bank-map-256k.json`). The GPU's share is what is left after the KV cache for one 262,144-token conversation (16 GiB)
and the measured peak of reading a 4,096-token prompt chunk (which does not grow with the context), plus 2 GiB.

### How experts are chosen for GPU memory

Which experts get the exact MXFP4 treatment is decided in three steps:

1. **How many per layer** is chosen by KLD. A simulation replays the routing of the KLD screen's text through the
   tiers and fits, for every layer, how much one more expert in GPU memory lowers KLD; the memory goes where that
   gain is largest (56 to 378 experts per layer instead of the same number everywhere).
2. **Which ones** is chosen by usage. Every expert is ranked by how strongly the official model picks it (its summed
   squared routing weight) over 65,536 tokens of varied text, separate from the KLD evaluation text; each layer's
   most-used experts start in GPU memory, the next ones on the RTX, the next ones in CPU memory.
3. **Then the choice follows the conversation.** While serving, the engine keeps a decaying score of each expert's
   recent picks and swaps the strongest experts that are not in GPU memory for the weakest ones that are,
   reassembling the exact MXFP4 weights from the 2-bit copy and its complement on the drive.

This is basic, and it is the part of the design we would change first. Usage is only a stand-in for what matters:
how much KLD it costs when a given expert is served from its 2-bit copy (or its average) instead of exactly. Those
differ a lot between experts. A frequently picked expert whose 2-bit copy is accurate gains little from being exact,
while a less frequently picked one whose 2-bit copy is poor, or whose output the next layers are sensitive to, can
cost more. Choosing the experts by measured KLD (each expert's cost in its fallback tier, per layer) rather than by
usage should buy KLD at no speed cost, or speed at the same KLD.

When decoding, a policy decides which CPU-memory picks are staged and which get the average (see
[Policies](#policies)). When reading a prompt, every CPU-memory pick is staged (each layer's copies stream in once per 4,096-token chunk),
the RTX gets each layer's band picks in one bulk request instead of one per eight tokens, and the tiers keep adapting
while the prompt is read, so the prompt's own rows are computed about as well as decoded ones.

The tiers follow the conversation. Every token, the engine scores which experts the recent tokens needed; the
strongest ones not yet in GPU memory are read from the drive and swap in for the weakest, the CPU-memory copies are
refreshed the same way, and experts leaving GPU memory move down to the 2-bit tiers. All of this runs in the
background, between steps, from the one model drive.

Two more things make decoding fast: the dense parts of the model run in 8-bit forms chosen per layer kind (int8,
FP8, MXFP4), and the DSpark drafter proposes 4 tokens per step that the model checks in one pass, so a step yields
about 2.45 tokens on average at the model card's sampling (temperature 1, top-p 0.95).

### Serving several conversations

An earlier build had a serious problem we did not see at first, and it shaped how this engine is measured.

**What went wrong.** Each decode step checks 5 rows per conversation: the next token and the drafter's 4 guesses.
The engine's budgets (which picks go to the RTX, how many copies are staged from CPU memory, the caps per draft row,
and the scores that decide which experts move into GPU memory) were all set per expert-layer call, and one call
covered every conversation in the step. With four conversations a step carried 20 rows, and past 16 rows the engine
fell back to a simpler path: the RTX was switched off, staging lost its per-row plan, rejected draft rows were scored
as if they had been accepted, and the experts in GPU memory stopped following the conversations. One conversation at
a time, everything was fine.

**What it did.** With four conversations at once, the engine's answers drifted about three times as far from the
official model as with one, it wrote tokens the official model would almost never write four to five times as
often, and 3.4% of its answers fell into a repetition loop that ran until the answer was out of room. On GPQA
Diamond, run four questions at a time, it scored 172 of 198 (the official model: 186.5).

**Why our measurements missed it.** KLD on fixed text reads one conversation at a time, on text the official model
wrote, and it read 0.062 for this build. A loop does not show in it either: once a model is repeating itself, the
official model agrees that the next token is "more of the same". The damage sits in the few tokens where a loop
starts, and an average over hundreds of thousands of tokens hides them.

**How we found it.** GPQA answers that never finished led us to read the engine's own answers back through the
official model. In 9 of 13 loops, the official model gave the token that started the loop under a 0.6% chance where
the engine gave it 50-100%; and the damage appeared only when a step carried 20 rows (four conversations with four
drafts each), not with 8 or 5.

**The fix.** With more than 8 rows in a step, each conversation's rows now run through the one-conversation path as
their own call (`K31_DECODE_SPLIT`), so every budget applies per conversation. Four at once are now as close to the
official model as one (distance 0.105 against 0.101 for the `fast` settings). That made four conversations about 13%
slower in total, so a joint path (`K31_DECODE_JOINT`) then shares what can be shared: the layer's dense parts and one
MegaMoE launch for all conversations, while each keeps its own staging plan and RTX call. Its results are bit for bit
those of the split; it brought four conversations of `default` from 94 to 111 tokens/s in total.

The closer `default` and `nortx` profiles came out of the same work: with four conversations measured properly,
staging more of each token's picks (ranks 0-7 instead of 0-3) and better average outputs paid for themselves in
GPQA, at about 76 tokens/s for one conversation instead of about 103.

## Policies

derpk31pro does not assume which shortcuts are safe: it measures them. A sensitivity study
([docs/sensitivity](docs/sensitivity/README.md): 31 passes, 748 variants of the model) priced every way of treating a
pick, by rank, by layer and by lateness: left out, served from a 2-bit copy, given the expert's average, or computed
exactly but one or more layers late. Profiles of the decode step record what each treatment costs in time. A policy turns those measurements into a table: for each of the 92 layers and
each of a token's 16 picks (strongest first), how a pick of an expert that is not in GPU memory is served.

| treatment | what happens | what it costs |
|---|---|---|
| RTX | computed on the RTX PRO 6000 from its 2-bit copy, while the GB300 computes the rest | close to exact; no GB300 time while the RTX keeps up |
| staged | its 2-bit copy is copied from CPU memory into a shared row on the GB300 and computed with the others | close to exact; tens of microseconds per copy, on the critical path |
| average | the expert's average output, scaled by the pick's weight, stands in for it | about half the error of leaving it out, for weak picks |
| left out | nothing | the whole pick's error |
| capped (resident) | a weak pick of an expert that *is* in GPU memory gets the average instead of its row | MegaMoE reads fewer experts per step; the pick's error is the average's |
| deferred (experimental) | computed exactly, one layer late | exact but late; not used by the included policies |

Staging more picks brings the engine closer to the official model and costs time; staging fewer is faster. Because
the measurements say where a microsecond buys the most KLD, a policy can aim at a speed or at a KLD, and every layer
gets its own rule: the least sensitive layers stage nothing, the most sensitive stage a token's four strongest picks.
Several policies are included, and we keep trying new ones:

| policy | hardware | what it does | trade |
|---|---|---|---|
| `policy-stage8.json` (**`default` profile**) | GB300 + RTX | a token's 8 strongest picks are staged whenever their expert has a 2-bit copy in CPU memory, in every layer; no resident caps; at most 8, 8, 4, 2, 2 staged ranks for the 5 rows of a verify block and 6 copies per call | measured: GPQA Diamond 186 of 198 (one run), distance 0.082 / 0.094 at 1 / 4 conversations; 76 tokens/s |
| `policy-stage4.json` (`quality` profile) | GB300 + RTX | the same with the 4 strongest picks | priced in the reference runner, not measured on its own |
| `policy-nortx-stage4.json` (**`nortx` profile**) | GB300 alone | a token's 4 strongest picks staged in every layer; at most 4, 4, 4, 2, 2 staged ranks per row and 6 copies per call | measured: GPQA Diamond 188 of 198 (one run), distance 0.111 / 0.125; 76 tokens/s |
| `policy-joint15.json` (**`fast` profile**) | GB300 + RTX | staged ranks and resident caps chosen together for about 1.5 ms less per step: in the least sensitive layers a token's weaker picks get the average even when their expert is in GPU memory | measured: +6% steps per second, KLD screen 0.0882 against 0.0702 without caps |
| `policy-joint12.json`, `policy-joint20.json` | GB300 + RTX | the same, for about 1.2 / 2.0 ms less | measured: joint20 +2% steps per second over joint15, KLD screen 0.1022 against 0.0882 (too far for the KLD target) |
| `policy-caps06.json`, `policy-caps12.json` | GB300 + RTX | resident caps only, the default's staged ranks | measured: caps12's KLD screen 0.0846 against joint12's 0.0833 |
| `policy-default.json` | GB300 + RTX | no caps; the strongest pick staged in most layers, the 2 strongest in layers 74, 76-80 and 83, the 4 strongest in 52-59 and 92, nothing in the 16 least sensitive layers | the base the others are compared with |
| `policy-quality.json` | GB300 + RTX | more staging | simulated: about 0.5 ms more per forward pass, about 4.9% lower KLD |
| `policy-balanced.json` | GB300 + RTX | as much staging as the default, placed by KLD per staged copy | simulated: same time, about 1.4% lower KLD |
| `policy-fast.json` | GB300 + RTX | less staging | simulated: about 0.6 ms less per forward pass, about 4.6% higher KLD |
| `policy-nortx-t92-allrows.json` (**`fast-nortx` profile**) | GB300 alone | staged ranks placed by KLD per microsecond of staging: the 4 strongest picks in the most sensitive layers, the strongest in most others, none in the cheapest | |
| `policy-nortx-joint03/06/09.json` | GB300 alone | resident caps without the RTX | measured: joint06 +2% steps per second, KLD screen 0.1005 against 0.0892 without caps |
| `policy-nortx.json`, `policy-nortx-fast.json`, `policy-nortx-t92.json` | GB300 alone | earlier GB300-alone tables | |

The RTX policies send every pick of an RTX expert to the RTX (at most 8 experts per call) and give the average to
the picks that are not staged. Choose a policy with `K31_POLICY`, for example
`K31_POLICY=/engine/profiles/policy-fast.json engine/k31.sh up`. The file format is in `engine/POLICY.md`.

Two settings change what every policy stages. A record staged for a token's strong pick also serves that expert's
weaker picks and its picks in the other rows of the verify block (`K31_STAGE_ALL`, on in every profile: the same
copies, more picks served). And `K31_STAGE_ROWS` caps the staged ranks per row of the verify block, because a later
draft row only counts when the drafts before it were accepted (about 69%, 43%, 24% and 14% of the time for rows 1-4):
`K31_STAGE_ROWS=4,4,2,1,1` measured +2.8% steps per second without the RTX, KLD screen 0.1006 against 0.0892; it is
the `fast-nortx` profile's default (512-sample KLD 0.0756). `K31_STAGE_CAP` caps the copies one call stages; the
`default` and `nortx` profiles stage at most 6.

## What made it fast, and what kept it close

The engine was built one change at a time, each measured on its own run where we could: decode speed on the bench
(median of 10 runs; it moves 3-5% between runs because the drafter's acceptance does), from late on also the time of
one forward pass in milliseconds (steadier), and KLD on the 32-sequence screen ("rot32"). Several changes went in
together; where a number covers a bundle, it says so. The KLD screen was made stricter part of the way (drafts accepted
at the drafter's real rates instead of always): the same engine reads about 10% higher under it, so KLD numbers are
compared only within one protocol. Most of this history built what are now the `fast` profiles; `default` and
`nortx` add the staging policies, the refitted averages and the decode split described above.

### Speed

| change | what it does | what it bought |
|---|---|---|
| **The RTX band** | The second card holds about 11,000 experts in 2-bit form and computes the ones a token picks while the GB300 runs its own experts; the GB300 adds the results in the kernel that finishes the layer. | The single largest change. It came with 8-bit dense layers and replaced deferral (below) in one step: 49.8 to 57.6 tokens/s, and KLD 0.0783 to 0.0723. The RTX's kernel takes 15 / 30 / 53 / 99 us for 1 / 4 / 8 / 16 experts per call. |
| **8-bit dense layers (W8A16)** | The dense layers' weights in FP8 with a scale per 128 values, the activations left in BF16, and a kernel of our own that splits each matrix across all of the GPU's cores. vLLM's FP8 kernels reached only 0.9-4.5 TB/s at the 3-5 rows a decode step has. | Per call, for example: 19.1 to 11.6 us (7168 x 3584), 19.2 to 9.5 us (2112 x 7168). In the engine, part of the step above. |
| **MXFP4 for the KDA projection** | The largest dense matrix (49,376 x 7,168) in 4-bit MXFP4, through DeepGEMM. | 209.5 to 96.3 us per call (111.1 with W8A16); +0.004 KLD in the reference runner. In the engine, part of a 57.9 to 64.7 tokens/s step with six other changes. |
| **CUTLASS FP8 for two layer kinds** | The shared experts' gate/up and the attention's output gate on CUTLASS's FP8 GEMM, which suits those shapes better. | 64.7 to 69.2 tokens/s (+7%) on its own; KLD 0.0765 to 0.0771. |
| **The MegaMoE fork** | Derpseek's changes to MegaMoE's scheduling, carried to this image's DeepGEMM; bit-identical output. | Part of a 69.2 to 76.7 tokens/s step with six other changes. |
| **Band entries by score** | The band's slots follow the conversation: an expert with a stronger recent score takes the slot of a weaker one, copied in from the record store. | The band served 13.6% of picks instead of 9-10%, and the GB300 had to stage 1.1% instead of 4.3-5.7%: 76.7 to about 85-88 tokens/s. |
| **4 draft tokens** | The DSpark drafter proposes 4 tokens per step instead of 3. | 84.7 to 87.4 tokens/s on its own (acceptance 2.47 to 2.54 tokens per step). |
| **Marlin** | Marlin's FP8 kernel for the 112 dense matrices where it measured faster, and for the drafter's copy of the output head. | Forward 28.1 to 27.8 ms; the head copy 298 to 233 us. |
| **A shorter tail after MegaMoE** | The routed output's norm and up-projection no longer fused in a slow kernel; the band's results collected in one round trip; the skipped picks' averages gathered in one pass. | Forward 27.8 to 26.2 ms (-1.6 ms) together. |
| **Better fork and join points** | Side work started where its inputs are ready and joined where its output is needed: the latent down-projection, the sparse attention's output gate, and one kernel that adds the band's rows and the averages in one rounding. | A sparse-attention layer 290.7 to 264.8 us; the forward about 26.0 ms. |
| **The staging table** | No staging in the 16 layers where a missing expert matters least; more ranks staged in the most sensitive layers. | Forward 26.12 to 25.69 ms (with a promotion-margin change in the same run). |
| **int8 on the main chain** | The int8 GEMV from the Hecate overlay for the attention's output and query projections, the routed output's up-projection, and the drafter: a per-row int8 weight, read at 4-6 TB/s. | Forward 26.04 to 25.41 ms and the drafter 1.284 to 1.189 ms, with KLD unchanged (0.1156 / 0.1158). Later also the routed output's up-projection: forward 24.60 to 24.23 ms and KLD slightly lower (paired -0.0011). On every dense layer at once it was slower: its whole-GPU launches starve the work beside them. |
| **Rows, band and bank sized per layer** | (In the KLD table.) More of a token's picks resident means less staging. | Forward 25.4 to 24.6 ms over the KLD screen (23.8 in the bench's part of the run); part of it is the stricter screen's more realistic rejected drafts, which need less staging. |
| **The final DSpark drafter** | Inferact's final drafter instead of the preview. Its sliding-window layers are handed to vLLM's KV manager as full attention (`K31_DRAFT_FULL_KV`); otherwise every chat request re-read its whole prompt and the first token took 1.1 s. | 83.6 to 93.1 tokens/s on lil-bench's decode cell (4 runs each, the same engine otherwise): 2.19 to 2.44 tokens per step; first token 0.12 s. |
| **Draft-side top-p** | The drafter's proposal is truncated by the request's own top-k / top-p before a draft is sampled (`K31_DRAFT_TOPP`); the verifier checks against the same truncated distribution, so the output distribution stays exact. | About +1% accepted tokens at top-p 0.95, for 14 us of masking per draft position. |
| **Fused sparse-attention kernels** | The DSA layers' small decode kernels (index key insert, index query quantization, cache insert, query padding, output gate) in three launches, bit-exact (`K31_DSA_FUSED`). | 18.0 to 5.4 us per DSA layer, 24 layers. |
| **int8 latent down-projection** | The routed experts' shared down-projection on the int8 GEMV. | +1.8% steps per second. |
| **Resident caps** | In the least sensitive layers a token's weaker picks get the expert's average even when the expert is in GPU memory, so MegaMoE reads fewer experts per step (`policy-joint15.json`). | 38.1 to 40.4 steps per second (+6%); the KLD it costs is in [Policies](#policies). |

### Reading prompts

| change | what it does | what it bought |
|---|---|---|
| **The prompt path** | Prompts in 4,096-token chunks: every CPU-memory pick of a chunk is staged (each layer's records stream in once per chunk), the RTX gets each layer's band picks in one bulk request, and the tiers keep adapting during the prompt. | From about 230 tokens/s (512-token chunks, the decode path) to 1,400-1,650 on 8K-32K prompts. |
| **Big-row dense kernels** | Prompt chunks through int8 and W8A16 kernels made for thousands of rows (`K31_I8_BIG_M_KERNEL`, `K31_W8A16_BIG_M`). | Together with the next row, below. |
| **A copy-engine lane** | Each chunk's CPU-memory records are copied by the GPU's copy engines into a scratch buffer while MegaMoE runs, instead of by the stager kernel (`K31_PROMPT_CELANE`). | With the big-row kernels, on 16K prompts: 1,600 to 1,808 tokens/s (+13%), and 1,585 to 1,904 (+20%) without the RTX. |
| **Row slices for sparse attention** | A chunk's sparse attention in slices of 2,048 rows (`K31_DSA_ROW_SLICE`), bit-identical. | No speed cost (1,816 tokens/s); a smaller peak. |
| **GPU memory sized by the prompt's peak** | The memory peak of a 4,096-token chunk was measured per step (`K31_MEM_PEAK`); it does not grow with the context, so the resident experts fill what the KV cache, that peak and 2 GiB leave. | A 260,000-token prompt reads at about 2,000 tokens/s; 2.8 GiB stay free at the peak. |

### KLD

| change | what it does | what it bought |
|---|---|---|
| **2-bit experts instead of skipping** | A pick that is not in GPU memory is computed from a 2-bit copy on the RTX or staged from CPU memory, instead of dropped. | In the reference runner, a band of 155 experts per layer instead of 35 took KLD from 0.0544 to 0.0390 (-28%); a 2-bit pick on time costs 7-15% of what skipping it costs. |
| **Staging a token's strongest misses** | The GB300 copies the 2-bit record of a strong pick it does not hold into a staging row for that step. | Staging ranks 0-2 instead of only rank 0: KLD 0.1203 to 0.0982 (early engine, same run otherwise). |
| **The average substitute** | A skipped pick adds its expert's average output times the pick's weight, instead of nothing. | Halved the cost of skipping in the [sensitivity study](docs/sensitivity/README.md) (0.53 on held-out text); in the engine only measured inside a bundle. |
| **Experts that follow the conversation** | Every token, experts that recent tokens needed move into GPU memory from the drive (0.1 per layer per token) and the CPU-memory copies are refreshed (0.08); an expert leaving GPU memory moves down to 2 bits. | The adaptive CPU-memory tier: 0.1317 to 0.1217 (paired -0.0100); copying the leaving expert down instead of re-reading it: 0.1018 to 0.0790 (paired -0.0228). |
| **Rows per layer** | GPU memory split between layers by how much one more resident expert lowers KLD (56 to 378 per layer instead of 105 each). | 0.1158 to 0.0973 (paired -0.0117, the largest single KLD step). |
| **No hand-offs, weighted credit** | A band expert promoted to GPU memory frees its slot for the best expert in CPU memory, instead of handing it to the expert it replaced; each layer's adaptation speed weighted by its sensitivity. | 0.0973 to 0.0854 (paired -0.0045). |
| **Band and CPU-memory sizes per layer** | The RTX band and the CPU-memory copies split between layers the same way, and 5,000 more CPU-memory copies. | 0.0854 to 0.0797 (paired -0.0050); predicted 512-sequence headline 0.061. |
| **Staged experts serve every row** | A record staged for a token's strong pick also serves that expert's other picks in the verify block (`K31_STAGE_ALL`), and the band's 8 slots per call go to the experts whose rows are most likely to be accepted (`K31_BAND_PRIOR`). | 0.0752 to 0.0702 at the same speed. |

Two choices raise the KLD we report on purpose: the engine learns which experts matter only from tokens the model
accepted (+0.0072 paired against learning from every draft), and the screen accepts drafts at the drafter's real
rates (about +10%). Both make the number what a user would see.

<details>
<summary>What was tried and left out</summary>

- **Deferral** (computing some picks one layer late, beside the next layer, so their cost leaves the critical path):
  before the RTX band it bought KLD (paired -0.0199) but cost speed (no deferral 63.5 tokens/s; 2, 4 and 8 deferred
  picks per layer: 55.8, 52.4, 45.3). The band computes the same misses on time, in parallel, so the first band run
  without deferral beat the best deferral run on both counts (57.6 / 0.0723 against 49.8 / 0.0783). The code stays as
  an option (`K31_DEFER`).
- **Band entries from CPU memory instead of the drive** (`K31_BAND_RING`): works (no drive reads for band entries),
  but spending the freed drive bandwidth on twice as many promotions cost 1.2 ms per forward and raised KLD. Off.
- **int8 on every dense layer**: slower (above). **Our own router kernels**: slower than vLLM's.
- **Copying the staged records ahead** (`K31_PRESTAGE`): the next layer's picks predicted before its attention and
  their records copied while the attention runs. The copies left the critical path, but the attention's kernels
  beside them slowed by as much (an int8 GEMV 18 to 88 us): 1.5% slower; with a narrower copy the attention could
  not hide it (19% slower). Copying from CPU memory is bound by what the copying SMs can move, not by when it runs.
- **vLLM's adaptive verification** (verify blocks sized from the drafter's confidence head) and **a sharper draft
  temperature** (0.85): no measured gain for one conversation.
- **The Grace CPU computing cold experts** (`K31_POOL`, `K31_LATE`): 64 us per record, too slow to finish on time.
- **MegaMoE's inputs without vLLM's prep kernel** (`K31_MEGAMOE_NOPREP`): no measured gain.
- **8,192-token prompt chunks**: they do not fit beside a 262,144-token KV cache.

</details>

<details>
<summary>The runs, in order</summary>

| run | change | tokens/s | forward ms | rot32 KLD |
|---|---|---|---|---|
| x08-d4-be | deferral, FP8 dense (first design) | 49.8 | | 0.0783 |
| bd-nd-w8 | RTX band, no deferral, W8A16 dense | 57.6 | | 0.0723 |
| bd-all | + 4 drafts, averages, MXFP4 projection, drafter in 8 bits, ... (7 changes) | 64.7 | | 0.0765 |
| bd-all-c8 | + CUTLASS FP8 for two layer kinds | 69.2 | | 0.0771 |
| bd-all-c8-s1 | + MegaMoE fork, fewer staged ranks in 12-35, ... (7 changes) | 76.7 | | 0.0806 |
| -fw | + band entries by score, staging table | 84.7 | | 0.1005 |
| -fw-d4 | + 4 drafts | 87.4 | 28.1 | 0.0995 |
| -mar | + Marlin | 92.1 | 27.8 | 0.1017 |
| -fu0 | + shorter tail | 91.3 | 26.0 | 0.1043 |
| C | + fork and join points, one-kernel tail | 97.1 | 26.1 | 0.1054 |
| E | + staging table T1 | 93.9 | 25.7 | 0.1049 |
| E, stricter screen | (same engine) | 95.2 | 26.0 | 0.1156 |
| G | + int8 on the main chain | 102.9 | 25.4 | 0.1158 |
| R1 | + rows per layer | 102.4 | 25.7 | 0.0973 |
| R2 | + no hand-offs, weighted credit | 95.1 | 25.7 | 0.0854 |
| R3 | + band and CPU-memory sizes per layer | 100.3 | 24.6 | 0.0797 |
| R3 + int8 up | + int8 routed up-projection | 99.5 | 24.2 | 0.0768 |

Forward ms is the median time of the main CUDA graph over the run. The bench's tokens/s moves with the drafter's
acceptance on its three prompts (2.4 to 2.7 tokens per step between runs), which is why some speed changes show in
the forward time but not in it.

From here on, speed is lil-bench's decode cell at the model card's sampling (temperature 1, top-p 0.95), the mean of
4 runs, and steps per second beside it (steadier: the drafter's acceptance moves tokens/s by about 4% per run).

| run | change | tokens/s | steps/s | rot32 KLD |
|---|---|---|---|---|
| XB | R3 + fused DSA kernels, staged experts serving every row, band prior (preview drafter) | 83.6 | 38.2 | 0.0702 |
| XBT | + draft-side top-p | 84.7 | 38.1 | |
| FX | final drafter with full-attention KV specs (instead of XBT) | 93.1 | 38.1 | |
| FXJ15T | + resident caps (joint15), draft-side top-p | 99.8 | 40.4 | 0.0882 |
| RF15T | + int8 latent down-projection | 101.7 | 41.0 | 0.0878 |
| RF | release map (9,390 experts, sized by the prompt's peak), prompt kernels: now the `fast` profile | 103.4 | 41.1 | 0.0896 |
| NF0T | without the RTX: the final drafter, staging table t92 for stage-all, fused DSA, int8 latent, top-p | 87.8 | 35.5 | 0.0892 |
| NF | release map: now the `fast-nortx` profile | 86.4 | 35.6 | |
| NRC1 | + row caps 4,4,2,1,1 | 88.3 | 36.6 | 0.1006 |
</details>

<details>
<summary>Settings (engine/profiles/default.env)</summary>

Every setting is an environment variable; `engine/profiles/default.env` sets the ones below, and any of them can be
overridden when starting (`K31_X=... engine/k31.sh up`).

| setting | default | what it does |
|---|---|---|
| `DRAFT` (config.env), `DRAFTS` | the final DSpark drafter, 4 | the drafter and its draft tokens per step |
| `KV_BYTES`, `MAXLEN` | 16 GiB, 262,144 | KV cache size and the longest conversation (one request at full length) |
| `K31_HBM_MAP` | `hbm-map-256k-final.json` | experts kept in GPU memory per layer (9,390 in all) |
| `K31_BAND_N`, `K31_BAND_MAP` | 118, `band-map.json` | experts on the RTX per layer (10,856 in all); 0 turns the RTX off |
| `K31_BANK_MAP` | `bank-map-256k.json` | 2-bit copies in CPU memory per layer (41,661; `bank-map-nortx.json`: 45,000 without the RTX; `bank-map.json`: 36,800 for hosts with less free memory) |
| `K31_POLICY` | `policy-stage8.json` (`nortx`: `policy-nortx-stage4.json`; `fast`: `policy-joint15.json`; `fast-nortx`: `policy-nortx-t92-allrows.json`) | staged ranks and resident caps per layer (see [Policies](#policies)) |
| `K31_STAGE_ALL`, `K31_STAGE_ROWS` | on, 8,8,4,2,2 (`nortx`: 4,4,4,2,2; `fast`: none; `fast-nortx`: 4,4,2,1,1) | a staged expert serves all of its picks; per-row staging caps |
| `K31_STAGE_CAP`, `K31_STAGE_SP`, `K31_STAGE_PRIOR` | 6, 8, 1,0.69,0.43,0.24,0.14 (`nortx`: 6, 4; `fast`: none) | at most this many copies staged per call; the ranks the staging plan considers; slots ordered by each row's chance of being accepted |
| `K31_DECODE_SPLIT`, `K31_DECODE_JOINT` | on, 2 | several conversations: each one's rows keep the one-conversation budgets; shared dense parts and one MegaMoE launch for all of them (bit for bit the split's results) |
| `K31_BAND_PRIOR` | 1,0.69,0.43,0.24,0.14 | the band's slots by each verify row's chance of being accepted |
| `K31_PROMOTE`, `K31_BANK_PROMOTE` | 0.1, 0.08 | how fast GPU memory and the CPU-memory copies follow the conversation (per layer per token) |
| `K31_HAND_FEED`, `K31_CREDIT_WEIGHTS` | 1, `credit-weights.json` | band slots refilled from CPU memory; adaptation speed weighted per layer |
| `K31_CUT_MEAN` | `/means/k31-fmeans.safetensors` (`fast`, `fast-nortx`: `k31-expert-means.safetensors`) | the average substitute for skipped picks (empty: none) |
| `K31_DENSE_FMT` | int8 / FP8 / MXFP4 / W8A16 per layer kind | the dense layers' 8-bit forms |
| `K31_DSA_FUSED` | on | the sparse-attention layers' small decode kernels fused (bit-exact) |
| `K31_DRAFT_TOPP`, `K31_DRAFT_FULL_KV` | on, on | the request's top-p applied to the drafts; the drafter's KV as full attention |
| `K31_I8_BIG_M_KERNEL`, `K31_W8A16_BIG_M`, `K31_PROMPT_CELANE`, `K31_DSA_ROW_SLICE` | on | the prompt path's big-row kernels, copy-engine lane and row slices (runtime flags: `engine/k31_engine/flags.py`) |

Other `K31_*` switches in `engine/k31_engine` (deferral, the CPU-memory ring for band entries, other dense formats,
pre-staging, the Grace pool, timing and checks) are experiments left off by default; each is described where it is
read. The `nortx` profile (`engine/profiles/nortx.env`) changes the band, bank, map and policy settings above.
</details>

## Measuring KLD

KLD (Kullback-Leibler divergence) measures how far the engine's next-token probabilities are from the official
model's, over the whole vocabulary: 0 means identical. derpk31pro reports it the way the engine is actually used:

- **The text:** brandonmusic's quantization KLD protocol, rebuilt for Kimi K3.1: 128 windows of 8,192 tokens in 11
  kinds of text (agentic and tool use, chat, code, legal, math, multilingual, pretraining text, and more), cut into
  512 sequences of 2,048 tokens. The headline is over all 512; quick screens use 32 of them (one per four windows,
  every kind of text).
- **Read as served:** each sequence starts with a 64-token prompt, then the rest is fed one decode step at a time, so
  the expert tiers adapt at decode speed, as they would in a conversation. The reference text is fed through the
  engine's own speculative decoding: the official tokens are offered as drafts and accepted at the drafter's measured
  per-position rates, and the rejected drafts are the drafter's own guesses, as in serving.
- **Exact:** the server computes the KL divergence against the official model's full-vocabulary log-probabilities at
  every position (`engine/k31_engine/force.py`). The reference comes from the official MXFP4 checkpoint with 8-bit
  activations, run layer by layer by the reference runner (`tools/runner`).
- **Generated rows:** the headline counts the positions where a model produces text (assistant output and plain
  text), not the positions inside system, user or tool messages, which a model reads but never writes.
- **Long context:** 32 sequences of 32,768 tokens of the same kinds of text, read as a prompt up to their last 2,048
  to 4,096 tokens, which are then decoded the same way; KLD over the decoded rows, and separately over the
  generated-text rows inside the long prompt.
- **Prompts:** the same 32 screen sequences with 1,024-token prompts, the KL computed exactly on every prompt row as
  the prompt path reads it; we report the generated-text rows (`kld_generated_prompt_rows_exact`).

**How far to trust it.** We adapted brandonmusic's protocol to this engine, and we have doubts about how accurately it
measures what derpk31pro changes. KLD can only see the experts that the evaluation text routes to. The engine's
shortcuts land hardest on the experts that are picked least often: those are the ones most often served from 2-bit
copies, given their average or left out, and the ones whose placement rests on a usage ranking (see
[How experts are chosen for GPU memory](#how-experts-are-chosen-for-gpu-memory)). Text the protocol does not cover may
depend on exactly those experts: obscure Chinese is the example we worry about most, and the same goes for other rare
languages, scripts and specialised subjects. There the engine may be further from the official model than the
headline says. Moving the experts a conversation needs into GPU memory as it goes should limit the damage in a long
conversation, but we have not measured that. This is one reason the task benchmarks above are reported beside KLD.

<details>
<summary>Running a KLD screen</summary>

The evaluation set and the reference files come from the reference runner (`tools/runner/README.md`). Then:

```bash
SUBSET=rot32 HMODE=decode PROMPT=64 KLREF=/kl/ref-w4a8-rot32.npy KLD_FAST=1 \
  engine/harness/sweep.sh default "screen1 K31_FORCE_DRAFTS=1 K31_FORCE_REJECT=drafter"
```

`KLD_DATA` (default `$STATE/kld-data`) holds the evaluation set, `KLD_DIR` the reference files; results land in
`$STATE/results`. For the full 512-sequence headline use `KLHIDDEN=/kl/hidden-512.npy` instead of `SUBSET` and `KLREF`;
for prompt rows `PROMPT=1024`; for the long-context set `L32=1 L32_HIDDEN=/kl/l32/hidden-l32.npy
L32_REF=/kl/l32/kld.pt` (its reference comes from the reference runner's long-sequence pass).
</details>

## Measuring what the engine writes

KLD checks the engine on text that somebody else wrote, one conversation at a time. Real use is different: the engine
writes its own answer, each token built on the ones before it, usually while other requests are running. A mistake
there is not averaged away, because the rest of the answer continues from it. An early build of this engine scored
well on KLD and still, with four requests running at once, sometimes wrote a token the official model would never
have written, and then kept repeating itself until the answer ran out of room. KLD could not see it: once a model is
inside such a loop, the official model agrees that the next token is "more of the same".

So there is a second measurement that runs the other way round:

1. The engine answers a fixed set of 120 prompts (science questions, math, code, chat, Japanese, a tool-using agent,
   requests it should refuse), with the usual sampling settings, up to 1,024 tokens each. It does this once alone and
   once with four requests running at the same time.
2. The official model then reads every answer and reports, token by token, how likely it would have been to write
   that same token at that point.

Three numbers come out of it, each per in-flight count:

- **Distance from the official model.** On average, how much more likely the engine thought its own tokens were than
  the official model did. 0 means the two agree. This is the number to compare configurations by.
- **Near-impossible tokens per 1,000.** How often the engine wrote a token to which the official model gave less than
  a 1 in 1,000 chance. The official model, sampling the same way, almost never does this: less than once per 100,000
  tokens on science questions and up to about once per 1,200 tokens on open-ended chat, where many continuations are
  possible. The table shows that floor beside the engine's number.
- **Confident mistakes per 1,000.** Tokens the engine was more than 50% sure of while the official model gave them
  less than 1%. The official model cannot make this kind of mistake at all. In the looping answers we collected,
  these were six times as frequent right where a loop began as elsewhere, so this is the number to watch for loops.

The loops themselves are counted separately, on long answers (up to 32,768 tokens) with four requests at once, because
most loops start later than the first 1,024 tokens.

**How far to trust it.** These numbers say how closely the engine follows the official model, not whether an answer
is right; the task benchmarks do that. The two counts are small numbers: a healthy configuration writes about one
near-impossible token per 1,000, so one run of the 120 prompts can tell "twice as many" apart but not "10% more".
Most near-impossible tokens are harmless (an unusual word, a different digit); we saw roughly one loop per 1,300 of
them, and a few loops started with no such token at all. The prompts cover one language besides English and no long
documents. And the measurement depends on how many requests run at once: we report 1 and 4, the most this
configuration serves; more would need its own run.

<details>
<summary>Exact definitions, settings and files</summary>

- Prompt set `tail-v1` (`tools/onpolicy/sets/tail-v1.json`: ids, seeds and hashes; the text is rebuilt locally and
  verified): 48 GPQA Diamond questions as lil-bench sends them, and conversation openings from the Nemotron-SFT
  sets (12 math, 12 code, 12 chat, 12 agentic, 12 multilingual, 6 science, 6 safety) that no KLD window uses.
- Sampling: temperature 1, top-p 0.95, one fixed seed per prompt, `max_tokens` 1,024, 1 and 4 requests in flight
  (`tools/onpolicy/tail_capture.sh`). The server returns the exact token ids and its log-probability of each.
- Scoring (`tools/onpolicy/tail_score.sh`): the reference runner (`tools/runner`, the official MXFP4 checkpoint with
  8-bit activations) reads prompt + answer and gives the official probability `p` of every sampled token and its full
  distribution at every position.
  - distance ("gap") = mean of `log q - log p` over the sampled tokens, `q` = the engine's own probability: a
    one-sample estimate of the reverse KL divergence of the served engine against the official model.
  - near-impossible = `p` < 0.1% (the score file also has < 1% and < 0.01%, tokens outside the official top-p
    nucleus, and "excess surprise", which needs no engine probabilities). The floor is the exact rate at which the
    official model, sampling from its own top-p 0.95 nucleus at the same positions, writes a token under the same
    threshold (`tools/onpolicy/tailfloor.py`).
  - confident mistake = `q` >= 50% and `p` < 1%.
  - "the official model, for scale" = the official model in one arithmetic layout sampled, in another scored.
- Intervals: 95% bootstrap over prompts; two configurations are compared on the same prompts and seeds
  (`tools/onpolicy/tailscore.py compare`).
- Loops: `tools/onpolicy/capture.sh` with a 32,768-token cap at 4 in flight; `capture.py` flags a stretch of at least
  1,024 tokens that repeats with a fixed period.
- Full specification: `docs/onpolicy-tail.md`.
</details>

## Credits

derpk31pro builds on:

- [vLLM](https://github.com/vllm-project/vllm) (its `kimi-k31` image) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)'s
  MegaMoE kernel, with the MegaMoE changes of [Derpseek](https://github.com/chronarion/derpseek) and Garner McCloud's
  [dgx-station-recipes](https://github.com/garnermccloud/dgx-station-recipes).
- Tomas Cupr's [Kimi K3.1 Pro recipe](https://github.com/tomascupr/dgx-station-recipes): its layer-streaming
  `kstream.py` under the reference runner and its `kstrip.py` for the stripped checkpoint.
- brandonmusic's quantization KLD protocol, which the evaluation follows.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp), whose graph and utility headers the int8 GEMV uses.
- Moonshot AI's Kimi K3.1 Pro and Inferact's DSpark drafter.

Papers behind what ships:

- Speculative decoding: Leviathan, Kalman and Matias, [Fast Inference from Transformers via Speculative
  Decoding](https://arxiv.org/abs/2211.17192); Chen et al., [Accelerating Large Language Model Decoding with
  Speculative Sampling](https://arxiv.org/abs/2302.01318). Verifying the drafts as a block: Sun et al., [Block
  Verification Accelerates Speculative Decoding](https://arxiv.org/abs/2403.10444).
- Keeping the experts in use on the GPU and the rest elsewhere: Eliseev and Mazur, [Fast Inference of
  Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238).
- Computing less critical cache-miss experts at low precision: Tang et al., [HOBBIT: A Mixed Precision Expert
  Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433).
- Skipping a token's weakest picks: Lu et al., [Not All Experts are Equal: Efficient Expert Pruning and Skipping for
  Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2402.14800); Chen et al., [You Only Need 2/3 of the
  Chosen Experts: An Empirical Study of Dynamic Expert Pruning in Fine-Grained MoE LLMs](https://arxiv.org/abs/2609.25809).
- A cheap stand-in for a removed expert's output: Zhang et al., [MoNE: Replacing Redundant Experts with Lightweight
  Novices for Structured Pruning of MoE](https://arxiv.org/abs/2507.00390).
- Judging a compressed model by how far its token probabilities move, not only by accuracy: Dutta et al., [Accuracy
  is Not All You Need](https://arxiv.org/abs/2407.09141).

## License

MIT (see LICENSE); third-party parts are listed in NOTICE.
