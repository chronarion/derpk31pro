# derpk31pro

Kimi K3.1 Pro on a single DGX Station GB300: about TBD tokens per second for one conversation, with next-token
probabilities within TBD (KLD) of the official model.

## What is derpk31pro

Kimi K3.1 Pro is a very large mixture-of-experts model: the official checkpoint is 1.56 TB, and every token is
computed by 16 of 896 experts in each of its 92 expert layers. A DGX Station GB300 has 252 GB of GPU memory, so most
of the model cannot sit on the GPU.

derpk31pro serves the model from one DGX Station anyway, by keeping each expert where it is cheapest to use:

- **GPU memory** holds about 9,700 experts in their official form: the ones the conversation uses most. Which ones
  changes as the conversation goes; experts move in from the drive in the background.
- **The RTX PRO 6000** (the Station's second card) holds about 11,000 more experts in a compact 2-bit form and computes
  them at the same time as the GB300 computes its own.
- **The CPU's memory** holds about 42,000 experts in 2-bit form. When a token strongly needs one of them, the GB300
  copies it in for that step.
- **Everything else is a trade we measure.** Not every pick matters equally: some layers barely notice when an
  expert is approximated, others carry the error forward, and a token's strongest picks carry most of its weight.
  derpk31pro's philosophy is to profile the impact of every shortcut, layer by layer and pick by pick, in KLD and in
  time, and to write the result down as a policy: which picks are computed on the RTX, which are copied in from CPU
  memory, and which get the expert's average output instead. Policies sit at different points between decode speed
  and KLD; several are included, and you can write your own (see [Policies](#policies)).

Everything is fed from one 4 TB NVMe drive. The engine runs inside vLLM's `kimi-k31` image and is built around
MegaMoE, the expert kernel from DeepGEMM, which computes a whole expert layer in one launch.

## How fast and how close

| profile | decode, 1 conversation | reading a prompt (8K / 16K / 32K tokens) | KLD vs the official model |
|---|---|---|---|
| `default` (GB300 + RTX PRO 6000) | TBD tokens/s | TBD tokens/s | TBD |
| `nortx` (GB300 alone) | TBD tokens/s | TBD tokens/s | TBD |

| benchmark (lil-bench) | derpk31pro (`default`) | for comparison |
|---|---|---|
| GPQA Diamond (`--test-profile gpqa-diamond`, all 198 questions) | TBD | 93.97 for the official MXFP4 checkpoint in NVIDIA's evaluation (their own protocol) |
| Tool calling (`--tool-eval`: tool-eval-bench, community leaderboard settings) | TBD | |

- **Speed:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) (lil-bench) 1.7.1,
  standard profile. Decode is its single-conversation cell (a chat prompt, 2,048 generated tokens); reading a prompt
  is the prompt's length divided by the time to the first token. Sampling is the model card's: temperature 1, top-p
  0.95, which is also the engine's default. The DSpark drafter proposes 4 tokens per step.
- **Benchmarks:** lil-bench's own GPQA Diamond and tool-calling runs, with their default settings, against the
  same server. They check what KLD cannot fully see (below).
- **KLD** (lower is closer): how far the next-token probabilities are from the official model's, measured on the
  tokens a model generates, over 512 text samples of 11 kinds of text, each read as the engine serves it (a short
  prompt, then one token at a time). See [Measuring KLD](#measuring-kld).

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
- The official checkpoint (`moonshotai/Kimi-K3.1-Pro`) and the DSpark drafter (`Inferact/Kimi-K3.1-Pro-DSpark-preview`).

## Getting started

```bash
cp config.example.env config.env     # then set the paths in it
./build.sh                           # builds the engine's and the sidecar's GPU libraries inside the vLLM image
docker build -t derpk31pro-runner tools/runner/docker
engine/store/run_kstrip.sh           # the stripped checkpoint (everything but the experts), ~107 GB
engine/store/run_build_store.sh      # the 2-bit records of every expert, ~1.4 TB
engine/k31.sh up                     # starts the server on 127.0.0.1:8113 (OpenAI-compatible API)
```

Without an RTX PRO 6000, start the `nortx` profile instead: `engine/k31.sh up nortx`. The first start takes several minutes: it reads the resident experts and 388 GB of 2-bit records into memory and
compiles kernels. `engine/k31.sh down` stops it; `engine/k31.sh logs` shows the log.

<details>
<summary>The average-output tables</summary>

Skipped picks are replaced by each expert's average output, from tables measured on held-out text and published on
Hugging Face ([chronarion/derpk31pro-means](https://huggingface.co/chronarion/derpk31pro-means), 591 MB):

```bash
hf download chronarion/derpk31pro-means k31-expert-means.safetensors --local-dir /models/k31-means
```

and set `MEANS_DIR=/models/k31-means` in `config.env`. Without them, start with `K31_CUT_MEAN=` (empty): skipped
picks then add nothing, which raises KLD. They can be rebuilt with the reference runner's means pass
(`tools/runner`, `--save-means`).
</details>

## How it works

Each of the model's 92 expert layers picks 16 of its 896 experts for every token, ranked by weight. Where each pick
is computed depends on where that expert currently is. The default profile places the 82,432 experts like this:

| Where | Experts | Per layer | Form | How a pick of it is computed |
|---|---:|---:|---|---|
| GPU memory | 9,980 (12%) | 56 to 378 | Moonshot's own MXFP4 | exactly as shipped, in one MegaMoE launch with the layer's other picks |
| RTX PRO 6000 | 10,856 (13%) | 48 to 216 | 2-bit copies | on the RTX, while the GB300 runs MegaMoE; the GB300 adds the results in the kernel that finishes the layer |
| CPU memory | 41,661 (51%) | 200 to 640 | 2-bit copies (387 GB) | copied into one of 128 shared staging rows on the GB300 and computed in the same MegaMoE launch |
| Nowhere in memory | 19,935 (24%) | 0 to 592 | | skipped: the expert's average output, scaled by the pick's weight, stands in for it |

Layers whose output matters more get more GPU memory (`engine/profiles/hbm-map-256k.json`, `band-map.json`,
`bank-map-256k.json`). The GPU's share is sized to fill what the KV cache for 262,144 tokens, the prompt chunks and a
safety margin leave.

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
about 2.6 tokens on average.

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
| deferred (experimental) | computed exactly, one layer late | exact but late; not used by the included policies |

Staging more picks brings the engine closer to the official model and costs time; staging fewer is faster. Because
the measurements say where a microsecond buys the most KLD, a policy can aim at a speed or at a KLD, and every layer
gets its own rule: the least sensitive layers stage nothing, the most sensitive stage a token's four strongest picks.
Several policies are included, and we keep trying new ones:

| policy | hardware | what it stages | trade (simulated, against `policy-default`) |
|---|---|---|---|
| `policy-default.json` | GB300 + RTX | the strongest pick in most layers, the 2 strongest in layers 74, 76-80 and 83, the 4 strongest in 52-59 and 92, nothing in the 16 least sensitive layers | |
| `policy-quality.json` | GB300 + RTX | more | about 0.5 ms more per forward pass, about 4.9% lower KLD |
| `policy-balanced.json` | GB300 + RTX | as much as the default, placed by KLD per staged copy | same time, about 1.4% lower KLD |
| `policy-fast.json` | GB300 + RTX | less | about 0.6 ms less per forward pass, about 4.6% higher KLD |
| `policy-nortx.json` | GB300 alone | the 4 strongest picks in layers 1-11 and 48-92, the strongest in 12-47 | |
| `policy-nortx-fast.json` | GB300 alone | the 2 strongest in layers 1-11 and 48-92, the strongest in 12-47 | faster than `policy-nortx`, higher KLD |
| `policy-nortx-t92.json` | GB300 alone | placed by KLD per microsecond of staging, for the GB300-alone speed target | |

The RTX policies send every pick of an RTX expert to the RTX (at most 8 experts per call) and give the average to
the picks that are not staged; the GB300-alone policies have no RTX. Choose a policy with `K31_POLICY`, or start a
profile that sets one (`engine/k31.sh up nortx`, `nortx-fast`, `nortx-t92`). The file format is in
`engine/POLICY.md`; the trades above are from the simulator, and measured ones will replace them.

## What made it fast, and what kept it close

The engine was built one change at a time, each measured on its own run where we could: decode speed on the bench
(median of 10 runs; it moves 3-5% between runs because the drafter's acceptance does), from late on also the time of
one forward pass in milliseconds (steadier), and KLD on the 32-sequence screen ("rot32"). Several changes went in
together; where a number covers a bundle, it says so. The KLD screen was made stricter part of the way (drafts accepted
at the drafter's real rates instead of always): the same engine reads about 10% higher under it, so KLD numbers are
compared only within one protocol.

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
</details>

<details>
<summary>Settings (engine/profiles/default.env)</summary>

Every setting is an environment variable; `engine/profiles/default.env` sets the ones below, and any of them can be
overridden when starting (`K31_X=... engine/k31.sh up`).

| setting | default | what it does |
|---|---|---|
| `K31_HBM_MAP` | `hbm-map-256k.json` | experts kept in GPU memory per layer (9,980 in all; `hbm-map.json`: 9,660, for a 65,536-token context) |
| `K31_BAND_N`, `K31_BAND_MAP` | 118, `band-map.json` | experts on the RTX per layer (10,856 in all); 0 turns the RTX off |
| `K31_BANK_MAP` | `bank-map-256k.json` | 2-bit copies in CPU memory per layer (41,661; `bank-map-large.json`: 41,800 beside `hbm-map.json`; `bank-map.json`: 36,800 for hosts with less free memory) |
| `K31_POLICY` | `policy-default.json` | which ranks may be staged in each layer; `policy-fast.json`, `policy-balanced.json`, `policy-quality.json` trade speed and KLD |
| `K31_PROMOTE`, `K31_BANK_PROMOTE` | 0.1, 0.08 | how fast GPU memory and the CPU-memory copies follow the conversation (per layer per token) |
| `K31_HAND_FEED`, `K31_CREDIT_WEIGHTS` | 1, `credit-weights.json` | band slots refilled from CPU memory; adaptation speed weighted per layer |
| `K31_CUT_MEAN` | `/means/k31-expert-means.safetensors` | the average substitute for skipped picks (empty: none) |
| `K31_DENSE_FMT` | int8 / FP8 / MXFP4 / W8A16 per layer kind | the dense layers' 8-bit forms |
| `DRAFTS` | 4 | draft tokens per step (DSpark drafter) |
| `KV_BYTES`, `MAXLEN` | 8 GiB, 65,536 | KV cache size and the longest conversation |

Other `K31_*` switches in `engine/k31_engine` (deferral, the CPU-memory ring for band entries, other dense formats,
timing and checks) are experiments left off by default; each is described where it is read.
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
`$STATE/results`. For the full 512-sequence headline use `KLHIDDEN=/kl/hidden-512.npy` instead of `SUBSET` and `KLREF`.
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
  Speculative Sampling](https://arxiv.org/abs/2302.01318).
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
