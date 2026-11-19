# Reference runner

A layer-streaming runner for the official Kimi K3.1 Pro checkpoint: it reads one layer at a time from the drive and
runs Moonshot AI's own modeling code (`third_party/moonshot-k3-ref`) on one GPU, so the published model's outputs can
be computed on a machine that cannot hold it. It builds on Tomas Cupr's `kstream.py` (`third_party/tomcupr`, MIT).

derpk31pro uses it for three data files, each one pass over the KLD evaluation set with the official model
(`universes/reference.json`: the official MXFP4 weights with 8-bit activations, as the checkpoint is meant to run):

| file | what it is for | pass option |
|---|---|---|
| reference log-probabilities of the 32 screen cuts (~21 GB) | exact KLD on quick screens (`KLREF`) | `--dump-logits` |
| final hidden states of all 512 cuts (~15 GB) | exact KLD on the full set; the server rebuilds the reference logits with the model's head (`KLHIDDEN`) | `--dump-hidden` |
| each expert's mean output per layer (~565 MB) | the average substitute for skipped picks (`MEANS_DIR`) | `--save-means` (on held-out text) |

<details>
<summary>Building the evaluation set and running a pass</summary>

The evaluation set follows brandonmusic's quantization KLD protocol (128 windows of 8,192 tokens in 11 kinds of text,
cut into 512 sequences of 2,048 tokens): `tools/bm_eval.py WINDOWS_JSON SRC TOKENIZER OUT` takes the protocol's
segment list and renders it with K3.1's tokenizer; `tools/role_labels.py` labels each row (generated text or not).

A pass (one GPU, the official checkpoint from `NATIVE` in `config.env`):

```bash
tools/runner/run_pass.sh NAME GPU_UUID TOKENS universes/reference.json OUT --dump-logits /scratch/OUT/ref.npy
```

`TOKENS` is the evaluation set (`.pt`). A full pass streams the 1.56 TB checkpoint once per batch of sequences and
takes hours.
</details>
