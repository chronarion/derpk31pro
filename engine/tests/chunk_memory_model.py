"""Memory model of one prompt chunk of T rows (Kimi K3.1 Pro in the kimi-k31 image), for the 8,192-row chunk: which
stage of the forward peaks, with and without K31_DSA_ROW_SLICE, and the persistent buffers that grow with
max_num_batched_tokens. Pure Python: `python3 -I chunk_memory_model.py [T ...]`.

Bytes per row are counted from vLLM's source (the image's files; r14/out/vllm-notes.txt has most file:line refs):
the transients of each stage plus what stays alive across layers (the residual stream). Measured points used:
  - MegaMoE's symmetric buffer (506 expert rows): 883 / 1,232 / 1,563 / 2,243 MiB at max_num_tokens 512 / 2,048 /
    4,096 / 8,192 (start-up logs, 11-18);
  - "Model loading took" 56.63 / 56.85 / 57.29 GiB at max_num_batched_tokens 2,048 / 4,096 / 8,192: buffers sized
    by the chunk (top-k indices, FlashKDA's workspace, drafter staging ...), ~0.107 MiB per row;
  - pp-D (4,096-row chunks, 11-18 21:14): nvidia-smi rose 3.82 GiB above the post-capture level during 4,096-row
    prompt chunks with 32K-token prompts. The model's transient peak for that case divided into it gives the factor
    from torch-allocated bytes to device memory (allocator rounding and caching, first-use workspaces).
"""
import sys

KB, MiB, GiB = 1000, 2 ** 20, 2 ** 30
HID = 7168 * 2                       # one [T, 7168] BF16 row

# what stays alive across layers (KimiLinearModel.forward; model.py:1240-1274)
BLOCK_RESIDUAL = 8 * HID             # AttnRes block states [T, 8, 7168] BF16, the whole forward
STREAM = 2 * HID                     # the previous layer's output (held by the caller) + prefix_sum
AUX = HID                            # per drafter aux state captured (after layers 75, 79, 83, 87, 91)

# stage transients, bytes per row
KDA = {"input (AttnRes output)": HID, "in_proj out [T, 49,376]": 49376 * 2, "g1 = f_b_proj [T, 12,288]": 12288 * 2,
       "core_attn_out [T, 96, 128]": 12288 * 2, "conv q, k, v [3, T, 12,288]": 3 * 12288 * 2}
DSA_FRONT = {"input": HID, "qkv_a [T, 2,112]": 2112 * 2, "q_c [T, 1,536]": 1536 * 2, "kv_c_normed [T, 512]": 512 * 2,
             "q [T, 96, 192]": 96 * 192 * 2}
DSA_INDEXER = {"index_q [T, 32, 128]": 4096 * 2, "q fp8": 4096, "weights fp32": 32 * 4}
INDEXER_LOGITS = 512 * MiB // 4 * (2 + 4)    # BF16 logits + FP32 copy, capped by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
DSA_OUT = 12288 * 2                            # forward_impl's output [T, 96 x 128] (allocated by MLAAttention.forward)
DSA_CORE = {"latent query [T, 96, 576] BF16": 96 * 576 * 2, "FP8 query [T, 96, 576]": 96 * 576,
            "FP8 query padded to 128 heads": 128 * 576, "latent output [T, 128, 512] BF16": 128 * 512 * 2,
            "physical top-k [T, 2,048] int32": 2048 * 4 + 4}
DSA_GATE = {"g_proj out": 12288 * 2, "sigmoid": 12288 * 2, "gated output": 12288 * 2}
MOE = {"input": HID, "router logits fp32": 896 * 4, "latent rows [T, 3,584]": 3584 * 2, "plan [T, 16] x ~8": 8 * 128,
       "MegaMoE out + round sum": 2 * 3584 * 2, "shared gate_up [T, 12,288]": 12288 * 2, "shared act": 6144 * 2,
       "fp8 act": 6144}
MLP0 = {"input": HID, "gate_up [T, 67,584]": 67584 * 2, "act [T, 33,792]": 33792 * 2, "fp8 act": 33792,
        "down out": HID}
DRAFT = {"last hidden": HID, "aux concat [T, 35,840]": 5 * HID, "context_proj out": HID, "context_norm out": HID}
MEGAMOE_BUF = ((512, 883), (2048, 1232), (4096, 1563), (8192, 2243))     # (max_num_tokens, MiB), measured
LOAD_PER_ROW = (57.29 - 56.85) * GiB / 4096                               # persistent bytes per max_num_batched_tokens
PPD_GROWTH = 3.82                                                          # GiB, nvidia-smi over post-capture, pp-D


def megamoe_mib(rows):
    pts = MEGAMOE_BUF
    for (a, ma), (b, mb) in zip(pts, pts[1:]):
        if rows <= b:
            return ma + (mb - ma) * (rows - a) / (b - a)
    (a, ma), (b, mb) = pts[-2], pts[-1]
    return mb + (mb - ma) * (rows - b) / (b - a)


def stages(T, slice_rows=0, P=None):
    """Peak bytes (torch-allocated, over the step's start) of each stage at T rows; P = keys of the context."""
    P = P or 4 * T
    late = BLOCK_RESIDUAL + STREAM                  # every layer
    s = {}
    s["KDA attention (layer 90: 4 aux states)"] = T * (late + 4 * AUX + sum(KDA.values()))
    front = sum(DSA_FRONT.values())
    logits = min(INDEXER_LOGITS, T * P * 6)
    s["DSA indexer (layer 92: 5 aux)"] = T * (late + 5 * AUX + front + sum(DSA_INDEXER.values())) + logits
    core = sum(DSA_CORE.values())
    rows = T if slice_rows <= 0 or T <= slice_rows else -(-T // -(-T // slice_rows))
    s["DSA sparse attention core (layer 92)"] = T * (late + 5 * AUX + front + DSA_OUT) + rows * core
    s["DSA output gate (layer 92)"] = T * (late + 5 * AUX + front + DSA_OUT + sum(DSA_GATE.values()))
    s["MoE + shared experts (layer 92)"] = T * (late + 5 * AUX + sum(MOE.values()))
    s["layer 0 dense MLP"] = T * (late + sum(MLP0.values()))
    s["drafter context store (after the forward)"] = T * (5 * AUX + sum(DRAFT.values()))
    return s


def report(T, P=None):
    print(f"\n=== prompt chunk of {T:,} rows (context {P or 4 * T:,} keys for the indexer) ===")
    print(f"per row: AttnRes block states {BLOCK_RESIDUAL / KB:.1f} KB, residual stream {STREAM / KB:.1f}, drafter aux "
          f"{AUX / KB:.1f} each; DSA core transient {sum(DSA_CORE.values()) / KB:.1f} KB; KDA layer "
          f"{sum(KDA.values()) / KB:.1f}; layer-0 MLP {sum(MLP0.values()) / KB:.1f}")
    base = None
    for sl in (0, 2048, 1024):
        st = stages(T, sl, P)
        top = max(st, key=st.get)
        tag = "unsliced" if sl == 0 else f"K31_DSA_ROW_SLICE={sl}"
        print(f"  {tag}: peak {st[top] / GiB:.2f} GiB at {top}")
        for k, v in sorted(st.items(), key=lambda kv: -kv[1]):
            print(f"      {v / GiB:6.2f}  {k}")
        if sl == 0:
            base = st[top]
    return base


def main():
    Ts = [int(a) for a in sys.argv[1:]] or [4096, 8192]
    peaks = {T: report(T) for T in Ts}
    ref = stages(4096, 0, 32768)
    f = PPD_GROWTH / (max(ref.values()) / GiB)
    print(f"\ncalibration: pp-D (4,096 rows, 32K prompts) grew {PPD_GROWTH} GiB on nvidia-smi; the model's peak there "
          f"is {max(ref.values()) / GiB:.2f} GiB allocated: factor {f:.2f}")
    print("\npredicted device memory against today's 4,096-row default (8 GiB KV run of pp-D as the reference):")
    print(f"{'chunk':>7} {'DSA slice':>10} {'MegaMoE rows':>13} {'buffers':>8} {'transient':>10} "
          f"{'total vs 4,096':>15}")
    for T in sorted(set(Ts) | {4096, 8192}):
        for sl in (0, 2048):
            for cap in (0, 4096):
                if cap and cap >= T:
                    continue
                pk = max(stages(T, sl, 4 * T if T > 4096 else 32768).values()) / GiB * f
                buf = (megamoe_mib(min(T, cap) if cap else T) - megamoe_mib(4096)) * MiB / GiB + \
                      (T - 4096) * LOAD_PER_ROW / GiB
                tot = buf + pk - PPD_GROWTH
                print(f"{T:7,} {sl if sl else '-':>10} {cap if cap else '-':>13} {buf:+8.2f} {pk:10.2f} {tot:+15.2f}")


if __name__ == "__main__":
    main()
