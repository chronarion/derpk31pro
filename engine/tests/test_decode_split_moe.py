"""K31_DECODE_SPLIT through the hook's own KimiMoE wrapper (hook._patch_model with K31_SHARED_SIDE=3), on a GPU.

A stand-in model module (gate, latent down-projection on its own stream with the module's event pair, shared experts,
routed output transform, grouped top-k, an experts object that calls hook._forward) is patched as the engine patches
vLLM's; the stand-in MegaMoE returns, per row, the sum of (row id + 1) x weight, so a layer's output tells which row
every pick was given.
1. Eager decode-only steps of 10, 15, 16 and 20 rows: the layer's output equals, block by block, the reference built
   from the same stand-in parts on the block's rows alone with the reference plan (resident caps, stage cap, row
   caps, K31_STAGE_ALL); one MegaMoE call, one stager call and one latent down-projection per block; a batch of 5
   rows and an eager batch outside a decode-only step are one call.
2. Two layers captured into one CUDA graph at 20 rows (each layer: four blocks, each with its fork of the shared
   experts, its latent down-projection on the side stream with events of its own, plan, staging and join): replays
   on new inputs equal the eager reference; the accepted-row buffers hold the 20 rows of the step.
usage: test_decode_split_moe.py   (ENGINE=/repo/engine; GPU).  Exit 0 and "DECODE SPLIT MOE: PASS".
"""
import os
import random
import sys
import types

os.environ["K31_MODE"] = "staged"
os.environ["K31_DECODE_SPLIT"] = "1"
os.environ["K31_SHARED_SIDE"] = "3"
for k in ("K31_STAGE_ALL", "K31_DEFER", "K31_PROMPT_PATH", "K31_RES_MAXRANK", "K31_TAIL1", "K31_BAND", "K31_PDL",
          "K31_BAND_FIN_SIDE", "K31_STAGE_ROWS", "K31_MEGAMOE_NOPREP", "K31_PRESTAGE", "K31_LATE", "K31_POLICY",
          "K31_CUT_MEAN", "K31_STAGE_CAP", "K31_STAGE_PRIOR", "K31_DECODE_SPLIT_MAX", "K31_DECODE_SPLIT_ROWS",
          "K31_WINDOW_TEST", "K31_FUSE_UP", "K31_SHARED_EARLY", "K31_BAND_SIDE", "K31_GATE_SIDE"):
    os.environ.pop(k, None)

import torch  # noqa: E402

sys.path.insert(0, os.environ.get("ENGINE", os.environ.get("K31_ENGINE_DIR", "/engine")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k31_engine import hook, unitk  # noqa: E402

dev = torch.device("cuda")
E, K, SL, H, D = 896, 16, unitk.SLOTS, 512, 128
R = hook.PEND_ROWS
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def ranks(wrow):
    return [sum(1 for j in range(K) if wrow[j] > wrow[k] or (wrow[j] == wrow[k] and j < k)) for k in range(K)]


def plan_ref(st, ids, w, strong, S, resmax, allrows, cap, rowcap):
    """The one-round plan of a call from its definition (see test_decode_split.py) -> (row ids, stage list)."""
    row, bidx = st.row, st.bank_idx
    T = len(ids)
    rk = [ranks(w[t]) for t in range(T)]
    cand = [[row[ids[t][k]] < 0 and bidx[ids[t][k]] >= 0 and rk[t][k] < strong and
             (rowcap is None or rk[t][k] < rowcap[t]) for k in range(K)] for t in range(T)]
    best = {}
    for t in range(T):
        for k in range(K):
            if cand[t][k]:
                best[ids[t][k]] = max(best.get(ids[t][k], -1.0), w[t][k])
    order = sorted(best, key=lambda e: (-best[e], e))[:cap]
    slot_of = {e: s for s, e in enumerate(order)}
    out = [[-1] * K for _ in range(T)]
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            if row[e] >= 0:
                out[t][k] = row[e] if rk[t][k] < resmax else -1
            elif e in slot_of and (cand[t][k] or allrows):
                out[t][k] = S + slot_of[e]
    return out, [bidx[e] for e in order] + [-1] * (SL - len(order))


def fgt(hidden_states, gating_output, topk, renormalize, e_score_correction_bias, num_expert_group, topk_group,
        scoring_func, routed_scaling_factor):
    s = torch.sigmoid(gating_output.float()) + e_score_correction_bias
    v, i = s.topk(topk, dim=1)
    return (v / v.sum(1, keepdim=True)).float().contiguous(), i.int().contiguous()


n_mega, n_stage, n_down = [0], [0], [0]


def fake_megamoe(self, hidden_states, topk_weights, ids, shared_experts=None):
    n_mega[0] += 1
    T = hidden_states.shape[0]
    return ((ids.float() + 1.0) * topk_weights).sum(1, keepdim=True).expand(T, D).to(torch.bfloat16).contiguous()


class Gate:
    def __init__(self, g):
        self.w = (torch.randn(E, H, generator=g) * 0.05).to(dev)
        self.e_score_correction_bias = types.SimpleNamespace(data=(torch.rand(E, generator=g) * 0.01).to(dev))

    def __call__(self, h):
        return h.float() @ self.w.t(), None


class Experts:
    top_k = K

    def __init__(self, st):
        self._k31 = st

    def __call__(self, routed_h, w, ids):
        return hook._forward(self, routed_h, w, ids)


class KimiRoutedOutputTransform:
    def forward(self, final, residual=None):
        return residual + final[:, :1].to(residual.dtype)

    __call__ = forward


class KimiMoE:
    use_mega_moe = True
    moe_renormalize, num_expert_group, topk_group, moe_router_activation_func, routed_scaling_factor = True, 1, 1, "sigmoid", 1.0

    def __init__(self, st, g):
        self.experts = Experts(st)
        self.gate = Gate(g)
        self.routed_output_transform = KimiRoutedOutputTransform()
        self._down_proj_stream = torch.cuda.Stream(device=dev)
        self._down_proj_events = (torch.cuda.Event(), torch.cuda.Event())

    def shared_experts(self, h):
        return h * 0.5

    def routed_expert_down_proj(self, h):
        n_down[0] += 1
        return h[:, :D].contiguous(), None

    def _maybe_overlap_router_and_down_proj(self, h):        # vLLM's joined form (calls of more than 16 rows)
        logits, _ = self.gate(h)
        w, ids = fgt(h, logits, K, True, self.gate.e_score_correction_bias.data, 1, 1, "sigmoid", 1.0)
        return self.routed_expert_down_proj(h)[0], w, ids

    def forward(self, hidden_states):                         # vLLM's own forward: never taken with K31_SHARED_SIDE
        raise AssertionError("the unpatched forward ran")


class KimiDecoderLayer:
    def forward(self, positions, hidden_states, residual, prefix_sum=None, **kw):
        return hidden_states


module = types.SimpleNamespace(KimiMoE=KimiMoE, KimiDecoderLayer=KimiDecoderLayer,
                               KimiRoutedOutputTransform=KimiRoutedOutputTransform, fused_grouped_topk=fgt,
                               attn_res=None)
assert hook.DECODE_SPLIT and hook.SHARED_SIDE_MODE == 3 and not hook.DEFER
hook._patch_model(module)
hook._orig["forward"] = fake_megamoe
P, block = 5, 5
hook._SPLIT["block"], hook._SPLIT["P"], hook._SPLIT["rows"] = block, P, 8


def make_layer(rng, g, idx, strong, resmax, cap):
    perm = list(range(E))
    rng.shuffle(perm)
    st = hook.LayerState(idx, perm[:rng.randint(80, 220)], perm[250:250 + rng.randint(250, 500)])
    st.row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
    st.bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
    st.arange = torch.arange(max(SL, K), dtype=torch.long, device=dev)
    st.band_map, st.cut_mean, st.strong, st.resmax, st.stage_cap = None, None, strong, resmax, cap
    st.score = torch.rand(E, device=dev)
    st.stats = torch.zeros(4, dtype=torch.int32, device=dev)
    st.pend = (torch.full((R, 16), -1, dtype=torch.int32, device=dev), torch.zeros((R, 16), dtype=torch.float32, device=dev),
               torch.zeros(1, dtype=torch.int32, device=dev))
    st.slog = torch.full((8, SL), -7, dtype=torch.int32, device=dev)
    st.ns = 0

    def stager(ptr, stage, st=st):
        st.slog[st.ns].copy_(stage)
        st.ns += 1
        n_stage[0] += 1
    st.stager = stager
    return st, KimiMoE(st, g)


def reference(moe, x, allrows, rc, blocks):
    """The layer on x, block by block, from the stand-in parts and the reference plan -> (y, stage lists)."""
    st = moe.experts._k31
    ys, stages = [], []
    for a, b in blocks:
        h = x[a:b]
        logits, _ = moe.gate(h)
        w, ids = fgt(h, logits, K, True, moe.gate.e_score_correction_bias.data, 1, 1, "sigmoid", 1.0)
        wi, ws = plan_ref(st, ids.tolist(), w.tolist(), st.strong, st.slot_base, st.resmax, allrows,
                          st.stage_cap or SL, rc)
        rid = torch.tensor(wi, dtype=torch.int32, device=dev)
        u = ((rid.float() + 1.0) * w).sum(1, keepdim=True).expand(b - a, D).to(torch.bfloat16)
        ys.append(h * 0.5 + u[:, :1])
        stages.append(ws)
    return torch.cat(ys), stages


# 1. eager decode-only steps
rng = random.Random(3)
g = torch.Generator().manual_seed(3)
bad, n_blocks = 0, 0
for trial in range(48):
    T = [10, 15, 16, 20, 5, 20][trial % 6]
    allrows = trial % 2 == 0
    hook.STAGE_ALL = unitk.STAGE_ALL = allrows
    rc = [[4, 4, 2, 1, 1][t % P] for t in range(16)] if trial % 3 == 0 else None
    hook._RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev) if rc else None
    # (an eager call of 5 rows that is not a block keeps today's rule: one round only if rows x ranks fit the slots)
    st, moe = make_layer(rng, g, rng.randint(1, 92), rng.randint(1, 3 if T == 5 else 4), rng.choice([16, 8]),
                         rng.choice([None, 3, 6]))
    x = torch.randn(T, H, generator=g).to(torch.bfloat16).to(dev)
    decode = trial % 6 != 5                                   # the sixth: an eager batch that is not a decode-only step
    n_mega[0] = n_stage[0] = n_down[0] = 0
    hook._SPLIT["decode"] = decode
    if T == 20 and not decode:                                # goes to the prompt path / legacy rounds: only its shape
        hook._SPLIT["decode"] = False
        ok = not hook._split_wanted(x)
        bad += not ok
        continue
    y = moe.forward(x)
    hook._SPLIT["decode"] = False
    blocks = hook.split_blocks(T, block, 8)
    want, stages = reference(moe, x, allrows, rc, blocks)
    ok = len(blocks) == (1 if T == 5 else len(blocks)) and n_mega[0] == n_stage[0] == n_down[0] == len(blocks)
    ok = ok and y.shape == (T, H) and torch.equal(y, want)
    ok = ok and [st.slog[c].tolist() for c in range(len(blocks))] == stages
    ok = ok and int(st.pend[2][0]) == T and not hook._SPLIT["sub"]
    n_blocks += len(blocks)
    bad += not ok
torch.cuda.synchronize()
check("eager decode-only steps through the patched KimiMoE forward: every block's rows planned alone (output = the "
      "reference of its rows; stage lists equal), one MegaMoE / stager / latent-down call per block",
      bad == 0, f"({n_blocks} blocks; {bad} batches wrong)")

# 2. two layers in one CUDA graph at 20 rows
T = 20
hook.STAGE_ALL = unitk.STAGE_ALL = True
rc = [[4, 4, 2, 1, 1][t % P] for t in range(16)]
hook._RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev)
st1, moe1 = make_layer(rng, g, 11, 4, 16, 6)
st2, moe2 = make_layer(rng, g, 12, 3, 8, None)
x = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
x.copy_(torch.randn(T, H, generator=g).to(torch.bfloat16))
hook._SPLIT["decode"] = True
moe2.forward(moe1.forward(x))                                 # compile outside the capture
hook._SPLIT["decode"] = False
torch.cuda.synchronize()
st1.ns = st2.ns = 0
n_mega[0] = n_stage[0] = n_down[0] = 0
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y1 = moe1.forward(x)
    y2 = moe2.forward(y1)
captured = (n_mega[0], n_stage[0], n_down[0])
blocks = hook.split_blocks(T, block, 8)
bad = 0
for seed in range(1, 11):
    x.copy_(torch.randn(T, H, generator=torch.Generator().manual_seed(100 + seed)).to(torch.bfloat16))
    st1.pend[2].zero_()
    st2.pend[2].zero_()
    graph.replay()
    torch.cuda.synchronize()
    w1, s1 = reference(moe1, x, True, rc, blocks)
    w2, s2 = reference(moe2, w1, True, rc, blocks)
    ok = torch.equal(y1, w1) and torch.equal(y2, w2)
    ok = ok and [st1.slog[c].tolist() for c in range(4)] == s1 and [st2.slog[c].tolist() for c in range(4)] == s2
    ok = ok and int(st1.pend[2][0]) == T and int(st2.pend[2][0]) == T
    bad += not ok
hook.STAGE_ALL = unitk.STAGE_ALL = False
hook._RCAP[0] = None
check("CUDA graph: two layers of four blocks each (shared-experts fork, latent down on the side stream with its own "
      "events, plan, staging, join per block): 10 replays on new inputs equal the reference",
      bad == 0 and captured == (8, 8, 8), f"(calls captured: MegaMoE / stager / latent down {captured}; {bad} wrong)")

print("DECODE SPLIT MOE: " + ("PASS" if not fails else f"FAIL ({len(fails)}: {'; '.join(fails)})"))
sys.exit(1 if fails else 0)
