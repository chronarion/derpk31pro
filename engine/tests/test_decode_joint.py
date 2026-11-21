"""K31_DECODE_JOINT through the hook's own KimiMoE wrapper (hook._patch_model with K31_SHARED_SIDE=3), on a GPU.

The joint path only reorders work: every verify block keeps its plan, its staged records and its band call. So the
test is equality with the split path (K31_DECODE_JOINT=0) on the same layer and rows, bit for bit, with stand-ins
whose result per row does not depend on the other rows of a call:
  - MegaMoE: out[t] = sum_k w[t, k] x value[row id], where a resident row has a fixed value and a staging row has the
    value of the record the stager stand-in last wrote into it (so a block reading another block's slot, or a slot
    overwritten before the call, changes the output);
  - stagers (the layer's own and one per slot group): write the staged records' values into the shared slot rows and
    log their stage lists;
  - the RTX band: a 4-slot mailbox and the one-tag state block of the fused kernels (send: the block's band rows into
    slot seq % 4, the tag into the state; finish: the rows of the state's tag + the mean sum added in one rounding,
    and the RMSNorm statistic), so a finish that reads another block's call changes the output;
  - the real unitk plan and the real cut-mean kernels.
1. dense_chunks: whole blocks, at most 16 rows, balanced.
2. No band (the RTX-less profile's shape), eager decode-only steps of 10, 15, 16, 20 and 40 rows, mean substitute on
   and off: levels 1 and 2 equal the split bit for bit; the same stage lists in the same order; MegaMoE calls B / B /
   1; one router and one latent down-projection per dense chunk; the accepted-row buffers hold the step's rows.
3. The same with the band and the fused tail (K31_TAIL1): levels 1 and 2 equal the split; a batch of more blocks
   than the mailbox has slots takes level 1.
4. Mutations: records staged into another slot group than the plan addresses, and finishes that all read the last
   call's tag, must change the output.
5. Two layers captured into one CUDA graph at 20 and at 10 rows, level 2, with and without the band: replays on new
   inputs equal the split run eagerly on the same inputs.
6. K31_DECODE_JOINT_CHECK: the in-engine check reports bit-equal outputs here.
usage: test_decode_joint.py   (ENGINE=/repo/engine; GPU).  Exit 0 and "DECODE JOINT: PASS".
"""
import os
import random
import sys
import types

os.environ["K31_MODE"] = "staged"
os.environ["K31_DECODE_SPLIT"] = "1"
os.environ["K31_SHARED_SIDE"] = "3"
os.environ["K31_BAND_FIN_SIDE"] = "1"          # the routed output transform is patched (the tail's statistic)
os.environ["K31_STAGE_ALL"] = "1"
for k in ("K31_DEFER", "K31_PROMPT_PATH", "K31_RES_MAXRANK", "K31_TAIL1", "K31_BAND", "K31_PDL", "K31_STAGE_ROWS",
          "K31_MEGAMOE_NOPREP", "K31_PRESTAGE", "K31_LATE", "K31_POLICY", "K31_CUT_MEAN", "K31_STAGE_CAP",
          "K31_STAGE_PRIOR", "K31_DECODE_SPLIT_MAX", "K31_DECODE_SPLIT_ROWS", "K31_WINDOW_TEST", "K31_FUSE_UP",
          "K31_SHARED_EARLY", "K31_BAND_SIDE", "K31_GATE_SIDE", "K31_DECODE_JOINT", "K31_DECODE_JOINT_CHECK",
          "K31_DECODE_JOINT_DENSE", "K31_BAND_HINT", "K31_BAND_SERVED", "K31_STAGE_SP"):
    os.environ.pop(k, None)

import torch  # noqa: E402

sys.path.insert(0, os.environ.get("ENGINE", os.environ.get("K31_ENGINE_DIR", "/engine")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k31_engine import hook, unitk, w8a16  # noqa: E402

dev = torch.device("cuda")
E, K, SL, H, D = 896, 16, unitk.SLOTS, 512, 128
R = hook.PEND_ROWS
G = hook.GROUPS
fails = []
logs = []
hook._LOG = lambda m: logs.append(m)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------------------------------------ 1. dense chunks
def chunks_ok(T, block=5, rows=8, dense=16):
    blocks = hook.split_blocks(T, block, rows)
    ch = hook.dense_chunks(blocks, dense)
    flat = [b for _a, _b, bl in ch for b in bl]
    sizes = [cb - ca for ca, cb, _ in ch]
    return flat == blocks and all(ca == bl[0][0] and cb == bl[-1][1] for ca, cb, bl in ch) and \
        all(s <= dense or len(bl) == 1 for s, (_a, _b, bl) in zip(sizes, ch)), sizes


c_ok = all(chunks_ok(T)[0] for T in range(9, 41))
sizes = {T: chunks_ok(T)[1] for T in (10, 15, 16, 20, 24, 32, 40)}
check("dense_chunks: whole blocks in order, at most 16 rows per chunk, for batches of 9..40 rows",
      c_ok and sizes[10] == [10] and sizes[15] == [15] and sizes[16] == [16] and sizes[20] == [10, 10]
      and sizes[40] == [15, 15, 10] and chunks_ok(20, dense=8)[1] == [5, 5, 5, 5] and chunks_ok(20, dense=40)[1] == [20],
      f"({sizes})")


# ------------------------------------------------------------------------------------------------ stand-ins
def fgt(hidden_states, gating_output, topk, renormalize, e_score_correction_bias, num_expert_group, topk_group,
        scoring_func, routed_scaling_factor):
    s = torch.sigmoid(gating_output.float()) + e_score_correction_bias
    v, i = s.topk(topk, dim=1)
    return (v / v.sum(1, keepdim=True)).float().contiguous(), i.int().contiguous()


n_mega, n_stage, n_down, n_gate, n_send, n_fin = [0], [0], [0], [0], [0], [0]
slot_val = torch.zeros(SL * G, D, device=dev)                 # the shared staging rows (all layers)
mb = torch.zeros(4, 8, D, device=dev)                         # the band mailbox's result slots
gbst = torch.zeros(20, dtype=torch.int64, device=dev)         # the fused kernels' state block: [0] seq, [1] tag
hook._BAND["gb"] = types.SimpleNamespace(st=gbst)
hook._BAND["fused"] = object()
MUT = {"tag": False}


def fake_megamoe(self, hidden_states, topk_weights, ids, shared_experts=None):
    n_mega[0] += 1
    st = self._k31
    T = hidden_states.shape[0]
    assert ids.shape == (T, K) and topk_weights.shape == (T, K) and hidden_states.shape == (T, D)
    val = torch.cat([slot_val, st.res_val])[ids.long().clamp_min(0)]             # [T, K, D]
    u = (val * (topk_weights.float() * (ids >= 0))[:, :, None]).sum(1)
    return (u + hidden_states.float() * 0.25).to(torch.bfloat16).contiguous()


def fake_send(st, x, topk_ids, topk_weights, hinted=False, served=False):
    n_send[0] += 1
    T = x.shape[0]
    assert T <= 8 and x.is_contiguous() and x.dtype == torch.bfloat16 and topk_ids.shape == (T, K)
    e = topk_ids.long()
    bp = (st.row_map[e] < 0) & (st.band_map[e] >= 0)
    sv = getattr(st, "band_served", None) if served is False else served
    if sv is not None:
        sv[:T * K].copy_(bp.reshape(-1).int())
    rows = (st.band_val[e] * (topk_weights.float() * bp)[:, :, None]).sum(1) + x.float() * 0.125 * bp.any(1, keepdim=True)
    some = bp.any()
    seq = gbst[0] + some.long()                               # a call without band picks sends nothing
    slot = (seq % 4).view(1)
    pad = torch.zeros(8, D, device=dev)
    pad[:T] = rows
    mb.index_copy_(0, slot, torch.where(some, pad, mb.index_select(0, slot)[0])[None])
    gbst[0] = seq
    gbst[1] = torch.where(some, seq * 1000 + T, torch.zeros_like(seq))


def fake_finish(st, out, msum=None, rstd=None, eps=0.0):
    n_fin[0] += 1
    T = out.shape[0]
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    tag = (gbst[0] * 1000 + T) if MUT["tag"] else gbst[1]      # (mutation: always the last call sent)
    rows = mb.index_select(0, ((tag // 1000) % 4).view(1))[0, :T]
    res = out.float() + torch.where(tag != 0, rows, torch.zeros_like(rows))
    if msum is not None:
        res = res + msum[:T]
    out.copy_(res.to(torch.bfloat16))
    if rstd is not None:
        rstd[:T].copy_(torch.rsqrt(out.float().pow(2).mean(1) + eps))


def fake_lnr(h, w, s, nw, eps, residual, rstd=None):
    """w8a16.linear_norm_res: here the statistic scales the routed sum's first column."""
    r = rstd[:h.shape[0]] if rstd is not None else torch.rsqrt(h.float().pow(2).mean(1) + eps)
    return residual + (h[:, :1].float() * r[:, None]).to(residual.dtype)


hook._band_send, hook._band_finish = fake_send, fake_finish
w8a16.linear_norm_res = fake_lnr


class Gate:
    def __init__(self, g):
        self.w = (torch.randn(E, H, generator=g) * 0.05).to(dev)
        self.e_score_correction_bias = types.SimpleNamespace(data=(torch.rand(E, generator=g) * 0.01).to(dev))

    def __call__(self, h):
        n_gate[0] += 1
        return h.float() @ self.w.t(), None


class Experts:
    top_k = K

    def __init__(self, st):
        self._k31 = st

    def __call__(self, routed_h, w, ids):
        return hook._forward(self, routed_h, w, ids)


class KimiRoutedOutputTransform:
    def __init__(self):
        self.up_proj = types.SimpleNamespace(_k31_w8=(1, 1), bias=None)
        self.norm = types.SimpleNamespace(weight=types.SimpleNamespace(data=None), variance_epsilon=1e-6)

    def forward(self, final, residual=None):
        return residual + final[:, :1].to(residual.dtype)

    def __call__(self, final, residual=None):
        return type(self).forward(self, final, residual)


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

    def _maybe_overlap_router_and_down_proj(self, h):
        logits, _ = self.gate(h)
        w, ids = fgt(h, logits, K, True, self.gate.e_score_correction_bias.data, 1, 1, "sigmoid", 1.0)
        return self.routed_expert_down_proj(h)[0], w, ids

    def forward(self, hidden_states):
        raise AssertionError("the unpatched forward ran")


class KimiDecoderLayer:
    def forward(self, positions, hidden_states, residual, prefix_sum=None, **kw):
        return hidden_states


module = types.SimpleNamespace(KimiMoE=KimiMoE, KimiDecoderLayer=KimiDecoderLayer,
                               KimiRoutedOutputTransform=KimiRoutedOutputTransform, fused_grouped_topk=fgt,
                               attn_res=None)
assert hook.DECODE_SPLIT and hook.SHARED_SIDE_MODE == 3 and not hook.DEFER and hook.BAND_FIN_SIDE
hook._patch_model(module)
hook._orig["forward"] = fake_megamoe
P, block = 5, 5
hook._SPLIT["block"], hook._SPLIT["P"], hook._SPLIT["rows"] = block, P, 8
hook._JT["msum"] = torch.zeros(hook.SPLIT_MAX, D, dtype=torch.float32, device=dev)


def set_band(on: bool):
    hook.TAIL1 = hook.BAND_SIDE = hook.BAND_SERVED = on


def make_layer(rng, g, idx, strong, resmax, cap, band, mean):
    perm = list(range(E))
    rng.shuffle(perm)
    nh, nb = rng.randint(60, 160), rng.randint(200, 400)
    st = hook.LayerState(idx, perm[:nh], perm[300:300 + nb])
    st.row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
    st.bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
    st.arange = torch.arange(max(SL, K), dtype=torch.long, device=dev)
    st.cut_mean, st.strong, st.resmax, st.stage_cap = None, strong, resmax, cap
    st.band_map = None
    if band:                                                  # the band: experts 160..299 of the order (disjoint)
        bm = [-1] * E
        for s, e in enumerate(perm[160:300]):
            bm[e] = s
        st.band_map = torch.tensor(bm, dtype=torch.int32, device=dev)
        st.band_val = torch.randn(E, D, generator=g).to(dev)
        st.band_served = torch.zeros(16 * 16, dtype=torch.int32, device=dev)
        st.msum = torch.zeros(8, D, dtype=torch.float32, device=dev)
        st.rstd = torch.zeros(16, dtype=torch.float32, device=dev)
        st.band_stats = torch.zeros(3, dtype=torch.int32, device=dev)
        st.jserved = torch.zeros((hook.BAND_NSLOT, 16 * 16), dtype=torch.int32, device=dev)
        st.jtags = torch.zeros(hook.BAND_NSLOT, dtype=torch.int64, device=dev)
        st.jrstd = torch.zeros(R, dtype=torch.float32, device=dev)
    if mean:
        st.cut_mean = (torch.randn(E, D, generator=g) * 0.1).to(dev, torch.bfloat16).contiguous()
    st.res_val = torch.randn(nh, D, generator=g).to(dev)
    st.bank_val = torch.randn(nb, D, generator=g).to(dev)
    st.score = torch.rand(E, device=dev)
    st.stats = torch.zeros(4, dtype=torch.int32, device=dev)
    st.pend = (torch.full((R, 16), -1, dtype=torch.int32, device=dev), torch.zeros((R, 16), dtype=torch.float32, device=dev),
               torch.zeros(1, dtype=torch.int32, device=dev))
    st.slog = torch.full((32, SL), -7, dtype=torch.int32, device=dev)
    st.ns = 0

    def stager_for(first):
        def stager(ptr, stage, st=st, first=first):
            st.slog[st.ns].copy_(stage)
            st.ns += 1
            n_stage[0] += 1
            v = st.bank_val[stage.long().clamp_min(0)]
            slot_val[first:first + SL] = torch.where((stage >= 0)[:, None], v, slot_val[first:first + SL])
        return stager
    st.stager = stager_for(st.slot_base)
    st.jstagers = [stager_for(SL * q) for q in range(G)]
    moe = KimiMoE(st, g)
    hook._MOE[idx] = moe
    return st, moe


def run(moe, x, level, decode=True):
    """One eager batch at joint level `level` (0 = the split) -> (output, stage lists, call counts, pending rows)."""
    st = moe.experts._k31
    hook.DECODE_JOINT = level
    st.ns = 0
    st.slog.fill_(-7)
    st.pend[2].zero_()
    slot_val.normal_()                                        # stale slot rows differ from run to run
    for c in (n_mega, n_stage, n_down, n_gate, n_send, n_fin):
        c[0] = 0
    hook._SPLIT["decode"] = decode
    try:
        y = moe.forward(x).clone()
    finally:
        hook._SPLIT["decode"] = False
        hook.DECODE_JOINT = 0
    torch.cuda.synchronize()
    return y, st.slog[:st.ns].tolist(), (n_mega[0], n_stage[0], n_down[0], n_gate[0], n_send[0], n_fin[0]), int(st.pend[2][0])


def eager_cases(band: bool, seed: int, sizes):
    set_band(band)
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    bad, n, levels, detail, staged = 0, 0, set(), "", 0
    for trial in range(40):
        T = sizes[trial % len(sizes)]
        rc = [[4, 4, 2, 1, 1][t % P] for t in range(16)] if trial % 3 == 0 else None
        hook._RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev) if rc else None
        st, moe = make_layer(rng, g, rng.randint(1, 92), rng.randint(1, 4), rng.choice([16, 8]), rng.choice([None, 3, 6]),
                             band, mean=trial % 4 != 3)
        x = torch.randn(T, H, generator=g).to(torch.bfloat16).to(dev)
        blocks = hook.split_blocks(T, block, 8)
        B, C = len(blocks), len(hook.dense_chunks(blocks))
        y0, s0, c0, p0 = run(moe, x, 0)
        y1, s1, c1, p1 = run(moe, x, 1)
        y2, s2, c2, p2 = run(moe, x, 2)
        hook.DECODE_JOINT = 2
        lv = hook._joint_level(st, blocks)
        hook.DECODE_JOINT = 0
        levels.add((B, lv))
        nb = B if band else 0
        ok = torch.equal(y1, y0) and torch.equal(y2, y0) and s1 == s0 and s2 == s0 and p0 == p1 == p2 == T
        ok = ok and c0 == (B, B, B, B, nb, nb) and c1 == (B, B, C, C, nb, nb)
        ok = ok and c2 == ((1 if lv == 2 else B), B, C, C, nb, nb)
        ok = ok and lv == (2 if (not band or B <= hook.BAND_NSLOT) else 1)
        ok = ok and not hook._SPLIT["sub"] and not hook._JT["on"] and hook._LD["ev"] is None and hook._FIN.get("ev") is None
        staged += sum(1 for lst in s0 for v in lst if v >= 0)
        if not ok and not detail:
            detail = f"first wrong: T {T}, layer {st.idx}, counts {c0} {c1} {c2}, level {lv}, equal " \
                     f"{torch.equal(y1, y0)} {torch.equal(y2, y0)}, stage lists {s1 == s0} {s2 == s0}"
        bad += not ok
        n += 1
    hook._RCAP[0] = None
    return bad, n, sorted(levels), detail, staged


# ------------------------------------------------------------------------------------------------ 2, 3. eager
bad, n, levels, detail, staged = eager_cases(False, 5, [10, 15, 16, 20, 40])
check("no band: eager batches of 10..40 rows at levels 1 and 2 equal the split bit for bit (same stage lists, "
      "accepted-row buffers; MegaMoE calls B / B / 1; router and latent down once per dense chunk)",
      bad == 0 and staged > 100, f"({n} batches, {staged} staged records, (blocks, level): {levels}; {detail})")
bad, n, levels, detail, staged = eager_cases(True, 6, [10, 15, 16, 20, 40])
check("band + fused tail: levels 1 and 2 equal the split bit for bit; one send and one finish per block; more "
      "blocks than mailbox slots take level 1",
      bad == 0 and (8, 1) in levels and (4, 2) in levels and (2, 2) in levels,
      f"({n} batches, (blocks, level): {levels}; {detail})")

# one dense chunk for the whole batch (K31_DECODE_JOINT_DENSE=40), no band: still the split's output
set_band(False)
rng, g = random.Random(7), torch.Generator().manual_seed(7)
hook.JOINT_DENSE_ROWS = 40
bad = n = 0
for trial in range(12):
    T = [20, 40, 16][trial % 3]
    st, moe = make_layer(rng, g, rng.randint(1, 92), rng.randint(1, 4), 16, rng.choice([None, 6]), False, trial % 2 == 0)
    x = torch.randn(T, H, generator=g).to(torch.bfloat16).to(dev)
    B = len(hook.split_blocks(T, block, 8))
    y0, s0, c0, _p = run(moe, x, 0)
    y2, s2, c2, _p = run(moe, x, 2)
    bad += not (torch.equal(y2, y0) and s2 == s0 and c2 == (1, B, 1, 1, 0, 0))
    n += 1
hook.JOINT_DENSE_ROWS = 16
check("K31_DECODE_JOINT_DENSE=40, no band: the whole batch as one dense chunk (one router, one latent down) equals "
      "the split", bad == 0, f"({n} batches, {bad} wrong)")

# ------------------------------------------------------------------------------------------------ 4. mutations
set_band(True)
rng, g = random.Random(9), torch.Generator().manual_seed(9)
diff_group = diff_tag = tried = 0
for trial in range(12):
    st, moe = make_layer(rng, g, rng.randint(1, 92), 4, 16, None, True, True)
    x = torch.randn(20, H, generator=g).to(torch.bfloat16).to(dev)
    y0 = run(moe, x, 0)[0]
    keep = st.jstagers
    st.jstagers = keep[1:] + keep[:1]                         # every block's records into the next slot group
    yg = run(moe, x, 2)[0]
    st.jstagers = keep
    MUT["tag"] = True
    yt = run(moe, x, 2)[0]
    MUT["tag"] = False
    tried += 1
    diff_group += not torch.equal(yg, y0)
    diff_tag += not torch.equal(yt, y0)
check("mutations change the output: records staged into another slot group than the plan's; finishes that read "
      "the last call's tag",
      diff_group >= tried - 1 and diff_tag >= tried - 1, f"({diff_group} and {diff_tag} of {tried} batches differ)")

# ------------------------------------------------------------------------------------------------ 5. CUDA graphs
for band in (False, True):
    for T in (20, 10):
        set_band(band)
        rng, g = random.Random(20 + T + band), torch.Generator().manual_seed(20 + T + band)
        rc = [[4, 4, 2, 1, 1][t % P] for t in range(16)]
        hook._RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev)
        st1, moe1 = make_layer(rng, g, 11, 4, 16, 6, band, True)
        st2, moe2 = make_layer(rng, g, 12, 3, 8, None, band, True)
        x = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
        x.copy_(torch.randn(T, H, generator=g).to(torch.bfloat16))
        hook.DECODE_JOINT = 2
        hook._SPLIT["decode"] = True
        moe2.forward(moe1.forward(x))                         # compile outside the capture
        hook._SPLIT["decode"] = False
        torch.cuda.synchronize()
        st1.ns = st2.ns = 0
        for c in (n_mega, n_stage, n_down, n_gate, n_send, n_fin):
            c[0] = 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y1 = moe1.forward(x)
            y2 = moe2.forward(y1)
        hook.DECODE_JOINT = 0
        B = len(hook.split_blocks(T, block, 8))
        C = len(hook.dense_chunks(hook.split_blocks(T, block, 8)))
        captured = (n_mega[0], n_stage[0], n_down[0], n_gate[0], n_send[0], n_fin[0])
        want_c = (2, 2 * B, 2 * C, 2 * C, 2 * B if band else 0, 2 * B if band else 0)
        bad = 0
        for seed in range(1, 11):
            x.copy_(torch.randn(T, H, generator=torch.Generator().manual_seed(300 + seed)).to(torch.bfloat16))
            st1.pend[2].zero_()
            st2.pend[2].zero_()
            slot_val.normal_()
            graph.replay()
            torch.cuda.synchronize()
            g1, g2 = y1.clone(), y2.clone()
            l1, l2 = st1.slog[:B].tolist(), st2.slog[:B].tolist()
            pt = (int(st1.pend[2][0]), int(st2.pend[2][0]))
            w1, s1, _c, _p = run(moe1, x, 0)
            w2, s2, _c, _p = run(moe2, w1, 0)
            ok = torch.equal(g1, w1) and torch.equal(g2, w2) and l1 == s1 and l2 == s2 and pt == (T, T)
            bad += not ok
        hook._RCAP[0] = None
        check(f"CUDA graph, {'band + tail' if band else 'no band'}, {T} rows: two layers at level 2 (one MegaMoE call "
              f"each); 10 replays on new inputs equal the split run eagerly",
              bad == 0 and captured == want_c,
              f"(captured MegaMoE / stager / latent down / router / send / finish {captured}, expected {want_c}; {bad} wrong)")

# ------------------------------------------------------------------------------------------------ 6. the check
set_band(True)
rng, g = random.Random(31), torch.Generator().manual_seed(31)
st, moe = make_layer(rng, g, 40, 4, 16, 6, True, True)
x = torch.randn(20, H, generator=g).to(torch.bfloat16).to(dev)
hook.DECODE_JOINT, hook.JOINT_CHECK = 2, 1
hook._SPLIT["decode"] = True
del logs[:]
ya = moe.forward(x).clone()
yb = moe.forward(x).clone()                                   # the second batch of the layer is not checked
hook._SPLIT["decode"] = False
hook.DECODE_JOINT, hook.JOINT_CHECK = 0, 0
line = [m for m in logs if m.startswith("decode joint check")]
check("K31_DECODE_JOINT_CHECK: one check line for the layer's first eager batch, all three comparisons bit-equal",
      len(line) == 1 and line[0].count("bit-equal") == 3 and torch.equal(ya, yb), f"({line[:1]})")

# ------------------------------------------------------------------------------------------------ 7. the counters
set_band(False)
rng, g = random.Random(41), torch.Generator().manual_seed(41)
st, moe = make_layer(rng, g, 17, 4, 16, None, False, True)
x = torch.randn(20, H, generator=g).to(torch.bfloat16).to(dev)
x[5:10] = x[0:5]                                              # the second block repeats the first block's rows
y0, s0, _c, _p = run(moe, x, 0)
hook.JOINT_STAT = True
hook._JT["stat"] = None
ys, ss, _c, _p = run(moe, x, 2)
hook.JOINT_STAT = False
n = [int(v) for v in hook._JT["stat"].tolist()]
rec = sum(1 for lst in s0 for v in lst if v >= 0)
first = sum(1 for v in s0[0] if v >= 0)
check("K31_DECODE_JOINT_STAT: the counters of one call (4 blocks, records staged; a repeated block's records all "
      "counted as staged twice) and an unchanged output",
      torch.equal(ys, y0) and n[:3] == [1, 4, rec] and n[3] >= first > 0 and 0 < n[5] < n[4], f"({n}, first block {first})")

print("DECODE JOINT: " + ("PASS" if not fails else f"FAIL ({len(fails)}: {'; '.join(fails)})"))
sys.exit(1 if fails else 0)
