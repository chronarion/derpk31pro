"""K31_DECODE_SPLIT (a decode batch of more than 8 rows as one MoE call per verify block) and K31_STAGE_CAP.

1. hook.split_block / split_blocks: a call is a whole number of verify blocks; the row ranges cover the batch in
   order, every call but the last is one block, no call has more than K31_DECODE_SPLIT_ROWS rows (8), a tail of up
   to that many rows (blocks + padding) is one call.
2. unitk with `cap`: against the reference plan (candidates' distinct experts by largest weight, ties by lower id, the
   first `cap` staged), with row caps, resident caps, K31_STAGE_ALL and the row prior of K31_STAGE_PRIOR (order by
   weight x prior), 300 random calls; the policy field
   `stage_cap` and the two quality policies.
3. unitk with `row0` and 64-row accepted-row buffers: a step of several requests planned block by block leaves in
   the buffers exactly the step's rows (expert ids, squared weights, row count) and no score change; commit() then
   gives the scores of the accepted rows (float64 reference), acc += accepted tokens. 16-row buffers as before.
4. hook._split_call over hook._forward (GPU; a stand-in MegaMoE that returns one reused buffer, stand-in stagers,
   small integer mean tables): for batches of 10, 15, 16, 20, 24 and 40 rows: one MegaMoE call per block, each with
   the row ids and the stage list of the reference plan of its rows alone (row caps by the row's place in its block,
   the stage cap per call); the assembled output has every row's own call's result and mean substitute; the pick
   counters count every row once; the split state is restored, also after an exception. hook._split_wanted: only
   calls of 9..K31_DECODE_SPLIT_MAX rows that are captured or part of a decode-only step, never a block itself.
5. GPU: the split captured in a CUDA graph (20 rows): replays on new picks equal the reference, block by block, and
   the accepted-row buffers after a replay commit to the reference scores.
CPU (parts 1-3 in the Triton interpreter): K31_TEST_CPU=1. GPU: every part.
    docker run --rm --network none [--device nvidia.com/gpu=...] -v "$PWD":/engine:ro --entrypoint python3 IMAGE \
        /engine/tests/test_decode_split.py
Exit 0 and "DECODE SPLIT: PASS" = every check passed.
"""
import os
import random
import sys

CPU = os.environ.get("K31_TEST_CPU", "0" if os.path.exists("/dev/nvidiactl") else "1") == "1"
if CPU:
    os.environ["TRITON_INTERPRET"] = "1"
os.environ["K31_MODE"] = "staged"
os.environ["K31_DECODE_SPLIT"] = "1"
for k in ("K31_STAGE_ALL", "K31_DEFER", "K31_PROMPT_PATH", "K31_RES_MAXRANK", "K31_TAIL1", "K31_BAND", "K31_PDL",
          "K31_BAND_FIN_SIDE", "K31_STAGE_ROWS", "K31_SHARED_SIDE", "K31_MEGAMOE_NOPREP", "K31_PRESTAGE", "K31_LATE",
          "K31_POLICY", "K31_CUT_MEAN", "K31_STAGE_CAP", "K31_DECODE_SPLIT_MAX", "K31_DECODE_SPLIT_ROWS",
          "K31_WINDOW_TEST"):
    os.environ.pop(k, None)

import torch  # noqa: E402
import triton.language as tl  # noqa: E402

if CPU:      # PDL instructions have no interpreter form; without the launch attribute they are no-ops on the GPU too
    tl.extra.cuda.gdc_wait = lambda *a, **k: None
    tl.extra.cuda.gdc_launch_dependents = lambda *a, **k: None
sys.path.insert(0, os.environ.get("ENGINE", os.environ.get("K31_ENGINE_DIR", "/engine")))
from k31_engine import hook, unitk  # noqa: E402

dev = torch.device("cpu" if CPU else "cuda")
E, K, SL = 896, 16, unitk.SLOTS
LAM = hook.LAM
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def make_layer(rng, n_res, n_bank, idx=1):
    perm = list(range(E))
    rng.shuffle(perm)
    st = hook.LayerState(idx, perm[:n_res], perm[n_res:n_res + n_bank])
    st.row_map = torch.tensor(st.row, dtype=torch.int32, device=dev)
    st.bank_map = torch.tensor(st.bank_idx, dtype=torch.int32, device=dev)
    st.arange = torch.arange(max(SL, K), dtype=torch.long, device=dev)
    return st


def make_call(rng, T, pool_n):
    pool = rng.sample(range(E), pool_n)
    ids = [rng.sample(pool, K) for _ in range(T)]
    for row in ids:
        if rng.random() < 0.08:
            row[rng.randrange(K)] = -1
    w = [[rng.randint(1, 6) / 8.0 for _ in range(K)] for _ in range(T)]
    return ids, w


def ranks(wrow):
    return [sum(1 for j in range(K) if wrow[j] > wrow[k] or (wrow[j] == wrow[k] and j < k)) for k in range(K)]


def plan_ref(st, ids, w, strong, S, resmax=16, allrows=False, cap=SL, rowcap=None, prior=None):
    """The one-round plan from its definition. Candidates: an expert id, no resident row, a bank record, rank <
    strong (and < rowcap[t]); staged: the candidates' distinct experts by their largest candidate weight (ties: lower
    id), the first `cap` in slots 0, 1, ...; rows: a resident pick of rank < resmax -> its row; a candidate of a
    staged expert (allrows: any non-resident pick of one) -> S + slot; else -1. -> (row ids [T][16], stage [16])"""
    row, bidx = st.row, st.bank_idx
    T = len(ids)
    rk = [ranks(w[t]) for t in range(T)]
    cand = [[ids[t][k] >= 0 and row[ids[t][k]] < 0 and bidx[ids[t][k]] >= 0 and rk[t][k] < strong and
             (rowcap is None or rk[t][k] < rowcap[t]) for k in range(K)] for t in range(T)]
    best = {}
    for t in range(T):
        for k in range(K):
            if cand[t][k]:
                e = ids[t][k]
                best[e] = max(best.get(e, -1.0), w[t][k] * (prior[t] if prior else 1.0))   # (K31_STAGE_PRIOR)
    order = sorted(best, key=lambda e: (-best[e], e))[:cap]
    slot_of = {e: s for s, e in enumerate(order)}
    stage = [bidx[e] for e in order] + [-1] * (SL - len(order))
    out = [[-1] * K for _ in range(T)]
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            if e < 0:
                continue
            if row[e] >= 0:
                out[t][k] = row[e] if rk[t][k] < resmax else -1
                continue
            s = slot_of.get(e, -1)
            if s >= 0 and (cand[t][k] or allrows):
                out[t][k] = S + s
    return out, stage


# 1. split_blocks
bad = 0
for rows in (8, 5, 2, 1):                    # K31_DECODE_SPLIT_ROWS: the largest batch that stays one call
    for P in (1, 2, 3, 4, 5, 6, 7, 8, 9, 17):
        blk = hook.split_block(P, rows)
        bad += not (1 <= blk <= rows and (blk % P == 0 if P <= rows else blk == rows))
        for T in range(1, 65):
            bl = hook.split_blocks(T, blk, rows)
            ok = bl[0][0] == 0 and bl[-1][1] == T and all(bl[i][1] == bl[i + 1][0] for i in range(len(bl) - 1)) \
                and all(0 < b - a <= rows for a, b in bl) and all(b - a == blk for a, b in bl[:-1]) \
                and (T > rows or len(bl) == 1)
            bad += not ok
check("split_blocks: ordered cover, whole blocks, at most 8 rows per call, a tail of up to 8 rows is one call",
      bad == 0 and hook.split_blocks(20, 5) == [(0, 5), (5, 10), (10, 15), (15, 20)]
      and hook.split_blocks(16, 5) == [(0, 5), (5, 10), (10, 16)] and hook.split_blocks(10, 5) == [(0, 5), (5, 10)]
      and hook.split_blocks(24, 5) == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 24)]
      and hook.split_blocks(16, 8) == [(0, 8), (8, 16)] and hook.split_block(5) == 5 and hook.split_block(2) == 8
      and hook.split_blocks(8, hook.split_block(2, 2), 2) == [(0, 2), (2, 4), (4, 6), (6, 8)], f"({bad} wrong)")

# 2. unitk with a stage cap
rng = random.Random(21)
bad, capped, n_pri = 0, 0, 0
for trial in range(120 if CPU else 300):
    st = make_layer(rng, rng.randint(40, 200), rng.randint(200, 600), idx=rng.randint(1, 92))
    T = rng.randint(1, 8)
    strong, resmax, allrows = rng.randint(1, 4), rng.choice([16, 16, 6, 10]), rng.random() < 0.5
    cap = rng.randint(1, 16)
    rc = [rng.choice([4, 4, 2, 1]) for _ in range(16)] if rng.random() < 0.5 else None
    ids, w = make_call(rng, T, rng.randint(40, 896))
    pri = [[1.0, 0.5, 0.5, 0.25, 0.125][t % 5] for t in range(16)] if trial % 3 == 0 else None   # exact products
    want_ids, want_stage = plan_ref(st, ids, w, strong, st.slot_base, resmax, allrows, cap, rc, pri)
    uncapped = plan_ref(st, ids, w, strong, st.slot_base, resmax, allrows, SL, rc)[1]
    capped += sum(b >= 0 for b in uncapped) > cap
    n_pri += pri is not None and want_stage != plan_ref(st, ids, w, strong, st.slot_base, resmax, allrows, cap, rc)[1]
    stage = torch.full((SL,), 77, dtype=torch.int32, device=dev)
    out, stage = unitk.plan(torch.tensor(ids, dtype=torch.int32, device=dev),
                            torch.tensor(w, dtype=torch.float32, device=dev), st.row_map, st.bank_map, strong,
                            st.slot_base, stage=stage, resmax=resmax, allrows=allrows, cap=cap,
                            rowcap=torch.tensor(rc, dtype=torch.int32, device=dev) if rc else None,
                            rowprior=torch.tensor(pri, dtype=torch.float32, device=dev) if pri else None)
    bad += out.tolist() != want_ids or stage.tolist() != want_stage
check("unitk cap: the plan with at most `cap` staged records equals the reference (row caps, resident caps, "
      "K31_STAGE_ALL, slots by weight x row prior)", bad == 0 and capped > 20 and n_pri > 5,
      f"({capped} calls with more candidates than the cap, {n_pri} where the prior changes the staged set; {bad} wrong)")

# 2b. the policy field and the quality policies
import json  # noqa: E402
import tempfile  # noqa: E402
from k31_engine import policy  # noqa: E402

PROF = os.path.join(os.environ.get("ENGINE", os.environ.get("K31_ENGINE_DIR", "/engine")), "profiles")
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "p.json")
    json.dump({"version": 1, "policies": {"a": {"stage": "0-3", "stage_cap": 5}, "b": {"stage": "0"}}, "default": "b",
               "layers": {"10-20": "a"}, "overrides": {"15": {"stage_cap": 2}}}, open(f, "w"))
    prm = policy.load(f)
    ok = prm[10].stage_cap == 5 and prm[15].stage_cap == 2 and prm[1].stage_cap is None and prm[10].strong == 4
    json.dump({"version": 1, "policies": {"a": {"stage": "0-3", "stage_cap": 17}}, "default": "a"}, open(f, "w"))
    try:
        policy.load(f)
        ok = False
    except policy.PolicyError:
        pass
for name, band in (("policy-stage4.json", True), ("policy-nortx-stage4.json", False)):
    prm = policy.load(os.path.join(PROF, name))
    ok = ok and all(prm[L].strong == 4 and prm[L].resmax == 16 and prm[L].band == band and prm[L].mean
                    and prm[L].stage_cap is None for L in policy.LAYERS)
check("policy: stage_cap per policy and per layer, refused outside 1..16; policy-stage4 / policy-nortx-stage4 stage "
      "ranks 0-3 in every layer with no resident cap", ok)

# 3. row0 + 64-row accepted-row buffers + commit
R = hook.PEND_ROWS
L = 4
bad, bad16, n_steps = 0, 0, 0
for trial in range(40 if CPU else 160):
    g = random.Random(1000 + trial)
    P = g.choice([2, 3, 5, 5, 5])
    block = P * (8 // P)
    nreq = g.randint(2, min(8, 40 // P))
    T = nreq * P
    Tpad = min(40, T + g.choice([0, 0, 1, 3]))                               # a graph's padding rows
    nsamp = [g.randint(1, P) for _ in range(nreq)]
    qsl = torch.tensor([r * P for r in range(nreq + 1)], dtype=torch.int32)
    score0 = torch.rand((L, E), generator=torch.Generator().manual_seed(trial)).to(dev)
    score = score0.clone()
    pid = torch.full((L, R, 16), -1, dtype=torch.int32, device=dev)
    pw = torch.zeros((L, R, 16), dtype=torch.float32, device=dev)
    pt = torch.zeros(L, dtype=torch.int32, device=dev)
    acc = torch.zeros(1, dtype=torch.int64, device=dev)
    stats = torch.zeros((L, 4), dtype=torch.int32, device=dev)
    ref = score0.double().cpu() * float(torch.tensor(LAM ** sum(nsamp), dtype=torch.float32))
    ok = True
    for j in range(L):
        st = make_layer(g, g.randint(40, 200), g.randint(200, 500), idx=j + 1)
        ids, w = make_call(g, Tpad, g.randint(60, 896))
        for a, b in hook.split_blocks(Tpad, block):
            out, stage = unitk.plan(torch.tensor(ids[a:b], dtype=torch.int32, device=dev),
                                    torch.tensor(w[a:b], dtype=torch.float32, device=dev), st.row_map, st.bank_map, 3,
                                    st.slot_base, score=score[j], lam=LAM, stats=stats[j],
                                    pend=(pid[j], pw[j], pt[j:j + 1]), row0=a)
            wi, ws = plan_ref(st, ids[a:b], w[a:b], 3, st.slot_base)
            ok = ok and out.tolist() == wi and stage.tolist() == ws
        ok = ok and int(pt[j]) == Tpad
        ok = ok and pid[j, :Tpad].tolist() == ids
        w2 = torch.where(torch.tensor(ids) >= 0, torch.tensor(w, dtype=torch.float32) ** 2, torch.zeros(()))
        ok = ok and torch.equal(pw[j, :Tpad].cpu(), w2)                      # (0 where the pick has no expert id)
        ok = ok and int(stats[j, 3]) == sum(e >= 0 for row in ids for e in row)
        for r in range(nreq):
            for t in range(r * P, r * P + nsamp[r]):
                for k in range(K):
                    if ids[t][k] >= 0:
                        ref[j, ids[t][k]] += w[t][k] ** 2
    ok = ok and torch.equal(score, score0)                                   # nothing added before the commit
    unitk.commit(score, pid, pw, pt, qsl.to(dev), torch.tensor(nsamp, dtype=torch.int32, device=dev), nreq, LAM, acc)
    ok = ok and torch.allclose(score.double().cpu(), ref, rtol=1e-5, atol=1e-5) and int(pt.abs().sum()) == 0 \
        and int(acc[0]) == sum(nsamp)
    bad += not ok
    n_steps += 1
    if trial < 40:                                                           # 16-row buffers, one call (as before)
        T16 = min(T, 15)
        nr = T16 // P
        if nr:
            sc = score0.clone()
            pid2 = torch.full((L, 16, 16), -1, dtype=torch.int32, device=dev)
            pw2 = torch.zeros((L, 16, 16), dtype=torch.float32, device=dev)
            pt2 = torch.zeros(L, dtype=torch.int32, device=dev)
            acc2 = torch.zeros(1, dtype=torch.int64, device=dev)
            ref2 = score0.double().cpu() * float(torch.tensor(LAM ** sum(nsamp[:nr]), dtype=torch.float32))
            for j in range(L):
                st = make_layer(g, 100, 300, idx=j + 1)
                ids, w = make_call(g, T16, 500)
                unitk.plan(torch.tensor(ids, dtype=torch.int32, device=dev),
                           torch.tensor(w, dtype=torch.float32, device=dev), st.row_map, st.bank_map, 3, st.slot_base,
                           score=sc[j], lam=LAM, stats=stats[j], pend=(pid2[j], pw2[j], pt2[j:j + 1]))
                for r in range(nr):
                    for t in range(r * P, r * P + nsamp[r]):
                        for k in range(K):
                            if ids[t][k] >= 0:
                                ref2[j, ids[t][k]] += w[t][k] ** 2
            unitk.commit(sc, pid2, pw2, pt2, qsl[:nr + 1].to(dev),
                         torch.tensor(nsamp[:nr], dtype=torch.int32, device=dev), nr, LAM, acc2)
            bad16 += not (torch.allclose(sc.double().cpu(), ref2, rtol=1e-5, atol=1e-5) and int(acc2[0]) == sum(nsamp[:nr]))
check("unitk row0: a step planned block by block leaves the step's rows in the 64-row buffers; commit() gives the "
      "accepted rows' scores", bad == 0, f"({n_steps} steps of up to 40 rows; {bad} wrong)")
check("commit with 16-row buffers and one call per layer: unchanged", bad16 == 0, f"({bad16} wrong)")

if CPU:
    print("(CPU: parts 4 and 5 need a GPU)")
else:
    # 4. hook._split_call over hook._forward
    D = 256
    calls, staged = [], []
    buf = torch.zeros(64, D, dtype=torch.bfloat16, device=dev)                # the stand-in MegaMoE's reused output

    def fake_forward(self, hidden_states, topk_weights, ids, shared_experts=None):
        calls.append(ids.clone())
        n = hidden_states.shape[0]
        buf[:n].fill_(float(len(calls)))                                     # the call's number in every element
        return buf[:n]

    class FakeMoE:
        def __init__(self, st):
            self._k31 = st

    hook._orig["forward"] = fake_forward
    assert hook.DECODE_SPLIT and hook.SPLIT_MAX == 40 and not hook._SPLIT["sub"]
    rng = random.Random(77)
    bad, n_calls, n_over, bad_state = 0, 0, 0, 0
    try:
        for trial in range(90):
            T = [10, 15, 16, 20, 24, 40][trial % 6]
            P = 5 if T != 16 or trial % 12 < 6 else 2
            block = P * (8 // P)
            hook._SPLIT["block"], hook._SPLIT["P"], hook._SPLIT["rows"] = block, P, 8
            st = make_layer(rng, rng.randint(50, 250), rng.randint(200, 500), idx=rng.randint(1, 92))
            st.band_map = None
            st.strong = rng.randint(1, 4)
            st.resmax = rng.choice([16, 16, 8])
            st.stage_cap = rng.choice([None, None, 2, 5])
            mean = torch.randint(-1, 2, (E, D)).to(torch.bfloat16)
            st.cut_mean = mean.to(dev).contiguous()
            allrows = trial % 2 == 0
            hook.STAGE_ALL = unitk.STAGE_ALL = allrows
            rc = [[4, 4, 2, 1, 1][min(t % P, 4)] for t in range(16)] if trial % 3 == 0 else None
            hook._RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev) if rc else None
            st.score = torch.rand(E, device=dev)
            sc0 = st.score.clone()
            st.stats = torch.zeros(4, dtype=torch.int32, device=dev)
            st.pend = (torch.full((R, 16), -1, dtype=torch.int32, device=dev),
                       torch.zeros((R, 16), dtype=torch.float32, device=dev),
                       torch.zeros(1, dtype=torch.int32, device=dev))
            ids, w = make_call(rng, T, rng.randint(40, 600))
            ids_d = torch.tensor(ids, dtype=torch.int32, device=dev)
            w_d = torch.tensor(w, dtype=torch.float32, device=dev)
            x = torch.zeros(T, 8, dtype=torch.bfloat16, device=dev)
            calls.clear()
            staged.clear()
            st.stager = lambda ptr, stage: staged.append(stage.clone().tolist())
            fm = FakeMoE(st)

            def one(rows):
                a = hook._SPLIT["row0"]
                assert hook._SPLIT["sub"]
                return hook._forward(fm, rows, w_d[a:a + rows.shape[0]], ids_d[a:a + rows.shape[0]])

            hook._SPLIT["decode"] = True                                     # an eager decode-only step
            ok = hook._split_wanted(x)
            y = hook._split_call(one, x, st.idx).float().cpu()
            hook._SPLIT["decode"] = False
            blocks = hook.split_blocks(T, block)
            ok = ok and len(calls) == len(blocks) == len(staged)
            n_calls += len(calls)
            if ok:
                for c, (a, b) in enumerate(blocks):
                    cap = st.stage_cap or SL
                    wi, ws = plan_ref(st, ids[a:b], w[a:b], st.strong, st.slot_base, st.resmax, allrows, cap, rc)
                    n_over += (b - a) * st.strong > SL
                    ok = ok and calls[c].tolist() == wi and staged[c] == ws
                    for t in range(a, b):                                    # the row's own call + its mean substitute
                        want = torch.full((D,), float(c + 1), dtype=torch.float64)
                        for k in range(K):
                            if ids[t][k] >= 0 and wi[t - a][k] < 0:
                                want += w[t][k] * mean[ids[t][k]].double()
                        ok = ok and torch.equal(y[t].double(), want)
                ok = ok and int(st.stats[3]) == sum(e >= 0 for row in ids for e in row)
                ok = ok and int(st.pend[2][0]) == T and st.pend[0][:T].tolist() == ids
                ok = ok and torch.equal(st.score, sc0)
            bad_state += hook._SPLIT["sub"] or hook._SPLIT["row0"] != 0
            bad += not ok
    finally:
        hook._SPLIT["decode"] = False
        hook.STAGE_ALL = unitk.STAGE_ALL = False
        hook._RCAP[0] = None
    check("hook: a decode batch as one call per verify block: each call plans its rows alone (row caps by the row's "
          "place in its block, the stage cap per call), every row gets its own call's output and mean substitute",
          bad == 0 and n_over > 20, f"({n_calls} calls; {n_over} eager blocks whose candidates could overflow the "
          f"slots still took the one-round plan; {bad} batches wrong)")
    try:
        hook._split_call(lambda rows: (_ for _ in ()).throw(ValueError("x")), torch.zeros(20, 8, device=dev), 1)
    except ValueError:
        pass
    check("hook: the split state is restored after every batch, also after an exception",
          bad_state == 0 and not hook._SPLIT["sub"] and hook._SPLIT["row0"] == 0)
    x20, x8, x41 = (torch.zeros(n, 8, device=dev) for n in (20, 8, 41))
    w0 = hook._split_wanted(x20)                                             # eager, not a decode-only step
    hook._SPLIT["decode"] = True
    w1, w2, w3 = hook._split_wanted(x20), hook._split_wanted(x8), hook._split_wanted(x41)
    hook._SPLIT["sub"] = True
    w4 = hook._split_wanted(x20)
    hook._SPLIT["sub"], hook._SPLIT["decode"] = False, False
    cap0 = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    w5 = hook._split_wanted(x20)
    torch.cuda.is_current_stream_capturing = cap0
    hook.DECODE_SPLIT = False
    hook._SPLIT["decode"] = True
    w6 = hook._split_wanted(x20)
    hook.DECODE_SPLIT, hook._SPLIT["decode"] = True, False
    check("hook._split_wanted: 9..40 rows, captured or in a decode-only step, never a block, only with the switch",
          (w0, w1, w2, w3, w4, w5, w6) == (False, True, False, False, False, True, False),
          f"{(w0, w1, w2, w3, w4, w5, w6)}")

    # 5. the split in a CUDA graph
    T, P, block = 20, 5, 5
    hook._SPLIT["block"], hook._SPLIT["P"], hook._SPLIT["rows"] = block, P, 8
    hook.STAGE_ALL = unitk.STAGE_ALL = True
    rc = [[4, 4, 2, 1, 1][t % P] for t in range(16)]
    hook._RCAP[0] = torch.tensor(rc, dtype=torch.int32, device=dev)
    rng = random.Random(5)
    st = make_layer(rng, 150, 400, idx=7)
    st.band_map, st.strong, st.resmax, st.stage_cap = None, 4, 10, 6
    mean = torch.randint(-1, 2, (E, D)).to(torch.bfloat16)
    st.cut_mean = mean.to(dev).contiguous()
    score = torch.rand((1, E), device=dev)
    st.score, st.stats = score[0], torch.zeros(4, dtype=torch.int32, device=dev)
    pid = torch.full((1, R, 16), -1, dtype=torch.int32, device=dev)
    pw = torch.zeros((1, R, 16), dtype=torch.float32, device=dev)
    pt = torch.zeros(1, dtype=torch.int32, device=dev)
    st.pend = (pid[0], pw[0], pt[0:1])
    ids_d = torch.zeros((T, K), dtype=torch.int32, device=dev)
    w_d = torch.zeros((T, K), dtype=torch.float32, device=dev)
    x = torch.zeros(T, 8, dtype=torch.bfloat16, device=dev)
    ilog = torch.zeros((8, 8, K), dtype=torch.int32, device=dev)
    slog = torch.zeros((8, SL), dtype=torch.int32, device=dev)
    cnt = {"i": 0, "s": 0}

    def fake_forward_g(self, hidden_states, topk_weights, ids, shared_experts=None):
        n = hidden_states.shape[0]
        ilog[cnt["i"], :n].copy_(ids)
        cnt["i"] += 1
        buf[:n].fill_(float(cnt["i"]))
        return buf[:n]

    def stager_g(ptr, stage):
        slog[cnt["s"]].copy_(stage)
        cnt["s"] += 1

    hook._orig["forward"] = fake_forward_g
    st.stager = stager_g
    fm = FakeMoE(st)

    def one_g(rows):
        a = hook._SPLIT["row0"]
        return hook._forward(fm, rows, w_d[a:a + rows.shape[0]], ids_d[a:a + rows.shape[0]])

    def load(seed):
        ids, w = make_call(random.Random(seed), T, 300)
        ids_d.copy_(torch.tensor(ids, dtype=torch.int32))
        w_d.copy_(torch.tensor(w, dtype=torch.float32))
        return ids, w

    load(0)
    hook._SPLIT["decode"] = True
    hook._split_call(one_g, x, 7)                                            # compile outside the capture
    hook._SPLIT["decode"] = False
    torch.cuda.synchronize()
    cnt["i"] = cnt["s"] = 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wanted = hook._split_wanted(x)
        yg = hook._split_call(one_g, x, 7)
    bad = 0
    for seed in range(1, 13):
        ids, w = load(seed)
        sc0 = score.clone()
        graph.replay()
        torch.cuda.synchronize()
        ok = wanted and cnt["i"] == 4 and cnt["s"] == 4
        y = yg.float().cpu()
        for c, (a, b) in enumerate(hook.split_blocks(T, block)):
            wi, ws = plan_ref(st, ids[a:b], w[a:b], 4, st.slot_base, 10, True, 6, rc)
            ok = ok and ilog[c, :b - a].tolist() == wi and slog[c].tolist() == ws
            for t in range(a, b):
                want = torch.full((D,), float(c + 1), dtype=torch.float64)
                for k in range(K):
                    if ids[t][k] >= 0 and wi[t - a][k] < 0:
                        want += w[t][k] * mean[ids[t][k]].double()
                ok = ok and torch.equal(y[t].double(), want)
        ok = ok and int(pt[0]) == T and pid[0, :T].tolist() == ids and torch.equal(score, sc0)
        g = random.Random(seed)
        nsamp = [g.randint(1, P) for _ in range(4)]
        ref = sc0[0].double().cpu() * float(torch.tensor(LAM ** sum(nsamp), dtype=torch.float32))
        for r in range(4):
            for t in range(r * P, r * P + nsamp[r]):
                for k in range(K):
                    if ids[t][k] >= 0:
                        ref[ids[t][k]] += w[t][k] ** 2
        acc = torch.zeros(1, dtype=torch.int64, device=dev)
        unitk.commit(score, pid, pw, pt, torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32, device=dev),
                     torch.tensor(nsamp, dtype=torch.int32, device=dev), 4, LAM, acc)
        ok = ok and torch.allclose(score[0].double().cpu(), ref, rtol=1e-5, atol=1e-5) and int(acc[0]) == sum(nsamp)
        bad += not ok
    hook.STAGE_ALL = unitk.STAGE_ALL = False
    hook._RCAP[0] = None
    check("CUDA graph: a captured 20-row batch (4 calls per layer) replays the reference plan of every block on new "
          "picks; the accepted-row buffers commit to the reference scores", bad == 0, f"(12 replays; {bad} wrong)")

print("DECODE SPLIT: " + ("PASS" if not fails else f"FAIL ({len(fails)}: {'; '.join(fails)})"))
sys.exit(1 if fails else 0)
