"""K31_STAGE_ALL (S1): every pick of an expert staged for the call uses its staging row, at any rank and in any row.

1. unitk against `plan_ref` below (every pick's row id and the stage list, exactly), 800 random calls (400 on the
   CPU; T 1..16, strong 1..4, resident caps, coarse weights for ties, experts repeated across rows, a few -1 picks,
   a third with more candidate experts than slots), with ALLROWS off (today's plan) and on; PRE mode (slots staged
   ahead) both ways.
2. Between the two modes on the same calls: the same stage list, and every pick's row id the same except the picks of
   a staged expert that today's plan cuts, which now hold that expert's slot; the residency inputs: the staged count
   up by the newly served picks and the cut count down by as many; the scores of both modes equal to the float64 sum
   (bit for bit where every partial sum is exact; rtol 1e-5 elsewhere, because the order of the GPU's atomic adds is
   not fixed); on the CPU the same check against float32 sums made in shuffled orders.
3. hook._forward's rounds of calls past 16 rows (the captured branch) and hook._eager_legacy (prompt chunks without
   the prompt path), with hook.STAGE_ALL off and on, a stand-in MegaMoE that records its row ids, stand-in stagers
   and small integer mean tables (exact sums): the same staging, the same row ids except the newly served picks
   (which point at a row holding their expert), and the mean substitute no longer added for those picks.
4. CPU only: unitk compiled for the GB300 (sm_103, Triton ahead of time, no device) in every mode, ALLROWS off / on.
5. GPU only: unitk with ALLROWS captured in a CUDA graph, replays on new inputs equal to eager calls; its time per
   call against ALLROWS off (printed).

CPU (the Triton interpreter; from the repository's engine/):
    docker run --rm --network none --cpuset-cpus 66 --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/engine:ro \
        --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_stage_all.py
GPU (GB300; parts 1-3 on the device, then part 5):
    docker run --rm --network none --device nvidia.com/gpu=$GB --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_stage_all.py
Exit 0 and "STAGE ALL: PASS" = every check passed. K31_TEST_CPU=1 / 0 forces the mode (default: GPU if visible).
"""
import os
import random
import sys

AOT = os.environ.get("K31_TEST_AOT") == "1"         # the child process of part 4
CPU = not AOT and os.environ.get("K31_TEST_CPU", "0" if os.path.exists("/dev/nvidiactl") else "1") == "1"
if CPU:
    os.environ["TRITON_INTERPRET"] = "1"
os.environ["K31_MODE"] = "staged"
for k in ("K31_STAGE_ALL", "K31_DEFER", "K31_PROMPT_PATH", "K31_RES_MAXRANK"):
    os.environ.pop(k, None)

import torch  # noqa: E402
import triton.language as tl  # noqa: E402

if CPU:      # PDL instructions have no interpreter form; without the launch attribute they are no-ops on the GPU too
    tl.extra.cuda.gdc_wait = lambda *a, **k: None
    tl.extra.cuda.gdc_launch_dependents = lambda *a, **k: None
sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import hook, unitk  # noqa: E402


def aot():
    """Compile unitk for sm_103 in every mode (no device needed); prints "AOT n_ok n_all cubin_off cubin_on"."""
    import inspect

    import triton
    from triton.backends.compiler import GPUTarget
    names = list(inspect.signature(unitk._unitk.fn).parameters)
    n_ok, n_all, sizes = 0, 0, {}
    for allrows in (False, True):
        for TB in (1, 8, 16):
            for score, pre, acc in ((False, False, False), (True, False, True), (False, True, False)):
                sig = {n: "*i32" for n in ("ids_ptr", "row_ptr", "bank_ptr", "out_ptr", "stage_ptr", "stats_ptr",
                                           "pre_ptr", "pid_ptr", "pt_ptr")}
                sig.update({"w_ptr": "*fp32", "T": "i32", "strong": "i32", "S": "i32", "resmax": "i32", "decay": "fp32",
                            "score_ptr": "*fp32" if score else "*i32", "pw_ptr": "*fp32" if acc else "*i32"})
                cx = {"TB": TB, "K": 16, "SPC": 4, "SLOTSC": 16, "NE": 1024, "SCORE": score, "PRE": pre, "ACC": acc,
                      "ALLROWS": allrows}
                sig.update({k: "constexpr" for k in cx})
                n_all += 1
                try:
                    src = triton.compiler.ASTSource(fn=unitk._unitk, signature=sig,
                                                    constexprs={(names.index(k),): v for k, v in cx.items()})
                    kern = triton.compile(src, target=GPUTarget("cuda", 103, 32),
                                          options={"num_warps": 4 if TB <= 4 else 8})
                    sizes[(allrows, TB, score)] = len(kern.asm["cubin"])
                    n_ok += 1
                except Exception as exc:                 # noqa: BLE001
                    print(f"  ALLROWS={allrows} TB={TB} SCORE={score} PRE={pre}: {type(exc).__name__}: {exc}"[:600])
    print(f"AOT {n_ok} {n_all} {sizes.get((False, 8, True), 0)} {sizes.get((True, 8, True), 0)}", flush=True)
    sys.exit(0 if n_ok == n_all else 1)


if AOT:
    aot()
dev = torch.device("cpu" if CPU else "cuda")
E, K, SL = 896, 16, unitk.SLOTS
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


def plan_ref(st, ids, w, strong, S, resmax=16, allrows=False, pre=None):
    """Mode staged's one-round plan from its definition. Candidates: an expert id, no resident row, a bank record,
    rank < strong (rank: by weight within the token, ties by lower position). Staged experts: the candidates' distinct
    experts by their largest candidate weight (ties: lower id), the first 16 in slots 0, 1, ... (PRE: the slots hold
    `pre`'s bank indices, a pick takes the highest matching slot). Rows: a resident pick of rank < resmax -> its row;
    a candidate of a staged expert -> S + slot; with allrows any non-resident pick of a staged expert -> S + slot;
    else -1. -> (row ids [T][16], stage list [16] or None in PRE mode)"""
    row, bidx = st.row, st.bank_idx
    T = len(ids)
    rk = [ranks(w[t]) for t in range(T)]
    cand = [[ids[t][k] >= 0 and row[ids[t][k]] < 0 and bidx[ids[t][k]] >= 0 and rk[t][k] < strong for k in range(K)]
            for t in range(T)]
    if pre is None:
        best = {}
        for t in range(T):
            for k in range(K):
                if cand[t][k]:
                    e = ids[t][k]
                    best[e] = max(best.get(e, -1.0), w[t][k])
        order = sorted(best, key=lambda e: (-best[e], e))[:SL]
        slot_of = {e: s for s, e in enumerate(order)}
        stage = [bidx[e] for e in order] + [-1] * (SL - len(order))

        def slot(e):
            return slot_of.get(e, -1)
    else:
        stage = None

        def slot(e):
            m = [s for s in range(SL) if pre[s] >= 0 and pre[s] == bidx[e]]
            return max(m) if m else -1
    out = [[-1] * K for _ in range(T)]
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            if e < 0:
                continue
            if row[e] >= 0:
                out[t][k] = row[e] if rk[t][k] < resmax else -1
                continue
            s = slot(e)
            if s >= 0 and (cand[t][k] or allrows):
                out[t][k] = S + s
    return out, stage


def kernel_plan(st, ids, w, strong, resmax=16, allrows=False, pre=None, score=None, stats=None, lam=0.9):
    ids_d = torch.tensor(ids, dtype=torch.int32, device=dev)
    w_d = torch.tensor(w, dtype=torch.float32, device=dev)
    stage = torch.full((SL,), 77, dtype=torch.int32, device=dev)
    out, stage = unitk.plan(ids_d, w_d, st.row_map, st.bank_map, strong, st.slot_base, stage=stage, resmax=resmax,
                            allrows=allrows, pre=pre, score=score, lam=lam if score is not None else 1.0, stats=stats)
    return out.tolist(), stage.tolist()


def score_base(sc0, lam, T):
    """The decayed scores as the kernel stores them before the adds: float32(sc0) x float32(lam^T)."""
    return sc0.float().cpu() * torch.tensor(float(lam ** T), dtype=torch.float32)


def score_ref(sc0, lam, ids, w):
    """The residency scores after one call, in float64: the decayed scores + w^2 of every pick with an expert id."""
    want = score_base(sc0, lam, len(ids)).double()
    for t, row in enumerate(ids):
        for k, e in enumerate(row):
            if e >= 0:
                want[e] += w[t][k] * w[t][k]
    return want


def score_f32(sc0, lam, ids, w, order_rng):
    """The same sum in float32 with the adds in a random order, as the GPU's atomics may make them."""
    acc = score_base(sc0, lam, len(ids)).clone()
    picks = [(t, k, e) for t, row in enumerate(ids) for k, e in enumerate(row) if e >= 0]
    order_rng.shuffle(picks)
    for t, k, e in picks:
        acc[e] = acc[e] + torch.tensor(w[t][k] * w[t][k], dtype=torch.float32)
    return acc


# 0. the GB300-only table for this mode (profiles/nortx-t92-allrows.env) compiles to the staged ranks it lists
from k31_engine import policy  # noqa: E402

prof = os.path.join(os.environ.get("ENGINE", "/engine"), "profiles")
pp = policy.load(os.path.join(prof, "policy-nortx-t92-allrows.json"))
want_strong = {L: 1 for L in range(1, 93)}
for spec, s in (("12-13,21,24,28", 0), ("8,41-43,45-47,50-51,61-63,68,76-77,83,90", 2),
                ("1,6,11,49,52-60,72-73,81-82,92", 4)):
    for L in policy.parse_layers(spec):
        want_strong[L] = s
env_ok = any(ln.startswith("export ") and "K31_STAGE_ALL=1" in ln.split()
             for ln in open(os.path.join(prof, "nortx-t92-allrows.env")).read().splitlines())
check("policy-nortx-t92-allrows.json: staged ranks per layer as listed, band off, mean on, hit all; profile sets "
      "K31_STAGE_ALL=1", all(pp[L].strong == want_strong[L] and not pp[L].band and pp[L].mean and pp[L].resmax == 16
                             and pp[L].defer is None for L in range(1, 93)) and env_ok,
      f"({sum(want_strong.values())} staged ranks over 92 layers)")

# 1. + 2. unitk against the reference, both modes, and the difference between the modes
rng = random.Random(5)
n_trials = 800 if not CPU else 400
bad = {False: 0, True: 0}
bad_pre, bad_delta, n_new, n_over = 0, 0, 0, 0
for trial in range(n_trials):
    if trial % 3 == 2:          # more candidate experts than slots: the 16th place and its ties decide
        st = make_layer(rng, rng.randint(0, 100), rng.randint(400, 700))
        T, strong = rng.choice([8, 12, 16]), 4
        resmax = rng.choice([16, 8])
        ids, w = make_call(rng, T, rng.randint(100, 400))
    else:
        st = make_layer(rng, rng.randint(0, 300), rng.randint(0, 500))
        T = rng.choice([1, 2, 3, 4, 5, 5, 5, 8, 8, 12, 16])
        strong = rng.randint(1, 4)
        resmax = rng.choice([16, 16, 16, 12, 8, 4])
        ids, w = make_call(rng, T, rng.randint(16, 200))
    res = {}
    for allrows in (False, True):
        got = kernel_plan(st, ids, w, strong, resmax, allrows)
        want = plan_ref(st, ids, w, strong, st.slot_base, resmax, allrows)
        res[allrows] = got
        if got != (want[0], want[1]):
            bad[allrows] += 1
            if bad[allrows] <= 3:
                nd = sum(a != b for ra, rb in zip(got[0], want[0]) for a, b in zip(ra, rb))
                print(f"  trial {trial} allrows={allrows} T={T} strong={strong} resmax={resmax}: {nd} picks differ, "
                      f"stage {got[1] == want[1]}")
    n_over += sum(1 for s in res[False][1] if s >= 0) == SL
    # the difference: same stage list; ids differ only where today's plan cuts a pick of a staged expert
    staged_bank = {s for s in res[False][1] if s >= 0}
    ok = res[False][1] == res[True][1]
    for t in range(T):
        for k in range(K):
            a, b = res[False][0][t][k], res[True][0][t][k]
            if a == b:
                continue
            e = ids[t][k]
            sb = st.slot_base
            new_ok = a == -1 and e >= 0 and st.row[e] < 0 and st.bank_idx[e] in staged_bank and \
                sb <= b < sb + SL and res[True][1][b - sb] == st.bank_idx[e]
            ok = ok and new_ok
            n_new += new_ok
    bad_delta += not ok
    # PRE mode: slots staged ahead (today's stage list with one slot dropped and one foreign record added)
    pre = list(res[False][1])
    used = [s for s in range(SL) if pre[s] >= 0]
    if used:
        pre[rng.choice(used)] = -1
    free = [s for s in range(SL) if pre[s] < 0]
    if free and st.bank:
        pre[rng.choice(free)] = rng.randrange(len(st.bank))
    pre_d = torch.tensor(pre, dtype=torch.int32, device=dev)
    for allrows in (False, True):
        got = kernel_plan(st, ids, w, strong, resmax, allrows, pre=pre_d)
        want = plan_ref(st, ids, w, strong, st.slot_base, resmax, allrows, pre=pre)
        if got[0] != want[0] or got[1] != [77] * SL:
            bad_pre += 1
check("unitk, today's plan (ALLROWS off) = the reference: every row id and the stage list", bad[False] == 0,
      f"({n_trials} calls, {n_over} with all 16 slots used; {bad[False]} differ)")
check("unitk with ALLROWS = the reference", bad[True] == 0, f"({bad[True]} differ)")
check("between the modes: the same stage list; only cut picks of staged experts change, to their expert's slot",
      bad_delta == 0 and n_new > 0, f"({n_new} picks newly served; {bad_delta} calls wrong)")
check("PRE mode (slots staged ahead), both modes = the reference; the stage list untouched", bad_pre == 0,
      f"({bad_pre} differ)")

# residency inputs: the newly served picks move from cut to staged; the scores do not depend on the mode. The scores
# are float32 sums made by one atomic add per pick, and experts repeat across rows: the GPU does not fix the order of
# the adds to one score, so with arbitrary values the last bits can differ between two launches of either mode (a
# partial sum that crosses a power of two rounds at another point). Bit equality is asked only where every partial sum
# is exact (no decay, values on a 1/64 grid: any order gives the same bits); elsewhere each mode is held to the float64
# sum at rtol 1e-5 (as tests/test_unitk.py), which still catches a lost or doubled add (>= 1/64) or a wrong decay.
bad_sc, bad_exact, bad_tol, worst = 0, 0, 0, 0.0
ord_rng = random.Random(17)
n_ord, n_ord_bits, bad_ord = 0, 0, 0
for trial in range(60):
    st = make_layer(rng, rng.randint(50, 300), rng.randint(100, 500))
    T = rng.choice([1, 3, 5, 8, 16])
    ids, w = make_call(rng, T, rng.randint(16, 120))
    for exact in (True, False):
        lam = 1.0 if exact else 0.9
        sc0 = (torch.randint(0, 512, (E,)).float() / 64 if exact else torch.rand(E)).to(dev)
        want = score_ref(sc0, lam, ids, w)
        got = {}
        for allrows in (False, True):
            score, stats = sc0.clone(), torch.zeros(4, dtype=torch.int32, device=dev)
            out, _ = kernel_plan(st, ids, w, 3, 16, allrows, score=score, stats=stats, lam=lam)
            got[allrows] = (score.cpu(), stats.tolist(), out)
        newly = sum(a != b for ra, rb in zip(got[False][2], got[True][2]) for a, b in zip(ra, rb))
        s0, s1 = got[False][1], got[True][1]
        bad_sc += not (s1[0] == s0[0] and s1[1] == s0[1] + newly and s1[2] == s0[2] - newly and s1[3] == s0[3])
        for allrows in (False, True):
            sc = got[allrows][0]
            if exact:
                bad_exact += not torch.equal(sc.double(), want)
            else:
                worst = max(worst, float(((sc.double() - want).abs() / want.abs().clamp_min(1e-6)).max()))
                bad_tol += not torch.allclose(sc.double(), want, rtol=1e-5, atol=1e-6)
        if exact:
            bad_exact += not torch.equal(got[False][0], got[True][0])
        if CPU:          # the check against other add orders: what the GPU's atomics may do
            for _ in range(3):
                alt = score_f32(sc0, lam, ids, w, ord_rng)
                n_ord += 1
                if exact:
                    bad_ord += not torch.equal(alt, got[False][0])
                else:
                    n_ord_bits += not torch.equal(alt, got[False][0])
                    bad_ord += not torch.allclose(alt.double(), want, rtol=1e-5, atol=1e-6)
check("residency inputs: staged + newly served, cut - newly served, the other counts equal", bad_sc == 0,
      f"({bad_sc} of 120 calls differ)")
check("residency scores, both modes = the float64 sum: bit-equal where every partial sum is exact, else rtol 1e-5",
      bad_exact == 0 and bad_tol == 0, f"({bad_exact} exact-grid and {bad_tol} decayed calls differ; worst relative "
      f"error {worst:.1e})")
if CPU:
    check("the score check holds for any add order (float32 sums in shuffled orders, as the GPU's atomics make them)",
          bad_ord == 0 and n_ord_bits > 0, f"({n_ord} orders: exact grid bit-equal, decayed values within the "
          f"tolerance; {n_ord_bits} of {n_ord // 2} decayed orders differ in the last bits from the kernel's, which "
          f"is why bit equality between two launches cannot be asked there; {bad_ord} fail)")

# 3. hook's torch rounds (calls past 16 rows)
D = 512
calls, staged = [], []


def fake_forward(self, hidden_states, topk_weights, ids, shared_experts=None):
    calls.append(ids.clone())
    return torch.zeros(hidden_states.shape[0], D, dtype=torch.bfloat16, device=hidden_states.device)


class FakeMoE:
    def __init__(self, st):
        self._k31 = st


def holder(st, rid, stage_list, base, nslots):
    """The expert row `rid` holds: a staging row base + s -> the expert of stage_list[s]; a resident row -> its
    expert; -1 -> -1."""
    if rid < 0:
        return -1
    if base <= rid < base + nslots:
        b = stage_list[rid - base]
        return st.bank[b] if b >= 0 else -2
    if rid >= st.slots:
        return st.hbm[rid - st.slots]
    return -3


def mean_ref(st, ids, w, served_mask, mean):
    """sum over the not-served picks with an expert id of w x mean[e] (exact: small integers x multiples of 1/8)."""
    T = len(ids)
    y = torch.zeros(T, D, dtype=torch.float64)
    for t in range(T):
        for k in range(K):
            e = ids[t][k]
            if e >= 0 and not served_mask[t][k]:
                y[t] += w[t][k] * mean[e].double().cpu()
    return y


hook._orig["forward"] = fake_forward
orig_capturing = torch.cuda.is_current_stream_capturing
rng = random.Random(9)
bad_cap, bad_eag, new_cap, new_eag, n_rounds2 = 0, 0, 0, 0, 0
try:
    for trial in range(24 if CPU else 60):
        captured = trial % 2 == 0
        big = not captured and trial % 6 == 1            # several eager rounds of 128 staged records
        st = make_layer(rng, rng.randint(0, 60) if big else rng.randint(50, 250),
                        rng.randint(600, 800) if big else rng.randint(200, 500), idx=rng.randint(1, 92))
        st.score = None
        st.band_map = None
        mean = torch.randint(-2, 3, (E, D)).to(torch.bfloat16)
        st.cut_mean = mean.to(dev).contiguous()
        T = rng.choice([17, 20, 24, 40]) if captured else (rng.choice([160, 240]) if big else rng.choice([20, 40, 64]))
        strong = rng.randint(1, 4) if captured else 4
        st.strong = strong
        ids, w = make_call(rng, T, rng.randint(600, 896) if big else rng.randint(40, 896 if not captured else 300))
        ids_d = torch.tensor(ids, dtype=torch.int32, device=dev)
        w_d = torch.tensor(w, dtype=torch.float32, device=dev)
        x = torch.zeros(T, 8, dtype=torch.bfloat16, device=dev)
        res = {}
        for mode in (False, True):
            hook.STAGE_ALL = mode
            calls.clear()
            staged.clear()
            st.stager = lambda ptr, stage: staged.append(stage.clone().tolist())
            st.stagers = [(lambda g: (lambda ptr, stage: staged.append((g, stage.clone().tolist()))))(g)
                          for g in range(hook.GROUPS)]
            torch.cuda.is_current_stream_capturing = (lambda: True) if captured else \
                ((lambda: False) if CPU else orig_capturing)
            y = hook._forward(FakeMoE(st), x, w_d, ids_d)
            res[mode] = ([c.tolist() for c in calls], list(staged), y.float().cpu())
        hook.STAGE_ALL = False
        torch.cuda.is_current_stream_capturing = orig_capturing
        (c0, s0, y0), (c1, s1, y1) = res[False], res[True]
        ok = s0 == s1 and len(c0) == len(c1)
        served = {False: [[False] * K for _ in range(T)], True: [[False] * K for _ in range(T)]}
        if ok:
            if captured:
                lists = [s0[0]]
                bases, nsl = [st.slot_base], SL
            else:                                        # 8 group calls per round, each 16 slots of the round's 128
                lists = []
                for r in range(len(c0)):
                    grp = sorted(s0[8 * r:8 * r + 8])
                    lists.append([b for _, lst in grp for b in lst])
                bases, nsl = [0] * len(c0), hook.STAGE_ROWS
                n_rounds2 += len(c0) > 1
            for r in range(len(c0)):
                for t in range(T):
                    for k in range(K):
                        a, b = c0[r][t][k], c1[r][t][k]
                        e = ids[t][k]
                        ha, hb = holder(st, a, lists[r], bases[r], nsl), holder(st, b, lists[r], bases[r], nsl)
                        if ha >= 0:
                            ok = ok and ha == e
                            served[False][t][k] = True
                        if hb >= 0:
                            ok = ok and hb == e
                            served[True][t][k] = True
                        if a != b:               # only a pick that today's plan leaves out, now on its expert's row
                            good = a == -1 and e >= 0 and st.row[e] < 0 and hb == e
                            ok = ok and good
                            if captured:
                                new_cap += good
                            else:
                                new_eag += good
            for mode in (False, True):
                want = mean_ref(st, ids, w, served[mode], mean)
                ok = ok and torch.equal(res[mode][2].double(), want)
        if captured:
            bad_cap += not ok
        else:
            bad_eag += not ok
finally:
    torch.cuda.is_current_stream_capturing = orig_capturing
    hook.STAGE_ALL = False
check("hook: calls past 16 rows in a captured graph (one round): same staging; only newly served picks change, "
      "onto their expert's row; no mean for them", bad_cap == 0 and new_cap > 0,
      f"({new_cap} picks newly served; {bad_cap} calls wrong)")
check("hook: eager rounds (_eager_legacy): the same, round by round", bad_eag == 0 and new_eag > 0,
      f"({new_eag} picks newly served, {n_rounds2} calls of more than one round; {bad_eag} calls wrong)")

# 4. CPU: the kernel compiled for the GB300 (sm_103, Triton ahead of time, no device) in every mode, ALLROWS off / on;
# in a child process without the interpreter (it refuses the PDL inline asm), see aot() at the top
if CPU:
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env["K31_TEST_AOT"] = "1"
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    line = ([ln for ln in r.stdout.splitlines() if ln.startswith("AOT ")] or ["AOT 0 1 0 0"])[-1].split()
    n_ok, n_all, s_off, s_on = (int(v) for v in line[1:5])
    if n_ok != n_all:
        print((r.stdout + r.stderr)[-3000:])
    check("unitk compiles for sm_103 (Triton ahead of time) in every mode, ALLROWS off and on", n_ok == n_all and
          r.returncode == 0, f"({n_ok} of {n_all}; cubin at 8 rows with residency inputs: {s_off} / {s_on} B)")

# 5. GPU: a CUDA graph and the time per call
if not CPU:
    st = make_layer(rng, 150, 400)
    bad_g = 0
    times = {}
    for T in (1, 3, 5, 8, 16):
        for allrows in (False, True):
            ids_b = torch.zeros((T, K), dtype=torch.int32, device=dev)
            w_b = torch.zeros((T, K), dtype=torch.float32, device=dev)
            out = torch.empty((T, K), dtype=torch.int32, device=dev)
            stage = torch.empty(SL, dtype=torch.int32, device=dev)

            def load():
                ids, w = make_call(rng, T, rng.randint(16, 200))
                ids_b.copy_(torch.tensor(ids, dtype=torch.int32))
                w_b.copy_(torch.tensor(w, dtype=torch.float32))
                return ids, w
            load()
            for _ in range(3):
                unitk.plan(ids_b, w_b, st.row_map, st.bank_map, 3, st.slot_base, out, stage, allrows=allrows)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                unitk.plan(ids_b, w_b, st.row_map, st.bank_map, 3, st.slot_base, out, stage, allrows=allrows)
            for rep in range(20):
                ids, w = load()
                g.replay()
                torch.cuda.synchronize()
                want = plan_ref(st, ids, w, 3, st.slot_base, 16, allrows)
                bad_g += (out.tolist(), stage.tolist()) != (want[0], want[1])
            best = 1e9
            for _ in range(50):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                best = min(best, a.elapsed_time(b) * 1e3)
            times[(T, allrows)] = best
    check("GPU: unitk with and without ALLROWS in a CUDA graph, 20 replays per size on new inputs = the reference",
          bad_g == 0, f"({bad_g} differ)")
    print("unitk in a CUDA graph, us (ALLROWS off / on): " +
          ", ".join(f"T={T} {times[(T, False)]:.1f} / {times[(T, True)]:.1f}" for T in (1, 3, 5, 8, 16)))
print(f"mode: {'CPU (Triton interpreter)' if CPU else 'GPU'}")
print("STAGE ALL:", "PASS" if not fails else f"FAIL ({', '.join(fails)})", flush=True)
sys.exit(1 if fails else 0)
