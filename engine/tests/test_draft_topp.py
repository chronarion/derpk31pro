"""K31_DRAFT_TOPP / K31_DRAFT_ARGMAX masking (drafttopp.py) on the CPU: the Triton kernels run in Triton's interpreter.

Run in the engine image, CPU only (from engine/):
    docker run --rm --network none --cpuset-cpus 50-51 --user $(id -u):$(id -g) -e HOME=/tmp -e TRITON_INTERPRET=1 \
        -e CUDA_VISIBLE_DEVICES= -v "$PWD":/engine:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /engine/tests/test_draft_topp.py
(CUDA_VISIBLE_DEVICES empty keeps vLLM's Triton import alive without a GPU.) Exit 0 = PASS. About 2 minutes.

1. The kernels against drafttopp.reference (the same cut in torch) on random rows of 163,840 logits: masks equal (a row
   may differ by one bin only where a cumulative mass is within float rounding of top_p x Z).
2. The cut against vLLM's sort-based apply_top_k_top_p_pytorch on the temperature-scaled logits (the target's
   semantics): untouched rows are bit-identical (top_p 1 and no top-k, temperature 0, padding); the kept set is a top
   set holding the argmax, ties kept or dropped together; top-k alone keeps the exact top-k set plus at most the rest
   of the boundary bin (a cut more than 256 bins = 15 nats below the maximum masks nothing); the top-p cut lands
   within one bin (1/17 nat) of the sorted nucleus's last token in >= 90% of rows and never more than two bins away;
   the kept mass against top_p (reported).
3. Argmax mode: exactly the tokens equal to the row's maximum survive.
4. The patched DSparkSpeculator._sample_logits (drafttopp.patch on vLLM's class) through vLLM's own gumbel_sample on
   rows of mixed requests: every draft of a truncated row lies in the kept set, the draft logits cache holds the
   masked row bit for bit (what the verifier reads), and the drafts and cache of untouched rows equal the stock
   method's (same seeds) bit for bit.
"""
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import drafttopp as dt  # noqa: E402

fails = []
V = 163840


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def rows(g, kind, n):
    """n rows of logits of one family (bf16)."""
    if kind == "gauss":
        x = torch.randn(n, V, generator=g) * (1.0 + 3.0 * torch.rand(n, 1, generator=g))
    elif kind == "peaked":                        # a few strong tokens over a broad bulk, as LLM next-token rows
        x = torch.randn(n, V, generator=g) * 2.0
        top = torch.randint(0, V, (n, 8), generator=g)
        x.scatter_(1, top, 12.0 + 4.0 * torch.rand(n, 8, generator=g))
    elif kind == "zipf":
        rank = torch.argsort(torch.rand(n, V, generator=g), 1).float() + 1.0
        x = -(0.8 + 0.8 * torch.rand(n, 1, generator=g)) * torch.log(rank) * 3.0
    elif kind == "ties":                          # few distinct values: ties at every cut
        x = torch.randint(0, 12, (n, V), generator=g).float() * 0.75
    elif kind == "neginf":                        # a reduced vocabulary written into full rows
        x = torch.randn(n, V, generator=g) * 3.0
        x[:, torch.rand(V, generator=g) < 0.6] = float("-inf")
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def params(g, n, mode):
    """Per-row request parameters: (temperature, top_p, top_k) lists."""
    T = [float(v) for v in torch.tensor([0.5, 0.8, 1.0, 1.0, 1.3])[torch.randint(0, 5, (n,), generator=g)]]
    if mode == "p":
        P = [float(v) for v in torch.tensor([0.5, 0.8, 0.9, 0.95, 0.95, 0.99])[torch.randint(0, 6, (n,), generator=g)]]
        K = [V] * n
    elif mode == "k":
        P = [1.0] * n
        K = [int(v) for v in torch.tensor([1, 5, 20, 50, 1000])[torch.randint(0, 5, (n,), generator=g)]]
    else:
        P = [float(v) for v in torch.tensor([0.8, 0.95])[torch.randint(0, 2, (n,), generator=g)]]
        K = [int(v) for v in torch.tensor([20, 100, 4000])[torch.randint(0, 3, (n,), generator=g)]]
    return T, P, K


def scaled(x, t):
    """The kernels' scaled logits: x * (1 / T) in float32 (an IEEE reciprocal, then the product)."""
    return x.float() * (torch.tensor(1.0) / torch.tensor(t, dtype=torch.float32))


def run(x, T, P, K, idx=None, argmax=False):
    n = x.shape[0]
    idx = torch.arange(n, dtype=torch.int64) if idx is None else idx
    temp = torch.tensor(T, dtype=torch.float32)
    topp = torch.tensor(P, dtype=torch.float32)
    topk = torch.tensor(K, dtype=torch.int32)
    y = x.clone()
    dt.apply(y, idx, temp, topp, topk, dt.Scratch(n, V, "cpu"), argmax=argmax)
    ref, info = dt.reference(x, idx, temp, topp, topk, argmax=argmax)
    return y, ref, info


g = torch.Generator().manual_seed(11)

# 1 + 2: kernels against the reference, the cut against the sorted nucleus
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch  # noqa: E402

stats = {"p": [], "k": [], "pk": []}
mism, mism_far, near_ok, n_cmp, cut_far = 0, 0, 0, 0, 0
for kind in ("gauss", "peaked", "zipf", "ties", "neginf"):
    for mode in ("p", "k", "pk"):
        n = 6
        x = rows(g, kind, n)
        T, P, K = params(g, n, mode)
        y, ref, info = run(x, T, P, K)
        for r in range(n):
            same = torch.equal(y[r].view(torch.int16), ref[r].view(torch.int16))
            if not same:
                kept_a, kept_r = int(torch.isfinite(y[r]).sum()), int(torch.isfinite(ref[r]).sum())
                ya = scaled(x[r], T[r])[torch.isfinite(y[r])].min()
                yr = scaled(x[r], T[r])[torch.isfinite(ref[r])].min()
                one_bin = abs(math.floor(float(ya) * dt.INV_W) - math.floor(float(yr) * dt.INV_W)) <= 1
                mism += 1
                print(f"  kernel/reference differ: {kind} {mode} row {r} kept {kept_a} vs {kept_r} "
                      f"({'one bin' if one_bin else 'MORE than one bin'})")
                if not one_bin:
                    mism_far += 1
            # against vLLM's sorted top-k / top-p on the scaled logits (fp32, as the target's sampler)
            xf = scaled(x[r:r + 1], T[r])
            ex = apply_top_k_top_p_pytorch(xf.clone(), torch.tensor([K[r]]) if K[r] < V else None,
                                           torch.tensor([P[r]]) if P[r] < 1 else None, allow_cpu_sync=True)[0]
            keep_e = torch.isfinite(ex) & torch.isfinite(xf[0])
            keep_a = torch.isfinite(y[r].float())
            fin = torch.isfinite(xf[0])
            yv = xf[0]
            top_ok = bool(keep_a[torch.argmax(torch.where(fin, yv, torch.tensor(-float("inf"))))])
            drop_a = fin & ~keep_a
            shape_ok = (not bool(drop_a.any())) or float(yv[drop_a].max()) < float(yv[keep_a].min())
            if not (top_ok and shape_ok):
                check(f"cut is a top set holding the argmax ({kind} {mode} row {r})", False)
            pr = torch.softmax(torch.where(fin, yv, torch.tensor(-float("inf"))), 0)
            m_a, m_e = float(pr[keep_a].sum()), float(pr[keep_e].sum())
            b = torch.floor(yv * dt.INV_W)
            top_bin = b[fin].max()
            if mode == "k":
                kth = torch.topk(yv[fin], min(K[r], int(fin.sum()))).values[-1]
                topk_set = fin & (yv >= kth)
                if top_bin - torch.floor(kth * dt.INV_W) >= dt.H:      # the cut lies beyond the bins: nothing masked
                    ok = bool(keep_a[fin].all())
                else:
                    extra = keep_a & ~topk_set
                    bound = b[keep_a].min()
                    ok = bool((keep_a | ~topk_set).all()) and bool((b[extra] == bound).all())
                if not ok:
                    check(f"top-k keeps the top-k set plus the boundary bin only ({kind} row {r})", False)
                stats["k"].append((int(keep_a.sum()), int(topk_set.sum())))
            elif top_bin - b[keep_e].min() >= dt.H - 1:                 # the nucleus reaches past the bins
                if not bool(keep_a[fin].all()):
                    check(f"a nucleus past the bins masks nothing ({kind} {mode} row {r})", False)
            else:
                d = int(b[keep_e].min() - b[keep_a].min())        # > 0: the cut keeps deeper bins than the nucleus
                stats[mode].append((d, m_a, m_e, P[r]))
                n_cmp += 1
                near_ok += abs(d) <= 1
                cut_far += abs(d) > 2
check("kernels = reference (masks bit-equal; a row may differ by one bin at a float tie)",
      mism_far == 0 and mism <= 1, f"({mism} rows differ)")
ds = [s[0] for s in stats["p"] + stats["pk"]]
check("top-p cut within one bin of the sorted nucleus in >= 90% of rows, never more than two",
      near_ok >= 0.9 * n_cmp and cut_far == 0,
      f"({near_ok}/{n_cmp}; bin offsets min {min(ds)} max {max(ds)})")
dm = [s[1] - s[2] for s in stats["p"]]
print(f"  top-p rows: kept mass minus the sorted nucleus's mass: mean {sum(dm) / len(dm):+.4f}, "
      f"min {min(dm):+.4f}, max {max(dm):+.4f}; kept mass - top_p: mean "
      f"{sum(s[1] - s[3] for s in stats['p']) / len(dm):+.4f}")
ks = stats["k"]
print(f"  top-k rows: kept / top-k set sizes {ks[:12]}")

# untouched rows: top_p 1 and no top-k, temperature 0, padding
x = rows(g, "gauss", 4)
y, ref, _ = run(x, [1.0, 0.0, 1.0, 0.7], [1.0, 0.9, 0.9, 1.0], [V, V, 20, V],
                idx=torch.tensor([0, 1, -1, 3], dtype=torch.int64))
check("untouched rows bit-identical (top_p 1 + no top-k, temperature 0, padding row)",
      torch.equal(y.view(torch.int16), x.view(torch.int16)))

# 3. argmax mode
x = rows(g, "ties", 3)
x[1] = rows(g, "peaked", 1)[0]
x[2, :5] = 50.0                                          # five-way tie at the top
y, ref, _ = run(x, [1.0, 0.8, 1.2], [1.0, 0.95, 0.9], [V, V, 7], argmax=True)
ok = True
for r in range(3):
    xm = x[r].float().max()
    ok &= torch.equal(torch.isfinite(y[r].float()), x[r].float() == xm) and torch.equal(y[r], ref[r])
check("argmax mode keeps exactly the tokens equal to the row maximum (ties kept)", ok)

# 4. the patched speculator method through vLLM's gumbel_sample and draft logits cache
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as dsm  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache  # noqa: E402

stock = dsm.DSparkSpeculator._sample_logits
dt.patch(dsm)
assert dsm.DSparkSpeculator._sample_logits is not stock


def speculator(n_slots, steps, temps, tops, topks, argmax):
    sp = object.__new__(dsm.DSparkSpeculator)
    sp.acceptance_estimator = None
    sp.draft_watermarker = None
    sp._d2t_scatter_index = None
    sp.use_fp64_gumbel = False
    sp.device = torch.device("cpu")
    sp.max_num_reqs = n_slots
    sp.vocab_size = V
    sp.temperature = torch.tensor(temps, dtype=torch.float32)
    sp.seeds = torch.randint(-2 ** 62, 2 ** 62, (n_slots,), generator=g, dtype=torch.int64)
    sp._step_cols = torch.arange(steps, dtype=torch.int32)
    sp.draft_logits = DraftLogitsCache(n_slots, steps, V, dtype=torch.bfloat16, fill=0.0, device="cpu")
    ss = types.SimpleNamespace(top_p=types.SimpleNamespace(gpu=torch.tensor(tops, dtype=torch.float32)),
                               top_k=types.SimpleNamespace(gpu=torch.tensor(topks, dtype=torch.int32)))
    st = dt.State(sp, ss, argmax)
    st.refresh()
    return sp, st


# request slots: 0 top-p 0.95, 1 top-p 1 (untouched), 2 top-k 20 + top-p 0.9, 3 temperature 0, 4 top-p 0.8 at T 0.7
temps, tops, topks = [1.0, 1.0, 1.0, 0.0, 0.7], [0.95, 1.0, 0.9, 0.95, 0.8], [V, V, 20, V, V]
for argmax in (False, True):
    sp_on, st = speculator(5, 2, temps, tops, topks, argmax)
    sp_on._k31_dtopp_state = st
    sp_off, _ = speculator(5, 2, temps, tops, topks, argmax)
    sp_off.seeds = sp_on.seeds.clone()
    idx = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64)          # batch rows -> request slots
    pos = torch.arange(5, dtype=torch.int64) * 37 + 1000
    ok_kept, ok_cache, ok_stock = True, True, True
    for step in range(2):
        x = torch.cat([rows(g, "peaked", 3), rows(g, "zipf", 2)])
        lo = x.clone()
        d_on = sp_on._sample_logits(lo, idx, pos + step, step)        # patched: masks lo in place, then vLLM samples
        d_off = sp_off._sample_logits(x.clone(), idx, pos + step, step)
        for b in range(5):
            slot = int(idx[b])
            touched = temps[slot] > 0 and (argmax or tops[slot] < 1 or topks[slot] < V)
            ok_kept &= bool(torch.isfinite(lo[b, d_on[b]].float()))
            ok_cache &= torch.equal(sp_on.draft_logits.values[slot, step].view(torch.int16), lo[b].view(torch.int16))
            if not touched:
                ok_stock &= int(d_on[b]) == int(d_off[b]) and torch.equal(
                    sp_on.draft_logits.values[slot, step].view(torch.int16),
                    sp_off.draft_logits.values[slot, step].view(torch.int16))
            else:
                ok_kept &= bool((~torch.isfinite(lo[b].float())).any()) or argmax
    tag = "argmax" if argmax else "top-k/top-p"
    check(f"patched _sample_logits ({tag}): every draft lies in its row's kept set", ok_kept)
    check(f"patched _sample_logits ({tag}): the draft logits cache holds the masked row bit for bit", ok_cache)
    check(f"patched _sample_logits ({tag}): untouched rows' drafts and cache = the stock method's", ok_stock)

print("DRAFT TOPP MASK:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
