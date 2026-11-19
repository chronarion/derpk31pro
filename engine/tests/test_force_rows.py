"""Row-range exact KL (k31_engine/force.py, requests with k31_kl_rows; harness/kld_long.py) against the whole-block
path the 2,048-token protocol uses.

A request flows through ForceTokens as vLLM drives it: add_request, apply_staged_writes, its prompt in chunks through
_prompt_kl (the wrapped prompt-logprob worker; another request's rows in the same forward), then verify steps through
apply (1 + 4 rows per step, a random number of them accepted, the rest with wrong inputs as after a rejection, another
request's rows in the batch), and the saved per-position KL (k31_kl_out) is compared:
  1. whole block (k31_kl_hidden as today) vs row range [0, T) of the same file: bit-equal, NaN at the same rows;
  2. row ranges [A, B) (inside the prompt, across the prompt's end, forced rows only; with prompt chunks of 2,048 /
     600 rows and of 700 / 1,024 / 333): bit-equal to (1) there, NaN elsewhere (the failing rows are listed); the GPU
     entry holds only the forced rows; the pinned host block is released after the prompt;
  3. the same reference as the runner's --dump-rows file (rows [O, T) + FILE.json) and as two halves (a comma list,
     a global block index): bit-equal to (1) where they hold rows;
  4. prefetch (k31_kl_next) of the next block, and a prefetch of the wrong block: equal to (1);
  5. the values against an independent float64 KL from the same reference and engine logits (rel < 1e-4);
  6. k31_kl_ref (full log-probabilities instead of hidden states): row range vs whole block, bit-equal;
  7. GPU only: the time of the prompt rows' KL per 1,000 rows, and the GPU memory each path holds per request.
Both devices: T 4,096 rows, a 3,000-token prompt. TEST_DEV=cpu: H 256, V 4,096, and SHAPE_EMU (below) on, since CPU
BLAS gives a row the same bits whatever the call's row count and the GPU does not. GPU (default): H 7,168, V 163,840
(the model's) on the GB300:

  docker run --rm --device nvidia.com/gpu=$GB --network none --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD/engine:/engine:ro -v /tmp/k31-test:/work -e TEST_DIR=/work \
    --entrypoint python3 vllm/vllm-openai:kimi-k31 /engine/tests/test_force_rows.py

(TEST_DIR: ~1 GB of reference files at GPU shapes; CPU: a few MB, default a temporary directory.) Exit 0 = PASS.
"""
import os
import random
import sys
import tempfile
import time
import types

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from k31_engine import force, hook  # noqa: E402

dev = torch.device(os.environ.get("TEST_DEV", "cuda"))
GPU = dev.type == "cuda"
H, V = (7168, 163840) if GPU else (256, 4096)
N, T, P = 3, 4096, 3000                  # the same geometry on both devices
SIZES = [2048, 600]                      # prompt chunk sizes, cycled: chunks [0, 2048), [2048, 2648), [2648, 3000)
S = 4                                    # drafts per verify step
# A GEMM's (and a row reduction's) result for one row depends on the call's shape on the GPU (cuBLAS picks the kernel,
# split-K and tiles by shape), not on CPU BLAS. SHAPE_EMU (default on CPU) emulates that: both LM-head products are
# scaled by 1 + rows x 2^-26, so a row computed in a call of another row count than the whole-block path's comes out
# different, as on the GPU.
SHAPE_EMU = os.environ.get("SHAPE_EMU", "0" if GPU else "1") == "1"
SLOT, OTHER, MAXLEN = 2, 0, 8192
WORK = os.environ.get("TEST_DIR") or tempfile.mkdtemp(prefix="k31-force-rows-")
os.makedirs(WORK, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def same(a, b):
    """Bit-equal float32 arrays (NaN at the same places count as equal)."""
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def differ(a, b, lo=0):
    """Which rows of a and b are not bit-equal: '' or ' / rows not bit-equal: N in [first, last]' (offset lo)."""
    d = np.flatnonzero(a.view(np.uint32) != b.view(np.uint32))
    return f" / rows not bit-equal: {d.size} in [{lo + d[0]}, {lo + d[-1]}]" if d.size else ""


g = torch.Generator().manual_seed(11)
W = (torch.randn(V, H, generator=g) * (2.5 / H ** 0.5)).to(torch.bfloat16)
hook.LM_HEAD[0] = W.to(dev)
Hr = torch.randn(N, T, H, generator=g).to(torch.bfloat16)                     # the reference's final hidden states
He = (Hr.float() + 0.15 * torch.randn(N, T, H, generator=g)).to(torch.bfloat16)   # the engine's
ids_all = torch.randint(0, V, (N, T), generator=g)


def write_ref(path, x, row_start=None, seq_len=None):
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=tuple(x.shape))
    mm[:] = x.view(torch.int16).numpy().view(np.uint16)
    mm.flush()
    del mm
    if row_start is not None:
        import json
        json.dump({"row_start": row_start, "row_end": row_start + x.shape[1], "seq_len": seq_len},
                  open(path + ".json", "w"))
    elif os.path.exists(path + ".json"):
        os.remove(path + ".json")


FULL = f"{WORK}/hidden.npy"
write_ref(FULL, Hr)
H0, H1 = f"{WORK}/hidden-h0.npy", f"{WORK}/hidden-h1.npy"
write_ref(H0, Hr[:1])
write_ref(H1, Hr[1:])
O = T - 1500                                                    # a --dump-rows file: rows [O, T) of every sequence
TAILF = f"{WORK}/hidden-tail.npy"
write_ref(TAILF, Hr[:, O:], row_start=O, seq_len=T)


def logits_fn(h):
    out = h.to(dev) @ hook.LM_HEAD[0].T                          # bf16 logits, as the LM head returns them
    return out.float() * (1 + h.shape[0] * 2 ** -26) if SHAPE_EMU else out


if SHAPE_EMU:
    _ref_logits_real = force._ref_logits

    def _ref_logits_emu(h, V_, chunk=16384):
        return _ref_logits_real(h, V_, chunk) * (1 + h.shape[0] * 2 ** -26)
    force._ref_logits = _ref_logits_emu


def schedule(n, seed, sizes):
    """The request's forwards: prompt chunks (sizes cycled), then verify steps of 1 + S rows from the prompt's last
    row on, a random number accepted per step."""
    rnd = random.Random(seed)
    chunks, cs, j = [], 0, 0
    while cs < P:
        ce = min(P, cs + sizes[j % len(sizes)])
        chunks.append((cs, ce))
        cs, j = ce, j + 1
    steps, F = [], P - 1
    while F <= T - 2:
        acc = rnd.choice([0, 1, 1, 2, 2, 3, 4])
        good = [F + d <= F + acc for d in range(S + 1)]        # rows past the first rejected draft see wrong inputs
        steps.append((F, good))
        F += acc + 1
    return chunks, steps


def step_logits(n, F, good, seed):
    """[rows, V] float32 logits of one verify step: this request's 1 + S rows at positions F .. F + S (wrong-input
    rows and rows past the sequence: other logits), then two rows of the other request."""
    gl = torch.Generator().manual_seed(seed)
    pos = [F + d for d in range(S + 1)]
    lg = logits_fn(He[n, [min(q, T - 1) for q in pos]]).float()
    junk = torch.randn(S + 1, V, generator=gl).to(dev) * 3
    keep = torch.tensor([ok and q < T for ok, q in zip(good, pos)], device=dev).unsqueeze(1)
    lg = torch.where(keep, lg, junk)
    other = torch.randn(2, V, generator=gl).to(dev) * 3
    return torch.cat([lg, other]), pos


def run(n, extra, out_path, seed=5, before=None, sizes=SIZES):
    """One request for sequence n through a fresh ForceTokens; returns (saved KL [T], the slot's entry after the
    prompt, peak GPU bytes the request's entry added)."""
    cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=MAXLEN))
    rs = types.SimpleNamespace(device=dev, max_num_reqs=4)
    ft = force.ForceTokens(cfg, rs)
    if before is not None:
        before(ft)
    ids = ids_all[n].tolist()
    xa = {"k31_force": ids[P:], "k31_start": P, "k31_kl_index": n, "k31_kl_out": out_path, **extra}
    if os.path.exists(out_path):
        os.remove(out_path)
    if GPU:
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
    ft.add_request(SLOT, types.SimpleNamespace(extra_args=xa))
    ft.add_request(OTHER, types.SimpleNamespace(extra_args={}))
    ft.apply_staged_writes()
    held = (torch.cuda.memory_allocated() - m0) if GPU else 0
    chunks, steps = schedule(n, seed, sizes)
    prompt_lens = np.zeros(4, dtype=np.int64)
    prompt_lens[SLOT], prompt_lens[OTHER] = P, 900
    for cs, ce in chunks:                       # the other request decodes one row in each of these forwards
        hs = torch.cat([He[(n + 1) % N, 900 + cs % 50:901 + cs % 50], He[n, cs:ce]]).to(dev)
        ib = types.SimpleNamespace(idx_mapping_np=np.array([OTHER, SLOT]),
                                   num_computed_prefill_tokens_np=np.array([950, cs]),
                                   num_scheduled_tokens=np.array([1, ce - cs]),
                                   query_start_loc_np=np.array([0, 1, 1 + ce - cs]))
        force._prompt_kl(logits_fn, hs, ib, prompt_lens)
    entry = dict(ft.kl[SLOT])
    for si, (F, good) in enumerate(steps):
        lg, pos = step_logits(n, F, good, seed * 100003 + si)
        rows_pos = torch.tensor(pos + [960 + si, 961 + si], device=dev)
        slots = torch.tensor([SLOT] * (S + 1) + [OTHER] * 2, device=dev, dtype=torch.int32)
        ctx = types.SimpleNamespace(expanded_idx_mapping=slots, pos=rows_pos, idx_mapping_np=np.array([SLOT, OTHER]),
                                    seq_lens_upper_bound_np=np.array([F + S + 1, 962 + si]))
        ft.apply(lg, ctx)
    return np.load(out_path), entry, held


def kl64(n, rows):
    """Independent KL(reference || engine) per row in float64: reference logits from Hr and the bf16 LM head in
    float64, engine logits as logits_fn gives them."""
    Wd = W.to(dev).double()
    out = []
    for a in range(0, len(rows), 256):
        r = torch.tensor(rows[a:a + 256])
        ref = torch.log_softmax(Hr[n, r].to(dev).double() @ Wd.T, -1)
        eng = torch.log_softmax(logits_fn(He[n, r]).double(), -1)
        out.append((ref.exp() * (ref - eng)).sum(-1).cpu())
    return torch.cat(out).numpy()


n = 1
t0 = time.time()
base, e_old, held_old = run(n, {"k31_kl_hidden": FULL}, f"{WORK}/old.npy")
check("whole-block path scores every row 0 .. T - 2", not np.isnan(base[:T - 1]).any(),
      f"({int(np.isnan(base[:T - 1]).sum())} NaN)")
rng, e_rng, held_rng = run(n, {"k31_kl_hidden": FULL, "k31_kl_rows": [0, T]}, f"{WORK}/rows.npy")
check("1. row range [0, T) = whole block, bit for bit", same(rng, base))
check("   the GPU holds only the forced rows", tuple(e_rng["ref"].shape) == (T - (P - 1), H) and e_rng["lo"] == P - 1,
      f"(ref {tuple(e_rng['ref'].shape)}, lo {e_rng['lo']}; whole block {tuple(e_old['ref'].shape)})")
check("   the host block is released after the prompt", e_rng["host"] is None)
for A, B in ((100, 1700), (P - 300, P + 500), (P + 7, T - 3)):
    r, e, _ = run(n, {"k31_kl_hidden": FULL, "k31_kl_rows": [A, B]}, f"{WORK}/r{A}.npy")
    inside = np.zeros(T, dtype=bool)
    inside[A:min(B, T - 1)] = True
    ok = same(r[inside], base[inside]) and np.isnan(r[:T - 1][~inside[:T - 1]]).all()
    check(f"2. rows [{A}, {B}): equal to the whole block there, NaN elsewhere", ok,
          f"(GPU rows {0 if e['ref'] is None else e['ref'].shape[0]}"
          f"{differ(r[A:min(B, T - 1)], base[A:min(B, T - 1)], A)})")
SIZES2 = [700, 1024, 333]                # chunk edges off the 512-row sub-chunk grid: ranges cut chunks and sub-chunks
base2, _, _ = run(n, {"k31_kl_hidden": FULL}, f"{WORK}/old2.npy", sizes=SIZES2)
bad = []
for A, B in ((100, 1700), (650, 2150), (700, 1224), (1211, 1212), (P - 300, P + 500)):
    r, _, _ = run(n, {"k31_kl_hidden": FULL, "k31_kl_rows": [A, B]}, f"{WORK}/s{A}.npy", sizes=SIZES2)
    inside = np.zeros(T, dtype=bool)
    inside[A:min(B, T - 1)] = True
    if not (same(r[inside], base2[inside]) and np.isnan(r[:T - 1][~inside[:T - 1]]).all()):
        bad.append(f"[{A}, {B}){differ(r[A:min(B, T - 1)], base2[A:min(B, T - 1)], A)}")
check(f"   prompt chunks of {'/'.join(map(str, SIZES2))} rows: five ranges (one row, chunk edges, across the prompt's "
      f"end) equal to the whole block", not bad, f"({'; '.join(bad)})" if bad else "")
r, _, _ = run(n, {"k31_kl_hidden": TAILF, "k31_kl_rows": [0, T]}, f"{WORK}/tail.npy")
ok = same(r[O:], base[O:]) and np.isnan(r[:O]).all()
check(f"3. a --dump-rows reference (rows [{O}, {T}) + .json): equal there, NaN before", ok, differ(r[O:], base[O:], O))
r, _, _ = run(n, {"k31_kl_hidden": f"{H0},{H1}", "k31_kl_rows": [0, T]}, f"{WORK}/halves.npy")
check("   two halves as one reference (global block index 1 = second file's block 0): equal", same(r, base))
r0, _, _ = run(0, {"k31_kl_hidden": f"{H0},{H1}", "k31_kl_rows": [0, T]}, f"{WORK}/halves0.npy")
b0, _, _ = run(0, {"k31_kl_hidden": FULL}, f"{WORK}/old0.npy")
check("   block 0 from the first half: equal", same(r0, b0))


def prefetch(block):
    def f(ft):                                  # an earlier request named `block` as the next one
        force._rows_start(FULL, block, 0, T)
    return f


r, _, _ = run(n, {"k31_kl_hidden": FULL, "k31_kl_rows": [0, T], "k31_kl_next": 2}, f"{WORK}/pf.npy", before=prefetch(n))
check("4. prefetched block: equal", same(r, base), f"(the next block queued: {force._RF['key']})")
r, _, _ = run(n, {"k31_kl_hidden": FULL, "k31_kl_rows": [0, T]}, f"{WORK}/pf2.npy", before=prefetch(0))
check("   a prefetch of another block is not used: equal", same(r, base))
rows5 = list(range(0, T - 1, 7 if GPU else 1))
x = kl64(n, rows5)
rel = float(np.abs(base[rows5] - x).max() / np.abs(x).max())
check("5. against an independent float64 KL", rel < 1e-4, f"(max abs diff / max {rel:.2e}; mean KL {x.mean():.4f})")

# 6. a full-log-probability reference (k31_kl_ref; runner --dump-logits: bf16 bits [N, T, V]), sequence 0 only
LP = f"{WORK}/logp.npy"
mm = np.lib.format.open_memmap(LP, mode="w+", dtype=np.uint16, shape=(1, T, V))
Wf = hook.LM_HEAD[0].float()
for a in range(0, T, 512):
    blk = torch.log_softmax(Hr[0, a:a + 512].to(dev).float() @ Wf.T, -1).to(torch.bfloat16)
    mm[0, a:a + 512] = blk.view(torch.int16).cpu().numpy().view(np.uint16)
mm.flush()
del mm, Wf
b6, _, _ = run(0, {"k31_kl_ref": LP}, f"{WORK}/lp-old.npy")
r6, _, _ = run(0, {"k31_kl_ref": LP, "k31_kl_rows": [0, T]}, f"{WORK}/lp-rows.npy")
check("6. k31_kl_ref: row range = whole block", same(r6, b6) and not np.isnan(b6[:T - 1]).any())
r6b, _, _ = run(0, {"k31_kl_ref": LP, "k31_kl_rows": [P + 1, T]}, f"{WORK}/lp-rows2.npy")
check("   k31_kl_ref rows [P + 1, T): equal there", same(r6b[P + 1:], b6[P + 1:]) and np.isnan(r6b[:P + 1]).all())

if GPU:                                         # 7. the cost of the prompt rows' KL and the GPU memory per request
    hs = He[n, :2048].to(dev)
    ib = types.SimpleNamespace(idx_mapping_np=np.array([SLOT]), num_computed_prefill_tokens_np=np.array([0]),
                               num_scheduled_tokens=np.array([2048]), query_start_loc_np=np.array([0, 2048]))
    cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=MAXLEN))
    ft = force.ForceTokens(cfg, types.SimpleNamespace(device=dev, max_num_reqs=4))
    xa = {"k31_force": ids_all[n, P:].tolist(), "k31_start": P, "k31_kl_index": n, "k31_kl_out": f"{WORK}/t.npy",
          "k31_kl_hidden": FULL, "k31_kl_rows": [0, T]}
    ft.add_request(SLOT, types.SimpleNamespace(extra_args=xa))
    ft.apply_staged_writes()
    pl = np.zeros(4, dtype=np.int64)
    pl[SLOT] = P
    host = ft.kl[SLOT]["host"]
    force._prompt_kl(logits_fn, hs, ib, pl)                            # warm-up
    ft.kl[SLOT]["host"] = host
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    t1 = time.time()
    force._prompt_kl(logits_fn, hs, ib, pl)
    torch.cuda.synchronize()
    dt = time.time() - t1
    peak = torch.cuda.max_memory_allocated() - m0
    print(f"INFO prompt-row KL: {1000 * dt / 2.048:.1f} ms per 1,000 rows (H {H}, V {V}), transient peak "
          f"{peak / 2**30:.2f} GiB; GPU bytes held per request: whole block {held_old / 2**20:.1f} MiB, row range "
          f"{held_rng / 2**20:.1f} MiB (this test's T = {T}; at 32,768 rows the whole block is "
          f"{32768 * H * 2 / 2**20:.0f} MiB, the row range keeps the forced rows only); host block pinned: "
          f"{host.is_pinned()}", flush=True)
    check("7. the row-range path's host block is pinned", host.is_pinned())
print(f"({time.time() - t0:.0f} s)")
print("FORCE ROWS:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
