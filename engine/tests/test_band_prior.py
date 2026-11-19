"""K31_BAND_PRIOR (B1): a band call past its cap keeps the experts with the largest weight x their row's prior chance
of being accepted; the kept experts keep today's work-list order.

1. band.parse_prior / band.prior_rows: row t of a call takes the prior of row t % P of a request's verify block.
2. CPU: the send's plan built for the host (tests/band_prior_host.cpp: the kernel's phases in loops, the B1 phase
   through the device functions of sidecar/band/band_prior.cuh) against the reference `plan_ref` below on random calls
   (T 1..8, caps 1..16, coarse weights for ties, experts repeated across rows): the work list, the served flags, the
   stats and the hint list. Also: without a prior = today's plan (largest weight first); a prior of all ones = no
   prior; a call under its cap sends the same work list with and without a prior.
3. GPU only (the GB300 alone, no sidecar: the mailbox segment is pinned host memory, as test_tail1.py's):
   librtxband_fused_v4.so's band_qsend and band_plan_hint with and without a prior against the same reference, bit for
   bit: the work list read back from the request slot (with its header), the served flags, the stats, the hint list;
   the MXFP8 rows unchanged by the prior; with the engine's setting (the prior in CUDA graphs only) an eager send keeps
   today's plan, and a send captured in a graph takes the prior, replayed on new inputs.

CPU only (from the repository root; part 3 is skipped without a GPU):
    docker run --rm --network none --cpuset-cpus 66 --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/repo:ro \
        -e ENGINE=/repo/engine --entrypoint python3 vllm/vllm-openai:kimi-k31 /repo/engine/tests/test_band_prior.py
GPU (after ./build.sh, which builds engine/band/librtxband_fused_v4.so with the prior; from the repository root):
    docker run --rm --network none --device nvidia.com/gpu=$GB --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/repo:ro -e ENGINE=/repo/engine --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /repo/engine/tests/test_band_prior.py
Exit 0 and "BAND PRIOR: PASS" = every check passed (the GPU line says whether part 3 ran).
"""
import ctypes
import os
import random
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

ENGINE = os.environ.get("ENGINE", "/engine")
REPO = os.path.dirname(os.path.abspath(ENGINE))
sys.path.insert(0, ENGINE)
from k31_engine import band  # noqa: E402

E, K, NE, MT = 896, 16, 16, 8
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def fbits(v) -> int:
    return int(np.array([v], dtype=np.float32).view(np.int32)[0])


def plan_ref(ids, w, row_map, band_map, T, cap, prior):
    """The B1 plan of one call, written from its definition: band picks = an expert id with no resident row and a band
    slot; key of an expert = its largest float32(w x prior[row]) (w without a prior); the first min(n, cap) experts by
    (key descending, id ascending) are kept; the work list holds the kept experts by (largest w descending, id
    ascending). -> (n_exp, work [16 + 128], served [T * 16], stats [3], hint [17])."""
    f32 = np.float32
    best_w, best_k, slot = {}, {}, {}
    for t in range(T):
        for k in range(K):
            e = int(ids[t][k])
            if not (0 <= e < E) or row_map[e] >= 0 or band_map[e] < 0:
                continue
            wv = f32(w[t][k])
            kv = f32(wv * f32(prior[t])) if prior is not None else wv
            best_w[e] = max(best_w.get(e, f32(-np.inf)), wv)
            best_k[e] = max(best_k.get(e, f32(-np.inf)), kv)
            slot[e] = int(band_map[e])
    ne = cap if 0 < cap < NE else NE
    kept = set(sorted(best_k, key=lambda e: (-float(best_k[e]), e))[:ne])
    order = sorted(kept, key=lambda e: (-float(best_w[e]), e))
    place = {e: p for p, e in enumerate(order)}
    work = [0] * (NE + NE * MT)
    for p, e in enumerate(order):
        work[p] = slot[e]
    served, stats = [0] * (T * K), [0, 0, 0]
    for t in range(T):
        for k in range(K):
            e = int(ids[t][k])
            stats[2] += e >= 0
            if e in best_w:
                if e in kept:
                    served[t * K + k] = 1
                    stats[0] += 1
                    work[NE + place[e] * MT + t] = fbits(w[t][k])
                else:
                    stats[1] += 1
    hint = [len(order)] + [slot[e] for e in order] + [0] * (NE - len(order))
    return len(order), work, served, stats, hint


def random_call(rng, T):
    """A call's routing: per row 16 distinct experts from a small pool (experts repeat across rows), a layer's maps
    with resident, band and other experts mixed, coarse weights (ties within and across rows), a few -1 picks."""
    pool = rng.sample(range(E), rng.randint(20, 120))
    roles = {e: rng.random() for e in pool}
    row_map = [-1] * E
    band_map = [-1] * E
    p_band = rng.uniform(0.03, 0.6)
    for i, e in enumerate(pool):
        if roles[e] < 0.25:
            row_map[e] = 128 + i
        elif roles[e] < 0.25 + p_band:
            band_map[e] = rng.randrange(0, 10000)
    ids, w = [], []
    for _ in range(T):
        row = rng.sample(pool, K)
        if rng.random() < 0.1:
            row[rng.randrange(K)] = -1
        ids.append(row)
        w.append([rng.randint(1, 12) / 16.0 for _ in range(K)])
    return ids, w, row_map, band_map


def random_prior(rng):
    kind = rng.random()
    if kind < 0.4:                                   # measured-like: 1, then falling
        p = [1.0]
        for _ in range(MT - 1):
            p.append(p[-1] * rng.uniform(0.3, 0.8))
        return p
    if kind < 0.7:                                   # coarse values: ties of w x prior across rows
        return [rng.choice([1.0, 0.5, 0.25, 0.125]) for _ in range(MT)]
    return [round(rng.uniform(0.0, 1.0), 3) for _ in range(MT)]


# 1. the row priors
ok1 = band.parse_prior("1, 0.61,0.31 ,0.14,0.06") == (1.0, 0.61, 0.31, 0.14, 0.06) and band.parse_prior("") == () \
    and band.prior_rows((1.0, 0.5, 0.25), 3) == [1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 0.5] \
    and band.prior_rows((1.0, 0.5), 5) == [1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5] \
    and band.prior_rows((1.0, 0.6, 0.3, 0.1, 0.05), 5)[:5] == [1.0, 0.6, 0.3, 0.1, 0.05] \
    and band.prior_rows((0.7,), 1) == [0.7] * 8
for bad in ("1,-0.5", "1,inf", "1,nan", "1,x"):
    try:
        band.parse_prior(bad)
        ok1 = False
    except ValueError:
        pass
check("parse_prior / prior_rows (row t = row t % P of a verify block; bad values refused)", ok1)

# 1b. the hook's binding: rows per request from vLLM's config, the priors handed to the fused send
from k31_engine import hook  # noqa: E402

NS = types.SimpleNamespace
rpr = [hook._rows_per_request(NS(vllm_config=NS(uniform_decode_query_len=5, speculative_config=None))),
       hook._rows_per_request(NS(vllm_config=NS(speculative_config=NS(num_speculative_tokens=3)))),
       hook._rows_per_request(NS(vllm_config=NS(speculative_config=None))),
       hook._rows_per_request(NS())]


class _FzStandIn:
    def set_prior(self, rows, graphs_only=True):
        self.rows, self.graphs_only = rows, graphs_only


saved = dict(hook._BAND), band.BAND_PRIOR
try:
    fz0 = _FzStandIn()
    hook._BAND.update(fused=fz0, rows_per_req=5)
    band.BAND_PRIOR = (1.0, 0.64, 0.36, 0.19, 0.09)
    hook._bind_prior()
    ok1b = rpr == [5, 4, 1, 1] and fz0.rows == [1.0, 0.64, 0.36, 0.19, 0.09, 1.0, 0.64, 0.36] and fz0.graphs_only
    hook._BAND["fused"] = None
    try:
        hook._bind_prior()
        ok1b = False
    except RuntimeError:
        pass
finally:
    hook._BAND.clear()
    hook._BAND.update(saved[0])
    band.BAND_PRIOR = saved[1]
check("hook: rows per request (5, 4, 1, 1), the row priors handed to the fused send (graphs only), refused without "
      "the fused send", ok1b, f"({rpr})")

# 2. the host build of the plan
so = None
try:
    tmp = tempfile.mkdtemp(prefix="bandprior")
    so = os.path.join(tmp, "libbandprior_host.so")
    src = os.path.join(ENGINE, "tests", "band_prior_host.cpp")
    inc = os.path.join(REPO, "sidecar", "band")
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-I", inc, "-o", so, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        so = None
except FileNotFoundError:
    so = None
check("host build of the plan (g++, sidecar/band/band_prior.cuh)", so is not None,
      "" if so is not None else f"(needs g++ and the repository's sidecar/ under {REPO})")


def host_plan(lib, ids, w, row_map, band_map, T, cap, prior):
    I32, F32 = ctypes.c_int * (T * K), ctypes.c_float * (T * K)
    a_ids = I32(*[int(v) for r in ids for v in r])
    a_w = F32(*[float(v) for r in w for v in r])
    a_row = (ctypes.c_int * E)(*row_map)
    a_band = (ctypes.c_int * E)(*band_map)
    a_pr = (ctypes.c_float * MT)(*prior) if prior is not None else None
    work = (ctypes.c_int * (NE + NE * MT))()
    served = (ctypes.c_int * (T * K))()
    stats = (ctypes.c_int * 3)()
    hint = (ctypes.c_int * (1 + NE))()
    n = lib.bandprior_plan_host(a_ids, a_w, a_row, a_band, T, cap, a_pr, work, served, stats, hint)
    return n, list(work), list(served), list(stats), list(hint)


if so is not None:
    lib = ctypes.CDLL(so)
    P = ctypes.c_void_p
    lib.bandprior_plan_host.argtypes = [P, P, P, P, ctypes.c_int, ctypes.c_int, P, P, P, P, P]
    lib.bandprior_plan_host.restype = ctypes.c_int
    rng = random.Random(7)
    bad, n_cap, n_moved, n_calls = 0, 0, 0, 0
    bad_off, bad_ones, bad_under = 0, 0, 0
    for trial in range(3000):
        T = rng.randint(1, MT)
        cap = rng.choice([1, 2, 3, 4, 6, 8, 8, 8, 12, 16])
        ids, w, row_map, band_map = random_call(rng, T)
        prior = random_prior(rng)
        got = host_plan(lib, ids, w, row_map, band_map, T, cap, prior)
        want = plan_ref(ids, w, row_map, band_map, T, cap, prior)
        n_calls += 1
        if got != want:
            bad += 1
            if bad <= 3:
                print(f"  trial {trial} T={T} cap={cap}: host n {got[0]} stats {got[3]} vs ref n {want[0]} {want[3]}")
        off_h = host_plan(lib, ids, w, row_map, band_map, T, cap, None)
        off_r = plan_ref(ids, w, row_map, band_map, T, cap, None)
        bad_off += off_h != off_r
        bad_ones += host_plan(lib, ids, w, row_map, band_map, T, cap, [1.0] * MT) != off_h
        n_band = len({int(e) for r in ids for e in r if 0 <= e < E and row_map[e] < 0 and band_map[e] >= 0})
        ne = cap if 0 < cap < NE else NE
        if n_band > ne:
            n_cap += 1
            n_moved += got[2] != off_h[2]
        else:
            bad_under += got != off_h
    check("host plan with a prior = the reference (work list, served flags, stats, hint list)", bad == 0,
          f"({n_calls} calls, {n_cap} past their cap, {n_moved} of those keep other experts than today; {bad} differ)")
    check("host plan without a prior = today's plan (largest weight, ties by lower id)", bad_off == 0,
          f"({bad_off} differ)")
    check("a prior of all ones = no prior", bad_ones == 0, f"({bad_ones} differ)")
    check("a call under its cap: the same work list with and without a prior", bad_under == 0,
          f"({bad_under} differ)")

# 3. the GPU kernels
GPU = torch.cuda.is_available()
lib_path = os.environ.get("FUSED_LIB", os.path.join(ENGINE, "band", "librtxband_fused_v4.so"))
if GPU and not os.path.exists(lib_path):
    check(f"GPU part: {lib_path} exists (./build.sh)", False)
elif GPU:
    dev = torch.device("cuda")
    OFF_REQ, REQ_BYTES, REQ_ROWS, ROW_IN, OFF_HINT = 4096, 32768, 1024, 3584 + 112, 384
    seg_host = torch.zeros(band.SEG_BYTES, dtype=torch.uint8).pin_memory()
    fake = types.SimpleNamespace(seg=ctypes.c_void_p(seg_host.data_ptr()),
                                 st=torch.zeros(20, dtype=torch.int64, device=dev), timeout_ns=50_000)
    fz = band.Fused(lib_path, fake)
    check("the library reads a prior (rtxband_fused_prior)", fz.has_prior, f"({lib_path})")
    hdr = [0, 0, band.FLAGS_MEGAMOE & 0xFFFF, band.SF_MIN_MEGAMOE, fbits(band.SITU_ALPHA), fbits(band.SITU_BETA), 0, 0]

    def req_call():
        seq = int(fake.st[0])
        o = OFF_REQ + (seq % band.NSLOT) * REQ_BYTES
        return seg_host[o:o + 4 * band.CALL_INTS].view(torch.int32).tolist(), o

    def run_send(ids, w, row_map, band_map, T, cap, prior, x, graphs_only=False):
        fz.set_prior(prior, graphs_only=graphs_only)
        ids_d = torch.tensor(ids, dtype=torch.int32, device=dev)
        w_d = torch.tensor(w, dtype=torch.float32, device=dev)
        rm = torch.tensor(row_map, dtype=torch.int32, device=dev)
        bm = torch.tensor(band_map, dtype=torch.int32, device=dev)
        stats = torch.zeros(3, dtype=torch.int32, device=dev)
        served = torch.full((256,), -1, dtype=torch.int32, device=dev)
        seq0 = int(fake.st[0])
        fz.qsend(ids_d, w_d, rm, bm, x, stats, 3, cap=cap, served=served)
        torch.cuda.synchronize()
        seq1, tag = int(fake.st[0]), int(fake.st[1])
        call, o = req_call() if seq1 != seq0 else (None, None)
        rows = seg_host[o + REQ_ROWS:o + REQ_ROWS + T * ROW_IN].clone() if o is not None else None
        hst = torch.zeros(3, dtype=torch.int32, device=dev)
        hsv = torch.full((256,), -1, dtype=torch.int32, device=dev)
        hseq0 = int(fake.st[3])
        fz.hint(ids_d, w_d, rm, bm, hst, 3, cap=cap, served=hsv)
        torch.cuda.synchronize()
        hint = seg_host[OFF_HINT:OFF_HINT + 4 * (1 + NE)].view(torch.int32).tolist() if int(fake.st[3]) != hseq0 \
            else None
        return dict(call=call, tag=tag, sent=seq1 != seq0, rows=rows, stats=stats.tolist(),
                    served=served.tolist(), hstats=hst.tolist(), hserved=hsv.tolist(), hint=hint)

    rng = random.Random(11)
    bad, bad_rows, bad_eager, n_cap, n_calls = 0, 0, 0, 0, 0
    rows_p = None
    for trial in range(400):
        T = rng.randint(1, MT)
        cap = rng.choice([1, 2, 3, 4, 8, 8, 12, 16])
        ids, w, row_map, band_map = random_call(rng, T)
        x = (torch.randn(T, 3584) * 0.5).to(torch.bfloat16).to(dev)
        for prior in (random_prior(rng), None):
            n, work, served, stats, hint = plan_ref(ids, w, row_map, band_map, T, cap, prior)
            got = run_send(ids, w, row_map, band_map, T, cap, prior, x)
            want_call = ([n, T] + hdr[2:] + work) if n > 0 else None
            okc = (got["call"] == want_call) if n > 0 else (not got["sent"] and got["tag"] == 0)
            oks = got["served"][:T * K] == served and got["served"][T * K:] == [-1] * (256 - T * K) \
                and got["stats"] == stats
            okh = got["hserved"][:T * K] == served and got["hstats"] == stats and \
                (got["hint"] == hint if n > 0 else got["hint"] is None)
            n_calls += 1
            if not (okc and oks and okh):
                bad += 1
                if bad <= 3:
                    print(f"  trial {trial} T={T} cap={cap} prior={'on' if prior else 'off'}: call {okc}, "
                          f"send flags/stats {oks}, hint {okh}; stats {got['stats']} vs {stats}")
            if prior is not None:
                rows_p = got["rows"]
            elif rows_p is not None and got["rows"] is not None and not torch.equal(rows_p, got["rows"]):
                bad_rows += 1
        n_cap += len({int(e) for r in ids for e in r if 0 <= e < E and row_map[e] < 0 and band_map[e] >= 0}) > \
            (cap if 0 < cap < NE else NE)
        # the engine's setting (graphs only): an eager send with a prior set is today's plan
        n, work, served, stats, hint = plan_ref(ids, w, row_map, band_map, T, cap, None)
        got = run_send(ids, w, row_map, band_map, T, cap, random_prior(rng), x, graphs_only=True)
        okc = (got["call"] == ([n, T] + hdr[2:] + work)) if n > 0 else (not got["sent"] and got["tag"] == 0)
        bad_eager += not (okc and got["served"][:T * K] == served and got["stats"] == stats and
                          got["hserved"][:T * K] == served)
    fz.set_prior(None)
    check("GPU: band_qsend / band_plan_hint with and without a prior = the reference, bit for bit", bad == 0,
          f"({n_calls} sends, {n_cap} of 400 calls past their cap; {bad} differ)")
    check("GPU: the MXFP8 rows do not depend on the prior", bad_rows == 0, f"({bad_rows} differ)")
    check("GPU: with the engine's setting (graphs only) an eager send keeps today's plan", bad_eager == 0,
          f"({bad_eager} of 400 differ)")

    # a send with a prior in a CUDA graph (the engine's setting: graphs only), replayed on new inputs copied into its
    # buffers
    T, cap = 5, 4
    prior = [1.0, 0.64, 0.36, 0.19, 0.09, 0.09, 0.09, 0.09]
    fz.set_prior(prior)
    ids_b = torch.zeros((T, K), dtype=torch.int32, device=dev)
    w_b = torch.zeros((T, K), dtype=torch.float32, device=dev)
    rm_b = torch.full((E,), -1, dtype=torch.int32, device=dev)
    bm_b = torch.full((E,), -1, dtype=torch.int32, device=dev)
    x_b = torch.zeros((T, 3584), dtype=torch.bfloat16, device=dev)
    st_b = torch.zeros(3, dtype=torch.int32, device=dev)
    sv_b = torch.zeros(256, dtype=torch.int32, device=dev)
    rng = random.Random(13)

    def load(ids, w, row_map, band_map):
        ids_b.copy_(torch.tensor(ids, dtype=torch.int32))
        w_b.copy_(torch.tensor(w, dtype=torch.float32))
        rm_b.copy_(torch.tensor(row_map, dtype=torch.int32))
        bm_b.copy_(torch.tensor(band_map, dtype=torch.int32))
        st_b.zero_()

    case = random_call(rng, T)
    load(*case)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fz.qsend(ids_b, w_b, rm_b, bm_b, x_b, st_b, 3, cap=cap, served=sv_b)      # warm
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fz.qsend(ids_b, w_b, rm_b, bm_b, x_b, st_b, 3, cap=cap, served=sv_b)
    bad_g = 0
    for rep in range(30):
        case = random_call(rng, T)
        load(*case)
        seq0 = int(fake.st[0])
        graph.replay()
        torch.cuda.synchronize()
        n, work, served, stats, hint = plan_ref(*case, T, cap, prior)
        call = req_call()[0] if int(fake.st[0]) != seq0 else None
        ok = (call == [n, T] + hdr[2:] + work if n > 0 else call is None) and sv_b[:T * K].tolist() == served \
            and st_b.tolist() == stats
        bad_g += not ok
    check("GPU: a send with a prior in a CUDA graph, 30 replays on new inputs = the reference", bad_g == 0,
          f"({bad_g} differ)")
    # the send's time in a CUDA graph (5 rows, cap 8) with and without the prior, and the plan / hint kernel's
    us = {}
    for name in ("qsend", "hint"):
        for on in (False, True):
            fz.set_prior(prior if on else None)
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                if name == "qsend":
                    fz.qsend(ids_b, w_b, rm_b, bm_b, x_b, st_b, 3, cap=8, served=sv_b)
                else:
                    fz.hint(ids_b, w_b, rm_b, bm_b, st_b, 3, cap=8, served=sv_b)
            best = 1e9
            for _ in range(50):
                a0, a1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a0.record()
                g2.replay()
                a1.record()
                a1.synchronize()
                best = min(best, a0.elapsed_time(a1) * 1e3)
            us[(name, on)] = best
    fz.set_prior(None)
    print(f"time in a CUDA graph, 5 rows, cap 8 (us, prior off / on): band_qsend {us[('qsend', False)]:.1f} / "
          f"{us[('qsend', True)]:.1f}, band_plan_hint {us[('hint', False)]:.1f} / {us[('hint', True)]:.1f}")
print(f"GPU part: {'ran' if GPU and os.path.exists(lib_path) else 'skipped (no GPU)' if not GPU else 'failed'}")
print("BAND PRIOR:", "PASS" if not fails else f"FAIL ({', '.join(fails)})", flush=True)
sys.exit(1 if fails else 0)
