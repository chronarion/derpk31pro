"""The RTX band through the engine's own kernels (k31_engine/band.py), end to end on real records.

Runs in a container with both GPUs (CDI) and --ipc host: a band sidecar subprocess on the RTX serves layers 1-2 (118
records each, ranks 105..222 of rank-pass1.json); this process works on the GB300 (CUDA_VISIBLE_DEVICES = its UUID).
  1. band.quant against rtxband.mxfp8_encode (bit-equal codes and scales).
  2. band.plan against a Python work list (experts, order, weights, counters).
  3. calls: random routings (resident, band and other experts mixed), plan + quant + send + finish, the rows added
     against the torch reference (reference.forward on the decoded hi2 records, MegaMoE's arithmetic).
  4. the same calls captured once in a CUDA graph and replayed with new inputs copied in.
Env: RTX, GB300 (UUIDs). Exit code 0 = PASS.
"""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/engine")
sys.path.insert(0, "/work/code")
from k31_engine import band  # noqa: E402
from band import reference  # noqa: E402
from band.rtxband import FLAGS_MEGAMOE, HI2_REC, SF_MIN_MEGAMOE, mxfp8_encode  # noqa: E402

SEG = "/dev/shm/k31band-test"
LAYERS, HBM, N = (1, 2), 105, 118
RANK = "/rank/rank-pass1.json"
# E2E_BAND_MAP=1: the sidecar takes --band-map with unequal counts (layer 1: 60, layer 2: 140; K31_BAND_MAP): layer 2's
# store slots start after layer 1's records
NMAP = {1: 60, 2: 140} if os.environ.get("E2E_BAND_MAP") == "1" else {1: N, 2: N}
MAPF = SEG + "-bandmap.json"
# E2E_RING=1: the sidecar gets K31_BAND_RING=4 and section 5 also enters a record from the ring (rec = row + 1)
RING = 4 if os.environ.get("E2E_RING") == "1" else 0
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["RTX"], CUDA_MODULE_LOADING="EAGER", K31_BAND_RING=str(RING))
with open(MAPF, "w") as f:
    json.dump({"layers": {str(k): v for k, v in NMAP.items()}}, f)
side = subprocess.Popen([sys.executable, "-m", "band.sidecar", "--build", "/work/build", "--hi2", "/hi2", "--rank", RANK,
                         "--hbm", str(HBM), "--n", str(N), "--seg", SEG, "--layers", f"{LAYERS[0]}-{LAYERS[-1]}",
                         "--status-s", "30", "--band-map", MAPF], cwd="/work/code", env=env)
try:
    dev = torch.device("cuda")
    torch.zeros(1, device=dev)
    g = torch.Generator().manual_seed(0)

    # 1. MXFP8 rows
    bad = 0
    for T in (1, 3, 8):
        x = (torch.randn(T, 3584, generator=g) * torch.exp(torch.randn(T, 1, generator=g))).to(torch.bfloat16)
        x[0, :32] = 0
        xq = torch.zeros((8, 3584), dtype=torch.uint8, device=dev)
        xs = torch.zeros((8, 112), dtype=torch.uint8, device=dev)
        band.quant(x.to(dev), xq, xs)
        q, s, _ = mxfp8_encode(x.float())
        bad += int((xq[:T].cpu() != q).sum()) + int((xs[:T].cpu() != s).sum())
    check("quant: MXFP8 codes and scales bit-equal to mxfp8_encode", bad == 0, f"({bad} differ)")

    # 2. work lists
    rank = json.load(open(RANK))["layers"]

    def routing(L, T, gen, p_native=0.4):
        ex = [int(e) for e in rank[str(L)]]
        native = set(ex[:HBM])
        banded = ex[HBM:HBM + NMAP[L]]
        ids = torch.empty((T, 16), dtype=torch.int32)
        for t in range(T):
            pool = torch.tensor(ex[:HBM + NMAP[L] + 200])
            ids[t] = pool[torch.randperm(len(pool), generator=gen)[:16]]
        w = torch.rand((T, 16), generator=gen) + 0.01
        w = w / w.sum(1, keepdim=True)
        row = torch.full((896,), -1, dtype=torch.int32)
        for r, e in enumerate(sorted(native)):
            row[e] = r
        return ids, w, row, banded

    def py_plan(ids, w, row, bmap, base=0):
        T = ids.shape[0]
        best = {}
        for t in range(T):
            for k in range(16):
                e = int(ids[t, k])
                if e >= 0 and row[e] < 0 and bmap[e] >= 0:
                    best[e] = max(best.get(e, -1.0), float(w[t, k]))
        order = sorted(best, key=lambda e: (-best[e], e))[base:base + 16]
        W = np.zeros((16, 8), dtype=np.float32)
        for j, e in enumerate(order):
            for t in range(T):
                for k in range(16):
                    if int(ids[t, k]) == e:
                        W[j, t] = float(w[t, k])
        return order, W

    m = None
    t0 = time.time()
    while not os.path.exists(SEG + ".ready"):
        assert side.poll() is None, f"the sidecar left: {side.returncode}"
        assert time.time() - t0 < 600, "no ready file after 600 s"
        time.sleep(1)
    m = np.load(SEG + ".map.npy")
    base, okm = 0, True
    for L in LAYERS:                                  # per-layer counts: layer L's slots follow the earlier layers'
        ex = [int(e) for e in rank[str(L)]]
        want = ex[HBM:HBM + NMAP[L]]
        okm &= int((m[L] >= 0).sum()) == NMAP[L] and sorted(int(m[L, e]) for e in want) == list(range(base, base + NMAP[L]))
        base += NMAP[L]
    check(f"store map: per-layer counts {NMAP} at consecutive slot bases", okm)
    bad = 0
    for trial in range(40):
        L = LAYERS[trial % 2]
        T = 1 + trial % 8
        ids, w, row, banded = routing(L, T, g)
        call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
        stats = torch.zeros(3, dtype=torch.int32, device=dev)
        band.plan(ids.to(dev), w.to(dev), row.to(dev), torch.tensor(m[L], device=dev), call, stats)
        c = call.cpu()
        order, W = py_plan(ids, w, row, m[L])
        got_slots = c[8:8 + int(c[0])].tolist()
        want_slots = [int(m[L, e]) for e in order]
        Wg = c[24:].view(torch.float32).view(16, 8).numpy()
        n_band = sum(1 for t in range(T) for k in range(16) if row[int(ids[t, k])] < 0 and m[L, int(ids[t, k])] >= 0)
        if got_slots != want_slots or not np.array_equal(Wg[:len(order)], W[:len(order)]) or int(c[1]) != T or \
                int(stats[0] + stats[1]) != n_band or int(stats[2]) != T * 16:
            bad += 1
            print(f"  trial {trial}: slots {got_slots[:6]} vs {want_slots[:6]}, stats {stats.tolist()} vs {n_band}")
    check("plan: work lists equal to the Python ones (40 routings, T 1-8)", bad == 0)

    # 3. calls through the mailbox
    gb = band.Gb("/work/build/librtxband_gb300.so", SEG, dev, timeout_us=20000)
    check("mailbox mapped, service running", gb.running(), f"(first seq {int(gb.st[0]) + 1})")
    recs = {}

    def experts_of(L, es):
        out = []
        for e in es:
            if (L, e) not in recs:
                with open(f"/hi2/L{L:02d}.hi2", "rb") as f:
                    f.seek(e * HI2_REC)
                    recs[(L, e)] = torch.frombuffer(bytearray(f.read(HI2_REC)), dtype=torch.uint8)
            out.append(reference.decode_hi2(recs[(L, e)].to(dev)))
        return out

    def ref_rows(L, x, ids, w, row, flags=FLAGS_MEGAMOE):
        order, W = py_plan(ids, w, row, m[L])
        if not order:
            return torch.zeros(x.shape[0], 3584, dtype=torch.float64)
        _, _, xv = mxfp8_encode(x.float().to(dev))
        y, _ = reference.forward(xv, experts_of(L, order), torch.tensor(W[:len(order)]).to(dev), flags,
                                 SF_MIN_MEGAMOE)
        return y.cpu()

    PRECISE = 8 | 16                                 # FLAG_PRECISE | FLAG_OUT_BF16: SiTU as torch computes it
    worst_p, rms_f, n_calls, n_work = 0.0, 0.0, 0, 0
    for trial in range(16):
        L = LAYERS[trial % 2]
        T = (1, 2, 3, 3, 4, 6, 8, 3)[trial % 8]
        ids, w, row, _ = routing(L, T, g)
        x = (torch.randn(T, 3584, generator=g) * 0.5).to(torch.bfloat16)
        call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
        stats = torch.zeros(3, dtype=torch.int32, device=dev)
        bmap = torch.tensor(m[L], device=dev)
        flags = PRECISE if trial % 2 else FLAGS_MEGAMOE
        band.plan(ids.to(dev), w.to(dev), row.to(dev), bmap, call, stats, flags=flags)
        band.quant(x.to(dev).contiguous(), gb.xq, gb.xs)
        a = gb.args(call)
        gb.dst.zero_()
        gb.send(a)
        gb.finish(a)
        torch.cuda.synchronize()
        got = gb.dst[:T].double().cpu()
        want = ref_rows(L, x, ids, w, row, flags)
        rms = float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt().clamp_min(1e-30))
        mx = float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))
        if flags == PRECISE:
            worst_p = max(worst_p, rms)
            mx_p = max(globals().get("mx_p", 0.0), mx)
            globals()["mx_p"] = mx_p
        else:
            rms_f = max(rms_f, rms)
        n_calls += 1
        n_work += 1 if py_plan(ids, w, row, m[L])[0] else 0     # calls with band work (a small band can draw none)
    cnt = gb.counters()
    # per-layer counts (E2E_BAND_MAP=1) route other experts: the fast-SiTU approximation error is data-dependent (2.3e-3
    # seen on layer 2 at 140 records); the exact-SiTU calls at 5e-4 are the correctness check of the layout
    check("calls: rows equal to the torch reference (rms error; 8 calls with torch's SiTU, 8 with MegaMoE's fast "
          "SiTU)", worst_p < 5e-4 and rms_f < (3e-3 if NMAP[1] != N else 2e-3) and cnt["n_added"] == n_work,
          f"(torch SiTU rms rel err {worst_p:.2e} (max elem {globals().get('mx_p', 0):.1e}), fast-SiTU rms rel err "
          f"{rms_f:.2e}; added {cnt['n_added']}, timeouts {cnt['n_timeout']}, down {cnt['n_down']})")

    # 4. one CUDA graph: plan + quant + send + finish, replayed with new inputs
    T = 3
    gi = torch.zeros((T, 16), dtype=torch.int32, device=dev)
    gw = torch.zeros((T, 16), dtype=torch.float32, device=dev)
    gx = torch.zeros((T, 3584), dtype=torch.bfloat16, device=dev)
    grow = torch.zeros(896, dtype=torch.int32, device=dev)
    gmap = torch.zeros(896, dtype=torch.int32, device=dev)
    gcall = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
    gstats = torch.zeros(3, dtype=torch.int32, device=dev)
    gout = torch.zeros((T, 3584), dtype=torch.float32, device=dev)
    ga = gb.args(gcall)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                       # warm the Triton kernels outside the capture
        band.plan(gi, gw, grow, gmap, gcall, gstats)
        band.quant(gx, gb.xq, gb.xs)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        band.plan(gi, gw, grow, gmap, gcall, gstats)
        band.quant(gx, gb.xq, gb.xs)
        gb.send(ga)
        gb.dst[:T].zero_()
        gb.finish(ga)
        gout.copy_(gb.dst[:T])
    worst = 0.0
    before = gb.counters()["n_added"]
    for trial in range(8):
        L = LAYERS[trial % 2]
        ids, w, row, _ = routing(L, T, g)
        x = (torch.randn(T, 3584, generator=g) * 0.5).to(torch.bfloat16)
        gi.copy_(ids.to(dev)); gw.copy_(w.to(dev)); gx.copy_(x.to(dev)); grow.copy_(row.to(dev))
        gmap.copy_(torch.tensor(m[L], device=dev))
        graph.replay()
        torch.cuda.synchronize()
        want = ref_rows(L, x, ids, w, row)
        err = float((gout.double().cpu() - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt().clamp_min(1e-30))
        worst = max(worst, err)
    added = gb.counters()["n_added"] - before
    check("graph: 8 replays equal to the reference", worst < 2e-3 and added == 8,
          f"(fast-SiTU rms rel err {worst:.2e}, added {added})")

    # 5. a record entry through the command ring: an expert outside the band takes a band slot, calls then compute it
    cmd = band.BandCmd(SEG + ".cmd", "/work/build/libbandhost.so")

    def entry_check(L, old, new, post_e=None, rec=0):
        """`new` enters `old`'s band slot (the command names post_e, default new; rec > 0: from ring row rec - 1), then
        calls that route `new` are checked against the reference computed from new's own record."""
        global m
        ex = [int(e) for e in rank[str(L)]]
        s_old = int(m[L, old])
        t0 = time.time()
        seq = cmd.post(L, new if post_e is None else post_e, s_old, rec=rec)
        done = []
        while not done and time.time() - t0 < 10:
            done = cmd.poll()
            time.sleep(0.001)
        entry_ms = 1e3 * (time.time() - t0)
        ok_entry = done == [(seq, 1)]
        m2 = m.copy()
        m2[L, old] = -1
        m2[L, new] = s_old
        worst = entry_rows(L, ex, new, m2)
        return ok_entry and worst < 2e-3, f"(ack {done}, {entry_ms:.1f} ms; fast-SiTU rms rel err {worst:.2e})"

    def entry_rows(L, ex, new, m2):
        global m
        worst = 0.0
        for trial in range(4):
            T = 3
            ids, w, row, _ = routing(L, T, g)
            for t in range(T):                                       # every row picks the entered expert, once
                r = [int(v) for v in ids[t] if int(v) != new][:15]
                spare = [v for v in ex[HBM + NMAP[L] + 100:] if v not in r and v != new]
                while len(r) < 15:
                    r.append(spare.pop(0))
                ids[t] = torch.tensor([new] + r, dtype=torch.int32)
            x = (torch.randn(T, 3584, generator=g) * 0.5).to(torch.bfloat16)
            call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
            stats = torch.zeros(3, dtype=torch.int32, device=dev)
            band.plan(ids.to(dev), w.to(dev), row.to(dev), torch.tensor(m2[L], device=dev), call, stats)
            band.quant(x.to(dev).contiguous(), gb.xq, gb.xs)
            a = gb.args(call)
            gb.dst.zero_()
            gb.send(a)
            gb.finish(a)
            torch.cuda.synchronize()
            m_save = m
            m = m2
            want = ref_rows(L, x, ids, w, row)
            m = m_save
            worst = max(worst, float((gb.dst[:T].double().cpu() - want).pow(2).mean().sqrt() /
                                     want.pow(2).mean().sqrt().clamp_min(1e-30)))
        return worst

    L = LAYERS[0]
    ex = [int(e) for e in rank[str(L)]]
    ok5, msg5 = entry_check(L, ex[HBM + 5], ex[HBM + NMAP[L] + 40])   # a band expert and one ranked past the band
    check("entry: an expert entered through the command ring is computed from its slot", ok5, msg5)
    if RING:
        # the ring row holds expert new3's record while the command names new2: the band must compute new3 (proof
        # that the record came from the ring, not from the drive)
        new2, new3 = ex[HBM + NMAP[L] + 41], ex[HBM + NMAP[L] + 42]
        ring = np.memmap(SEG + ".rec", dtype=np.uint8, mode="r+").reshape(-1, HI2_REC)
        with open(f"/hi2/L{L:02d}.hi2", "rb") as f:
            f.seek(new3 * HI2_REC)
            ring[2][:] = np.frombuffer(f.read(HI2_REC), dtype=np.uint8)
        ring.flush()
        ok5b, msg5b = entry_check(L, ex[HBM + 6], new3, post_e=new2, rec=3)
        check("ring entry: the record comes from ring row rec - 1, not from the drive", ok5b, msg5b)

    # 6. the fused kernels (one send, one finish into bfloat16) against the unfused path: the same rows, bit for bit
    fused_lib = os.environ.get("K31_BAND_FUSED_LIB", "librtxband_fused.so")
    fused_path = next((p for p in (f"/engine/band/{fused_lib}", f"/work/build/{fused_lib}") if os.path.exists(p)),
                      f"/work/build/{fused_lib}")             # the engine's own build first
    fz = band.Fused(fused_path, gb)
    bad, n_cmp = 0, 0
    for trial in range(12):
        L = LAYERS[trial % 2]
        T = (1, 3, 3, 5, 8, 2)[trial % 6]
        ids, w, row, _ = routing(L, T, g)
        x = (torch.randn(T, 3584, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        base = (torch.randn(T, 3584, generator=g) * 0.1).to(torch.bfloat16).to(dev)
        bmap = torch.tensor(m[L], device=dev)
        ids_d, w_d, row_d = ids.to(dev).contiguous(), w.to(dev).float().contiguous(), row.to(dev)
        call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
        st1 = torch.zeros(3, dtype=torch.int32, device=dev)
        band.plan(ids_d, w_d, row_d, bmap, call, st1)
        band.quant(x, gb.xq, gb.xs)
        a = gb.args(call)
        gb.dst.zero_()
        gb.send(a)
        gb.finish(a)
        out1 = base.clone()
        out1.add_(gb.dst[:T])
        st2 = torch.zeros(3, dtype=torch.int32, device=dev)
        out2 = base.clone()
        fz.qsend(ids_d, w_d, row_d, bmap, x, st2, L)
        fz.finish(out2)
        torch.cuda.synchronize()
        n_cmp += 1
        if not torch.equal(out1, out2) or not torch.equal(st1, st2):
            bad += 1
            print(f"  trial {trial}: max diff {float((out1.float() - out2.float()).abs().max())}, stats {st1.tolist()} "
                  f"vs {st2.tolist()}")
    cnt = gb.counters()
    check("fused: qsend + finish_bf16 bit-equal to plan + quant + send + finish + add", bad == 0,
          f"({n_cmp} calls; state: added {cnt['n_added']}, timeouts {cnt['n_timeout']}, empty {cnt['n_empty']})")

    # 7. K31_BAND_MAXE: the fused send capped at CAP experts serves the CAP strongest band experts only: bit-equal to
    # the unfused path with every other band expert's picks at weight 0 (they then sort last and add exact zeros)
    CAP = 4
    bad, n_cmp, n_capped = 0, 0, 0
    V3 = any(v in os.environ.get("K31_BAND_FUSED_LIB", "") for v in ("v3", "v4"))
    n_flags, bad_flags = 0, 0
    old_cap = band.BAND_MAXE
    band.BAND_MAXE = CAP
    try:
        for trial in range(12):
            L = LAYERS[trial % 2]
            T = (8, 5, 8, 3, 8, 5)[trial % 6]
            ids, w, row, _ = routing(L, T, g)
            x = (torch.randn(T, 3584, generator=g) * 0.5).to(torch.bfloat16).to(dev)
            base = (torch.randn(T, 3584, generator=g) * 0.1).to(torch.bfloat16).to(dev)
            bmap = torch.tensor(m[L], device=dev)
            ids_d, w_d, row_d = ids.to(dev).contiguous(), w.to(dev).float().contiguous(), row.to(dev)
            # the call's distinct band experts by their largest weight (ties: lower id), as the planner orders them
            best = {}
            for t in range(T):
                for j in range(16):
                    e = int(ids[t, j])
                    if e >= 0 and int(row[e]) < 0 and m[L][e] >= 0:
                        best[e] = max(best.get(e, float("-inf")), float(w[t, j]))
            order = sorted(best, key=lambda e: (-best[e], e))
            keep = set(order[:CAP])
            n_capped += len(order) > CAP
            w_ref = w.clone()
            for t in range(T):
                for j in range(16):
                    e = int(ids[t, j])
                    if e in best and e not in keep:
                        w_ref[t, j] = 0.0
            wr_d = w_ref.to(dev).float().contiguous()
            call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
            st1 = torch.zeros(3, dtype=torch.int32, device=dev)
            band.plan(ids_d, wr_d, row_d, bmap, call, st1)
            band.quant(x, gb.xq, gb.xs)
            a = gb.args(call)
            gb.dst.zero_()
            gb.send(a)
            gb.finish(a)
            out1 = base.clone()
            out1.add_(gb.dst[:T])
            st2 = torch.zeros(3, dtype=torch.int32, device=dev)
            out2 = base.clone()
            sv = torch.full((256,), -1, dtype=torch.int32, device=dev) if V3 else None
            fz.qsend(ids_d, w_d, row_d, bmap, x, st2, L, served=sv)
            fz.finish(out2)
            torch.cuda.synchronize()
            n_cmp += 1
            served_ok = int(st2[0]) == sum(1 for t in range(T) for j in range(16) if int(ids[t, j]) in keep)
            if V3:                     # v3: the per-pick served flags are exactly the picks of the kept experts
                ref = [1 if int(ids[t, j]) in keep else 0 for t in range(T) for j in range(16)]
                n_flags += 1
                if sv[:T * 16].cpu().tolist() != ref:
                    served_ok = False
                    bad_flags += 1
            if not torch.equal(out1, out2) or not served_ok:
                bad += 1
                print(f"  cap trial {trial}: max diff {float((out1.float() - out2.float()).abs().max())}, stats "
                      f"{st2.tolist()}, kept {len(keep)} of {len(order)}")
    finally:
        band.BAND_MAXE = old_cap
    check(f"cap: the fused send with K31_BAND_MAXE={CAP} equals the unfused path on its {CAP} strongest band experts",
          bad == 0 and n_capped > 0, f"({n_cmp} calls, {n_capped} with more than {CAP} band experts"
          + (f"; v3 served flags equal the kept experts' picks in {n_flags - bad_flags} of {n_flags}" if V3 else "")
          + ")")

    # 8. v4 (K31_TAIL1): finish with msum and rstd = bf16(base + the call's rows + msum) in one rounding, bit for bit
    # against the unfused rows, and the RMSNorm statistic of the result
    if "v4" in os.environ.get("K31_BAND_FUSED_LIB", ""):
        bad8, worst8, n8 = 0, 0.0, 0
        for trial in range(10):
            L = LAYERS[trial % 2]
            T = (1, 3, 5, 8, 4)[trial % 5]
            ids, w, row, _ = routing(L, T, g)
            x = (torch.randn(T, 3584, generator=g) * 0.5).to(torch.bfloat16).to(dev)
            base = (torch.randn(T, 3584, generator=g) * 0.1).to(torch.bfloat16).to(dev)
            msum = (torch.randn(T, 3584, generator=g) * 0.05).to(dev)
            bmap = torch.tensor(m[L], device=dev)
            ids_d, w_d, row_d = ids.to(dev).contiguous(), w.to(dev).float().contiguous(), row.to(dev)
            call = torch.zeros(band.CALL_INTS, dtype=torch.int32, device=dev)
            st1 = torch.zeros(3, dtype=torch.int32, device=dev)
            band.plan(ids_d, w_d, row_d, bmap, call, st1)
            band.quant(x, gb.xq, gb.xs)
            a = gb.args(call)
            gb.dst.zero_()
            gb.send(a)
            gb.finish(a)
            torch.cuda.synchronize()
            want = (base.float() + gb.dst[:T].float() + msum).to(torch.bfloat16)
            rs_want = torch.rsqrt(want.float().pow(2).mean(-1) + 1e-6)
            out2 = base.clone()
            rstd = torch.full((16,), float("nan"), dtype=torch.float32, device=dev)
            st2 = torch.zeros(3, dtype=torch.int32, device=dev)
            fz.qsend(ids_d, w_d, row_d, bmap, x, st2, L)
            fz.finish(out2, msum=msum.contiguous(), rstd=rstd, eps=1e-6)
            torch.cuda.synchronize()
            n8 += 1
            if not torch.equal(out2, want):
                bad8 += 1
                print(f"  v4 trial {trial}: max diff {float((out2.float() - want.float()).abs().max())}")
            worst8 = max(worst8, float(((rstd[:T] - rs_want).abs() / rs_want).max()))
        check("v4: finish with msum and rstd = bf16(base + rows + msum), statistic of the result", bad8 == 0 and
              worst8 < 2e-6, f"({n8} calls, rstd rel err {worst8:.1e})")
finally:
    if side.poll() is None:
        side.send_signal(signal.SIGTERM)
        try:
            side.wait(timeout=60)
        except subprocess.TimeoutExpired:
            side.kill()
    print(f"sidecar exit code {side.returncode}", flush=True)
print("BAND E2E:", "PASS" if not fails else f"FAIL ({', '.join(fails)})", flush=True)
sys.exit(1 if fails else 0)
