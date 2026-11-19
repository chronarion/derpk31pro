"""derpk31pro fork gates (b), (c) and the host part of (d): Derpseek's host tier in the MegaMoE fork on the kimi-k31
image, at Kimi K3.1's geometry on the GB300. Ported from Derpseek's tests/step_cost_test.

One bank of E = D + NH experts is loaded into HBM through vLLM's DeepGemmMegaMoEExperts. Its first D experts are also
copied to pinned host memory: those are the call's host rows. Two forms of the same call, same routing, same inputs:
  ref     every expert in HBM: the bank's E rows, no control block (the stock-equivalent call)
  split   kernel rows [0, D) are the host rows (read through the fork's re-based tensor-map clones), rows [D, E) the
          bank's HBM rows D.. through launcher tensors that start at bank row D and claim E rows (Derpseek's
          spanning()); Derpseek's control block rides in cumulative_local_expert_recv_stats
Gate (b): split == ref bit for bit for k = 0..D picked host experts, T = 1, 4, 16, 64, routing that gives host rows
one or many tokens; then two controls: a zeroed host row must change the output (the bytes do come from the host
copy), and the picks per row land at words 16.. of the control block while the lane counters return to 0.
Gate (c), SF6=1: every scale bank of the split call is 6-bit coded (Derpseek's sf6.py: HBM rows, host rows, the coded
maps after the clone table); split == the uncoded ref bit for bit.
Gate (d): call time of ref and split k as CUDA-graph replays of back-to-back calls (k31_fork_common.timeit: 10 calls
per replay while a replay stays under 1.5 ms, fewer above; min / median of 30), including vLLM's prepare kernel as in
the earlier vLLM 0.32 port.

env: D (8), NH (120), TMAX (64), TS_EXACT (1,4,16,64: row counts of the bit-equality part), TS (1,4) and NDIST (8,16:
distinct HBM experts per T) of the timing part, KS (0,1,2,3,4,6,8), LANE (4), SF6 (0), SF6_PY (directory of sf6.py),
EXACT (1: the bit-equality part), TIMING (1), NO_TABLE (1: no clone table of the layer's own, control words 14-15 = 0),
DUMP, PDL, NSMS, and k31_fork_common.import_dg's MODE, FORK, FORK_VIA, DEFINES, VARIANT
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k31_fork_common as c  # noqa: E402

MODE = os.environ.get("MODE", "fork")
D, NH = int(os.environ.get("D", "8")), int(os.environ.get("NH", "120"))
E = D + NH
TMAX = int(os.environ.get("TMAX", "64"))
LANE = int(os.environ.get("LANE", "4"))
SF6 = os.environ.get("SF6", "0") == "1"
EXACT = os.environ.get("EXACT", "1") == "1"
TIMING = os.environ.get("TIMING", "1") == "1"
KS = [int(v) for v in os.environ.get("KS", "0,1,2,3,4,6,8").split(",")]
dev = c.dev
g = torch.Generator(device=dev).manual_seed(0)
assert MODE == "fork", "the host tier exists in the fork only"

dg, desc = c.import_dg()
print(f"device {torch.cuda.get_device_name(0)}; headers: {desc}; nvidia-smi before: {c.smi()}", flush=True)
group = c.init_group(29591)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs  # noqa: E402

exp = c.make_experts(E, g, TMAX, "r5.step.experts")
kw = dict(exp._kernel_kwargs)
b4 = c.bank(exp)                                    # the whole bank in HBM (the ref call's tensors)
host = [c.pinned_like(t, D) for t in b4]            # host rows = bank rows 0..D-1, pinned, in the bank's own layout
hbm = [t[D:] for t in b4]                           # the split call's HBM rows
sp = [c.spanning(t, E) for t in hbm]                # launcher tensors of the split call: start at row D, claim E rows
symm = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, c.K, c.H, c.I, activation="situ", mma_type="fp8xfp4")
tail = 0
if SF6:
    sys.path.insert(0, os.environ.get("SF6_PY", "/d/engine/derpseek_engine"))
    import sf6
    assert (sf6.E0, sf6.LAYOUT) == (96, 2), "sf6.py must match the fork header's DRP_SF6_E0 / DRP_SF6_LAYOUT"
    tail = sf6.TAIL_BYTES
desc_tbl = torch.zeros(c.CLONE_BYTES + tail, dtype=torch.uint8, device=dev)
if os.environ.get("NO_TABLE") == "1":             # control words 14-15 = 0: the clone table inside the JIT module
    assert not SF6, "a coded call needs the layer's own table (its coded maps follow it)"
    desc_tbl = None
buf, ctl = c.new_ctl(E, d=D, lane=LANE, host=host, desc_tbl=desc_tbl)
if SF6:
    def code(t):
        coded = sf6.encode(t)
        assert torch.equal(sf6.decode(coded[:2], t.shape[1], t.shape[2]), t[:2]), "sf6 reference decode"
        return coded

    c1, c2 = code(hbm[1]), code(hbm[3])                                  # coded HBM scale banks (rows D..)
    raw1, raw3 = (tuple(hbm[1].shape), tuple(hbm[1].stride())), (tuple(hbm[3].shape), tuple(hbm[3].stride()))
    sp[1], sp[3] = sf6.span(c1, E, *raw1), sf6.span(c2, E, *raw3)        # what the launcher checks: uncoded shapes
    h1, h2 = code(b4[1][:D]), code(b4[3][:D])
    host[1], host[3] = c.pinned_like(h1), c.pinned_like(h2)              # coded host scale rows
    del h1, h2
    buf, ctl = c.new_ctl(E, d=D, lane=LANE, host=host, desc_tbl=desc_tbl)
    buf[11] = sf6.ctl_word(prof=False)
    sf6.write_tail(desc_tbl[c.CLONE_BYTES:], host[1], host[3], c1, c2, raw1[0][1:], raw3[0][1:])
    print(f"6-bit scale codes: {c1[0].numel() + c2[0].numel()} B of scales per row (uncoded "
          f"{(hbm[1][0].numel() + hbm[3][0].numel()) * 4})", flush=True)
print(f"D {D} host rows ({sum(h[0].numel() * h.element_size() for h in host) / 1e6:.2f} MB each, pinned: "
      + ", ".join(f"{tuple(h.shape)} {h.dtype} strides {tuple(h.stride())}" for h in host)
      + f") + NH {NH} HBM rows; lane {LANE} clusters; SF6 {int(SF6)}; launcher tensors of the split call: "
      + ", ".join(f"{tuple(t.shape)} strides {tuple(t.stride())} contiguous {t.is_contiguous()}" for t in sp), flush=True)


def call(x, ids, w, mode, T):
    prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, c.H, dtype=torch.bfloat16, device=dev)
    if mode == "ref":
        dg.fp8_fp4_mega_moe(y, (b4[0], b4[1]), (b4[2], b4[3]), symm, fast_math=True, **kw)
    else:
        dg.fp8_fp4_mega_moe(y, (sp[0], sp[1]), (sp[2], sp[3]), symm, cumulative_local_expert_recv_stats=ctl,
                            fast_math=True, **kw)
    return y


def host_ids(base, k, T):
    """`base` with k distinct host rows picked once each: host row j at the last slot of token j % T, then earlier slots"""
    ids = base.clone()
    for j in range(k):
        ids[j % T, c.K - 1 - j // T] = j
    return ids


bad = 0
if EXACT:
    # (T, routing over kernel rows, description): host rows with one pick, with many, as the majority, several m blocks
    cases = []
    for T in (int(t) for t in os.environ.get("TS_EXACT", "1,4,16,64").split(",")):
        nd = min(NH, 8 * T if T <= 4 else 96)
        base = c.make_ids(T, NH, nd, 16, g, "cyc", lo=D)
        for k in (0, 1, 2, 4, D):
            if k <= T * 4:
                cases.append((T, host_ids(base, k, T), f"{nd} HBM experts, {k} host experts with one pick each"))
        cases.append((T, c.make_ids(T, E, min(E, 16 * T), 16, g, "rnd"), "random over all rows"))
        cases.append((T, c.make_ids(T, E, min(E, 16 * T), 11, g, "rnd"), "random over all rows, 5 picks cut"))
    TB = min(TMAX, max(64, max(t for t, _, _ in cases)))   # the widest call of this run
    cases.append((16, c.make_ids(16, D, D, min(D, c.K), g, "rnd"), "host rows only"))
    cases.append((TB, c.make_ids(TB, D + 16, D + 16, 16, g, "rnd"),
                  f"{D} host + 16 HBM rows, ~{TB * 16 // (D + 16)} tokens each (several token blocks per row)"))
    cases.append((TB, c.make_ids(TB, D, D, min(D, c.K), g, "rnd"),
                  f"host rows only, ~{TB * min(D, c.K) // D} tokens each (several token blocks per row)"))
    for ci, (T, ids, what) in enumerate(cases):
        w_raw = torch.softmax(torch.randn(T, c.K, device=dev, generator=g), -1).float()
        x = torch.randn(T, c.H, dtype=torch.bfloat16, device=dev, generator=g)
        w = torch.where(ids >= 0, w_raw, torch.zeros_like(w_raw))
        n_host = int(((ids >= 0) & (ids < D)).sum())
        ya = call(x, ids, w, "ref", T)
        buf[c.HDR:].zero_()
        yb = call(x, ids, w, "split", T)
        torch.cuda.synchronize()
        expect = torch.bincount(ids[ids >= 0].flatten(), minlength=E).to(torch.int32)
        eq = c.same(ya, yb)
        counts = bool(torch.equal(buf[c.HDR:], expect)) and int(buf[12]) == 0 and int(buf[13]) == 0
        nz = float((ya.float() != 0).float().mean())
        fin = bool(torch.isfinite(ya.float()).all())
        ctrl = ""
        if n_host and what.startswith(("random over all rows", "host rows only")) and T in (4, 16):
            # control: zero the host copy of one picked host row's first-GEMM weights; the output must change, and
            # only for the tokens that picked it
            r = int(ids[(ids >= 0) & (ids < D)][0])
            keep = host[0][r].clone()
            host[0][r].zero_()
            yz = call(x, ids, w, "split", T)
            torch.cuda.synchronize()
            host[0][r].copy_(keep)
            rows_hit = (ids == r).any(1)
            changed = (c.bits(yz) != c.bits(ya)).any(1)
            ok_ctrl = bool(changed[rows_hit].all()) and not bool(changed[~rows_hit].any()) and bool(rows_hit.any())
            yr = call(x, ids, w, "split", T)
            torch.cuda.synchronize()
            ok_ctrl = ok_ctrl and c.same(yr, ya)
            ctrl = f"; control (host row {r} zeroed in host memory): {'changes exactly its tokens' if ok_ctrl else 'FAILED'}"
            bad += not ok_ctrl
        bad += (not eq) + (not counts) + (not fin) + (nz < 0.9)
        print(f"T={T:2d} {what} ({n_host} host picks): split against ref {'BIT-EQUAL' if eq else 'DIFFERS'}; "
              f"counts at words 16.. {'right' if counts else 'WRONG'}, lane counters {int(buf[12])} {int(buf[13])}; "
              f"finite {fin}, nonzero {nz:.3f}{ctrl}", flush=True)
        if os.environ.get("DUMP"):
            os.makedirs(os.environ["DUMP"], exist_ok=True)
            torch.save({"ref": ya.cpu(), "split": yb.cpu(), "ids": ids.cpu()},
                       os.path.join(os.environ["DUMP"], f"step-sf6{int(SF6)}-lane{LANE}-case{ci:02d}-T{T}.pt"))
    print(f"exactness part: {'ALL EQUAL' if bad == 0 else f'{bad} FAILURES'} ({len(cases)} cases)", flush=True)

if TIMING:
    NDIST = [int(v) for v in os.environ.get("NDIST", "8,16").split(",")]
    for i, T in enumerate(int(t) for t in os.environ.get("TS", "1,4").split(",")):
        nd = NDIST[i]
        valid = min(c.K, nd)
        base = c.make_ids(T, NH, nd, valid, g, "cyc", lo=D)
        x = torch.randn(T, c.H, dtype=torch.bfloat16, device=dev, generator=g)
        w_raw = torch.softmax(torch.randn(T, c.K, device=dev, generator=g), -1).float()
        w = torch.where(base >= 0, w_raw, torch.zeros_like(w_raw))
        ts = c.timeit(lambda: call(x, base, w, "ref", T), reps=30)
        line = f"T={T} {nd} distinct HBM experts: ref {ts[0]:6.1f}/{ts[1]:6.1f} us"
        res = {}
        for k in KS:
            if k > min(D, T * 4):
                continue
            ids = host_ids(base, k, T)
            wk = torch.where(ids >= 0, w_raw, torch.zeros_like(w_raw))
            eq = c.same(call(x, ids, wk, "ref", T), call(x, ids, wk, "split", T))
            bad += not eq
            tk = c.timeit(lambda: call(x, ids, wk, "split", T), reps=30)
            res[k] = tk[0]
            line += f" | split{k} {tk[0]:6.1f}/{tk[1]:6.1f}{'' if eq else ' DIFFERS'}"
        print(line, flush=True)
        ks = sorted(k for k in res if k >= 1)
        if len(ks) >= 2:
            print(f"T={T}: host tier present, no host pick: +{res.get(0, float('nan')) - ts[0]:.1f} us over ref; first host row "
                  f"+{res[ks[0]] - res.get(0, ts[0]):.1f} us; each further host row "
                  f"{(res[ks[-1]] - res[ks[0]]) / (ks[-1] - ks[0]):.1f} us ({ks[0]} -> {ks[-1]} rows, min times) = "
                  f"{c.row_bytes(host) / 1e3 / ((res[ks[-1]] - res[ks[0]]) / (ks[-1] - ks[0])):.0f} GB/s", flush=True)
print(f"done: {'ALL EQUAL' if bad == 0 else f'{bad} FAILURES'}; peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; "
      f"nvidia-smi after: {c.smi()}", flush=True)
sys.exit(1 if bad else 0)
