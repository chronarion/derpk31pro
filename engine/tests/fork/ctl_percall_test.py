"""derpk31pro fork: one control block rewritten between the calls of one CUDA graph by kernels of the same stream
(what a policy kernel or a staging lane does: another host region, another D, another lane width per call), with one
clone table for all of them. The fork reads every control word below its dependency wait, so each call
must see the header written just before it, with PDL (as vLLM serves) and without.

One bank of E experts in HBM; two pinned host copies: A = bank rows 0..7, B = bank rows 8..15. Launcher tensors start
at bank row 8 and claim E rows. Three headers, written into the block by the kernel launched right before each call:
  h0  D = 8, bases A, lane 4     kernel row r < 8 -> bank row r (host A),     r >= 8 -> bank row r
  h1  D = 8, bases B, lane 0     kernel row r < 8 -> bank row 8 + r (host B), r >= 8 -> bank row r
  h2  D = 0,          lane 4     kernel row r     -> bank row 8 + r (HBM)
The graph runs h0, h1, h2, h0, ... ROUNDS times with one routing per call; every output must equal, bit for bit, the
all-HBM call with the routing mapped to bank rows. Also without a graph (eager calls).
env: E (128), T (4), ROUNDS (4), PDL (0|1), SF6 is not used here; k31_fork_common.import_dg's MODE=fork etc."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k31_fork_common as c  # noqa: E402

E, T, D = int(os.environ.get("E", "128")), int(os.environ.get("T", "4")), 8
ROUNDS = int(os.environ.get("ROUNDS", "4"))
dev = c.dev
dg, desc = c.import_dg()
print(f"device {torch.cuda.get_device_name(0)}; headers: {desc}", flush=True)
group = c.init_group(29611)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs  # noqa: E402

g = torch.Generator(device=dev).manual_seed(0)
exp = c.make_experts(E, g, 64, "r5.percall.experts")
kw = dict(exp._kernel_kwargs)
b4 = c.bank(exp)
host_a = [c.pinned_like(t, D) for t in b4]
host_b = [c.pinned_like(t[D:], D) for t in b4]
sp = [c.spanning(t[D:], E) for t in b4]
symm = dg.get_symm_buffer_for_mega_moe(group, E, 64, c.K, c.H, c.I, activation="situ", mma_type="fp8xfp4")
desc_tbl = torch.zeros(c.CLONE_BYTES, dtype=torch.uint8, device=dev)
hdrs = []
for d, lane, host in ((D, 4, host_a), (D, 0, host_b), (0, 4, host_a)):
    hb, _ = c.new_ctl(E, d=d, lane=lane, host=host, desc_tbl=desc_tbl)
    hdrs.append(hb[:c.HDR].clone())
buf, ctl = c.new_ctl(E, d=D, lane=4, host=host_a, desc_tbl=desc_tbl)
n = 3 * ROUNDS
gen = torch.Generator(device=dev).manual_seed(11)
# kernel rows 0..E-9, so that 8 + r is a bank row; every call routes to rows below 8 too
ids = [c.make_ids(T, E - D, min(E - D, 12 * T), 16, gen, "rnd") for _ in range(n)]
for i in range(n):
    ids[i][i % T, 3] = i % D                          # at least one pick of a row below 8 (not already in that token's picks)
    dup = (ids[i][i % T] == i % D)
    dup[3] = False
    ids[i][i % T][dup] = -1
xs = [torch.randn(T, c.H, dtype=torch.bfloat16, device=dev, generator=gen) for _ in range(n)]
ws = [torch.where(ids[i] >= 0, torch.softmax(torch.randn(T, c.K, device=dev, generator=gen), -1).float(), torch.zeros(T, c.K, device=dev))
      for i in range(n)]


def bank_ids(i):
    v, r = i % 3, ids[i]
    if v == 0:
        return r
    if v == 1:
        return torch.where((r >= 0) & (r < D), r + D, r)
    return torch.where(r >= 0, r + D, r)


def ref(i):
    r = bank_ids(i)
    prepare_megamoe_inputs(xs[i], ws[i], r, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, c.H, dtype=torch.bfloat16, device=dev)
    dg.fp8_fp4_mega_moe(y, (b4[0], b4[1]), (b4[2], b4[3]), symm, fast_math=True, **kw)
    return y


zero = torch.zeros((), dtype=torch.int32, device=dev)


def call(i):
    prepare_megamoe_inputs(xs[i], ws[i], ids[i], symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, c.H, dtype=torch.bfloat16, device=dev)
    # the header is written by the kernel launched right before the MegaMoE call (an elementwise add, not a memcpy):
    # the worst case for a call that starts under PDL before its predecessor's writes are visible
    torch.add(hdrs[i % 3], zero, out=buf[:c.HDR])
    dg.fp8_fp4_mega_moe(y, (sp[0], sp[1]), (sp[2], sp[3]), symm, cumulative_local_expert_recv_stats=ctl, fast_math=True, **kw)
    return y


refs = [ref(i) for i in range(n)]
torch.cuda.synchronize()
assert all(bool(torch.isfinite(r.float()).all()) for r in refs)
distinct = sum(not c.same(refs[i], refs[j]) for i in range(n) for j in range(i))
eager = [call(i) for i in range(n)]
torch.cuda.synchronize()
bad_e = [i for i in range(n) if not c.same(eager[i], refs[i])]
print(f"eager: {n - len(bad_e)} of {n} calls BIT-EQUAL to the all-HBM reference{'' if not bad_e else f' (differ: {bad_e})'}", flush=True)

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        call(i)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
gr = torch.cuda.CUDAGraph()
outs = []
with torch.cuda.graph(gr):
    for i in range(n):
        outs.append(call(i))
bad_g = set()
for rep in range(20):
    for o in outs:
        o.zero_()
    gr.replay()
    torch.cuda.synchronize()
    bad_g |= {i for i in range(n) if not c.same(outs[i], refs[i])}
print(f"graph of {n} calls (headers h0, h1, h2 in turn, each written by the kernel right before its call), 20 replays: "
      f"{n - len(bad_g)} of {n} calls BIT-EQUAL to the all-HBM reference every time"
      f"{'' if not bad_g else f' (differ: {sorted(bad_g)})'}; the {n} references are {'pairwise distinct' if distinct == n * (n - 1) // 2 else 'NOT all distinct'}",
      flush=True)
ok = not bad_e and not bad_g and distinct == n * (n - 1) // 2
print("per-call control block:", "ALL EQUAL" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
