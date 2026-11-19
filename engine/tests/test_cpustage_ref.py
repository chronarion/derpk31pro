"""The Grace pool's kernel against the project's own references, on real hi2 records (CPU only: torch CPU, no GPU).
Needs engine/band/libcpustage.so (./build.sh or engine/pool/build.sh) and the band package (sidecar/band).

Run in the engine image from the repository root, with the record store (HI2_DIR) mounted:
    docker run --rm --network none --cpuset-cpus 67-69 --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/repo:ro -v /models/k31-hi2:/hi2:ro --entrypoint python3 vllm/vllm-openai:kimi-k31 \
        /repo/engine/tests/test_cpustage_ref.py /hi2/L01.hi2
Checks, for 3 records x 5 rows and 8 records x 5 rows (exit code 0 = PASS):
  q  the library's quantization of x (int8 with a power-of-two scale per 32) equals the rule in torch, bit for bit
  a  the kernel (cpustage_run_single) against a float64 torch pipeline built on band.reference.decode_hi2 (the record
     decode that sidecar/tests/test_native.py ties to the official weights) with the pool's activation rule: relative
     L2 per row < 5e-3 (float32 sums against float64: a few rounding flips)
  b  against band.reference.forward with MegaMoE's arithmetic (MXFP8 input rows and intermediate, scale floor 105,
     outputs rounded to bfloat16): relative L2 per row < 6e-2; expected ~3.5e-2, the distance hook._selfcheck measures
     between MegaMoE and exact activations. This is what a staged pick sees when the pool computes it instead of
     MegaMoE: the activation rounding only (a layout error reads ~1).
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("REPO", "/repo")
sys.path.insert(0, os.path.join(REPO, "engine"))
sys.path.insert(0, os.path.join(REPO, "sidecar"))
from band import reference as ref  # noqa: E402
from band.rtxband import FLAGS_MEGAMOE, SF_MIN_MEGAMOE, mxfp8_encode  # noqa: E402
from k31_engine import cpupool as cp  # noqa: E402

H, REC = 3584, cp.REC_BYTES
path = sys.argv[1] if len(sys.argv) > 1 else "/hi2/L01.hi2"
pool = cp.CpuPool(threads=1, core0=min(os.sched_getaffinity(0)))     # the library and run_single only: never started
fd = os.open(path, os.O_RDONLY)
try:
    recs = np.frombuffer(os.pread(fd, 8 * REC, 0), dtype=np.uint8).reshape(8, REC).copy()
finally:
    os.close(fd)
experts = [ref.decode_hi2(torch.from_numpy(recs[p])) for p in range(8)]


def quant(a: torch.Tensor):
    """The pool's rule (cpustage_seg2.h): int8 with a power-of-two scale per 32, dequantized (float64)."""
    v = a.float().reshape(-1, 32)
    amax = v.abs().amax(-1, keepdim=True)
    e = (((amax.view(torch.int32) + 0x1FFFF) >> 23) - 133).clamp(-100, 120)
    q = torch.round(v.double() * torch.exp2(-e.double())).clamp(-127, 127)       # round half to even
    return (q * torch.exp2(e.double())).reshape(a.shape), q.reshape(a.shape), e


def case(name, P, w, seed):
    T = w.shape[1]
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((T, H), generator=g) * 0.5).bfloat16().float()
    out, xq, _ = pool.run_single([recs[p] for p in range(P)], x.numpy(), w.numpy(), want_q=True)
    got = torch.from_numpy(out).double()
    xd, xq_ref, _ = quant(x)
    ok_q = bool((xq_ref.to(torch.int8) == torch.from_numpy(xq)).all())
    want = torch.zeros(T, H, dtype=torch.float64)
    for p in range(P):
        w1, w3, w2 = experts[p]
        gate = (xd @ w1.double().T).float().bfloat16().float()
        up = (xd @ w3.double().T).float().bfloat16().float()
        a = ref.situ(gate, up) * w[p, :T].view(-1, 1)
        ad, _, _ = quant(a)
        y = (ad @ w2.double().T).float().bfloat16().double()
        used = (w[p, :T] != 0).view(-1, 1)
        want += torch.where(used, y, torch.zeros_like(y))
    rows = [t for t in range(T) if bool((w[:, t] != 0).any())]
    rel_a = max(float((got[t] - want[t]).norm() / want[t].norm()) for t in rows)
    w8 = torch.zeros(P, 8)
    w8[:, :T] = w
    y_mm, _ = ref.forward(mxfp8_encode(x)[2], experts[:P], w8, FLAGS_MEGAMOE, SF_MIN_MEGAMOE)
    rel_b = max(float((got[t] - y_mm[t]).norm() / y_mm[t].norm()) for t in rows)
    good = ok_q and rel_a < 5e-3 and rel_b < 6e-2
    print(f"{'PASS' if good else 'FAIL'} {name}: q {'equal' if ok_q else 'DIFFERENT'}; a against the float64 pipeline "
          f"{rel_a:.2e} (< 5e-3); b against MegaMoE's arithmetic {rel_b:.2e} (< 6e-2, expected ~3.5e-2)", flush=True)
    c_line(name, x, w, P, rows, got, y_mm)
    return good


def exact(x, w, P):
    """No activation quantization (float64 x and intermediate), the same bfloat16 roundings of gate / up / outputs."""
    T = x.shape[0]
    y = torch.zeros(T, H, dtype=torch.float64)
    for p in range(P):
        w1, w3, w2 = experts[p]
        gate = (x.double() @ w1.double().T).float().bfloat16().float()
        up = (x.double() @ w3.double().T).float().bfloat16().float()
        a = ref.situ(gate, up) * w[p, :T].view(-1, 1)
        yp = (a.double() @ w2.double().T).float().bfloat16().double()
        y += torch.where((w[p, :T] != 0).view(-1, 1), yp, torch.zeros_like(yp))
    return y


def c_line(name, x, w, P, rows, got, y_mm):
    """c (information, no gate): the pool's and MegaMoE's rows against exact activations; with the same rows given a
    few outlier channels (x 20 on 1 in 64 channels), which a power-of-two int8 scale per 32 handles worse than E4M3."""
    out = []
    for label, xx, g, m in (("rows as above", x, got, y_mm), ("outlier channels", None, None, None)):
        if xx is None:
            xx = x.clone()
            xx[:, ::64] *= 20.0
            xx = xx.bfloat16().float()
            g = torch.from_numpy(pool.run_single([recs[p] for p in range(P)], xx.numpy(), w.numpy())).double()
            w8 = torch.zeros(P, 8)
            w8[:, :w.shape[1]] = w
            m, _ = ref.forward(mxfp8_encode(xx)[2], experts[:P], w8, FLAGS_MEGAMOE, SF_MIN_MEGAMOE)
        ex = exact(xx, w, P)
        rp = max(float((g[t] - ex[t]).norm() / ex[t].norm()) for t in rows)
        rm = max(float((m[t] - ex[t]).norm() / ex[t].norm()) for t in rows)
        out.append(f"{label}: pool {rp:.2e}, MegaMoE {rm:.2e}")
    print(f"     c {name} against exact activations: " + "; ".join(out), flush=True)


w3 = torch.zeros(3, 5)
w3[0, 0], w3[0, 3] = 0.31, 0.12
w3[1, 1], w3[1, 2], w3[1, 4] = 0.22, 0.05, 0.40
w3[2, 0] = 0.02
w8 = torch.zeros(8, 5)
gen = torch.Generator().manual_seed(1)
for p in range(8):
    for t in torch.randperm(5, generator=gen)[:2].tolist():
        w8[p, t] = float(torch.rand(1, generator=gen)) * 0.3 + 0.02
ok = case("3 records x 5 rows", 3, w3, 0)
ok = case("8 records x 5 rows", 8, w8, 1) and ok
print("CPUSTAGE REF: PASS" if ok else "CPUSTAGE REF: FAIL")
sys.exit(0 if ok else 1)
