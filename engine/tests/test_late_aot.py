"""The late tier's Triton kernels (late.py) compiled ahead of time for the GB300 (sm_103), on the CPU: Triton's full
pipeline down to the cubin, for every dtype and switch the engine launches them with. No GPU (run beside
test_late_cpu.py, which runs the same kernels in Triton's interpreter):

    docker run --rm --network none --cpuset-cpus 70 --ulimit core=0 --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/repo -w /repo \
      --entrypoint python3 vllm/vllm-openai:kimi-k31 engine/tests/test_late_aot.py

Exit 0 = every variant compiled (registers and shared memory printed)."""
import os
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from k31_engine import late  # noqa: E402

target = GPUTarget("cuda", 103, 32)
fails = []


def build(name, fn, sig, consts, num_warps):
    try:
        src = ASTSource(fn=fn, signature={**sig, **{k: "constexpr" for k in consts}}, constexprs=consts)
        k = triton.compile(src, target=target, options={"num_warps": num_warps})
        md = k.metadata
        print(f"PASS {name}: shared {md.shared} B, cubin {len(k.asm['cubin'])} B", flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}", flush=True)
        fails.append(name)


for ids_t in ("*i32",):
    for w_t in ("*fp32", "*bf16"):
        for band in (False, True):
            for pdl in (False, True):
                build(f"plan w {w_t} band {band} pdl {pdl}", late._lplan,
                      {"ids_ptr": ids_t, "w_ptr": w_t, "row_ptr": "*i32", "bank_ptr": "*i32", "band_ptr": "*i32",
                       "rows_ptr": "*i32", "lslot_ptr": "*i32", "lw_ptr": "*fp32", "lmask_ptr": "*i32",
                       "cnt_ptr": "*i64", "T": "i32", "lfrom": "i32", "lto": "i32", "D": "i32"},
                      {"NE": late.E, "TB": late.MAXT, "KC": late.K, "PM": late.PMAX, "HAS_BAND": band,
                       "USE_PDL": pdl}, 8)
for ids_t in ("*i32", "*i64"):
    for w_t in ("*fp32", "*bf16"):
        for pdl in (False, True):
            build(f"mean ids {ids_t} w {w_t} pdl {pdl}", late._lmean,
                  {"out_ptr": "*fp32", "lmask_ptr": "*i32", "ids_ptr": ids_t, "w_ptr": w_t, "mean_ptr": "*bf16",
                   "D": "i32", "NE": "i32"}, {"KC": late.K, "BD": 512, "USE_PDL": pdl}, 4)
for z_t in ("*fp32", "*bf16"):
    for pdl in (False, True):
        build(f"add z {z_t} pdl {pdl}", late._ladd,
              {"dst_ptr": "*bf16", "out_ptr": "*bf16", "z_ptr": z_t, "y_ptr": "*bf16", "coef_ptr": "*fp32",
               "HC": "i32", "RSD": "i32", "RSO": "i32"}, {"BH": 1024, "USE_PDL": pdl}, 4)
print("LATE AOT " + ("PASS" if not fails else f"FAIL ({len(fails)})"), flush=True)
sys.exit(0 if not fails else 1)
