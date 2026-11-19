"""K31_SF6=1: the MXFP4 rows in GPU memory keep their block scales as 6-bit codes (sf6.py), 17,289,216 B per expert
row instead of 17,547,264 (-1.5%: the bytes of 9,518 rows hold 9,660).

What changes for a MoE layer (mode staged):
  - its two scale tensors are coded banks, uint8 [rows, 516,096] and [rows, 258,048] (shared staging rows first, as
    the weight tensors: vmm.SharedSlots); the module's scale parameters become spans over them that claim the
    uncoded shape (the launcher checks that shape and builds a tensor map it never loads through in a coded call);
  - every MegaMoE call gets the fork's control block (the `cumulative_local_expert_recv_stats` argument: magic, no
    host rows, word 11 = coded) with the layer's clone table and the four coded scale maps after it;
  - everything that writes or reads a row's scales goes through stager2.py (staging, the copy-engine lane, promotion
    merges, write-back extracts) or `write_scales` here (rows loaded from the checkpoint).
The fork's kernel moves E0 to the token scales and unpacks the codes in the scale copy; its output is bit-identical
to uncoded rows (tests/fork/step_cost_k31.py SF6=1 with host rows; tests/fork/sf6_rows_test.py for the engine's form:
no host rows, this module's control block). Needs the MegaMoE fork (engine/overlay/megamoe_k31, MEGAMOE_FORK=1).
"""
import os
import re

import torch

from . import sf6

ON = os.environ.get("K31_SF6", "0") == "1"
MAGIC = 0x53505244              # "DRPS": derpseek_host::kMagic
HDR = 16                        # control block header words; cumulative picks per kernel row follow
KERNEL_REL = "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh"


def check_fork(dg_root: str) -> None:
    """The DeepGEMM headers in use must be the fork's, with this module's E0 and layout."""
    path = os.path.join(dg_root, KERNEL_REL)
    try:
        h = open(path).read()
    except OSError as ex:
        raise RuntimeError(f"K31_SF6=1: cannot read {path} ({ex})")
    if "namespace derpseek_sf6" not in h:
        raise RuntimeError(f"K31_SF6=1 needs the MegaMoE fork's kernel (MEGAMOE_FORK=1); {path} is the image's")

    def define(name, default):
        m = re.search(rf"^#define {name} (\d+)", h, re.M)
        return int(m.group(1)) if m else default
    e0, layout = define("DRP_SF6_E0", 96), define("DRP_SF6_LAYOUT", 2)
    if (e0, layout) != (sf6.E0, sf6.LAYOUT):
        raise RuntimeError(f"K31_SF6=1: the fork is built for E0 {e0}, layout {layout}; sf6.py has {sf6.E0}, {sf6.LAYOUT}")


def row_bytes(raw: torch.Tensor) -> int:
    """Bytes of one expert's coded scales for an uncoded scale tensor [rows, N, KB]."""
    return sf6.row_bytes(raw.shape[1], raw.shape[2])


def prototype(raw: torch.Tensor) -> torch.Tensor:
    """A one-row coded bank shaped for vmm.SharedSlots (uint8 [1, row bytes]) from the uncoded scale parameter."""
    return torch.empty((1, row_bytes(raw)), dtype=torch.uint8, device=raw.device)


def attach(mod, coded13: torch.Tensor, coded2: torch.Tensor) -> None:
    """Make `mod` (a DeepGemmMegaMoEExperts whose scale parameters still have the uncoded shape and strides) run on
    the coded banks coded13 / coded2 (uint8 [rows, row bytes], one row per kernel row)."""
    dev = coded13.device
    rows = coded13.shape[0]
    raws = (mod.w13_weight_scale, mod.w2_weight_scale)
    assert coded2.shape[0] == rows and coded13.dtype == coded2.dtype == torch.uint8
    assert coded13.shape[1] == row_bytes(raws[0]) and coded2.shape[1] == row_bytes(raws[1]), "coded row sizes"
    shapes = [tuple(r.shape[1:]) for r in raws]                       # (N, KB) of the two matrices
    spans = [sf6.span(c, rows, r.shape, r.stride()) for c, r in zip((coded13, coded2), raws)]
    tbl = torch.zeros(sf6.CLONE_BYTES + sf6.TAIL_BYTES, dtype=torch.uint8, device=dev)
    assert tbl.data_ptr() % 128 == 0, "the clone table must be 128-byte aligned"
    # no host rows here: the two host maps are never loaded through; they are encoded over the HBM banks
    sf6.write_tail(tbl[sf6.CLONE_BYTES:], coded13, coded2, coded13, coded2, shapes[0], shapes[1])
    buf = torch.zeros(HDR + rows, dtype=torch.int32, device=dev)
    buf[0] = MAGIC
    buf[11] = sf6.ctl_word(prof=False)
    buf[:HDR].view(torch.int64)[7] = tbl.data_ptr()
    for name, span in zip(("w13_weight_scale", "w2_weight_scale"), spans):
        if isinstance(mod, torch.nn.Module):
            setattr(mod, name, torch.nn.Parameter(span, requires_grad=False))
        else:
            setattr(mod, name, span)
    object.__setattr__(mod, "_k31_sf6", (coded13, coded2))
    object.__setattr__(mod, "_k31_sf6_ctl", (buf, tbl))               # kept alive with the module
    mod._kernel_kwargs = dict(mod._kernel_kwargs, cumulative_local_expert_recv_stats=buf[:rows])


_TMP: dict = {}


def write_scales(mod, r: int, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor, dev) -> None:
    """One expert's three checkpoint scale tensors (uint8 [out, in / 32]) into kernel row r of the coded banks: laid
    out by the module's own loader into uncoded scratch rows, then coded."""
    coded = mod._k31_sf6
    key = (str(dev), tuple(mod.w13_weight_scale.shape[1:]), tuple(mod.w2_weight_scale.shape[1:]))
    tmp = _TMP.get(key)
    if tmp is None:
        def scratch(raw):                                             # [1, N, KB] int32 over [KB][N] memory
            n, kb = raw.shape[1], raw.shape[2]
            return torch.zeros((1, kb, n), dtype=torch.int32, device=dev).transpose(1, 2)
        tmp = _TMP[key] = (scratch(mod.w13_weight_scale), scratch(mod.w2_weight_scale))
    mod._load_weight(tmp[0][0], w1.to(dev), "w1", True)
    mod._load_weight(tmp[0][0], w3.to(dev), "w3", True)
    mod._load_weight(tmp[1][0], w2.to(dev), "w2", True)
    coded[0][r] = sf6.encode(tmp[0])[0]
    coded[1][r] = sf6.encode(tmp[1])[0]
