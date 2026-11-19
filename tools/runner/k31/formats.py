"""Weight and activation formats for the derpk31pro runner.

- The official MXFP4 experts: e2m1 codes (low nibble = even column) with one UE8M0 scale per 32 inputs.
- Bit-plane truncation of those codes (sign, high exponent bit, low exponent bit, mantissa: depth p keeps the first p
  planes) with two reconstructions:
    "centroid": per matrix, each cell of magnitudes that share the kept bits is reconstructed at its mean magnitude,
                weighted by the blocks' squared scales (the MSE-optimal table for that matrix);
    "onesfill": the missing bits are filled with ones and the matrix gets one least-squares gain.
  Depth 4 is the official weight exactly.
- MX fake quantizers (quantize, then dequantize) along the last dimension in blocks of 32 with a shared power-of-two
  scale, as in the OCP MX spec: "mxfp8" (FP8 E4M3 elements, the activation format K3.1 was trained with) and "fp6"
  (FP6 E2M3 elements, a dense-weight candidate).
"""
import torch

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
MAG = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
# each magnitude code with its dropped low bits set to one (stats1.py ONES), per depth
ONES = {1: [6.0] * 8, 2: [1.5] * 4 + [6.0] * 4, 3: [.5, .5, 1.5, 1.5, 3, 3, 6, 6]}
# "trick": what the engine's stager can do with no multiply: each cell of kept bits maps to one fixed e2m1 magnitude
# code, and a power-of-two gain is folded into the scale code. The maps are the best fixed ones on real K3.1 experts
# (searched 11-16 over all code maps x gains 1/4..2): depth 2 = codes 3, 7 at gain 1/2 -> 0.75 / 3.0 (the earlier design's
# nibble (pair << 2) | 3 with the scale code lowered by one; weight MSE 0.148 against the centroids' 0.135); depth 3 =
# codes 1, 2, 4, 6 at gain 1 -> 0.5 / 1 / 2 / 4 (MSE 0.065 against 0.042; filling with ones gives 0.134)
TRICK_MAP = {1: (1.0, [3]), 2: (0.5, [3, 7]), 3: (1.0, [1, 2, 4, 6])}


def _trick_table(depth, device):
    gain, codes = TRICK_MAP[depth]
    cell = torch.arange(8, device=device) >> (4 - depth)
    return gain * MAG.to(device)[torch.tensor(codes, device=device)][cell]


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """[out, in/2] uint8 -> e2m1 codes [out, in] uint8 (low nibble = even column)."""
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def _apply_scale(values: torch.Tensor, scale: torch.Tensor, dtype) -> torch.Tensor:
    s = torch.exp2(scale.float() - 127)
    return (values.view(*scale.shape, 32) * s.unsqueeze(-1)).view(values.shape).to(dtype)


def decode_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """The official weight: packed [out, in/2] uint8, scale [out, in/32] uint8 (E8M0) -> [out, in]. Every value is
    exact in bfloat16."""
    codes = unpack(packed).long()
    return _apply_scale(FP4.to(packed.device, torch.float32)[codes], scale, dtype)


def plane_table(codes: torch.Tensor, scale: torch.Tensor, depth: int, recon: str = "centroid") -> torch.Tensor:
    """Reconstructed magnitude (in units of the block scale) for each of the 8 magnitude codes at `depth` planes,
    fitted to this matrix. Returns float64 [8]."""
    if depth >= 4:
        return MAG.clone()
    dev = codes.device
    mag = (codes & 7).long().flatten()
    rel = scale.float() - scale.float().max()
    energy = torch.exp2(2.0 * rel).repeat_interleave(32, dim=-1).flatten().double()   # squared block scale
    count = torch.zeros(8, dtype=torch.float64, device=dev).index_add_(0, mag, energy)
    m = MAG.to(dev)
    if recon == "centroid":
        cell = torch.arange(8, device=dev) >> (4 - depth)       # depth 1: one cell; 2: two; 3: four
        table = torch.zeros(8, dtype=torch.float64, device=dev)
        for c in range(1 << (depth - 1)):
            sel = cell == c
            den = count[sel].sum()
            table[sel] = (count[sel] * m[sel]).sum() / den if den > 0 else 0.0
        return table
    if recon == "onesfill":
        fill = torch.tensor(ONES[depth], dtype=torch.float64, device=dev)
        gain = (count * m * fill).sum() / (count * fill * fill).sum()
        return gain * fill
    if recon == "trick":
        return _trick_table(depth, dev)
    raise ValueError(f"unknown reconstruction {recon!r}")


def decode_planes(packed: torch.Tensor, scale: torch.Tensor, depth: int, recon: str = "centroid",
                  dtype=torch.bfloat16) -> torch.Tensor:
    """The expert weight as stored at `depth` bit planes (signs always kept)."""
    if depth >= 4:
        return decode_mxfp4(packed, scale, dtype)
    codes = unpack(packed)
    table = plane_table(codes, scale, depth, recon).float()
    mag = table[(codes & 7).long()]
    sign = 1.0 - 2.0 * (codes >> 3).float()
    return _apply_scale(mag * sign, scale, dtype)


def plane_tables(packed: torch.Tensor, scale: torch.Tensor, depth: int, recon: str = "centroid",
                 chunk: int = 8) -> torch.Tensor:
    """plane_table for a stack of experts at once: packed [E, out, in/2], scale [E, out, in/32] -> float32 [E, 8]
    (no host syncs; `chunk` experts at a time bound the transient memory)."""
    E, dev = packed.shape[0], packed.device
    m = MAG.to(dev)
    if depth >= 4:
        return m.float().expand(E, 8).contiguous()
    if recon == "trick":                          # fixed: no statistics needed
        return _trick_table(depth, dev).float().expand(E, 8).contiguous()
    cell = torch.arange(8, device=dev) >> (4 - depth)
    ncell = 1 << (depth - 1)
    fill = torch.tensor(ONES[depth], dtype=torch.float64, device=dev) if recon == "onesfill" else None
    count = _energy_counts(packed, scale)                                     # [E, 8]
    if recon == "centroid":
        num = torch.zeros(E, ncell, dtype=torch.float64, device=dev).index_add_(1, cell, count * m)
        den = torch.zeros(E, ncell, dtype=torch.float64, device=dev).index_add_(1, cell, count)
        out = (num / den.clamp_min(1e-300))[:, cell]
    elif recon == "onesfill":
        gain = (count * m * fill).sum(1, keepdim=True) / (count * fill * fill).sum(1, keepdim=True)
        out = gain * fill
    else:
        raise ValueError(f"unknown reconstruction {recon!r}")
    return out.float()


def decode_stack_torch(packed: torch.Tensor, scale: torch.Tensor, table: torch.Tensor,
                       dtype=torch.bfloat16) -> torch.Tensor:
    """packed [C, out, in/2], scale [C, out, in/32], table [C, 8] magnitudes -> [C, out, in] (PyTorch reference)."""
    codes = unpack(packed)
    C = codes.shape[0]
    mag = (codes & 7).long()
    val = torch.gather(table.view(C, 1, 8).expand(C, codes.shape[1], 8), 2, mag)
    val = torch.where(codes >= 8, -val, val)
    return _apply_scale(val, scale, dtype)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_kernel(packed_ptr, scale_ptr, table_ptr, out_ptr, ROWS, COLS,
                       BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        e = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
        rmask = (rows < ROWS)[:, None]
        e64 = e.to(tl.int64)
        r64 = rows.to(tl.int64)[:, None]
        p = tl.load(packed_ptr + e64 * ROWS * (COLS // 2) + r64 * (COLS // 2) + (cols // 2)[None, :], mask=rmask,
                    other=0)
        nib = tl.where((cols % 2 == 0)[None, :], p & 15, p >> 4)
        mag = tl.load(table_ptr + e * 8 + (nib & 7).to(tl.int32))
        val = tl.where(nib >= 8, -mag, mag)
        s = tl.load(scale_ptr + e64 * ROWS * (COLS // 32) + r64 * (COLS // 32) + (cols // 32)[None, :], mask=rmask,
                    other=127)
        val = val * tl.exp2(s.to(tl.float32) - 127.0)
        tl.store(out_ptr + e64 * ROWS * COLS + r64 * COLS + cols[None, :], val.to(tl.bfloat16), mask=rmask)

    @triton.jit
    def _hist_kernel(packed_ptr, scale_ptr, smax_ptr, hist_ptr, ROWS, COLS,
                     BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        """hist int32 [E, 256]: counts of (binades below the expert matrix's largest scale, clipped at 31) x 8 + mag.
        Full tiles only (ROWS % BLOCK_R == 0, COLS % BLOCK_C == 0: the caller checks)."""
        e = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
        e64 = e.to(tl.int64)
        r64 = rows.to(tl.int64)[:, None]
        p = tl.load(packed_ptr + e64 * ROWS * (COLS // 2) + r64 * (COLS // 2) + (cols // 2)[None, :])
        nib = tl.where((cols % 2 == 0)[None, :], p & 15, p >> 4)
        s = tl.load(scale_ptr + e64 * ROWS * (COLS // 32) + r64 * (COLS // 32) + (cols // 32)[None, :]).to(tl.int32)
        d = tl.minimum(tl.load(smax_ptr + e).to(tl.int32) - s, 31)
        key = d * 8 + (nib & 7).to(tl.int32)
        h = tl.histogram(tl.reshape(key, (BLOCK_R * BLOCK_C,)), 256)
        tl.atomic_add(hist_ptr + e * 256 + tl.arange(0, 256), h)

    HAVE_TRITON = True
except Exception:                                 # no triton (or no GPU backend): the PyTorch path does the work
    HAVE_TRITON = False


def _energy_counts(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """float64 [E, 8]: per expert matrix, the count of each magnitude code weighted by its block's squared scale
    relative to the matrix's largest (the weights plane_table fits); one Triton histogram pass on a GPU."""
    E, R, half = packed.shape
    cols = half * 2
    if packed.is_cuda and HAVE_TRITON and cols % 256 == 0 and R % 32 == 0:
        smax = scale.view(E, -1).amax(1).to(torch.int32).contiguous()
        hist = torch.zeros(E, 256, dtype=torch.int32, device=packed.device)
        _hist_kernel[(E, triton.cdiv(R, 32), cols // 256)](packed.contiguous(), scale.contiguous(), smax, hist, R, cols,
                                                           BLOCK_R=32, BLOCK_C=256)
        w = 4.0 ** -torch.arange(32, dtype=torch.float64, device=packed.device)
        return (hist.view(E, 32, 8).double() * w.view(1, 32, 1)).sum(1)
    out = torch.empty(E, 8, dtype=torch.float64, device=packed.device)
    for a in range(0, E, 8):                     # PyTorch reference, 8 experts at a time
        p, s = packed[a:a + 8], scale[a:a + 8]
        C = p.shape[0]
        mag = (unpack(p) & 7).long().view(C, -1)
        rel = s.float() - s.float().amax(dim=(1, 2), keepdim=True)
        energy = torch.exp2(2.0 * rel).repeat_interleave(32, dim=-1).view(C, -1).double()
        out[a:a + C] = torch.zeros(C, 8, dtype=torch.float64, device=packed.device).scatter_add_(1, mag, energy)
    return out


def decode_stack(packed: torch.Tensor, scale: torch.Tensor, table: torch.Tensor,
                 dtype=torch.bfloat16) -> torch.Tensor:
    """Decode a stack of experts with per-expert magnitude tables (MAG = the official weights): one Triton kernel on a
    GPU (bfloat16 out), the PyTorch reference otherwise."""
    C, R, half = packed.shape
    cols = half * 2
    if packed.is_cuda and HAVE_TRITON and dtype == torch.bfloat16 and cols % 256 == 0:
        out = torch.empty((C, R, cols), dtype=torch.bfloat16, device=packed.device)
        grid = (C, triton.cdiv(R, 32), cols // 256)
        _decode_kernel[grid](packed.contiguous(), scale.contiguous(), table.float().contiguous(), out, R, cols,
                             BLOCK_R=32, BLOCK_C=256)
        return out
    return decode_stack_torch(packed, scale, table, dtype)


_E2M3 = torch.tensor(sorted({m / 8 for m in range(8)} | {(1 + m / 8) * 2.0 ** (e - 1) for e in (1, 2, 3)
                                                          for m in range(8)}))          # 32 magnitudes, max 7.5
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])                       # FP4 magnitudes, max 6
_EMAX = {"mxfp8": 8, "fp6": 2, "mxfp4": 2}                                              # exponent of the largest normal
_MAXV = {"mxfp8": 448.0, "fp6": 7.5, "mxfp4": 6.0}                                      # largest element magnitude


def _round_e2m1(y: torch.Tensor) -> torch.Tensor:
    grid = _E2M1.to(y.device)
    mid = (grid[1:] + grid[:-1]) / 2
    q = grid[torch.bucketize(y.abs().clamp(max=6.0), mid)]
    return torch.where(y < 0, -q, q)


def _round_e2m3(y: torch.Tensor) -> torch.Tensor:
    grid = _E2M3.to(y.device)
    mid = (grid[1:] + grid[:-1]) / 2
    q = grid[torch.bucketize(y.abs().clamp(max=7.5), mid)]
    return torch.where(y < 0, -q, q)


def mx_fake_quant(x: torch.Tensor, fmt: str, block: int = 32, scale_rule: str = "ceil") -> torch.Tensor:
    """Quantize then dequantize x along its last dimension in blocks of `block` sharing a power-of-two scale;
    elements round to nearest and saturate. Returns x's dtype.
    scale_rule "ceil": the smallest power of two with amax / scale <= the element maximum (nothing clips; what most
    MXFP8 kernels do). "ocp": 2^(floor(log2(amax)) - emax) as the OCP MX spec writes it (E4M3 then clips the part of
    a block between 448 and 512)."""
    if fmt in (None, "bf16", "none"):
        return x
    shape = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    if scale_rule == "ceil":
        s = torch.exp2(torch.ceil(torch.log2(amax / _MAXV[fmt])))
    elif scale_rule == "ocp":
        s = torch.exp2(torch.floor(torch.log2(amax)) - _EMAX[fmt])
    else:
        raise ValueError(f"unknown scale rule {scale_rule!r}")
    y = xf / s
    if fmt == "mxfp8":
        q = y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    elif fmt == "fp6":
        q = _round_e2m3(y)
    elif fmt == "mxfp4":
        q = _round_e2m1(y)
    else:
        raise ValueError(f"unknown MX format {fmt!r}")
    return (q * s).reshape(shape).to(x.dtype)


def nvfp4_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 (as Blackwell's kind::mxf4nvf4 and vLLM's/ModelOpt's checkpoints store it): E2M1 elements in blocks of 16
    along the last dimension, each block's scale an FP8 E4M3 value, times one float32 scale per tensor
    (global = amax / (448 * 6)). Quantize then dequantize; returns x's dtype."""
    shape = x.shape
    xf = x.float()
    g = (xf.abs().amax().clamp_min(2.0 ** -126) / (448.0 * 6.0))
    xb = xf.reshape(-1, 16)
    s = (xb.abs().amax(-1, keepdim=True) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    q = _round_e2m1((xb / (s * g)).clamp(-6.0, 6.0))
    return (q * s * g).reshape(shape).to(x.dtype)


def relative_mse(w: torch.Tensor, ref: torch.Tensor) -> float:
    w, ref = w.double(), ref.double()
    return float(((w - ref) ** 2).sum() / (ref ** 2).sum())
