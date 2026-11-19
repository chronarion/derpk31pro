"""Torch reference of the band's arithmetic, independent of the kernel and of the band record layout.

- `expand()`: the stager's expansion of hi2 code bytes into native MXFP4 bytes (copied from
  engine/store/build_store.py: nibble = (pair << 2) | 3).
- `decode_hi2()`: one hi2 record -> the dequantized float weights w1 (gate), w3 (up), w2, exactly what MegaMoE
  computes with after the stager wrote the record into a native row (e2m1 value of the staged nibble x
  2^(stored scale code - 127)), mapped back from the kernel layout to the checkpoint's rows.
- `forward()`: y[t] = sum_e w[e, t] * w2_e @ Q8(SiTU(w1_e @ x[t], w3_e @ x[t])) with the GEMMs in float64 and
  the activation / quantization steps in float32 as the kernel (and MegaMoE, and the runner) do them; the flags
  select the same variants as the kernel's (SiTU is always torch's: the kernel's fast-math form has no torch twin).
"""
import torch

from .rtxband import (CODES13, CODES2, FLAG_ACT_BF16, FLAG_NO_BF16, FLAG_OUT_BF16, FLAG_WEIGHT_AFTER, H, I, KQ1, KQ2,
                      MAXT, SF13, SF2, SITU_ALPHA, SITU_BETA, mxfp8_encode, scale_row)

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def expand(rec: torch.Tensor) -> torch.Tensor:
    """The stager's expansion: [n, B/2] record bytes -> [n, B] native code bytes, nibble = (pair << 2) | 3."""
    p = [(rec >> (2 * j)) & 3 for j in range(4)]
    n = [(q << 2) | 3 for q in p]
    return torch.stack((n[0] | (n[1] << 4), n[2] | (n[3] << 4)), dim=-1).reshape(rec.shape[0], -1)


def _dequant(codes: torch.Tensor, sf: torch.Tensor, rows: int, kq: int) -> torch.Tensor:
    """codes uint8 [rows, K/4] record bytes, sf uint8 [kq, rows(scale order), 4] -> float32 [rows, K] in kernel rows."""
    native = expand(codes)                                                     # [rows, K/2], low nibble = even column
    nib = torch.stack((native & 15, native >> 4), dim=-1).flatten(-2).long()   # [rows, K]
    val = FP4.to(codes.device)[nib]
    s = sf.permute(1, 0, 2).reshape(rows, kq * 4)[scale_row(rows, codes.device)]   # [rows natural, K/32]
    return val * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)


def decode_hi2(rec: torch.Tensor):
    """One hi2 record uint8 [HI2_REC] -> (w1 [3072, 3584], w3 [3072, 3584], w2 [3584, 3072]) float32."""
    c13 = rec[:CODES13].view(6144, 896)
    c2 = rec[CODES13:CODES13 + CODES2].view(3584, 768)
    o = CODES13 + CODES2
    s13 = rec[o:o + SF13].view(KQ1, 6144, 4)
    s2 = rec[o + SF13:o + SF13 + SF2].view(KQ2, 3584, 4)
    w13 = _dequant(c13, s13, 6144, KQ1)
    w2 = _dequant(c2, s2, 3584, KQ2)
    r = torch.arange(I, device=rec.device)
    gate_row = 16 * (r // 8) + r % 8                                           # _kernel_layout_shard: w1 rows
    return w13[gate_row], w13[gate_row + 8], w2


def situ(gate: torch.Tensor, up: torch.Tensor, alpha: float = SITU_ALPHA, beta: float = SITU_BETA) -> torch.Tensor:
    """SituAndMul of modeling_kimi_linear.py in float32: alpha tanh(gate / alpha) sigmoid(gate) * beta tanh(up / beta)."""
    gate, up = gate.float(), up.float()
    a = alpha * torch.tanh(gate / alpha) * torch.sigmoid(gate)
    if beta:
        up = beta * torch.tanh(up / beta)
    return a * up


def forward(xv: torch.Tensor, experts, w: torch.Tensor, flags: int = 0, sf_min: int = 105, alpha: float = SITU_ALPHA,
            beta: float = SITU_BETA):
    """xv float [T, 3584]: the dequantized MXFP8 input rows. experts: list of (w1, w3, w2) float32. w [E, T] routing
    weights (0 = the row does not use the expert). Returns (y float64 [T, 3584], per expert (codes uint8 [T, 3072],
    scale codes uint8 [T, 96]) of the quantized intermediate)."""
    T = xv.shape[0]
    x64 = xv.double()
    y = torch.zeros(T, H, dtype=torch.float64, device=xv.device)
    inter = []
    for e, (w1, w3, w2) in enumerate(experts):
        we = w[e, :T].to(xv.device).float()
        gate = (x64 @ w1.double().T).float()
        up = (x64 @ w3.double().T).float()
        if not flags & FLAG_NO_BF16:
            gate, up = gate.bfloat16().float(), up.bfloat16().float()
        a = situ(gate, up, alpha, beta)
        if flags & FLAG_ACT_BF16:
            a = a.bfloat16().float()
        if flags & FLAG_WEIGHT_AFTER:
            a = torch.where(we.view(-1, 1) != 0, a, torch.zeros_like(a))
        else:
            a = a * we.view(-1, 1)
        codes, sf, av = mxfp8_encode(a, sf_min)
        out = av.double() @ w2.double().T
        if flags & FLAG_OUT_BF16:
            out = out.float().bfloat16().double()
        if flags & FLAG_WEIGHT_AFTER:
            out = (out.float() * we.view(-1, 1)).double()
        y += out
        inter.append((codes, sf))
    return y, inter
