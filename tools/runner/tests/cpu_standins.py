"""Plain-PyTorch stand-ins for the fla kernels Moonshot's K3 reference calls (fla ships them for Triton / CUDA only),
so the runner's self-test can run on the CPU. Both sides of the self-test (the reference's own full forward and the
layer-streaming pass) use the same stand-ins, so they only need to be causal and deterministic; they follow fla's
documented semantics closely but are not bit-compatible with it."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConvolution(nn.Conv1d):
    """Causal depthwise convolution over time, then the activation; same parameters as fla's (weight [C, 1, K])."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias,
                         padding=kernel_size - 1)
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kwargs):
        T = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), self.weight.to(x.dtype),
                     None if self.bias is None else self.bias.to(x.dtype),
                     padding=self.padding[0], groups=self.groups)[..., :T].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        return y, None


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, activation="sigmoid", **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps, self.activation = eps, activation

    def forward(self, x, g, **kwargs):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        gate = torch.sigmoid(g.float()) if self.activation == "sigmoid" else F.silu(g.float())
        return (y * gate).to(x.dtype)


def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              safe_gate=False, lower_bound=None, transpose_state_layout=False, cu_seqlens=None, **kwargs):
    from fla.ops.kda.naive import naive_recurrent_kda
    dtype = v.dtype
    B, T, H, D = q.shape
    q, k, v = q.float(), k.float(), v.float()
    if use_qk_l2norm_in_kernel:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    if use_gate_in_kernel:
        a = A_log.float().exp().view(1, 1, H, 1)
        x = g.float() + dt_bias.float().view(H, -1)
        g = lower_bound * torch.sigmoid(a * x) if lower_bound is not None else -a * F.softplus(x)
    else:
        g = g.float()
    b = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta.float()
    o, state = naive_recurrent_kda(q, k, v, g, b, initial_state=initial_state, output_final_state=output_final_state)
    return o.to(dtype), state


def patch(ref):
    ref.ShortConvolution = ShortConvolution
    ref.FusedRMSNormGated = FusedRMSNormGated
    ref.chunk_kda = chunk_kda
    ref.fused_recurrent_kda = chunk_kda
