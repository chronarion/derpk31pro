"""K31_DENSE_FP6=1: dense linears as MXFP6 E2M3 weights (group-32 E8M0 scales) with BF16 inputs (W6A16), computed by
Humming (humming 0.1.18 in the image; vLLM's HummingMxFp6LinearKernel uses the same library), instead of vLLM's online
FP8 block weights with per-token-group FP8 inputs (W8A8 + a separate activation quant kernel per GEMM).

Runner pass `dense-kld` (gen32, generated rows): FP6 weights 0.0048 = FP8 block 0.0048 (paired -0.0000 [-0.0002,
+0.0001]); 25% fewer weight bytes and no activation quant kernels. Hooks Fp8PerBlockOnlineLinearMethod: for a layer
whose prefix matches K31_FP6_LAYERS (comma-separated substrings), process_weights_after_loading quantizes the BF16
weight it was about to cast to FP8 into a HummingLayer (N padded to 256, K to 128) and drops the weight; apply runs
the Humming GEMM. Layers outside the list keep FP8. Not for layers whose weight other code reads directly (MLA's
fused q/kv-a projections and kv_b_proj, the DSA indexer)."""
import os

import torch

DEFAULT = "in_proj_qkvgfab,f_b_proj,self_attn.o_proj,shared_experts,routed_expert_down_proj,routed_expert_up_proj"
_N = [0, 0, 0]
# Humming weight formats (inputs stay BF16: W*A16, no activation quant kernels)
FORMATS = {
    "fp6": {"dtype": "float6e2m3", "scale_dtype": "float8e8m0", "group_size": 32, "weight_scale_type": "group"},
    "nvfp4": {"dtype": "float4e2m1", "scale_dtype": "float8e4m3", "group_size": 16, "weight_scale_type": "group",
              "weight_scale_2_type": "tensor"},
    "mxfp4": {"dtype": "float4e2m1", "scale_dtype": "float8e8m0", "group_size": 32, "weight_scale_type": "group"},
    "fp8": {"dtype": "float8e4m3", "scale_dtype": "float8e8m0", "group_size": 32, "weight_scale_type": "group"},
}


def _format(prefix: str):
    """K31_DENSE_FMT: ordered "substring:fmt" pairs ("*" = every other layer of the default list; fmt "fp8b" = keep
    vLLM's FP8 block path); else K31_DENSE_FP6=1 = fp6 for K31_FP6_LAYERS. Never the skip list."""
    skip = [p for p in os.environ.get("K31_FP6_SKIP", "draft,mtp,kv_b_proj").split(",") if p]
    if any(s in prefix for s in skip):
        return None
    spec = os.environ.get("K31_DENSE_FMT", "")
    if spec:
        pats = [p for p in DEFAULT.split(",") if p]
        for pair in spec.split(","):
            pat, _, fmt = pair.rpartition(":")
            if (pat == "*" and any(p in prefix for p in pats)) or (pat and pat != "*" and pat in prefix):
                return None if fmt == "fp8b" else fmt
        return None
    pats = [p for p in os.environ.get("K31_FP6_LAYERS", DEFAULT).split(",") if p]
    return "fp6" if any(p in prefix for p in pats) else None


def install(log) -> None:
    from vllm.model_executor.layers.quantization.online import fp8 as of
    from vllm.model_executor.utils import replace_parameter
    cls = of.Fp8PerBlockOnlineLinearMethod
    if getattr(cls, "_k31_fp6", False):
        return
    orig_pw, orig_apply = cls.process_weights_after_loading, cls.apply

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        prefix = getattr(layer, "prefix", "")
        fmt = _format(prefix)
        if fmt is None:
            return orig_pw(self, layer)
        if fmt == "mxfp4dg":                 # W4A8 through DeepGEMM's FP8 x FP4 GEMM (mxfp4dg.py)
            from . import mxfp4dg
            w = self.get_weight_for_quantization(layer)
            N, K = w.shape
            if K % 128:
                return orig_pw(self, layer)
            q, sft = mxfp4dg.prepare(w)
            layer._k31_dg4 = (q, sft, N)
            layer.input_scale = None
            replace_parameter(layer, "weight", torch.empty((0, K), dtype=torch.bfloat16, device=w.device))
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True
            _N[0] += 1
            _N[1] += q.numel() + sft.numel() * 4
            if _N[0] % 50 == 1:
                log(f"MXFP4 (DeepGEMM W4A8) dense: {_N[0]} layers ({_N[1] / 1e9:.1f} GB), last {prefix} [{N} x {K}]")
            return
        if fmt == "fp8c":                    # FP8 128 x 128 block weights on vLLM's CUTLASS blockwise GEMM (W8A8, 1 x 128
            w = self.get_weight_for_quantization(layer)   # activation groups): faster than DeepGEMM and W8A16 on the
            N, K = w.shape                                 # wide decode shapes (11-18 bench: 12288 x 7168 19.4 us vs
            if N % 128 or K % 128:                         # W8A16 29.4 / DeepGEMM 33.0)
                return orig_pw(self, layer)
            wb = w.float().view(N // 128, 128, K // 128, 128)
            sw = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)
            wq = (wb / sw[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(N, K).contiguous()
            del wb
            layer._k31_c8 = (wq, sw.contiguous(), N)
            layer.input_scale = None
            replace_parameter(layer, "weight", torch.empty((0, K), dtype=torch.bfloat16, device=w.device))
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True
            _N[0] += 1
            _N[1] += wq.numel() + sw.numel() * 4
            if _N[0] % 50 == 1:
                log(f"CUTLASS FP8 dense: {_N[0]} layers ({_N[1] / 1e9:.1f} GB), last {prefix} [{N} x {K}]")
            return
        if fmt == "w8i":                     # int8 per-row (Hecate's GEMV, w8i.py)
            from . import w8i
            w = self.get_weight_for_quantization(layer)
            N, K = w.shape
            if w8i.supported(N, K):
                iw = w8i.I8W(w)
                layer._k31_i8 = (iw, N)
                layer.input_scale = None
                replace_parameter(layer, "weight", torch.empty((0, K), dtype=torch.bfloat16, device=w.device))
                self.release_requantization_source_weights(layer)
                layer._already_called_process_weights_after_loading = True
                _N[0] += 1
                _N[1] += iw.nbytes
                if _N[0] % 50 == 1:
                    log(f"int8 GEMV dense: {_N[0]} layers ({_N[1] / 1e9:.1f} GB), last {prefix} [{N} x {K}]")
                return
            fmt = "w8a16"                    # a shape the kernel does not take: the split-K W8A16 instead
        if fmt == "w8a16":                   # FP8 weights, per-row 128-group scales, BF16 inputs, split-K (w8a16.py)
            from . import w8a16
            w = self.get_weight_for_quantization(layer)
            N, K = w.shape
            if K % w8a16.GK or not w8a16.preferred(N, K):        # keep FP8 where DeepGEMM measured faster
                return orig_pw(self, layer)
            q, sc = w8a16.quantize(w)
            nb = q.numel() + sc.numel() * 2
            # K31_W8_MARLIN: Marlin's layout where it measured faster (not the routed output's up-projection while
            # K31_FUSE_UP fuses it with its norm and residual add in the split-K kernel)
            if not ((os.environ.get("K31_FUSE_UP", "0") == "1" or os.environ.get("K31_TAIL1", "0") == "1")
                    and "routed_expert_up_proj" in prefix):
                q, sc = w8a16.maybe_marlin(q, sc)
            layer._k31_w8 = (q, sc, N)
            layer.input_scale = None
            replace_parameter(layer, "weight", torch.empty((0, K), dtype=torch.bfloat16, device=w.device))
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True
            _N[0] += 1
            _N[1] += nb
            if isinstance(q, w8a16.MarlinW):
                _N[2] += 1
            if _N[0] % 50 == 1:
                log(f"W8A16 dense: {_N[0]} layers ({_N[1] / 1e9:.1f} GB, {_N[2]} on Marlin), last {prefix} [{N} x {K}]")
            return
        if fmt == "marlin4":                 # MXFP4 (mxfp4dg's codes and UE8M0 scales) on vLLM's Marlin GEMM, BF16
            from . import mxfp4dg             # inputs as they come (W4A16: no activation quantization)
            w = self.get_weight_for_quantization(layer)
            N, K = w.shape
            if K % 128:
                return orig_pw(self, layer)
            mw = mxfp4dg.MarlinW4(w)
            layer._k31_m4 = (mw, N)
            layer.input_scale = None
            replace_parameter(layer, "weight", torch.empty((0, K), dtype=torch.bfloat16, device=w.device))
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True
            _N[0] += 1
            _N[1] += mw.nbytes
            log(f"MXFP4 (Marlin W4A16) dense: {prefix} [{N} x {K}]") if _N[0] % 50 == 1 else None
            return
        from humming.layer import HummingLayer
        w = self.get_weight_for_quantization(layer)
        N, K = w.shape
        hum = HummingLayer(shape_n=N, shape_k=K, weight_config=dict(FORMATS[fmt]),
                           torch_dtype=torch.bfloat16, pad_n_to_multiple=256, pad_k_to_multiple=128).to(w.device)
        hum.load_from_unquantized(w.to(torch.bfloat16))
        hum.transform()
        layer._k31_fp6 = (hum, N)
        layer.input_scale = None
        replace_parameter(layer, "weight", torch.empty((0, K), dtype=torch.bfloat16, device=w.device))
        self.release_requantization_source_weights(layer)
        layer._already_called_process_weights_after_loading = True
        _N[0] += 1
        _N[1] += sum(p.numel() * p.element_size() for p in hum.parameters())
        if _N[0] % 50 == 1:
            log(f"Humming dense: {_N[0]} layers ({_N[1] / 1e9:.1f} GB), last {prefix} [{N} x {K}] {fmt}")

    def apply(self, layer, x, bias=None):
        c8 = getattr(layer, "_k31_c8", None)
        if c8 is not None:
            from vllm import _custom_ops as ops
            from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
            wq, sw, N = c8
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
            xq, sx = per_token_group_quant_fp8(x2, 128, column_major_scales=True, use_ue8m0=False)
            y = ops.cutlass_scaled_mm(xq, wq.t(), sx, sw.t(), torch.bfloat16)
            if bias is not None:
                y = y + bias
            return y.reshape(*lead, N)
        i8 = getattr(layer, "_k31_i8", None)
        if i8 is not None:
            iw, N = i8
            lead = x.shape[:-1]
            y = iw(x.reshape(-1, x.shape[-1]))
            if bias is not None:
                y = y + bias
            return y.reshape(*lead, N)
        m4 = getattr(layer, "_k31_m4", None)
        if m4 is not None:
            mw, N = m4
            lead = x.shape[:-1]
            y = mw(x.reshape(-1, x.shape[-1]).to(torch.bfloat16))
            if bias is not None:
                y = y + bias
            return y.reshape(*lead, N)
        d4 = getattr(layer, "_k31_dg4", None)
        if d4 is not None:
            from . import mxfp4dg
            q, sft, N = d4
            lead = x.shape[:-1]
            y = mxfp4dg.linear(x.reshape(-1, x.shape[-1]).to(torch.bfloat16), q, sft, N)
            if bias is not None:
                y = y + bias
            return y.reshape(*lead, N)
        w8 = getattr(layer, "_k31_w8", None)
        if w8 is not None:
            from . import w8a16
            q, sc, N = w8
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) < x2.shape[1]):
                x2 = x2.contiguous()             # row-strided inputs go as they are (KDA's f_a slice: no copy kernel)
            y = w8a16.linear(x2, q, sc)
            if bias is not None:
                y = y + bias
            return y.reshape(*lead, N)
        f = getattr(layer, "_k31_fp6", None)
        if f is None:
            return orig_apply(self, layer, x, bias)
        hum, N = f
        lead = x.shape[:-1]
        y = hum(x.reshape(-1, x.shape[-1]).to(torch.bfloat16))[:, :N]
        if bias is not None:
            y = y + bias
        return y.reshape(*lead, N)

    # The checkpoint's shards pass through the GPU while the dense weights are converted, and what each conversion
    # frees stays in torch's cache. With PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True that cache grows until a
    # mapping fails with the device full (the allocator then releases it and retries: a warning with ~17 MB free,
    # twice per load). K31_LOAD_TRIM=n (default 32; 0 = off) returns the cache to the device every n converted
    # layers, so a load never runs the device full.
    trim = int(os.environ.get("K31_LOAD_TRIM", "32") or 0)
    calls = [0]

    def process_weights_trim(self, layer):
        out = process_weights_after_loading(self, layer)
        calls[0] += 1
        if trim > 0 and calls[0] % trim == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    cls.process_weights_after_loading = process_weights_trim
    cls.apply = apply
    cls._k31_fp6 = True
    log("Humming dense (W*A16): " + (os.environ.get("K31_DENSE_FMT") or
                                     "fp6 for linears matching K31_FP6_LAYERS=" + os.environ.get("K31_FP6_LAYERS", DEFAULT)))
