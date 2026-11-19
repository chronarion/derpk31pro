"""_head_w8: an FP8 W8A16 copy of an LM head used by the head's quant_method.apply for <= 16 rows: only inside the
drafter's compute_draft_logits (K31_HEAD_W8=0), or for every call (K31_HEAD_W8=1); longer calls and other heads keep
the original apply."""
import os
import sys

sys.path.insert(0, "/engine")
import torch  # noqa: E402

from k31_engine import hook, w8a16  # noqa: E402

dev = "cuda"
torch.manual_seed(0)


class Method:                                    # UnquantizedEmbeddingMethod stand-in
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class Head(torch.nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(V, H, device=dev, dtype=torch.bfloat16) * 0.02,
                                         requires_grad=False)
        self.quant_method = Method()


class Drafter:
    def __init__(self, head):
        self.lm_head = head

    def compute_draft_logits(self, h):
        return self.lm_head.quant_method.apply(self.lm_head, h)


V, H = 32768, 7168
head, other = Head(V, H), Head(4096, H)
slow = Head(8192, H)
# the benchmark table decides: W8A16 measured faster than BF16 for the head's shape, slower for `slow`'s
w8a16._TABLE["t"] = {"w8a16_best_us": {f"{V}x{H}": {"3": 50.0}, f"8192x{H}": {"3": 30.0}},
                     "bf16_us": {f"{V}x{H}": {"3": 80.0}, f"8192x{H}": {"3": 20.0}}}
assert not hook._head_w8(slow) and getattr(slow, "_k31_w8h", None) is None, "slower shape must stay BF16"
x = torch.randn(3, H, device=dev, dtype=torch.bfloat16)
ref = torch.nn.functional.linear(x, head.weight)
q, sc = w8a16.quantize(head.weight.data)
w8ref = w8a16.linear(x, q, sc)
res = {}
for mode in ("drafter-only", "every-call"):
    hook.HEAD_W8 = mode == "every-call"
    if mode == "drafter-only":
        assert hook._head_w8(head)
        d = Drafter(head)
        orig = d.compute_draft_logits

        def cdl(h, _orig=orig):
            hook._DRAFT_HEAD[0] = True
            try:
                return _orig(h)
            finally:
                hook._DRAFT_HEAD[0] = False
        d.compute_draft_logits = cdl
    tgt = head.quant_method.apply(head, x)
    drf = d.compute_draft_logits(x)
    long = head.quant_method.apply(head, torch.randn(32, H, device=dev, dtype=torch.bfloat16))
    oth = other.quant_method.apply(other, x)
    res[mode] = (torch.equal(drf, w8ref), torch.equal(tgt, w8ref) if hook.HEAD_W8 else torch.equal(tgt, ref),
                 long.shape == (32, V), torch.equal(oth, torch.nn.functional.linear(x, other.weight)))
rel = float((w8ref.float() - ref.float()).norm() / ref.float().norm())
print("drafter-only:", res["drafter-only"], "every-call:", res["every-call"], f"FP8 head rel err {rel:.2e}")
ok = all(all(v) for v in res.values()) and rel < 4e-2      # FP8 e4m3 per-row 128-group: ~2.6% on N(0, 1) weights
print("HEAD W8:", "PASS" if ok else "FAIL")
