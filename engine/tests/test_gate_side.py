"""K31_GATE_SIDE mechanics on a stand-in DSA attention: the output gate forked on the side stream, joined in the
indexer's score, consumed by the wrapper's g_proj call after the attention; eager and under CUDA graph capture, the
result equals the unpatched forward bit for bit."""
import os
import sys
import types

os.environ["K31_GATE_SIDE"] = "1"
sys.path.insert(0, "/engine")
import torch  # noqa: E402

from k31_engine import hook  # noqa: E402

dev = "cuda"
torch.manual_seed(0)


class GP(torch.nn.Module):                      # ColumnParallelLinear stand-in: returns (out, bias)
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(256, 512, bias=False, device=dev, dtype=torch.bfloat16)

    def forward(self, x):
        return self.lin(x), None


class Indexer:
    def score(self, x):
        return x.float().sum(-1, keepdim=True)


class Attn(torch.nn.Module):                    # the unpacked wrapper path: front, score, attention, gate, o_proj
    def __init__(self):
        super().__init__()
        self._packed = None
        self.g_proj = GP()
        self.front = torch.nn.Linear(256, 512, bias=False, device=dev, dtype=torch.bfloat16)
        self.o = torch.nn.Linear(512, 256, bias=False, device=dev, dtype=torch.bfloat16)
        self.idx = Indexer()

    def forward(self, positions, hidden_states):
        h = self.front(hidden_states)
        for _ in range(20):                      # a front long enough for the side stream to matter
            h = torch.tanh(h)
        s = self.idx.score(h)
        attn = h * torch.sigmoid(s).to(h.dtype)
        attn = attn * self.g_proj(hidden_states)[0].sigmoid()
        return self.o(attn)


mod = types.SimpleNamespace(KimiK3DSAAttention=Attn, KimiK3DSAIndexer=Indexer)
a = Attn()
x = torch.randn(3, 256, device=dev, dtype=torch.bfloat16)
ref = a(None, x).clone()
orig_fwd = Attn.forward
hook._gate_side(mod)
assert Attn.forward is not orig_fwd
y = a(None, x)
ok_eager = torch.equal(y, ref)
xs = x.clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = a(None, xs)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = a(None, xs)
xs.copy_(x)
g.replay()
torch.cuda.synchronize()
ok_graph = torch.equal(out, ref)
x2 = torch.randn(3, 256, device=dev, dtype=torch.bfloat16)
ref2 = Attn.__mro__[0].__dict__["forward"]  # patched; compute the unpatched result by hand
a2 = a
h = a2.front(x2)
for _ in range(20):
    h = torch.tanh(h)
r2 = a2.o(h * torch.sigmoid(h.float().sum(-1, keepdim=True)).to(h.dtype) * a2.g_proj.lin(x2).sigmoid())
xs.copy_(x2)
g.replay()
torch.cuda.synchronize()
ok_replay2 = torch.equal(out, r2)
big = torch.randn(32, 256, device=dev, dtype=torch.bfloat16)   # > 16 rows: the unpatched path
ok_big = a(None, big).shape == (32, 256) and hook._GATE["ev"] is None and a.g_proj._k31_pre is None
print(f"eager equal {ok_eager}, graph equal {ok_graph}, replay with new input equal {ok_replay2}, wide path {ok_big}")
print("GATE SIDE:", "PASS" if ok_eager and ok_graph and ok_replay2 and ok_big else "FAIL")
