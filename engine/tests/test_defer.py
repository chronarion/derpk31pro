"""defer.plan and defer.diff against plain-torch references (the hook's v1 code)."""
import sys
import torch
sys.path.insert(0, "/engine")
from k31_engine import defer

E, K = 896, 16
dev = torch.device("cuda")
torch.manual_seed(0)


def ref_plan(ids, w, row_map, bank_map, dfrom, dto, D, base):
    T = ids.shape[0]
    valid = ids >= 0
    e = ids.long().clamp_min(0)
    native, bank = row_map[e], bank_map[e]
    wf = w.float()
    order = wf.argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(K, device=dev).expand(T, K))
    cand = valid & (native < 0) & (bank >= 0) & (rank >= dfrom) & (rank < dto)
    energy = torch.zeros(E, dtype=torch.float32, device=dev)
    energy.index_add_(0, e.reshape(-1), torch.where(cand, wf * wf, 0.0).reshape(-1))
    # deterministic order: energy desc, then expert id asc
    key = torch.where(energy > 0, energy, torch.tensor(-1.0, device=dev))
    order_e = sorted(range(E), key=lambda x: (-float(key[x]), x))
    top = torch.tensor(order_e[:16], device=dev)
    v = energy[top]
    ok = (v > 0) & (torch.arange(16, device=dev) < D)
    slot_of = torch.full((E,), -1, dtype=torch.long, device=dev)
    slot_of[top[ok]] = torch.arange(16, device=dev)[ok]
    ds = slot_of[e]
    deferred = cand & (ds >= 0)
    stage = torch.where(ok, bank_map[top], -1).to(torch.int32)
    pids = torch.where(deferred, base + ds, -1).to(torch.int32)
    pw = torch.where(deferred, w, torch.zeros_like(w))
    return stage, pids, pw, int(deferred.sum())


bad = 0
for case in range(200):
    T = [1, 2, 3, 4, 5, 6, 7, 8][case % 8]
    row_map = torch.where(torch.rand(E, device=dev) < 0.15, torch.arange(E, device=dev, dtype=torch.int32), -1).to(torch.int32)
    bank_map = torch.where(torch.rand(E, device=dev) < 0.6, torch.randint(0, 400, (E,), device=dev, dtype=torch.int32), -1).to(torch.int32)
    pool = torch.randperm(E, device=dev)[:40]                      # tokens share experts
    ids = pool[torch.randint(0, 40, (T, K), device=dev)].to(torch.int32)
    for t in range(T):                                              # distinct picks per token
        ids[t] = pool[torch.randperm(40, device=dev)[:K]].to(torch.int32)
    if case % 5 == 0:
        ids[0, 3] = -1
    w = torch.rand(T, K, device=dev).sort(-1, descending=True).values.to(torch.bfloat16)
    dfrom, dto, D = [(3, 12, 8), (0, 16, 16), (4, 8, 3)][case % 3]
    cnt = torch.zeros((), dtype=torch.int64, device=dev)
    s1, p1, w1 = defer.plan(ids, w, row_map, bank_map, dfrom, dto, D, 112, cnt)
    s2, p2, w2, n2 = ref_plan(ids, w, row_map, bank_map, dfrom, dto, D, 112)
    if not (torch.equal(s1, s2) and torch.equal(p1, p2) and torch.equal(w1, w2) and int(cnt) == n2):
        bad += 1
        if bad <= 3:
            print("MISMATCH case", case, T, dfrom, dto, D, s1.tolist(), s2.tolist(), int(cnt), n2)
print(f"plan: {200 - bad}/200 cases equal")
ok_d = True
for T in (1, 3, 8):
    u = torch.randn(T, 3584, device=dev).to(torch.bfloat16)
    lt = (0.3 * torch.randn(T, 3584, device=dev)).to(torch.bfloat16)
    lt[0] = 0
    g = (1 + 0.1 * torch.randn(3584, device=dev)).to(torch.bfloat16)
    eps = 1e-6
    def rmsn(v):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * g.float()
    ref = (rmsn(u.float() + lt.float()) - rmsn(u.float())).to(torch.bfloat16)
    got = defer.diff(u, lt, g, eps)
    err = (got.float() - ref.float()).abs().max().item()
    zero_row = got[0].abs().max().item()
    ok_d &= err < 1e-2 and zero_row == 0.0
    print(f"diff T={T}: max |err| {err:.2e}, row without a late share -> {zero_row}")
print("PASS" if bad == 0 and ok_d else "FAIL")
