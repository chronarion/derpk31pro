"""stager.extract_record: a native row -> hi2 and lo2 records; merging them back (the promotion kernel) must give the
row byte for byte; and the hi2 record must equal store/build_store.py's pairs() + lowered scales."""
import sys

import torch

sys.path.insert(0, "/engine")
from k31_engine.stager import extract_record, merge  # noqa: E402

dev = torch.device("cuda")
layout = {"record_bytes": 9289728, "codes_bytes": 8257536, "scale_bytes": 1032192,
          "native_bytes": {"w13": 11010048, "w13s": 688128, "w2": 5505024, "w2s": 344064}}


class M:
    pass


def mk(rows):
    m = M()
    m.w13_weight = torch.randint(-128, 127, (rows, 6144, 1792), dtype=torch.int8, device=dev)
    m.w2_weight = torch.randint(-128, 127, (rows, 3584, 1536), dtype=torch.int8, device=dev)
    m.w13_weight_scale = torch.randint(0, 2 ** 31 - 1, (rows, 28, 6144), dtype=torch.int32, device=dev).transpose(1, 2)
    m.w2_weight_scale = torch.randint(0, 2 ** 31 - 1, (rows, 24, 3584), dtype=torch.int32, device=dev).transpose(1, 2)
    return m


src = mk(4)
dst = mk(4)
for t in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
    getattr(dst, t).zero_()
hi = torch.empty(layout["record_bytes"], dtype=torch.uint8, device=dev)
lo = torch.empty_like(hi)
ok = True
for row in (0, 2, 3):
    extract_record(src, row, layout, hi)
    extract_record(src, row, layout, lo, low=True)
    merge(hi.data_ptr(), lo.data_ptr(), dst, row, layout, blocks=64)
    torch.cuda.synchronize()
    for t in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        a, b = getattr(src, t)[row], getattr(dst, t)[row]
        same = torch.equal(a.contiguous(), b.contiguous())
        ok &= same
        if not same:
            print(f"row {row} {t}: {int((a != b).sum())} elements differ")
    # against build_store's formula
    from torch import cat
    def flat(t, r):
        per = t[0].numel()
        return torch.as_strided(t, (t.shape[0], per), (per, 1))[r].view(torch.uint8)
    codes = cat((flat(src.w13_weight, row), flat(src.w2_weight, row)))
    l, h = (codes & 0x0F) >> 2, codes >> 6
    b = (l | (h << 2)).view(-1, 2)
    want = cat((b[:, 0] | (b[:, 1] << 4), torch.where((sf := cat((flat(src.w13_weight_scale, row),
                                                                     flat(src.w2_weight_scale, row)))) > 0, sf - 1, sf)))
    ok &= torch.equal(hi, want)
print("extract_record (hi2 + lo2 -> merge == row; hi2 == build_store):", "PASS" if ok else "FAIL")

# the fused kernel (extract_hi2) == the torch path, byte for byte
from k31_engine.stager import extract_hi2  # noqa: E402
fast = torch.empty_like(hi)
ok2 = True
for row in (0, 1, 3):
    extract_record(src, row, layout, hi)
    fast.fill_(0)
    extract_hi2(src, row, layout, fast)
    torch.cuda.synchronize()
    ok2 &= torch.equal(hi, fast)
print("extract_hi2 (fused) == extract_record:", "PASS" if ok2 else "FAIL")
