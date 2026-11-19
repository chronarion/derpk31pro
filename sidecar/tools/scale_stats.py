"""Scale-code statistics of hi2 records (what a compact scale code in the band must cover). Standard library only.
    python tools/scale_stats.py FILE.hi2 [n_records] [stride]"""
import os
import sys

REC, CODES = 9289728, 8257536
fd = os.open(sys.argv[1], os.O_RDONLY)
total = os.fstat(fd).st_size // REC
n = int(sys.argv[2]) if len(sys.argv) > 2 else total
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
hist = [0] * 256
lo_min, lo_max, hi_min, hi_max, span_max, zeros_max, count = 255, 0, 255, 0, 0, 0, 0
for e in range(0, min(n * stride, total), stride):
    sf = os.pread(fd, REC - CODES, e * REC + CODES)
    zeros = sf.count(0)
    present = [v for v in range(1, 256) if sf.count(v)]
    for v in present:
        hist[v] += sf.count(v)
    hist[0] += zeros
    lo, hi = present[0], present[-1]
    lo_min, lo_max, hi_min, hi_max = min(lo_min, lo), max(lo_max, lo), min(hi_min, hi), max(hi_max, hi)
    span_max, zeros_max, count = max(span_max, hi - lo + 1), max(zeros_max, zeros), count + 1
tot = sum(hist)
top = sorted(((c, v) for v, c in enumerate(hist) if c), reverse=True)[:3]
print(f"{os.path.basename(sys.argv[1])}: {count} records; stored codes (native - 1) {min(v for v, c in enumerate(hist) if c and v)}"
      f"..{max(v for v, c in enumerate(hist) if c)}; per record min {lo_min}..{lo_max}, max {hi_min}..{hi_max}, widest span "
      f"{span_max} values; zero codes per record max {zeros_max}; top " + ", ".join(f"{v}: {100 * c / tot:.1f}%" for c, v in top))
