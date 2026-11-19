"""Summary of a tests/test_band.py log by record kind and flag set."""
import collections, re, sys
agg = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0, True, 0.0, 0.0])
for l in open(sys.argv[1]):
    if not re.match(r"^(synthetic|real L01)", l):
        continue
    name = l[:10].strip()
    f = l.split("|")
    flags = int(l[10:].split()[2])
    rel, worst = (float(v) for v in f[1].split())
    d, tot, sfd = int(f[2].split()[0]), int(f[2].split()[2]), int(f[2].split()[3])
    k = (name, flags)
    a = agg[k]
    a[0] += 1; a[1] += d; a[2] += tot; a[3] = max(a[3], rel); a[4] = max(a[4], worst); a[5] += sfd
    a[6] &= "True" in f[3]
    if len(f) > 4:
        g, b = (float(v) for v in f[4].split()[:2])
        a[7] = max(a[7], abs(g)); a[8] = max(a[8], abs(b))
names = {16: "MegaMoE preset (fast SiTU)", 24: "MegaMoE preset, precise SiTU", 29: "runner preset", 21: "runner preset, fast SiTU",
         0: "no output rounding, fast SiTU", 8: "no output rounding, precise SiTU", 10: "pure fp32 (no bf16 anywhere), precise",
         13: "weight after, act bf16, precise"}
print(f"{'records':10s} {'flags':>5s} {'arithmetic':36s} {'cases':>5s} {'FP8 codes differing':>26s} {'scales':>6s} {'max rel err':>11s} {'worst elem':>10s}  same bits  {'max |gain|':>10s} {'|offset|':>9s}")
for (name, flags), a in sorted(agg.items()):
    print(f"{name:10s} {flags:5d} {names.get(flags, ''):36s} {a[0]:5d} {a[1]:8d} of {a[2]:8d} ({a[1] / max(a[2], 1):.1e}) {a[5]:6d} {a[3]:11.2e} {a[4]:10.2e}  {str(a[6]):9s}  {a[7]:10.1e} {a[8]:9.1e}")
