"""k31_engine.policy (K31_POLICY): the shipped tables compile to the per-layer parameters their notes describe;
lookup tables compile like named policies; what the v1 kernels cannot run is refused with the reason; the env section
reads back. CPU only."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("ENGINE", "/engine"))
from k31_engine import policy as P  # noqa: E402

PROF = os.path.join(os.environ.get("ENGINE", "/engine"), "profiles")
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def write(obj):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


# 1-3. the shipped tables compile; the default's stated structure; the env section reads back
for name in ("default", "balanced", "fast", "quality"):
    try:
        P.load(os.path.join(PROF, f"policy-{name}.json"))
        check(f"policy-{name} compiles", True)
    except P.PolicyError as e:
        check(f"policy-{name} compiles", False, str(e)[:80])
p = P.load(os.path.join(PROF, "policy-default.json"))
s0 = sorted(L for L in P.LAYERS if p[L].strong == 0)
ok = len(s0) == 16 and all(p[L].strong == 4 for L in list(range(52, 60)) + [92]) and \
    all(p[L].strong == 2 for L in (74, 76, 77, 78, 79, 80, 83)) and all(p[L].band and p[L].band_cap == 8 and p[L].mean
                                                                         for L in P.LAYERS)
check("policy-default: no staging in 16 layers, ranks 0-3 in 52-59 and 92, ranks 0-1 in 74, 76-80, 83, band cap 8",
      ok, f"(no staging in {s0})")
env = P.env_section(os.path.join(PROF, "policy-default.json"))
check("policy-default env section", env == {"K31_BAND_FEED": "0.25", "BAND_READAHEAD": "8"}, str(env))
# 4. a lookup table compiles like the named policy it spells out; overrides apply on top of the assignment
row = ["hit", "band", "stage", "mean"]
rows = [row] * 2 + [["hit", "band", "mean"]] * 14            # stage ranks 0-1
f = write({"version": 1, "policies": {"s": {"stage": "0-2", "mean": "all"}}, "default": "s",
           "overrides": {"5": {"stage": "0-3"}}, "table": {"7-8": rows}})
p = P.load(f)
ok = p[7].strong == 2 and p[8].strong == 2 and p[7].mean and p[5].strong == 4 and p[6].strong == 3 and \
    p[7].policy == "table" and p[5].policy == "s+override"
check("lookup table and overrides", ok, f"(L5 {p[5].strong}, L6 {p[6].strong}, L7 {p[7].strong} [{p[7].policy}])")
# 5. what the v1 kernels cannot run is refused, each with its reason
bad = {
    "hit not a prefix": {"hit": "0-3,8"},
    "stage beyond rank 3": {"stage": "0-4"},
    "stage not a prefix": {"stage": "1-2"},
    "band on some ranks": {"band": "0-7"},
    "mean on some ranks": {"mean": "8-15"},
    "defer not contiguous": {"defer": "3-5,8"},
    "unknown field": {"stagee": "0"},
    "rank out of range": {"stage": "0-16"},
}
for name, fields in bad.items():
    try:
        P.load(write({"version": 1, "policies": {"x": fields}, "default": "x"}))
        check(f"refused: {name}", False, "(accepted)")
    except P.PolicyError as e:
        check(f"refused: {name}", True, f"({str(e)[:70]})")
# 6. deferral compiles to (from, to) after the staged ranks; skip-only and no-band layers
p = P.load(write({"version": 1, "policies": {"d": {"stage": "0-2", "defer": "2-11", "mean": "all"},
                                             "off": {"band": "none", "hit": "0-11"}},
                  "default": "d", "layers": {"90-92": "off"}}))
ok = p[10].defer == (3, 12) and p[91].band is False and p[91].resmax == 12 and p[91].strong == 0 and \
    p[91].mean is False and p[91].defer is None
check("deferral range, band off, resident cap", ok, f"(L10 defer {p[10].defer}, L91 resmax {p[91].resmax})")
print(P.describe(P.load(os.path.join(PROF, "policy-default.json"))))
print("POLICY:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
