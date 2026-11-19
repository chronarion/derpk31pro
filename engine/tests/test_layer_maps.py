"""K31_HBM_MAP + K31_BAND_MAP + K31_BANK_MAP in hook._rowmap (CPU only): each layer's resident rows, RTX band and
Grace bank are the first h_L, the next n_L and the next g_L experts of its rank list, disjoint; without the band and
bank maps every layer takes K31_BAND_N and K31_BANK. Runs itself once per case in a subprocess (hook reads its
switches at import). Env: RANKF (default /rank/rank-native.json)."""
import json
import os
import subprocess
import sys

RANKF = os.environ.get("RANKF", "/rank/rank-native.json")
P = "/engine/profiles"
CASES = {
    "maps": dict(K31_HBM_MAP=f"{P}/hbm-map.json", K31_BAND_MAP=f"{P}/band-map.json",
                 K31_BANK_MAP=f"{P}/bank-map.json"),
    "rows only": dict(K31_HBM_MAP=f"{P}/hbm-map.json"),
}

if len(sys.argv) > 1:                                      # one case: check _rowmap against the maps
    sys.path.insert(0, "/engine")
    from k31_engine import hook
    rank = json.load(open(RANKF))["layers"]
    rm = hook._rowmap()

    def cnt(path, L, d):
        return json.load(open(path))["layers"].get(str(L), d) if path else d
    bad = []
    for L, v in rank.items():
        L = int(L)
        h = cnt(os.environ.get("K31_HBM_MAP"), L, hook.HBM)
        nb = cnt(os.environ.get("K31_BAND_MAP"), L, hook.BAND_N)
        g = cnt(os.environ.get("K31_BANK_MAP"), L, hook.BANK)
        r = rm[L]
        ok = (r["hbm"] == sorted(v[:h]) and r["band"] == [int(e) for e in v[h:h + nb]]
              and r["bank"] == sorted(v[h + nb:h + nb + g])
              and len(set(r["hbm"]) | set(r["band"]) | set(r["bank"])) == h + nb + g)
        if not ok:
            bad.append(L)
    tot = [sum(len(r[k]) for r in rm.values()) for k in ("hbm", "band", "bank")]
    print(f"{sys.argv[1]}: {len(rm)} layers, rows {tot[0]}, band {tot[1]}, bank {tot[2]}; layers off: {bad[:8]}")
    sys.exit(1 if bad else 0)

fails = []
for name, env in CASES.items():
    e = dict(os.environ, K31_MODE="staged", K31_RANK=RANKF, K31_HBM="105", K31_BAND="/dev/shm/none", K31_BAND_N="118",
             K31_BANK="400", K31_BANK_EXCL="1", K31_PROMOTE="0.1", **env)
    rc = subprocess.run([sys.executable, __file__, name], env=e).returncode
    print(f"{'PASS' if rc == 0 else 'FAIL'} {name}", flush=True)
    if rc:
        fails.append(name)
print("LAYER MAPS:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
