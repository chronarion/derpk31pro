"""Prompt-reading speed of several prompt-path configurations on one running server, switched at runtime.

Each configuration is a set of K31_* runtime flags (engine/k31_engine/flags.py) written to the control file the
server re-reads between steps; for each, cold prompts (token ids of real text, a fresh window each time, so the
prefix cache cannot hit) are timed with max_tokens 1. Configurations are visited round-robin `--rounds` times so the
residency's slow drift falls on all of them alike.

usage: python3 -I prefill_bisect.py URL FLAGS_FILE CORPUS.pt CONFIGS.json [--len 16384] [--rounds 2] [--out OUT.json]
  FLAGS_FILE  the control file as the host sees it (the server reads the same file at K31_FLAGS_FILE)
  CORPUS.pt   a torch.save'd dict with "windows" (or "tokens") [N, W] token ids (read without torch: pure-Python zip reader)
  CONFIGS     {"name": {"K31_...": value, ...}, ...}; the first is the baseline
"""
import argparse
import json
import os
import pickle
import struct
import sys
import time
import urllib.request
import zipfile


def _windows(path):
    """Token ids [N][W] from a torch.save'd dict's "windows" tensor (int32 / int64), no torch needed."""
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]
    meta = {}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return lambda storage, off, shape, stride, *a: ("T", storage, off, shape)
            if mod == "torch" and name.endswith("Storage"):
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            return (pid[1], pid[2])                         # (storage kind, key)

    d = U(z.open(f"{prefix}/data.pkl")).load()
    _, (kind, key), off, shape = d["windows"] if "windows" in d else d["tokens"]
    size = {"IntStorage": 4, "LongStorage": 8}[kind]
    fmt = {"IntStorage": "i", "LongStorage": "q"}[kind]
    raw = z.read(f"{prefix}/data/{key}")
    N, W = shape
    return [list(struct.unpack_from(f"<{W}{fmt}", raw, (off + n * W) * size)) for n in range(N)]


def cold_prompt(url, toks):
    body = json.dumps({"model": "derpk31pro", "prompt": toks, "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out.get("usage", {}).get("prompt_tokens", len(toks)), time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("flags_file"); ap.add_argument("corpus"); ap.add_argument("configs")
    ap.add_argument("--len", type=int, default=16384); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--first-window", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    configs = json.load(open(a.configs))
    wins = _windows(a.corpus)
    w = a.first_window
    res = {name: [] for name in configs}
    for r in range(a.rounds):
        for name, fl in configs.items():
            tmp = a.flags_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(fl, f)
            os.replace(tmp, a.flags_file)               # the server picks it up at its next step
            cold_prompt(a.url.rstrip("/"), wins[w % len(wins)][:64])   # one short step applies the flags
            w += 1
            toks = []
            while len(toks) < a.len:
                toks += wins[w % len(wins)]
                w += 1
            try:
                n, dt = cold_prompt(a.url.rstrip("/"), toks[:a.len])
            except Exception as ex:                     # noqa: BLE001  (a configuration that fails: record, go on)
                print(f"round {r} {name}: FAILED ({ex})", flush=True)
                res[name].append(float("nan"))
                continue
            res[name].append(n / dt)
            print(f"round {r} {name}: {n} tokens in {dt:.2f} s = {n / dt:.0f} tok/s", flush=True)
    if os.path.exists(a.flags_file):
        os.remove(a.flags_file)
    base = list(configs)[0]
    ok = lambda v: [x for x in v if x == x]             # drop failed rounds (NaN)
    b = sum(ok(res[base])) / max(1, len(ok(res[base])))
    for name, v in res.items():
        m = sum(ok(v)) / len(ok(v)) if ok(v) else float("nan")
        print(f"{name:24s} {m:7.0f} tok/s  ({m / b:.3f} x {base}){'' if len(ok(v)) == len(v) else '  (failed rounds)'}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
