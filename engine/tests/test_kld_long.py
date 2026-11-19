"""The KLD harness on long sequences (harness/kld.py --long, kld_long.py) and the 2,048-token protocol's numbers, CPU
only: a fake server answers /v1/completions as vLLM with the k31_engine.force processor does (prompt log-probs with
the prompt token first, forced decoding with generated top log-probs, the exact per-row KL file at k31_kl_out, rows
limited by k31_kl_rows), from deterministic per-row distributions over a 512-token vocabulary.
  A. 2,048-token protocol: main's kld.py (OLD_KLD) and this one, same arguments, same server: OUT.json byte-equal and
     OUT.tokens.pt equal tensor for tensor (prefill; decode with exact KL; KLD_FAST's --decode-logprobs 1 --pipeline
     --prefetch; --subset rot128; --windows; --dump-ref);
  B. the long mode's definitions on the same 2,048-token data (--long --prompt-rows estimate --block 4): kld.py's
     keys within float rounding of main's (rel < 1e-6; bootstrap CIs from the same resamples);
  C. L32-shaped (2 x 32,768 tokens, per-sequence prompt from the corpus `tail`, corpus labels, two reference halves):
     the requests (prompt lengths, prompt log-probs 1, k31_kl_rows, k31_kl_index / k31_kl_next), and every reported
     group (headline = generated forced rows, generated prompt rows, by distance, by kind, top-1, NLL) against an
     independent computation from the server's per-row values; --kl-rows; --cuts; a missing exact row and a roles
     file that does not match the corpus are refused;
  D. with L32_CORPUS (the real corpus/k31-l32.pt mounted): all 32 x 32,768 tokens through the harness (token ids
     folded into the fake vocabulary): prompt lengths = 32,768 - tail, generated forced / prompt rows per kind = the
     corpus labels' counts, every row scored, the harness's time and peak memory (its container has 32 GB).

  git show c47cf79:engine/harness/kld.py > /tmp/kld_main.py
  docker run --rm --network none --cpuset-cpus <cores> --user $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=1 \
    -v $PWD/engine:/engine:ro -v /tmp/kld_main.py:/old/kld.py:ro -e OLD_KLD=/old/kld.py \
    -v $STATE/kld/corpus:/corpus:ro -e L32_CORPUS=/corpus/k31-l32.pt \
    --entrypoint python3 derpk31pro-runner:0.2 /engine/tests/test_kld_long.py

Exit 0 = PASS. ~3 min on one core, ~8 with D (~1.2 GB of files in TEST_DIR, default a temporary directory).
"""
import hashlib
import http.server
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

ENGINE = os.environ.get("ENGINE", "/engine")
NEW = f"{ENGINE}/harness/kld.py"
OLD = os.environ.get("OLD_KLD", "/old/kld.py")
WORK = os.environ.get("TEST_DIR") or tempfile.mkdtemp(prefix="k31-kld-long-")
VS = 512                                 # the fake model's vocabulary
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def seq_key(ids):
    return int.from_bytes(hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).digest()[:8], "little")


def dists(ids, t0, t1):
    """Rows [t0, t1) of a sequence: (engine log-probs, reference log-probs), float64 [rows, VS]; deterministic in the
    sequence's tokens and the row (so the server and the test agree)."""
    eng, ref = [], []
    key = seq_key(ids)
    for b in range(t0 // 1024, (t1 - 1) // 1024 + 1):
        g = np.random.default_rng([key, b])
        base = g.standard_normal((1024, VS)) * 2.0
        e = base + 0.35 * g.standard_normal((1024, VS))
        r = base + 0.35 * g.standard_normal((1024, VS))
        lo, hi = max(t0, 1024 * b), min(t1, 1024 * (b + 1))
        eng.append(e[lo - 1024 * b:hi - 1024 * b])
        ref.append(r[lo - 1024 * b:hi - 1024 * b])
    def lsm(x):
        mx = x.max(1, keepdims=True)
        return x - mx - np.log(np.exp(x - mx).sum(1, keepdims=True))
    return lsm(np.concatenate(eng)), lsm(np.concatenate(ref))


def exact_kl(ids, t0, t1):
    e, r = dists(ids, t0, t1)
    return (np.exp(r) * (r - e)).sum(1)


SEEN = []                                # the request bodies, for checks
NAN_ROW = {"row": None}                  # a row the server leaves without an exact KL (error-path check)


class Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({k: v for k, v in body.items() if k not in ("prompt",)} | {"prompt_len": len(body["prompt"])})
        xa = body.get("vllm_xargs") or {}
        prompt = body["prompt"]
        force = xa.get("k31_force")
        ids = prompt + (force or [])
        T, P = len(ids), len(prompt)
        kp = body.get("prompt_logprobs")
        kl_ = body.get("logprobs") or 0
        e, _ = dists(ids, 0, T)
        top = np.argsort(-e, axis=1, kind="stable")
        fid = [x % VS for x in ids]           # real token ids (section D) folded into the fake vocabulary
        pl = [None]
        for i in range(1, P):                 # entry i: the distribution after token i - 1; the prompt token first
            t = i - 1
            ent = {str(ids[i]): {"logprob": float(e[t, fid[i]]), "rank": 1, "decoded_token": "x"}}
            for tok in top[t, :kp]:
                ent.setdefault(str(int(tok)), {"logprob": float(e[t, tok]), "rank": 1, "decoded_token": "x"})
            pl.append(ent)
        ch = {"prompt_logprobs": pl, "text": ""}
        if force is not None:
            toks, tl = [], []
            for t in range(P - 1, T - 1):     # forced rows: the sampled (= forced) token first
                d = {f"token_id:{ids[t + 1]}": float(e[t, fid[t + 1]])}
                for tok in top[t, :kl_]:
                    d.setdefault(f"token_id:{int(tok)}", float(e[t, tok]))
                toks.append(f"token_id:{ids[t + 1]}")
                tl.append(d)
            ch["logprobs"] = {"tokens": toks, "top_logprobs": tl}
            if xa.get("k31_kl_out"):
                out = np.full(T, np.nan, dtype=np.float32)
                A, B = xa.get("k31_kl_rows", [0, T])
                A, B = max(A, 0), min(B, T - 1)
                if B > A:
                    out[A:B] = exact_kl(ids, A, B).astype(np.float32)
                if NAN_ROW["row"] is not None:
                    out[NAN_ROW["row"]] = np.nan
                np.save(xa["k31_kl_out"].replace("/SERVER", WORK), out)
        data = json.dumps({"choices": [ch]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}"
os.makedirs(f"{WORK}/out", exist_ok=True)


def harness(script, args, out):
    cmd = [sys.executable, script, URL, *args[:2], out, *args[2:], "--kl-dir", f"{WORK}/out",
           "--kl-server-dir", "/SERVER/out"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return p


def write_ref(path, ids_rows, K=64):
    """A runner kld.pt as the harness reads it: universe 0's top-64 log-probabilities per position."""
    N, T = len(ids_rows), len(ids_rows[0])
    ref_lp = torch.zeros((N, T, K), dtype=torch.float32)
    ref_id = torch.zeros((N, T, K), dtype=torch.int32)
    for n, ids in enumerate(ids_rows):
        _, r = dists(ids, 0, T)
        o = np.argsort(-r, axis=1, kind="stable")[:, :K]
        ref_id[n] = torch.from_numpy(o.astype(np.int32))
        ref_lp[n] = torch.from_numpy(np.take_along_axis(r, o, 1).astype(np.float32))
    torch.save({"ref_lp": ref_lp, "ref_id": ref_id}, path)


def same_pt(a, b):
    A, B = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    if set(A) != set(B):
        return False, f"keys {sorted(set(A) ^ set(B))}"
    for k in A:
        x, y = A[k], B[k]
        if isinstance(x, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(torch.nan_to_num(x, nan=7.25),
                                                                            torch.nan_to_num(y, nan=7.25)):
                return False, f"tensor {k}"
        elif x != y:
            return False, f"value {k}"
    return True, ""


t00 = time.time()
# ------------------------------------------------------------------------- A. the 2,048-token protocol, byte-equal
g = torch.Generator().manual_seed(3)
N2, T2 = 16, 2048
toks = torch.randint(0, VS, (N2, T2), generator=g, dtype=torch.int32)
kinds = ["agentic", "code", "swe", "math"] * 4
corpus = f"{WORK}/cuts.pt"
torch.save({"tokens": toks, "kinds": kinds, "sources": ["x"] * N2}, corpus)
lab = torch.randint(0, 7, (N2 // 4, 4 * T2), generator=g, dtype=torch.int8)      # windows of 4 cuts, as bm-eval's
lab[lab > 4] = 0                                                                     # more generated rows
roles = f"{WORK}/roles.pt"
torch.save({"labels": lab, "names": ["assistant", "text", "system", "user", "tool", "hdr", "pre"]}, roles)
ref2 = f"{WORK}/ref2.pt"
write_ref(ref2, toks.tolist())
H2 = f"{WORK}/h2.npy"                            # a reference header to name (the fake server does not read it)
np.lib.format.open_memmap(H2, mode="w+", dtype=np.uint16, shape=(N2, T2, 8)).flush()
cases = {
    "prefill": [corpus, ref2, "--roles", roles],
    "decode": [corpus, ref2, "--mode", "decode", "--prompt", "64", "--kl-hidden", H2, "--roles", roles],
    "fast": [corpus, ref2, "--mode", "decode", "--prompt", "64", "--kl-hidden", H2, "--roles", roles,
             "--decode-logprobs", "1", "--pipeline", "--prefetch"],
    "rot128": [corpus, ref2, "--subset", "rot128", "--mode", "decode", "--prompt", "300", "--kl-hidden", H2,
               "--roles", roles],
    "windows": [corpus, ref2, "--windows", "8", "--roles", roles],
}
if not os.path.exists(OLD):
    check("A. main's kld.py (OLD_KLD) present", False, f"({OLD}: see the docstring)")
else:
    for name, args in cases.items():
        res = []
        for tag, script in (("old", OLD), ("new", NEW)):
            out = f"{WORK}/{name}-{tag}.json"
            p = harness(script, args, out)
            res.append((p, out))
        (po, oo), (pn, on) = res
        if po.returncode or pn.returncode:
            check(f"A. {name}: both harnesses ran", False, f"(old {po.returncode}: {po.stderr[-400:]} | new "
                                                            f"{pn.returncode}: {pn.stderr[-400:]})")
            continue
        okj = open(oo, "rb").read() == open(on, "rb").read() and po.stdout == pn.stdout
        okt, why = same_pt(oo.replace(".json", ".tokens.pt"), on.replace(".json", ".tokens.pt"))
        check(f"A. {name}: OUT.json byte-equal, OUT.tokens.pt equal", okj and okt, why)
    for tag, script in (("old", OLD), ("new", NEW)):
        p = harness(script, [corpus, f"{WORK}/dump-{tag}.pt", "--dump-ref", "--windows", "3"], f"{WORK}/dump.json")
    okd, why = same_pt(f"{WORK}/dump-old.pt", f"{WORK}/dump-new.pt")
    check("A. --dump-ref: equal", okd, why)

# --------------------------------------------------- B. the long mode's definitions on the 2,048-token data (decode)
p = harness(NEW, cases["decode"] + ["--long", "--prompt-rows", "estimate", "--block", "4"], f"{WORK}/decode-long.json")
if p.returncode:
    check("B. --long on 2,048-token cuts ran", False, p.stderr[-800:])
else:
    old = json.load(open(f"{WORK}/decode-old.json" if os.path.exists(f"{WORK}/decode-old.json")
                         else f"{WORK}/decode-new.json"))
    lng = json.load(open(f"{WORK}/decode-long.json"))
    worst, bad = 0.0, []

    def cmp(path, a, b):
        global worst
        if isinstance(a, dict):
            for k in a:
                if k not in b:
                    bad.append(f"{path}.{k} missing")
                else:
                    cmp(f"{path}.{k}", a[k], b[k])
        elif isinstance(a, list):
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(f"{path}[{i}]", x, y)
        elif isinstance(a, float):
            if b is None:
                bad.append(f"{path} None")
                return
            r = abs(a - b) / max(abs(a), 1e-12)
            # kld.py's percentiles interpolate in float32 (the quantile's index included): 1e-4 there
            tol = 1e-4 if path.split(".")[0] in ("median", "p90", "p99", "p999", "generated_median") else 1e-6
            worst = max(worst, r) if tol == 1e-6 else worst
            if r > tol:
                bad.append(f"{path} {a} vs {b}")
        elif a != b:
            bad.append(f"{path} {a} vs {b}")
    keys = ["kld", "kld_ci95", "median", "p90", "p99", "p999", "top1", "cover", "nll_in_topk", "windows", "positions",
            "by_position", "kld_topk_estimate", "kld_decode_rows", "by_domain", "kld_generated", "kld_generated_ci95",
            "generated_rows", "generated_median", "generated_top1", "by_role", "kld_generated_decode_rows",
            "kld_prompt_rows_exact", "kld_generated_prompt_rows_exact", "generated_prompt_rows", "generated_by_domain"]
    for k in keys:
        cmp(k, old.get(k), lng.get(k))
    check("B. --long's definitions = kld.py's on 2,048-token cuts (estimate on prompt rows, blocks of 4)", not bad,
          f"(worst rel {worst:.1e} outside the percentiles; {bad[:4]})")

# ------------------------------------------------------------------------------------------ C. L32-shaped sequences
T, NL = 32768, 2
gl = torch.Generator().manual_seed(9)
toksL = torch.randint(0, VS, (NL, T), generator=gl, dtype=torch.int32)
labL = torch.randint(0, 7, (NL, T), generator=gl, dtype=torch.int8)
labL[0, : T // 2] = 4                                # sequence 0: a tool-heavy first half
tail = torch.tensor([4096, 2048])
kindsL = ["agentic", "code"]
corpusL = f"{WORK}/l32.pt"
torch.save({"tokens": toksL, "labels": labL, "label_names": ["assistant", "text", "system", "user", "tool", "hdr",
                                                                "pre"], "kinds": kindsL, "tail": tail}, corpusL)
refL0, refL1 = f"{WORK}/l32-ref0.pt", f"{WORK}/l32-ref1.pt"
write_ref(refL0, toksL[:1].tolist())
write_ref(refL1, toksL[1:].tolist())
HL = f"{WORK}/l32-h0.npy,{WORK}/l32-h1.npy"
for i, f in enumerate(HL.split(",")):            # headers only (the fake server does not read them); [1, T, 8] each
    np.lib.format.open_memmap(f, mode="w+", dtype=np.uint16, shape=(1, T, 8)).flush()
cuts = f"{WORK}/cuts-arm.json"
json.dump({"kld_generated": 0.05, "generated_by_domain": {"agentic": 0.04, "code": 0.06, "math": 0.2}}, open(cuts, "w"))
SEEN.clear()
argsL = [corpusL, f"{refL0},{refL1}", "--mode", "decode", "--prompt", "tail", "--kl-hidden", HL,
         "--decode-logprobs", "1", "--pipeline", "--prefetch", "--cuts", cuts]
t1 = time.time()
p = harness(NEW, argsL, f"{WORK}/l32.json")
if p.returncode:
    check("C. L32-shaped run", False, p.stderr[-1500:])
else:
    print(f"(L32-shaped run: {time.time() - t1:.0f} s for 2 x 32,768 tokens)")
    R = json.load(open(f"{WORK}/l32.json"))
    Lr = R["long"]
    reqs = sorted(SEEN, key=lambda s: s["vllm_xargs"]["k31_kl_index"])
    okr = [s["prompt_len"] for s in reqs] == [T - 4096, T - 2048] and all(s["prompt_logprobs"] == 1 for s in reqs) \
        and all(s["vllm_xargs"]["k31_kl_rows"] == [0, T] for s in reqs) and all(s["logprobs"] == 1 for s in reqs) \
        and [s["vllm_xargs"]["k31_kl_next"] for s in reqs] == [1, -1] \
        and all(s["vllm_xargs"]["k31_kl_hidden"] == HL for s in reqs)
    check("C. requests: prompt = T - tail, prompt log-probs 1, k31_kl_rows [0, T], prefetch of the next block", okr,
          f"({[(s['prompt_len'], s['prompt_logprobs'], s['vllm_xargs'].get('k31_kl_rows')) for s in reqs]})")
    # the independent computation
    ex = np.stack([exact_kl(toksL[n].tolist(), 0, T - 1) for n in range(NL)])         # [NL, T - 1], as the server
    ex = np.maximum(ex.astype(np.float32).astype(np.float64), 0.0)                     # writes it (float32)
    rows = np.arange(T - 1)
    dec = np.stack([rows >= T - int(tail[n]) - 1 for n in range(NL)])
    gen = labL[:, 1:].numpy() <= 1
    eng_top, nll, ref_top = np.zeros((NL, T - 1), dtype=np.int64), np.zeros((NL, T - 1)), np.zeros((NL, T - 1))
    for n in range(NL):
        ids = toksL[n].tolist()
        e, r = dists(ids, 0, T - 1)
        eng_top[n] = e.argmax(1)
        ref_top[n] = np.argsort(-r, axis=1, kind="stable")[:, 0]
        nll[n] = -e[rows, np.array(ids[1:])]
    t1m = eng_top == ref_top
    m = lambda mask: float(ex[mask].mean())                                              # noqa: E731
    want = {"headline": m(gen & dec), "generated_prompt": m(gen & ~dec), "all": m(np.ones_like(dec)),
            "forced": m(dec), "prompt": m(~dec), "generated": m(gen)}
    got = {"headline": Lr["headline"]["kld"], "generated_prompt": Lr["groups"]["generated_prompt"]["kld"],
           "all": R["kld"], "forced": R["kld_decode_rows"], "prompt": R["kld_prompt_rows_exact"],
           "generated": R["kld_generated"]}
    worst = max(abs(got[k] - want[k]) / want[k] for k in want)
    check("C. group means (headline, generated prompt rows, all, forced, prompt, generated) = independent",
          worst < 1e-6,
          f"(worst rel {worst:.1e}; headline {got['headline']:.5f} over {Lr['headline']['rows']} rows, want "
          f"{int((gen & dec).sum())})")
    check("   row counts", Lr["headline"]["rows"] == int((gen & dec).sum()) and R["positions"] == NL * (T - 1)
          and R["generated_prompt_rows"] == int((gen & ~dec).sum()))
    okt = abs(Lr["headline"]["top1"] - float(t1m[gen & dec].mean())) < 1e-12 and \
        abs(Lr["headline"]["nll"] - float(nll[gen & dec].mean())) < 1e-9 and \
        abs(Lr["groups"]["generated_prompt"]["top1"] - float(t1m[gen & ~dec].mean())) < 1e-12
    check("   top-1 agreement and NLL (headline, generated prompt rows)", okt)
    off = rows[None, :] - (T - tail.numpy()[:, None] - 1)
    okd = True
    for kname, (d0, d1) in (("0-256", (0, 256)), ("256-1024", (256, 1024)), ("1024-2048", (1024, 2048)),
                            ("2048-4096", (2048, 4096))):
        mm = dec & (off >= d0) & (off < d1)
        e_ = Lr["by_distance"][kname]
        okd &= abs(e_["kld"] - m(mm)) < 1e-9 and e_["rows"] == int(mm.sum()) and \
            abs(e_["generated"] - m(mm & gen)) < 1e-9
    check("   by distance from the prompt's end (0-256, 256-1024, 1024-2048, 2048-4096)", okd and
          "4096-8192" not in Lr["by_distance"])
    okp = all(abs(Lr["by_position"][f"{b0}-{b1}"]["generated"] - m(gen & (rows >= b0)[None] & (rows < b1)[None]))
              < 1e-9 for b0, b1 in ((0, 256), (2048, 4096), (16384, 32767)))
    check("   by position (absolute, doubling buckets to 32,767)", okp, f"({list(Lr['by_position'])})")
    seq = lambda n: (np.arange(NL) == n)[:, None]                                      # noqa: E731
    okk = all(abs(Lr["by_kind"][k]["generated_forced"] - m(seq(n) & gen & dec)) < 1e-9 and
              abs(Lr["by_kind"][k]["generated_prompt"] - m(seq(n) & gen & ~dec)) < 1e-9 for n, k in enumerate(kindsL))
    check("   by kind", okk)
    c = Lr["cuts"]
    wa, wc = int((gen & dec)[0].sum()), int((gen & dec)[1].sum())
    base = (wa * 0.04 + wc * 0.06) / (wa + wc)
    check("   --cuts: the cuts' per-kind values weighted by this set's generated forced rows",
          abs(c["cuts_generated_same_kinds"] - base) < 1e-12 and abs(c["ratio"] - want["headline"] / base) < 1e-9
          and c["kinds"] == ["agentic", "code"], f"(ratio {c['ratio']:.3f})")
    TP = torch.load(f"{WORK}/l32.tokens.pt", weights_only=False)
    check("   OUT.tokens.pt: per-row KL = the server's, eng_ids of a forced row hold the engine's top-1",
          np.array_equal(TP["kl"].numpy(), ex.astype(np.float32)) and
          int(TP["eng_ids"][1, T - 10, 0]) == int(eng_top[1, T - 10]) and TP["prompt"] == [T - 4096, T - 2048])
    # --kl-rows: the last 2,049 rows only
    SEEN.clear()
    p = harness(NEW, argsL[:-2] + ["--kl-rows=-2049:"], f"{WORK}/l32-rows.json")
    if p.returncode:
        check("C. --kl-rows -2049:", False, p.stderr[-800:])
    else:
        R2 = json.load(open(f"{WORK}/l32-rows.json"))
        sc = rows >= T - 2049
        ok = abs(R2["kld"] - m(np.broadcast_to(sc, dec.shape))) < 1e-9 and R2["positions"] == NL * int(sc.sum()) \
            and all(s["vllm_xargs"]["k31_kl_rows"] == [T - 2049, T] for s in SEEN) \
            and abs(R2["long"]["headline"]["kld"] - m(gen & dec & sc[None])) < 1e-9
        check("C. --kl-rows -2049: only those rows scored (prompt rows of sequence 0 before them left out)", ok,
              f"(positions {R2['positions']})")
    # a row the server did not score inside the range: refused
    NAN_ROW["row"] = 1000
    p = harness(NEW, argsL[:-2], f"{WORK}/l32-nan.json")
    NAN_ROW["row"] = None
    check("C. a scored row without an exact KL is refused", p.returncode != 0 and "without an exact KL" in p.stderr,
          f"(exit {p.returncode})")
    # the 2,048-token cuts' roles file with a 32K corpus that has no labels: refused (same label count, wrong layout)
    nol = f"{WORK}/l32-nolabels.pt"
    big = torch.randint(0, VS, (32, T), generator=gl, dtype=torch.int32)
    torch.save({"tokens": big, "kinds": ["x"] * 32, "tail": torch.full((32,), 2048)}, nol)
    lab512 = f"{WORK}/roles512.pt"
    torch.save({"labels": torch.zeros((128, 8192), dtype=torch.int8), "names": ["assistant"]}, lab512)
    p = harness(NEW, [nol, "-", "--mode", "decode", "--prompt", "tail", "--kl-hidden", "/nonexistent/h.npy",
                      "--roles", lab512, "--windows", "1"], f"{WORK}/l32-badroles.json")
    check("C. a roles file that does not match the corpus's sequences is refused",
          p.returncode != 0 and "not whole" in p.stderr, f"(exit {p.returncode}: {p.stderr.strip()[-160:]})")

# ------------------------------------------------------------------------------- D. the real L32 corpus, full size
LC = os.environ.get("L32_CORPUS", "")
if LC and os.path.exists(LC):
    import resource
    c = torch.load(LC, weights_only=False)
    tk, lb, tl, kd = c["tokens"], c["labels"], c["tail"], c["kinds"]
    ND, TD = tk.shape
    refD = f"{WORK}/real-ref.pt"
    write_ref(refD, tk.tolist())
    HD = f"{WORK}/real-h0.npy,{WORK}/real-h1.npy"
    for f, n in zip(HD.split(","), (ND // 2, ND - ND // 2)):
        np.lib.format.open_memmap(f, mode="w+", dtype=np.uint16, shape=(n, TD, 8)).flush()
        json.dump({"row_start": 0, "row_end": TD, "seq_len": TD}, open(f + ".json", "w"))
    SEEN.clear()
    t1 = time.time()
    p = harness(NEW, [LC, refD, "--mode", "decode", "--prompt", "tail", "--kl-hidden", HD, "--decode-logprobs", "1",
                      "--pipeline", "--prefetch"], f"{WORK}/real.json")
    secs = time.time() - t1
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    if p.returncode:
        check("D. the real L32 corpus through the harness", False, p.stderr[-1500:])
    else:
        R = json.load(open(f"{WORK}/real.json"))
        Lr = R["long"]
        reqs = sorted(SEEN, key=lambda s: s["vllm_xargs"]["k31_kl_index"])
        Pw = [TD - int(v) for v in tl.tolist()]
        rows = np.arange(TD - 1)
        dec = np.stack([rows >= q - 1 for q in Pw])
        gen = lb[:, 1:].numpy() <= 1
        okc = [s["prompt_len"] for s in reqs] == Pw and Lr["protocol"]["prompt"]["by_sequence"] == Pw
        cnt = {k: (int((gen & dec)[[i for i in range(ND) if kd[i] == k]].sum()),
                   int((gen & ~dec)[[i for i in range(ND) if kd[i] == k]].sum())) for k in sorted(set(kd))}
        okk = all((Lr["by_kind"][k]["generated_forced_rows"], Lr["by_kind"][k]["generated_prompt_rows"]) == v
                  for k, v in cnt.items())
        check("D. real L32 corpus: prompt lengths = 32,768 - tail; generated rows per kind = the labels' counts; "
              "every row scored", okc and okk and R["positions"] == ND * (TD - 1),
              f"({sorted(set(Pw))}; generated forced / prompt rows by kind {cnt})")
        print(f"INFO harness on 32 x 32,768 (fake server): {secs:.0f} s in all, peak RSS {rss:.2f} GiB; headline rows "
              f"{Lr['headline']['rows']}, generated prompt rows {Lr['groups']['generated_prompt']['rows']}", flush=True)
print(f"({time.time() - t00:.0f} s)")
print("KLD LONG:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
srv.shutdown()
sys.exit(1 if fails else 0)
