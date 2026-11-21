"""KLD sets from an on-policy capture (capture.py), in the harness's corpus format.

    build_sets.py TOKENIZER_DIR OUT_DIR CAPTURE.jsonl [CAPTURE2.jsonl ...] [--main-first N] [--main-gen 4096]
                  [--main-budget TOKENS] [--loop-back 4096] [--loop-keep 2048] [--loops-json FILE] [--gpus 4]

Main set: each selected answer as one sequence = the rendered prompt's token ids (as the server reported them) + the
first --main-gen generated tokens. The first --main-first records of the first capture file are taken whole (0 =
all), then more records in file order while the generated tokens stay under --main-budget.

Loop set: each answer in which capture.loop_scan (or --loops-json: {"id": {"loop_start": i, "period": p} | null}
overrides) finds a loop: prompt + everything generated up to the loop's start + the loop's first tokens (--loop-keep,
or two periods + 512 if that is more, at most 6,144). The engine reads the sequence's beginning through its prompt path
and is forced token by token from --loop-back tokens before the loop's start (`forced_from`).

Sequences have different lengths; the harness and the reference runner take rectangular [N, T] files. Each set is
split into buckets of similar length, padded at the END with the sequence's last token (the model is causal: rows
before the padding do not see it; padded rows are labelled `pad` and never scored). The split minimises padded tokens
for a number of buckets chosen so that --gpus reference passes in parallel finish soonest.

Per bucket OUT_DIR/<set>-b<k>.pt: {"tokens" int32 [N, T], "labels" int8 [N, T] (tools/runner/tools/role_labels.py's
rules on the whole sequence: 0 assistant, 1 text, 2 system, 3 user, 4 tool, 5 hdr, 6 pre, 7 pad), "label_names",
"kinds", "ids" (request ids), "lens" (real tokens), "prompt" (prompt tokens = index of the first generated token),
"forced_from" (index of the first token the engine is forced on), "loop_start" / "period" (loop set; absolute index),
"protocol"}. OUT_DIR/sets.json: every sequence (bucket, row, lengths, finish reason, loop), the buckets and the
per-GPU schedule for the reference passes.
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import loop_scan  # noqa: E402

NAMES = ["assistant", "text", "system", "user", "tool", "hdr", "pre", "pad"]
FIXED_S, PER_TOKEN_S = 600.0, 0.0055       # a reference pass: seconds per pass and per (padded) token, one GPU


def label_codes(tok, ids):
    """role_labels.py's labels of every token: message framing = hdr, else the enclosing message's role (channels
    inside an assistant message count as assistant), `pre` before the first header."""
    toks = tok.convert_ids_to_tokens(ids)
    T = len(toks)
    lab = ["pre"] * T
    cur = "pre"
    i = 0
    while i < T:
        if toks[i] == "<|open|>":
            j = i + 1
            while j < T and toks[j] != "<|sep|>" and j - i < 48:
                j += 1
            head = tok.decode(ids[i + 1:j])
            m = re.search(r'role="([^"]+)"', head)
            if m:
                cur = next((k for k in ("assistant", "system", "user", "tool") if m.group(1).startswith(k)), "user")
            else:
                cur = "assistant"                        # a channel inside an assistant message
            for k in range(i, min(j + 1, T)):
                lab[k] = "hdr"
            i = j + 1
            continue
        if toks[i] in ("<|close|>", "<|end_of_msg|>"):
            lab[i] = "hdr"
            i += 1
            continue
        lab[i] = cur
        i += 1
    if all(x == "pre" for x in lab):                         # no message header at all: plain text
        lab = ["text"] * T
    return [NAMES.index(x) for x in lab]


def split_buckets(lens, k):
    """Sorted lengths into k contiguous groups minimising the padded tokens (sum of count x group maximum)."""
    n = len(lens)
    INF = float("inf")
    cost = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    cost[0][0] = 0
    for g in range(1, k + 1):
        for j in range(g, n + 1):
            best, arg = INF, g - 1
            for i in range(g - 1, j):
                c = cost[g - 1][i] + (j - i) * lens[j - 1]
                if c < best:
                    best, arg = c, i
            cost[g][j], cut[g][j] = best, arg
    out, j = [], n
    for g in range(k, 0, -1):
        i = cut[g][j]
        out.append((i, j))
        j = i
    return out[::-1]


def schedule(groups, lens, gpus):
    """Longest pass first onto the least loaded GPU; returns (makespan seconds, [[group index, ...] per GPU])."""
    secs = [FIXED_S + PER_TOKEN_S * (j - i) * lens[j - 1] for i, j in groups]
    load, plan = [0.0] * gpus, [[] for _ in range(gpus)]
    for g in sorted(range(len(groups)), key=lambda x: -secs[x]):
        t = min(range(gpus), key=lambda x: load[x])
        load[t] += secs[g]
        plan[t].append(g)
    return max(load), plan, secs


def pack(seqs, gpus, max_buckets=12):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["tokens"]))
    lens = [len(seqs[i]["tokens"]) for i in order]
    best = None
    for k in range(1, min(max_buckets, len(seqs)) + 1):
        groups = split_buckets(lens, k)
        span, plan, secs = schedule(groups, lens, gpus)
        if best is None or span < best[0] - 1:
            best = (span, groups, plan, secs)
    span, groups, plan, secs = best
    return order, lens, groups, plan, secs, span


def write_set(kind, seqs, tok, out_dir, gpus, protocol, suffix=""):
    name = kind + suffix
    if not seqs:
        return {"buckets": [], "plan": [], "sequences": []}
    order, lens, groups, plan, secs, span = pack(seqs, gpus)
    info = {"buckets": [], "plan": [[f"{name}-b{g}" for g in gp] for gp in plan], "makespan_s_est": span,
            "sequences": []}
    for b, (i, j) in enumerate(groups):
        T = lens[j - 1]
        rows = [seqs[order[x]] for x in range(i, j)]
        tokens = torch.zeros((len(rows), T), dtype=torch.int32)
        labels = torch.full((len(rows), T), NAMES.index("pad"), dtype=torch.int8)
        for r, s in enumerate(rows):
            n = len(s["tokens"])
            tokens[r, :n] = torch.tensor(s["tokens"], dtype=torch.int32)
            tokens[r, n:] = s["tokens"][-1]
            labels[r, :n] = torch.tensor(label_codes(tok, s["tokens"]), dtype=torch.int8)
            info["sequences"].append({"id": s["id"], "kind": s["kind"], "bucket": f"{name}-b{b}", "row": r, "len": n,
                                      "prompt": s["prompt"], "forced_from": s["forced_from"],
                                      "generated_total": s["generated_total"], "finish_reason": s["finish_reason"],
                                      "loop": s.get("loop")})
        d = {"tokens": tokens, "labels": labels, "label_names": NAMES, "kinds": [s["kind"] for s in rows],
             "ids": [s["id"] for s in rows], "lens": [len(s["tokens"]) for s in rows],
             "prompt": [s["prompt"] for s in rows], "forced_from": [s["forced_from"] for s in rows],
             "protocol": protocol}
        if kind == "loop":
            d["loop_start"] = [s["loop_abs"] for s in rows]
            d["period"] = [s["loop"]["period"] for s in rows]
        torch.save(d, os.path.join(out_dir, f"{name}-b{b}.pt"))
        info["buckets"].append({"name": f"{name}-b{b}", "sequences": len(rows), "T": T,
                                "tokens": int(sum(d["lens"])), "padded": len(rows) * T, "seconds_est": secs[b],
                                "path": "long" if T > 2048 else "dense"})
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tokenizer"); p.add_argument("out"); p.add_argument("captures", nargs="+")
    p.add_argument("--main-first", type=int, default=0)
    p.add_argument("--main-gen", type=int, default=4096)
    p.add_argument("--main-budget", type=int, default=0)
    p.add_argument("--loop-back", type=int, default=4096)
    p.add_argument("--loop-keep", type=int, default=2048)
    p.add_argument("--loops-json", default="")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--sets", default="main,loop")
    p.add_argument("--suffix", default="", help="added to the bucket names (<set><suffix>-b<k>), e.g. for a second "
                   "capture's loops beside the first's")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    os.makedirs(a.out, exist_ok=True)
    recs, skipped = [], []
    for fi, f in enumerate(a.captures):
        for line in open(f):
            r = json.loads(line)
            c = r.get("checks") or {}
            if r.get("error") or not r.get("ids") or not r.get("prompt_ids") or not c.get("logprob_ids_match", True):
                skipped.append((r.get("id"), r.get("error") or "incomplete record"))
                continue
            r["_file"] = fi
            recs.append(r)
    seen = set()
    recs = [r for r in recs if not (r["id"] in seen or seen.add(r["id"]))]
    over = json.load(open(a.loops_json)) if a.loops_json else {}
    main_seqs, loop_seqs, budget = [], [], 0
    first = [r for r in recs if r["_file"] == 0]
    n_first = len(first) if a.main_first == 0 else a.main_first
    for n, r in enumerate(recs):
        P, ids = len(r["prompt_ids"]), r["ids"]
        whole = r["_file"] == 0 and first.index(r) < n_first
        g = min(len(ids), a.main_gen)
        if whole or (a.main_budget and budget + g <= a.main_budget):
            budget += g
            main_seqs.append({"id": r["id"], "kind": r["kind"], "tokens": r["prompt_ids"] + ids[:g], "prompt": P,
                              "forced_from": P, "generated_total": len(ids), "finish_reason": r["finish_reason"]})
        lp = over[r["id"]] if r["id"] in over else loop_scan(ids)
        if lp:
            keep = min(max(a.loop_keep, 2 * lp["period"] + 512), 6144)
            end = min(len(ids), lp["loop_start"] + keep)
            loop_seqs.append({"id": r["id"], "kind": r["kind"], "tokens": r["prompt_ids"] + ids[:end], "prompt": P,
                              "forced_from": max(P, P + lp["loop_start"] - a.loop_back),
                              "generated_total": len(ids), "finish_reason": r["finish_reason"], "loop": lp,
                              "loop_abs": P + lp["loop_start"]})
    info = {"captures": a.captures, "records": len(recs), "skipped": skipped, "main_gen": a.main_gen,
            "loop_back": a.loop_back, "loop_keep": a.loop_keep}
    want = a.sets.split(",")
    if "main" in want:
        info["main"] = write_set("main", main_seqs, tok, a.out, a.gpus,
                                 f"on-policy main set: prompt + the first {a.main_gen} generated tokens of each answer",
                                 a.suffix)
    if "loop" in want:
        info["loop"] = write_set("loop", loop_seqs, tok, a.out, a.gpus,
                                 f"on-policy loop set: prompt + everything up to the loop's start + its first tokens; "
                                 f"forced from {a.loop_back} tokens before the loop's start", a.suffix)
    json.dump(info, open(os.path.join(a.out, "sets.json"), "w"), indent=1)
    for name in want:
        s = info.get(name) or {}
        bs = s.get("buckets", [])
        print(f"{name}: {len(s.get('sequences', []))} sequences, {sum(b['tokens'] for b in bs)} tokens "
              f"({sum(b['padded'] for b in bs)} padded) in {len(bs)} buckets "
              f"{[(b['name'], b['sequences'], b['T']) for b in bs]}; reference est. {s.get('makespan_s_est', 0):.0f} s "
              f"on {a.gpus} GPUs: {s.get('plan')}")
    print(f"records {len(recs)}, skipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    main()
