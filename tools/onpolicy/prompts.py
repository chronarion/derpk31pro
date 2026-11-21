"""Prompts for the on-policy capture (capture.py): one JSON line per request.

    prompts.py OUT.jsonl [--gpqa GPQA.jsonl --gpqa-template TEMPLATE.json --gpqa-samples N --gpqa-first N]
               [--src SRC_DIR --other kind=count,...] [--seed-base S] [--max-chars N]
               [--manifest-out SET.json | --recipe SET.json]

A frozen set: --manifest-out writes SET.json = the recipe (the options above), the sha256 of every source file and,
per request, its id, kind, seed and the sha256 of its messages and tools. --recipe SET.json rebuilds the set from the
same sources with the manifest's options and fails unless every request matches its recorded hash, so the prompt file
itself (which holds GPQA plaintext) never has to be published: the manifest is the set.

GPQA Diamond exactly as lil-bench's gpqa-diamond profile sends it: GPQA.jsonl is lil-bench's canonical file (its
builder's output: question, options shuffled per record id, answer letter; the plaintext must stay local), and the one
user message is lil-bench's multiple-choice template (TEMPLATE.json: {"template": ..., "letters": ...}) with the options
as "A. ...". --gpqa-samples N writes N requests per question (sample 0 first for every question, then sample 1, ...),
each with its own seed.

Other prompts: the conversation openings (system message if any, tools if any, the first user turn) of rows of the
KLD protocol's source files in SRC_DIR (tools/runner/tools/bm_eval.py's names), taken from the END of each file so they
are not rows the evaluation windows use. Kinds: math, code, chat_if, agentic, science, multilingual, safety.

Line: {"id", "kind", "sample", "seed", "messages", "tools" (optional), "meta"}.
"""
import argparse
import hashlib
import json
import os
import sys
import zlib

SOURCES = {   # kind: file name in SRC_DIR (bm_eval.py's src_name)
    "math": "nvidia__Nemotron-SFT-Math-v4__data__train-00011-of-00012.parquet",
    "code": "nvidia__Nemotron-SFT-Competitive-Programming-v2__data__competitive_programming_python_00.jsonl",
    "chat_if": "nvidia__Nemotron-SFT-Instruction-Following-Chat-v3__data__instruction_following.jsonl",
    "agentic": "nvidia__Nemotron-SFT-Agentic-v2__data__search.jsonl",
    "science": "nvidia__Nemotron-SFT-Science-v2__syn_mcq.jsonl",
    "multilingual": "nvidia__Nemotron-SFT-Multilingual-v2__ultra-v3_stem_ja_translated_postedit_final.jsonl",
    "safety": "nvidia__Nemotron-SFT-Safety-v2__data__train.jsonl",
}


def seed_of(base, rid, sample):
    return (base + zlib.crc32(f"{rid}#{sample}".encode())) % (2 ** 31)


def fix_tools(tools):
    out = []
    for t in tools or []:
        t = dict(t)
        fn = dict(t.get("function", {}))
        if isinstance(fn.get("parameters"), str):
            fn["parameters"] = json.loads(fn["parameters"])
        t["function"] = fn
        out.append(t)
    return out or None


def opening(rec):
    """Messages up to and including the first user turn (text content only), and the row's tools."""
    msgs = []
    for m in rec["messages"]:
        if m["role"] not in ("system", "user") or not isinstance(m.get("content"), str):
            break
        msgs.append({"role": m["role"], "content": m["content"]})
        if m["role"] == "user":
            return msgs, fix_tools(rec.get("tools"))
    return None, None


def tail_rows(path, n):
    """The last rows of a source file, last first (jsonl, or parquet when pyarrow is there)."""
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(path)]
        return [(i, rows[i]) for i in range(len(rows) - 1, max(-1, len(rows) - 1 - 4 * n), -1)]
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    g = f.num_row_groups - 1
    start = f.metadata.num_rows - f.metadata.row_group(g).num_rows
    rows = f.read_row_group(g).to_pylist()
    return [(start + i, rows[i]) for i in range(len(rows) - 1, max(-1, len(rows) - 1 - 4 * n), -1)]


def prompt_hash(e):
    body = {"messages": e["messages"], "tools": e.get("tools")}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--gpqa"); p.add_argument("--gpqa-template"); p.add_argument("--gpqa-samples", type=int, default=1)
    p.add_argument("--src"); p.add_argument("--other", default="")
    p.add_argument("--max-chars", type=int, default=24000, help="skip openings longer than this")
    p.add_argument("--seed-base", type=int, default=20261119)
    p.add_argument("--gpqa-first", type=int, default=0, help="only the first N GPQA records (0 = all)")
    p.add_argument("--manifest-out", help="write the frozen set's manifest here")
    p.add_argument("--recipe", help="rebuild a frozen set from its manifest and verify every request against it")
    a = p.parse_args()
    man = None
    if a.recipe:
        man = json.load(open(a.recipe))
        r = man["recipe"]
        a.other, a.gpqa_samples, a.gpqa_first = r["other"], r["gpqa_samples"], r["gpqa_first"]
        a.max_chars, a.seed_base = r["max_chars"], r["seed_base"]
    out = []
    if a.other:
        for item in a.other.split(","):
            kind, _, n = item.partition("=")
            n = int(n)
            path = os.path.join(a.src, SOURCES[kind])
            got = 0
            for row, rec in tail_rows(path, n):
                msgs, tools = opening(rec)
                if msgs is None or sum(len(m["content"]) for m in msgs) > a.max_chars:
                    continue
                rid = f"{kind}-{row}"
                e = {"id": f"{rid}#0", "kind": kind, "sample": 0, "seed": seed_of(a.seed_base, rid, 0),
                     "messages": msgs, "meta": {"file": SOURCES[kind], "row": row}}
                if tools:
                    e["tools"] = tools
                out.append(e)
                got += 1
                if got == n:
                    break
            if got < n:
                raise SystemExit(f"{kind}: {got} of {n} openings")
    gp = []
    if a.gpqa:
        t = json.load(open(a.gpqa_template))
        rows = [json.loads(line) for line in open(a.gpqa)]
        if a.gpqa_first:
            rows = rows[:a.gpqa_first]
        for s in range(a.gpqa_samples):
            for row in rows:
                opts = "\n".join(f"{t['letters'][i]}. {o}" for i, o in enumerate(row["options"]))
                rid = f"gpqa-{row['record_id']}"
                gp.append({"id": f"{rid}#{s}", "kind": "gpqa", "sample": s, "seed": seed_of(a.seed_base, rid, s),
                           "messages": [{"role": "user", "content": t["template"].format(question=row["question"],
                                                                                         options=opts)}],
                           "meta": {"category": row["category"], "answer": row["answer"]}})
    n0 = len([e for e in gp if e["sample"] == 0])
    final = gp[:n0] + out + gp[n0:]             # GPQA sample 0, the other prompts, then the GPQA resamples
    used = {SOURCES[k.partition("=")[0]]: os.path.join(a.src, SOURCES[k.partition("=")[0]])
            for k in a.other.split(",") if k}
    if a.gpqa:
        used.update({"gpqa": a.gpqa, "gpqa-template": a.gpqa_template})
    if man is not None:
        want = [(x["id"], x["kind"], x["seed"], x["sha256"]) for x in man["prompts"]]
        got = [(e["id"], e["kind"], e["seed"], prompt_hash(e)) for e in final]
        bad = [w[0] for w, g in zip(want, got) if w != g]
        if len(want) != len(got) or bad:
            src = [n for n, pth in used.items() if man["sources"].get(n) != file_hash(pth)]
            sys.exit(f"the rebuilt set does not match {a.recipe}: {len(got)} requests for {len(want)}, "
                     f"{len(bad)} differ (first: {bad[:3]}); source files with another sha256: {src}")
        print(f"verified against {a.recipe} ({man['name']}): {len(got)} requests match their hashes")
    if a.manifest_out:
        m = {"name": os.path.basename(a.manifest_out).rsplit(".", 1)[0],
             "recipe": {"other": a.other, "gpqa_samples": a.gpqa_samples, "gpqa_first": a.gpqa_first,
                        "max_chars": a.max_chars, "seed_base": a.seed_base},
             "sources": {n: file_hash(pth) for n, pth in used.items()},
             "kinds": {}, "prompts": [{"id": e["id"], "kind": e["kind"], "seed": e["seed"], "sha256": prompt_hash(e)}
                                      for e in final]}
        for e in final:
            m["kinds"][e["kind"]] = m["kinds"].get(e["kind"], 0) + 1
        m["sha256"] = hashlib.sha256(json.dumps(m["prompts"], sort_keys=True).encode()).hexdigest()
        json.dump(m, open(a.manifest_out, "w"), indent=1)
    with open(a.out, "w") as f:
        for e in final:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    kinds = {}
    for e in final:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"{len(final)} requests: {kinds}")


if __name__ == "__main__":
    main()
