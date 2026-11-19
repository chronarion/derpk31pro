"""Step 1 of the KLD corpus: fetch small samples of public text, one JSONL file per kind, for pack_corpus.py.

The kinds and sources follow derpseek's tools/varied_corpus.py (Wikipedia in many languages and scripts, web text,
Rosetta Code in many languages, math with solutions, multi-turn chats, biomedical articles, generated structured
data), so K3.1's corpus is comparable with the DeepSeek one. Plain-text kinds are stored as {"text": ...}; chat kinds
as {"messages": [{"role", "content"}, ...]}. Runs where the `datasets` library is installed, with network access; it
runs no model code.

usage: fetch_texts.py OUT_DIR [--chars 60000]
"""
import argparse
import json
import os
import random
import sys

import datasets

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--chars", type=int, default=60000, help="characters to collect per kind (roughly 3+ windows)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = random.Random(20261116)


def stream(name, cfg=None, split="train", skip=0):
    ds = datasets.load_dataset(name, cfg, split=split, streaming=True)
    return iter(ds.skip(skip)) if skip else iter(ds)


def save(kind, source, records, chars=None):
    """records: iterable of {"text"} or {"messages"}; keeps them until `chars` characters are collected."""
    chars = chars or a.chars
    out, n = [], 0
    try:
        for r in records:
            size = len(r["text"]) if "text" in r else sum(len(m["content"]) for m in r["messages"])
            if size == 0:
                continue
            out.append(r)
            n += size
            if n >= chars:
                break
    except Exception as e:                                       # a source that fails is reported and skipped
        sys.stderr.write(f"{kind}: {e!r}\n")
    if out:
        with open(os.path.join(a.out, f"{kind}.jsonl"), "w") as f:
            f.write(json.dumps({"kind": kind, "source": source}) + "\n")
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sys.stderr.write(f"{kind}: {len(out)} records, {n} chars\n")


for lang in ("en", "de", "fr", "es", "ru", "zh", "ja", "ar", "hi", "ko", "sw", "fi", "tr", "he", "th", "vi"):
    it = stream("wikimedia/wikipedia", f"20231101.{lang}", skip=rng.randrange(200, 3000))
    save(f"wiki_{lang}", f"wikimedia/wikipedia 20231101.{lang}", ({"text": r["text"].strip()} for r in it))
for cfg in ("rus_Cyrl", "cmn_Hani", "jpn_Jpan", "spa_Latn"):
    save(f"web_{cfg.split('_')[0]}", f"HuggingFaceFW/fineweb-2 {cfg}",
         ({"text": r["text"].strip()} for r in stream("HuggingFaceFW/fineweb-2", cfg)))
save("web_eng", "HuggingFaceFW/fineweb sample-10BT",
     ({"text": r["text"].strip()} for r in stream("HuggingFaceFW/fineweb", "sample-10BT")), chars=2 * a.chars)

CODE = ["Python", "C", "C++", "Java", "JavaScript", "Go", "Rust", "Haskell", "SQL", "UNIX Shell", "Fortran", "Lua"]
code = {c: [] for c in CODE}
try:
    for r in stream("christopher/rosetta-code"):
        lang = r.get("language_name") or (r.get("language_url") or "").lstrip("#").replace("_", " ")
        if lang in code and sum(len(x) for x in code[lang]) < a.chars:
            code[lang].append(f"// {r['task_name']}\n{r.get('code') or ''}")
        if all(sum(len(x) for x in v) >= a.chars for v in code.values()):
            break
except Exception as e:
    sys.stderr.write(f"rosetta: {e!r}\n")
for lang, texts in code.items():
    rng.shuffle(texts)
    name = "".join(ch for ch in lang.lower().replace("c++", "cpp") if ch.isalnum())
    save(f"code_{name}", f"christopher/rosetta-code {lang}", ({"text": t} for t in texts))

save("math_chat", "HuggingFaceH4/MATH-500",
     ({"messages": [{"role": "user", "content": r["problem"]}, {"role": "assistant", "content": r["solution"]}]}
      for r in stream("HuggingFaceH4/MATH-500", split="test")), chars=2 * a.chars)
save("chat_ultra", "HuggingFaceH4/ultrachat_200k",
     ({"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]}
      for r in stream("HuggingFaceH4/ultrachat_200k", split="train_sft")), chars=2 * a.chars)
try:
    by = {}
    for r in stream("OpenAssistant/oasst1"):
        if r["lang"] in ("en", "es", "ru", "de", "zh", "fr") and sum(len(x) for x in by.get(r["lang"], [])) < a.chars:
            by.setdefault(r["lang"], []).append((r["role"], r["text"]))
    for lang, turns in by.items():                       # consecutive messages as alternating turns
        msgs = [{"role": "user" if role == "prompter" else "assistant", "content": text} for role, text in turns]
        save(f"chat_oasst_{lang}", f"OpenAssistant/oasst1 {lang}",
             ({"messages": msgs[i:i + 6]} for i in range(0, len(msgs), 6)))
except Exception as e:
    sys.stderr.write(f"oasst: {e!r}\n")
save("sci_pubmed", "ccdv/pubmed-summarization",
     ({"text": r["article"].strip()} for r in stream("ccdv/pubmed-summarization", "document")))

recs = [r for _, r in zip(range(400), stream("HuggingFaceH4/MATH-500", split="test"))]
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
save("struct_json", "generated JSON records", ({"text": json.dumps(
    {"id": i, "subject": recs[i % len(recs)]["subject"], "level": recs[i % len(recs)]["level"],
     "answer": recs[i % len(recs)]["answer"], "tags": rng.sample(words, 3), "score": round(rng.random(), 4)},
    indent=2, ensure_ascii=False)} for i in range(5000)))
save("struct_csv", "generated CSV", ({"text": "id,subject,level,score,ratio\n" + "\n".join(
    f"{i},{recs[i % len(recs)]['subject']},{recs[i % len(recs)]['level']},{rng.randint(0, 999)},{rng.random():.5f}"
    for i in range(k * 40, k * 40 + 40))} for k in range(500)))
save("struct_yaml", "generated YAML", ({"text": (
    f"service:\n  name: {rng.choice(words)}-{k}\n  replicas: {rng.randint(1, 9)}\n  ports:\n    - "
    f"{rng.randint(1024, 65535)}\n  env:\n    LEVEL: {rng.choice(words)}\n    TIMEOUT_MS: {rng.randint(100, 9000)}\n"
    f"  enabled: {rng.choice(['true', 'false'])}")} for k in range(5000)))
save("struct_log", "generated logs", ({"text": "\n".join(
    f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T{rng.randint(10, 23)}:{rng.randint(10, 59)}:"
    f"{rng.randint(10, 59)}Z {rng.choice(['INFO', 'WARN', 'ERROR', 'DEBUG'])} {rng.choice(words)}-svc "
    f"request_id={rng.randrange(16 ** 8):08x} latency_ms={rng.randint(1, 4000)} status={rng.choice([200, 404, 500])}"
    for _ in range(30))} for _ in range(3000)))
sys.stderr.write("done\n")
