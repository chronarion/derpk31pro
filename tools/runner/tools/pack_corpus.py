"""Step 2 of the KLD corpus: tokenize fetch_texts.py's samples with Kimi K3.1's own tokenizer and chat format
(tokenization_kimi.py + encoding_k3.py from the checkpoint; chats rendered by its apply_chat_template, which adds
K3.1's default thinking-effort note as served traffic has it) and pack them into teacher-forced windows of 2,048
tokens ([BOS] + 2,047), each from consecutive records of one kind, a window starting on a fresh record.

Writes OUT (all windows: {"tokens" int64 [N, 2048], "kinds", "sources", "first"}) and OUT with "-first" before the
suffix: the first-pass selection, one window from each of 32 kinds across text, code, chat, math, science and
structured data; every other window is held out for later validation. Runs without network.

usage: pack_corpus.py RAW_DIR K31_DIR OUT.pt [--per 3]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import torch
from transformers import AutoTokenizer

FIRST = ["wiki_en", "wiki_de", "wiki_ru", "wiki_zh", "wiki_ja", "wiki_ar", "wiki_hi", "wiki_ko",
         "web_eng", "web_cmn", "web_spa", "web_rus",
         "code_python", "code_cpp", "code_java", "code_rust", "code_haskell", "code_sql", "code_unixshell", "code_go",
         "math_chat", "math_chat", "chat_ultra", "chat_ultra", "chat_oasst_en", "chat_oasst_zh", "chat_oasst_es",
         "sci_pubmed", "struct_json", "struct_csv", "struct_yaml", "struct_log"]
TOKENIZER_FILES = ("tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py", "encoding_k3.py", "config.json",
                   "configuration_kimi_k3.py")

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("k31")
ap.add_argument("out")
ap.add_argument("--per", type=int, default=3, help="windows per kind at most")
ap.add_argument("--T", type=int, default=2048)
a = ap.parse_args()

tokdir = tempfile.mkdtemp(prefix="k31tok-")
for f in TOKENIZER_FILES:
    shutil.copy(os.path.join(a.k31, f), tokdir)
tok = AutoTokenizer.from_pretrained(tokdir, trust_remote_code=True)
BOS = tok.bos_token_id
assert BOS is not None, "no BOS token"
T = a.T

windows, kinds, sources = [], [], []
for path in sorted(glob.glob(os.path.join(a.raw, "*.jsonl"))):
    lines = open(path, encoding="utf-8").read().splitlines()
    meta, recs = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    buf, made = [], 0
    for r in recs:
        if made >= a.per:
            break
        if "text" in r:
            ids = tok.encode(r["text"] + "\n\n", add_special_tokens=False)
        else:
            ids = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
        buf += list(ids)
        if len(buf) >= T - 1:
            windows.append([BOS] + buf[: T - 1])
            kinds.append(meta["kind"])
            sources.append(meta["source"])
            made, buf = made + 1, []
    sys.stderr.write(f"{meta['kind']}: {made} windows\n")

first, used = [], set()
for k in FIRST:
    i = next((i for i, kk in enumerate(kinds) if kk == k and i not in used), None)
    if i is None:
        sys.stderr.write(f"first pass: no window left for {k}\n")
        continue
    first.append(i)
    used.add(i)
tokens = torch.tensor(windows, dtype=torch.int64)
torch.save({"tokens": tokens, "kinds": kinds, "sources": sources, "first": first}, a.out)
stem, ext = os.path.splitext(a.out)
torch.save({"tokens": tokens[first], "kinds": [kinds[i] for i in first], "sources": [sources[i] for i in first]},
           f"{stem}-first{ext}")
sample = tok.apply_chat_template([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
                                 tokenize=False, add_generation_prompt=False)
print(f"{len(windows)} windows of {T} tokens from {len(set(kinds))} kinds; first pass {len(first)} windows; "
      f"held out {len(windows) - len(first)}")
print("chat format sample:", repr(sample[:400]))
