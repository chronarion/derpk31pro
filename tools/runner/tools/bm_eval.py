"""brandonmusic's KLD evaluation set, rebuilt for Kimi K3.1.

The protocol of brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits (and its TrellisMX cards): 128 windows of 8,192
tokens in 11 domains, each filled from its own category with conversations from public datasets rendered with the
model's own chat encoding; the headline KLD is over the 512 cuts of 2,048 tokens (cut C = window C // 4 at offset
(C % 4) x 2048, run as a fresh sequence from position 0), KL(official || candidate) over the full vocabulary at every
next-token row. This script takes the same segments (dataset, file, row id, rendering mode) from that dataset's
windows-8192.json, renders them with K3.1's tokenizer (`apply_chat_template` defaults for "thinking": K3.1's served
thinking-effort note; thinking=False for "chat"; plain text for "text", BOS at the window start), fills each window to
8,192 K3.1 tokens (when the listed segments are shorter, the following rows of the last segment's file, same mode),
and cuts it into four 2,048-token sequences.

    bm_eval.py WINDOWS_JSON SRC_DIR TOKENIZER_DIR OUT_PT

OUT_PT: {"tokens": int32 [512, 2048], "kinds": [domain per cut], "sources": [...], "windows": int32 [128, 8192],
"segments": per cut [(start, end, segment index, dataset, row, mode)], "protocol": ...}.
"""
import argparse
import json
import os

import torch

W_TOK, CUT = 8192, 2048


def src_name(ds, f):
    return f"{ds.replace('/', '__')}__{f.replace('/', '__')}"


class Rows:
    def __init__(self, src):
        self.src, self.cache = src, {}

    def get(self, ds, f, row):
        key = (ds, f)
        if key not in self.cache:
            path = os.path.join(self.src, src_name(ds, f))
            if f.endswith(".parquet"):
                import pyarrow.parquet as pq
                self.cache[key] = ("pq", pq.ParquetFile(path))
            else:
                self.cache[key] = ("jsonl", [json.loads(line) for line in open(path)])
        kind, obj = self.cache[key]
        if kind == "jsonl":
            return obj[row]
        # parquet: find the row group holding `row`
        start = 0
        for g in range(obj.num_row_groups):
            n = obj.metadata.row_group(g).num_rows
            if row < start + n:
                return obj.read_row_group(g).slice(row - start, 1).to_pylist()[0]
            start += n
        raise IndexError(f"{ds} {f} has no row {row}")


def messages_of(rec):
    if "trajectory" in rec:                              # nebius SWE-agent trajectories
        out = []
        for s in rec["trajectory"]:
            role = {"ai": "assistant"}.get(s["role"], s["role"])
            text = s.get("system_prompt") if s["role"] == "system" else s.get("text")
            out.append({"role": role, "content": text or ""})
        return out, None
    msgs = rec["messages"]
    tools = rec.get("tools") or None
    if tools:
        fixed = []
        for t in tools:
            t = dict(t)
            fn = dict(t.get("function", {}))
            if isinstance(fn.get("parameters"), str):
                fn["parameters"] = json.loads(fn["parameters"])
            t["function"] = fn
            fixed.append(t)
        tools = fixed
    clean = []
    for m in msgs:
        m = {k: v for k, v in m.items() if v is not None}
        if isinstance(m.get("tool_calls"), list):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        clean.append(m)
    return clean, tools


def render(tok, rec, mode):
    if mode == "text":
        return tok.encode(rec["text"], add_special_tokens=False)
    msgs, tools = messages_of(rec)
    kw = {} if mode == "thinking" else {"thinking": False}
    ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=False, **kw)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("windows_json"); p.add_argument("src"); p.add_argument("tokenizer"); p.add_argument("out")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    bos = tok.bos_token_id
    spec = json.load(open(a.windows_json))
    rows = Rows(a.src)
    windows, segs_w, kinds_w, extended = [], [], [], 0
    for w in spec["windows"]:
        ids, segs = [bos], []
        for si, s in enumerate(w["segments"]):
            if len(ids) >= W_TOK:
                break
            t = render(tok, rows.get(s["dataset"], s["file"], s["row_id"]), s["mode"])
            segs.append((len(ids), min(W_TOK, len(ids) + len(t)), si, s["dataset"], s["row_id"], s["mode"]))
            ids += t
        last = w["segments"][-1]
        r = last["row_id"]
        while len(ids) < W_TOK:                            # K3.1 renders shorter: the file's next rows, same mode
            r += 1
            t = render(tok, rows.get(last["dataset"], last["file"], r), last["mode"])
            segs.append((len(ids), min(W_TOK, len(ids) + len(t)), -1, last["dataset"], r, last["mode"]))
            ids += t
            extended += 1
        windows.append(ids[:W_TOK])
        segs_w.append(segs)
        kinds_w.append(w["domain"])
    win = torch.tensor(windows, dtype=torch.int32)
    cuts, kinds, sources, segments = [], [], [], []
    for wi in range(len(windows)):
        for c in range(W_TOK // CUT):
            o = c * CUT
            cuts.append(win[wi, o:o + CUT])
            kinds.append(kinds_w[wi])
            sources.append(f"bm-window {wi} cut {c}")
            segments.append([(max(0, s0 - o), min(CUT, s1 - o), si, ds, row, mode)
                             for (s0, s1, si, ds, row, mode) in segs_w[wi] if s1 > o and s0 < o + CUT])
    out = {"tokens": torch.stack(cuts), "kinds": kinds, "sources": sources, "windows": win, "segments": segments,
           "protocol": "brandonmusic DS-V4.1 teacher-logits windows-8192.json rebuilt for K3.1 (bm_eval.py): 128 "
                       "windows x 8192, headline = 512 cuts x 2048 run fresh; KL(official || candidate) full vocab"}
    torch.save(out, a.out)
    by = {}
    for k in kinds_w:
        by[k] = by.get(k, 0) + 1
    print(f"{len(windows)} windows, {len(cuts)} cuts of {CUT}; {extended} extra rows to fill windows; domains {by}")


if __name__ == "__main__":
    main()
