"""On-policy capture: sample answers from the served engine and keep exactly what it produced.

    capture.py URL PROMPTS.jsonl OUT.jsonl [--concurrency 4] [--max-tokens 32768] [--temperature 1] [--top-p 0.95]
               [--reasoning-effort E] [--first N] [--skip N] [--stop-file F] [--deadline EPOCH] [--no-stream] [--smoke]

Each line of PROMPTS.jsonl (prompts.py) is sent as one chat completion (messages, tools if any, its own seed), streamed,
with `logprobs` (the sampled token's raw log-probability and the engine's top-1), `return_tokens_as_token_ids` and
`return_token_ids`, so the record holds the exact token ids of the rendered prompt and of everything generated
(reasoning included), not re-tokenized text. One JSON line per finished request goes to OUT.jsonl:

  id, kind, sample, seed, prompt_ids, ids (generated), lp (log-prob of each generated token), top_id / top_lp (the
  engine's most likely token at that step), finish_reason, usage, reasoning / content (text), seconds, ttft,
  checks (the record's self-consistency), loop (loop_scan below), meta, error, settings (temperature, top_p,
  max_tokens, reasoning_effort, in_flight: what this capture was sampled with).

Checks per record: generated ids == usage.completion_tokens, one log-prob entry per generated token naming the same
token id, prompt ids == usage.prompt_tokens. --smoke sends the first request only (max 256 tokens), prints the checks
and exits 0 only if they all hold (try --no-stream if the stream fails them).

Requests already in OUT.jsonl (same id, no error) are skipped, so a capture can be resumed or extended. --stop-file:
no new request starts once the file exists (running ones finish). Standard library only.
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse


def _ngram_lags(ids, n):
    """lag[i] = distance from position i back to the previous end of the same n-gram (0 = not seen before)."""
    L = len(ids)
    last = {}
    lag = [0] * L
    key = tuple(ids[:n])
    for i in range(n - 1, L):
        if i >= n:
            key = key[1:] + (ids[i],)
        j = last.get(key)
        if j is not None:
            lag[i] = i - j
        last[key] = i
    return lag


def loop_scan(ids, n_rep=16, n_lag=4, win=512, thr_rep=0.8, thr_lag=0.5, min_len=1024, rounds=3, step=64):
    """Repetition scan of a token sequence. A window of `win` tokens is repetitive when at least thr_rep of its
    positions end an n_rep-gram seen before (verbatim repetition), or when one single lag (the distance back to the
    previous occurrence of the n_lag-gram ending there) covers at least thr_lag of its positions (a cycle of fixed
    length, even with some tokens changing each round). A loop is the first stretch of repetitive windows (stride
    `step`) that is at least min_len tokens long and at least `rounds` times its period (text quoted once or twice
    is not a loop). Returns None, or {"rep_start": the first position of the stretch that repeats at the period,
    "period": the stretch's most common lag, "loop_start": rep_start - n_lag + 1 - period clipped at 0 (where the
    first copy of the repeated text begins), "end": the stretch's end, "to_end": it runs to the sequence's end,
    "frac": the share of the stretch's positions that end a repeated n_rep-gram, "lag_frac": the share at the
    period's lag}."""
    L = len(ids)
    if L < max(min_len, win, n_rep):
        return None
    lag = _ngram_lags(ids, n_lag)
    rep = [1 if d else 0 for d in _ngram_lags(ids, n_rep)]
    pre = [0] * (L + 1)
    for i in range(L):
        pre[i + 1] = pre[i] + rep[i]
    starts = list(range(0, L - win + 1, step))
    if starts[-1] != L - win:
        starts.append(L - win)

    def lags(a, b):
        cnt = {}
        for d in lag[a:b]:
            if d:
                cnt[d] = cnt.get(d, 0) + 1
        return cnt

    def repetitive(a):
        if pre[a + win] - pre[a] >= thr_rep * win:
            return True
        cnt = lags(a, a + win)
        return bool(cnt) and max(cnt.values()) >= thr_lag * win

    run_a = prev = None
    for a in starts + [None]:
        ok = a is not None and repetitive(a)
        if ok and run_a is None:
            run_a = a
        if not ok and run_a is not None:
            b = prev + win
            cnt = lags(run_a, b)
            period = max(cnt, key=cnt.get) if cnt else 0
            if period and b - run_a >= max(min_len, rounds * period):
                first = next(i for i in range(run_a, b) if lag[i] == period)
                return {"rep_start": first, "period": period, "loop_start": max(0, first - n_lag + 1 - period),
                        "end": b, "to_end": b >= L, "frac": (pre[b] - pre[run_a]) / (b - run_a),
                        "lag_frac": cnt[period] / (b - run_a)}
            run_a = None
        prev = a
    return None


def _tid(s):
    return int(s.split(":", 1)[1]) if isinstance(s, str) and s.startswith("token_id:") else None


def _take_logprobs(lp, rec):
    for e in (lp or {}).get("content") or []:
        rec["lp_ids"].append(_tid(e.get("token")))
        rec["lp"].append(e.get("logprob"))
        top = e.get("top_logprobs") or []
        if top:
            best = max(top, key=lambda t: t.get("logprob", -1e9))
            rec["top_id"].append(_tid(best.get("token")))
            rec["top_lp"].append(best.get("logprob"))
        else:
            rec["top_id"].append(None)
            rec["top_lp"].append(None)


def request(url, item, a, max_tokens):
    u = urllib.parse.urlparse(url)
    body = {"model": a.model, "messages": item["messages"], "max_tokens": max_tokens, "temperature": a.temperature,
            "top_p": a.top_p, "seed": item["seed"], "logprobs": True, "top_logprobs": 1,
            "return_tokens_as_token_ids": True, "return_token_ids": True, "stream": not a.no_stream}
    if item.get("tools"):
        body["tools"] = item["tools"]
    if a.reasoning_effort:
        body["reasoning_effort"] = a.reasoning_effort
    if not a.no_stream:
        body["stream_options"] = {"include_usage": True}
    rec = {"id": item["id"], "kind": item["kind"], "sample": item.get("sample", 0), "seed": item["seed"],
           "prompt_ids": None, "ids": [], "lp_ids": [], "lp": [], "top_id": [], "top_lp": [], "finish_reason": None,
           "usage": None, "reasoning": "", "content": "", "tool_calls": 0, "meta": item.get("meta"), "error": None,
           "settings": {"temperature": a.temperature, "top_p": a.top_p, "max_tokens": max_tokens,
                        "reasoning_effort": a.reasoning_effort or None, "in_flight": a.concurrency}}
    t0 = time.time()
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=a.stall)
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            rec["error"] = f"HTTP {r.status}: {r.read(600).decode(errors='replace')}"
            return rec
        if a.no_stream:
            d = json.loads(r.read())
            ch = d["choices"][0]
            rec["prompt_ids"] = d.get("prompt_token_ids")
            rec["ids"] = list(ch.get("token_ids") or [])
            _take_logprobs(ch.get("logprobs"), rec)
            m = ch.get("message") or {}
            rec["reasoning"] = m.get("reasoning") or m.get("reasoning_content") or ""
            rec["content"] = m.get("content") or ""
            rec["tool_calls"] = len(m.get("tool_calls") or [])
            rec["finish_reason"] = ch.get("finish_reason")
            rec["usage"] = d.get("usage")
        else:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):
                    rec["error"] = json.dumps(d["error"])[:600]
                    break
                if d.get("prompt_token_ids") is not None and rec["prompt_ids"] is None:
                    rec["prompt_ids"] = d["prompt_token_ids"]
                if d.get("usage"):
                    rec["usage"] = d["usage"]
                for ch in d.get("choices") or []:
                    if ch.get("token_ids"):
                        if not rec["ids"]:
                            rec["ttft"] = time.time() - t0
                        rec["ids"] += ch["token_ids"]
                    _take_logprobs(ch.get("logprobs"), rec)
                    dl = ch.get("delta") or {}
                    rec["reasoning"] += dl.get("reasoning") or dl.get("reasoning_content") or ""
                    rec["content"] += dl.get("content") or ""
                    rec["tool_calls"] += len(dl.get("tool_calls") or [])
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
    except Exception as e:                                   # noqa: BLE001 (recorded; the caller decides)
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    rec["seconds"] = time.time() - t0
    return rec


def checks(rec):
    us = rec.get("usage") or {}
    n = len(rec["ids"])
    c = {"has_prompt_ids": bool(rec["prompt_ids"]),
         "prompt_len": bool(rec["prompt_ids"]) and len(rec["prompt_ids"]) == us.get("prompt_tokens"),
         "completion_len": n > 0 and n == us.get("completion_tokens"),
         "one_logprob_per_token": len(rec["lp"]) == n and all(v is not None for v in rec["lp"]),
         "logprob_ids_match": rec["lp_ids"] == rec["ids"],
         "finished": rec["finish_reason"] is not None and rec["error"] is None}
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url"); p.add_argument("prompts"); p.add_argument("out")
    p.add_argument("--model", default="kimi-k3.1-pro")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--reasoning-effort", default="", help="sent as `reasoning_effort` (empty: the server's default)")
    p.add_argument("--skip", type=int, default=0, help="skip the first N prompt lines")
    p.add_argument("--first", type=int, default=0, help="only the first N prompt lines after --skip (0 = all)")
    p.add_argument("--stop-file", default="", help="start no new request once this file exists")
    p.add_argument("--deadline", type=float, default=0, help="start no new request after this epoch time")
    p.add_argument("--stall", type=float, default=1800, help="seconds without a byte before a request is given up")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    items = [json.loads(line) for line in open(a.prompts)][a.skip:]
    if a.first:
        items = items[:a.first]
    if a.smoke:
        rec = request(a.url, items[0], a, 256)
        c = checks(rec)
        c["finished"] = rec["error"] is None and rec["finish_reason"] is not None
        print(json.dumps({"checks": c, "error": rec["error"], "finish_reason": rec["finish_reason"],
                          "usage": rec["usage"], "prompt_ids": len(rec["prompt_ids"] or []), "ids": len(rec["ids"]),
                          "lp_head": rec["lp"][:4], "ids_head": rec["ids"][:8], "seconds": rec.get("seconds"),
                          "reasoning_chars": len(rec["reasoning"]), "content_chars": len(rec["content"])}))
        sys.exit(0 if all(c.values()) else 1)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not d.get("error"):
                done.add(d["id"])
    todo = [it for it in items if it["id"] not in done]
    lock = threading.Lock()
    state = {"next": 0, "ok": 0, "err": 0, "tokens": 0, "loops": 0, "bad_checks": 0, "capped": 0, "t0": time.time()}
    out = open(a.out, "a")

    def log(msg):
        sys.stderr.write(f"{time.strftime('%F %T')} {msg}\n")
        sys.stderr.flush()

    def worker():
        while True:
            with lock:
                if state["next"] >= len(todo):
                    return
                if (a.stop_file and os.path.exists(a.stop_file)) or (a.deadline and time.time() > a.deadline):
                    return
                it = todo[state["next"]]
                state["next"] += 1
            rec = request(a.url, it, a, a.max_tokens)
            rec["checks"] = checks(rec)
            rec["loop"] = loop_scan(rec["ids"]) if rec["ids"] else None
            del rec["lp_ids"]
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                good = all(rec["checks"].values())
                state["ok" if not rec["error"] else "err"] += 1
                state["tokens"] += len(rec["ids"])
                state["loops"] += rec["loop"] is not None
                state["bad_checks"] += not good
                state["capped"] += rec["finish_reason"] == "length"
                lp = rec["loop"]
                log(f"{rec['id']}: {len(rec['ids'])} tokens, {rec['finish_reason']}, {rec.get('seconds', 0):.0f} s"
                    + (f", LOOP from {lp['loop_start']} (period {lp['period']})" if lp else "")
                    + ("" if good else f", CHECKS FAILED {[k for k, v in rec['checks'].items() if not v]}")
                    + (f", ERROR {rec['error'][:200]}" if rec["error"] else "")
                    + f" | done {state['ok']} of {len(todo)}, errors {state['err']}, loops {state['loops']}, "
                      f"capped {state['capped']}, {state['tokens']} tokens, {time.time() - state['t0']:.0f} s")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"capture done: {state['ok']} ok, {state['err']} errors, {state['bad_checks']} with failed checks, "
        f"{state['loops']} loops, {state['capped']} at the token cap, {state['tokens']} tokens, "
        f"{time.time() - state['t0']:.0f} s; not started {len(todo) - state['next']}")


if __name__ == "__main__":
    main()
