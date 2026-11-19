"""Decode speed of a served engine at one request: streams PROMPTS in turn, greedy, and reports tokens per second
after the first token (median over runs), as tomcupr's vbench does. Prompts are chat messages (the server applies
K3.1's template); reasoning output counts as tokens.
    bench.py URL [--runs 6] [--max-tokens 256]"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = [
    "Explain how a B-tree keeps itself balanced during inserts, with a small worked example.",
    "Write a Python function that parses ISO 8601 durations like P3DT4H5M into seconds, with tests.",
    "Summarize the causes and consequences of the 1929 stock market crash in about 300 words.",
]


def one(url, prompt, max_tokens):
    body = {"model": "derpk31pro", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning_content")
                                     or d["choices"][0]["delta"].get("reasoning")):
                chunks += 1
                if first is None:
                    first = time.time()
    t1 = time.time()
    n = usage["completion_tokens"] if usage else chunks
    return n, first - t0, (n - 1) / (t1 - first) if first and t1 > first else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url"); p.add_argument("--runs", type=int, default=6); p.add_argument("--max-tokens", type=int, default=256)
    a = p.parse_args()
    one(a.url, "Hello", 8)                                   # warm
    rates = []
    for i in range(a.runs):
        n, ttft, rate = one(a.url, PROMPTS[i % len(PROMPTS)], a.max_tokens)
        rates.append(rate)
        print(f"run {i}: {n} tokens, first after {ttft:.2f} s, {rate:.1f} tok/s", flush=True)
    print(f"median {statistics.median(rates):.1f} tok/s over {a.runs} runs (min {min(rates):.1f}, max {max(rates):.1f})")


if __name__ == "__main__":
    main()
