"""K31_PROMPT_PROF=1 (runtime flag, flags.py; off by default): where a prompt chunk's time goes, read on the GPU's own
clock without a profiler.

A prompt forward (an eager step of more than 16 rows) marks five points of every MoE layer on the main stream with
timed CUDA events:
  0  the layer's expert step begins (attention, norms, router and latent down-projection are queued before it)
  1  after the residency's in-forward tick (host work; the stream idles through it)
  2  after the plan and the layer's host sync (the staging rounds start here)
  3  after the last round's MegaMoE call
  4  the expert step ends (band result added, cut picks' means added)
and the step's first and last points. At the end of the step the stream is synchronized once (this mode only) and one
line is logged per step:
  rows, the step's time on the stream, and its split: dense (point 4 of one layer to point 0 of the next: the previous
  layer's tail, this layer's attention and dense front), tick (0 -> 1), plan (1 -> 2), rounds (2 -> 3), tail (3 -> 4);
  the layers' rounds and staged records; the host's wall time since the previous prompt step began; the step's peak
  of torch's allocator (allocated, reserved) and the device's free memory after it.
Every K31_PROMPT_PROF_SUM steps (default 0 = never) a summary by layer class is logged too: layers with a tick, DSA
layers, the rest.
The marks cost one event record each (about 460 per forward); the synchronize at the end of a step is the mode's only
change to the forward's timing (a prompt step ends with nothing queued behind it, so it moves no work).
"""
import sys
import time

import torch

from . import flags

_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
_S: dict = {"on": False, "marks": [], "rows": 0, "t0": None, "prev_t0": None, "rounds": 0, "records": 0, "steps": 0,
            "first": None, "sum": {}}
NAMES = ("tick", "plan", "rounds", "tail")


def on() -> bool:
    return _S["on"]


def step_begin(rows: int) -> None:
    """Called before a step's forward: arms the marks for a prompt step when the flag is on."""
    want = rows > 16 and flags.on("K31_PROMPT_PROF")
    _S["on"] = want
    if not want:
        _S["prev_t0"] = None
        return
    _S["marks"] = []
    _S["rows"] = rows
    _S["rounds"] = _S["records"] = 0
    _S["t0"] = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()             # the step's own peak (torch's allocator)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream())
        _S["first"] = ev
    except Exception as ex:                              # noqa: BLE001  (a timer must never stop a step)
        _S["on"] = False
        _LOG(f"prompt prof: off for this step ({type(ex).__name__}: {ex})")


def mark(layer: int, point: int) -> None:
    try:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream())
        _S["marks"].append((layer, point, ev))
    except Exception:                                    # noqa: BLE001
        _S["on"] = False


def count(rounds: int, records: int) -> None:
    _S["rounds"] += rounds
    _S["records"] += records


def _memory() -> str:
    """The step's memory: torch's peak allocated and reserved, and what the device has free now, in MiB."""
    try:
        free, total = torch.cuda.mem_get_info()
        m = 2 ** 20
        return (f"; memory MiB: torch peak allocated {torch.cuda.max_memory_allocated() / m:.0f}, peak reserved "
                f"{torch.cuda.max_memory_reserved() / m:.0f}, device free now {free / m:.0f} of {total / m:.0f}")
    except Exception:                                    # noqa: BLE001
        return ""


def step_end() -> None:
    """After the step's forward: wait for the stream, read the marks, log the step's line. Never raises."""
    if not _S["on"]:
        return
    _S["on"] = False
    try:
        _step_end()
    except Exception as ex:                              # noqa: BLE001  (a timer must never stop a step)
        _LOG(f"prompt prof: this step's line is lost ({type(ex).__name__}: {ex})")


def _step_end() -> None:
    marks = _S["marks"]
    if not marks:
        return
    last = torch.cuda.Event(enable_timing=True)
    s = torch.cuda.current_stream()
    last.record(s)
    last.synchronize()
    t_host = time.perf_counter()
    by = {}
    for layer, point, ev in marks:
        by.setdefault(layer, {})[point] = ev
    layers = sorted(by)
    tot = dict.fromkeys(("dense",) + NAMES, 0.0)
    per = []
    prev_end = _S["first"]
    for L in layers:
        p = by[L]
        if len(p) < 5:
            continue
        d = prev_end.elapsed_time(p[0])
        seg = [p[i].elapsed_time(p[i + 1]) for i in range(4)]
        tot["dense"] += d
        for n, v in zip(NAMES, seg):
            tot[n] += v
        per.append((L, d, seg))
        prev_end = p[4]
    total = _S["first"].elapsed_time(last)
    closing = prev_end.elapsed_time(last)
    wall = None if _S["prev_t0"] is None else 1e3 * (_S["t0"] - _S["prev_t0"])
    _S["prev_t0"] = _S["t0"]
    _S["steps"] += 1
    _LOG(f"prompt prof: {_S['rows']} rows, {total:.1f} ms on the stream = dense {tot['dense']:.1f} + tick "
         f"{tot['tick']:.1f} + plan {tot['plan']:.1f} + rounds {tot['rounds']:.1f} + tail {tot['tail']:.1f} + after "
         f"the last layer {closing:.1f}; {len(per)} layers, {_S['rounds']} rounds, {_S['records']} records; host: "
         f"forward returned after {1e3 * (t_host - _S['t0']):.1f} ms"
         + (f", {wall:.1f} ms since the previous prompt step began" if wall is not None else "") + _memory())
    every = flags.num("K31_PROMPT_PROF_SUM", 0)
    if every > 0:
        acc = _S["sum"]
        for L, d, seg in per:
            kind = "tick" if L % 4 == 1 else ("dsa" if L % 4 == 3 else "kda")
            a = acc.setdefault(kind, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            a[0] += 1
            a[1] += d
            for i, v in enumerate(seg):
                a[2 + i] += v
        if _S["steps"] % every == 0:
            for kind, a in sorted(acc.items()):
                n = max(1, a[0])
                _LOG(f"prompt prof, {kind} layers over {every} steps ({a[0]} layer steps), ms each: dense "
                     f"{a[1] / n:.2f}, tick {a[2] / n:.2f}, plan {a[3] / n:.2f}, rounds {a[4] / n:.2f}, tail "
                     f"{a[5] / n:.2f}")
            acc.clear()
