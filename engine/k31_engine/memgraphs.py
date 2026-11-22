"""K31_MEM_SNAPSHOT=1 (start-up; a measurement): where torch's reserved-but-unallocated memory sits.

Two things are logged at the start-up milestones of K31_MEM_REPORT (hook.py):
- per CUDA graph capture: who captured it (the first caller outside torch), its memory pool, and how much torch's
  reserved and allocated memory grew during the capture. A captured graph's intermediates live in the pool it was
  captured into; that memory stays reserved after the capture although no tensor holds it, and
  torch.cuda.empty_cache() cannot return it while the graph is alive. Captures into one pool reuse each other's
  blocks; captures into pools of their own add up.
- torch's memory snapshot by pool, stream and segment kind: segments, reserved, allocated, cached.
It also counts replays per graph (its capture's index), so the graphs that a workload never replays show.
Nothing here changes what is computed; the capture and replay wrappers cost a few attribute reads.
"""
import os
import sys
import traceback

import torch

_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731
_CAP: list = []            # (index, site, pool id, pool given, reserved delta, allocated delta)
_REPLAY: dict = {}         # capture index -> replays
_SEEN = [0]


def _site() -> str:
    for fr in reversed(traceback.extract_stack(limit=16)[:-2]):
        fn = fr.filename
        if "/torch/" in fn or os.path.basename(fn) == "memgraphs.py":
            continue
        return f"{os.path.basename(os.path.dirname(fn))}/{os.path.basename(fn)}:{fr.name}"
    return "?"


def install() -> None:
    G = torch.cuda.CUDAGraph
    if getattr(G, "_k31_mem", False):
        return
    ob, oe, orp = G.capture_begin, G.capture_end, G.replay

    def capture_begin(self, *a, **k):
        try:
            pool = k.get("pool", a[0] if a else None)
            self._k31_m = (torch.cuda.memory_reserved(), torch.cuda.memory_allocated(), _site(), pool is not None)
        except Exception:                                # noqa: BLE001
            self._k31_m = None
        return ob(self, *a, **k)

    def capture_end(self):
        out = oe(self)
        try:                                             # a measurement must never stop a capture
            m = getattr(self, "_k31_m", None)
            if m is not None:
                try:
                    pid = tuple(self.pool())
                except Exception:                        # noqa: BLE001
                    pid = None
                self._k31_i = len(_CAP)
                _CAP.append((len(_CAP), m[2], pid, m[3], torch.cuda.memory_reserved() - m[0],
                             torch.cuda.memory_allocated() - m[1]))
        except Exception:                                # noqa: BLE001
            pass
        return out

    def replay(self):
        i = getattr(self, "_k31_i", None)
        if i is not None:
            _REPLAY[i] = _REPLAY.get(i, 0) + 1
        return orp(self)

    replay._k31 = getattr(orp, "_k31", False)            # timing.py's mark, if its wrapper is underneath

    G.capture_begin, G.capture_end, G.replay = capture_begin, capture_end, replay
    G._k31_mem = True
    _LOG("K31_MEM_SNAPSHOT: CUDA graph captures and torch's memory snapshot are reported at the start-up milestones")


def report(tag: str) -> None:
    """The captures since the last report, by site and pool; then torch's snapshot by pool / stream / kind."""
    m = 2 ** 20
    new = _CAP[_SEEN[0]:]
    _SEEN[0] = len(_CAP)
    by: dict = {}
    for i, site, pid, given, dr, da in new:
        a = by.setdefault((site, pid, given), [0, 0, 0, i, i])
        a[0] += 1
        a[1] += dr
        a[2] += da
        a[4] = i
    for (site, pid, given), a in sorted(by.items(), key=lambda kv: kv[1][3]):
        _LOG(f"graph captures {tag}: {a[0]} by {site} into pool {pid}{' (a pool passed in)' if given else ' (its own)'}"
             f", captures {a[3]}-{a[4]}: reserved +{a[1] / m:.0f} MiB, allocated {a[2] / m:+.0f} MiB")
    big = sorted(new, key=lambda r: -r[4])[:8]
    if big:
        _LOG(f"graph captures {tag}, the largest: " + "; ".join(f"#{i} {site} +{dr / m:.0f} MiB" for i, site, _, _, dr, _ in big))
    try:
        snap = torch.cuda.memory_snapshot()
    except Exception as ex:                              # noqa: BLE001
        _LOG(f"memory snapshot {tag}: not available ({type(ex).__name__}: {ex})")
        return
    seg: dict = {}
    for s in snap:
        key = (tuple(s.get("segment_pool_id", (0, 0))), s.get("stream", 0), s.get("segment_type", "?"))
        a = seg.setdefault(key, [0, 0, 0, 0])
        a[0] += 1
        a[1] += s.get("total_size", 0)
        a[2] += s.get("allocated_size", 0)
        a[3] += s.get("active_size", 0)
    tot = [sum(a[k] for a in seg.values()) for k in range(4)]
    _LOG(f"memory snapshot {tag}: {tot[0]} segments, reserved {tot[1] / m:.0f} MiB, allocated {tot[2] / m:.0f} MiB, "
         f"cached {(tot[1] - tot[2]) / m:.0f} MiB; by pool / stream / kind (reserved over 16 MiB):")
    pools: dict = {}
    for (pid, stream, kind), a in sorted(seg.items(), key=lambda kv: -kv[1][1]):
        p = pools.setdefault(pid, [0, 0])
        p[0] += a[1]
        p[1] += a[2]
        if a[1] >= 16 * m:
            _LOG(f"  pool {pid} stream {stream:#x} {kind}: {a[0]} segments, reserved {a[1] / m:.0f} MiB, allocated "
                 f"{a[2] / m:.0f} MiB, cached {(a[1] - a[2]) / m:.0f} MiB")
    # the unallocated blocks inside the segments, by size: many small holes between live tensors cannot be returned
    # (a free range goes back to the device only in whole mapping units), a few large ones can
    try:
        edges = (1, 4, 20, 100)
        hist, big, exp = [[0, 0] for _ in range(len(edges) + 1)], 0, 0
        for sg in snap:
            exp += 1 if sg.get("is_expandable") else 0
            for b in sg.get("blocks", ()):
                if b.get("state") != "inactive":
                    continue
                sz = b.get("size", 0)
                k = sum(sz >= e * m for e in edges)
                hist[k][0] += 1
                hist[k][1] += sz
                big = max(big, sz)
        names = ("under 1 MiB", "1-4", "4-20", "20-100", "100 MiB and more")
        _LOG(f"memory snapshot {tag}: free blocks inside the segments ({exp} of {len(snap)} segments expandable): "
             + "; ".join(f"{nm}: {c} blocks, {t / m:.0f} MiB" for nm, (c, t) in zip(names, hist))
             + f"; the largest {big / m:.0f} MiB")
    except Exception as ex:                              # noqa: BLE001
        _LOG(f"memory snapshot {tag}: free-block sizes not read ({type(ex).__name__}: {ex})")
    priv = {pid: v for pid, v in pools.items() if tuple(pid) != (0, 0)}
    _LOG(f"memory snapshot {tag}: the default pool reserves {pools.get((0, 0), [0, 0])[0] / m:.0f} MiB (allocated "
         f"{pools.get((0, 0), [0, 0])[1] / m:.0f}); {len(priv)} graph pools reserve {sum(v[0] for v in priv.values()) / m:.0f} "
         f"MiB (allocated {sum(v[1] for v in priv.values()) / m:.0f})")


def replays() -> None:
    """The replays per captured graph so far (the captures never replayed are the candidates to drop)."""
    if not _CAP:
        return
    m = 2 ** 20
    never = [r for r in _CAP if _REPLAY.get(r[0], 0) == 0]
    _LOG(f"graph replays so far: {sum(_REPLAY.values())} of {len(_CAP)} captured graphs, {len(never)} never replayed "
         f"(their captures reserved +{sum(r[4] for r in never) / m:.0f} MiB in all); replays by site: "
         + "; ".join(f"{site} {n}" for site, n in sorted(
             {s: sum(_REPLAY.get(r[0], 0) for r in _CAP if r[1] == s) for s in {r[1] for r in _CAP}}.items())))
