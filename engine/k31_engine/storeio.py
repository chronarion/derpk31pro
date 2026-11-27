"""Record reads from the model drive, their timing, a start-up check of the drive, and a warning when the drive holds
the engine back.

Reads. Every hi2 / lo2 record the engine and the band sidecar read while serving goes through `Reader.read`:
  K31_ODIRECT=1 (default)  O_DIRECT: the whole record goes to the drive at once, whatever the host's read-ahead
                           settings, and nothing lands in the page cache. Used when the buffer, offset and length are
                           4 KiB-aligned (records are 4 KiB multiples; pinned buffers are page-aligned).
  otherwise, or when the file system refuses O_DIRECT (at open or at read: EINVAL, EFAULT, EOPNOTSUPP), buffered reads:
                           POSIX_FADV_RANDOM on the file (K31_FADV_RANDOM=1: no read-ahead past a record) and a
                           POSIX_FADV_WILLNEED per 128 KiB piece of the record before one pread. Linux serves a
                           FADV_RANDOM read in pieces of max(read_ahead_kb, max_sectors_kb), one after another (512 KiB
                           and ~17 waits per 8 MiB record on a stock host); the per-piece hints put them all in flight.
Each read's duration is kept for the residency's 30 s line and the runtime warning (`window()`).

Start-up check (`check()`, K31_STORE_CHECK=1 by default): before the model loads its experts, a few dozen records
are read cold, mostly records the engine will read often, the way the engine reads them: one at a time, then with the
promotion workers in parallel. It runs on its own thread and gives up after K31_STORE_CHECK_S (default 3.5) seconds on
any drive; it never stops the start. One log line with the numbers and the drive's settings, and a plain-language
warning when the drive is slow enough to matter.
K31_STORE_TEST_DELAY_MS (testing): each serving record read takes this much longer, a stand-in for a slow drive.
"""
from __future__ import annotations

import errno
import mmap
import os
import random
import threading
import time

ODIRECT = os.environ.get("K31_ODIRECT", "1") == "1"
FADV_RANDOM = os.environ.get("K31_FADV_RANDOM", "0") == "1"
PIECE = int(os.environ.get("K31_READ_PIECE_KB", "128") or 0) * 1024     # buffered reads: WILLNEED piece size
ALIGN = 4096
TEST_DELAY = float(os.environ.get("K31_STORE_TEST_DELAY_MS", "0") or 0) / 1e3
# Rates of the parallel reads (GB/s) below which the start-up check and the runtime warning speak up. A Gen5 NVMe drive
# reads ~10-14 GB/s this way, a good Gen4 drive ~6-7.
NOTE_GBS = float(os.environ.get("K31_STORE_NOTE_GBS", "6"))
WARN_GBS = float(os.environ.get("K31_STORE_WARN_GBS", "3"))
BAD_GBS = 1.0
WARN_MS = float(os.environ.get("K31_STORE_WARN_MS", "15"))   # one cold record (median) slower than this: warning

_DIRECT_OK = [True]                                             # False once the file system refused O_DIRECT
_LOCK = threading.Lock()
_ACC = {"n": 0, "s": 0.0, "bytes": 0, "times": []}            # since the last window(); times capped at 100,000


def _log(msg: str) -> None:
    try:
        from .hook import _LOG
        _LOG(msg)
    except Exception:                                           # noqa: BLE001
        print(f"K31 {msg}", flush=True)


def _direct_refused(path: str, ex: OSError) -> None:
    if _DIRECT_OK[0]:
        _DIRECT_OK[0] = False
        _log(f"store reads: the file system refused O_DIRECT for {path} ({ex.strerror}); buffered reads from now on")


class Reader:
    """One thread's file descriptors (buffered and O_DIRECT per path). Not shared between threads."""

    def __init__(self, direct: bool | None = None, timed: bool = True):
        self.direct = ODIRECT if direct is None else direct
        self.timed = timed
        self.fds: dict = {}

    def open(self, path: str, direct: bool) -> int:
        key = (path, direct)
        fd = self.fds.get(key)
        if fd is None:
            if direct:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            else:
                fd = os.open(path, os.O_RDONLY)
                if FADV_RANDOM:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self.fds[key] = fd
        return fd

    def use_direct(self, arr, pos: int) -> bool:
        return self.direct and _DIRECT_OK[0] and arr.ctypes.data % ALIGN == 0 and pos % ALIGN == 0 and \
            arr.nbytes % ALIGN == 0

    def read(self, path: str, arr, pos: int) -> int:
        """Fill numpy uint8 array `arr` (contiguous) from `path` at byte `pos`. Returns the bytes read."""
        n = arr.nbytes
        t0 = time.perf_counter()
        view = memoryview(arr).cast("B")
        done = False
        if self.use_direct(arr, pos):
            try:
                _read_all(self.open(path, True), view, pos, path)
                done = True
            except OSError as ex:
                if ex.errno not in (errno.EINVAL, errno.EFAULT, errno.EOPNOTSUPP):
                    raise
                _direct_refused(path, ex)
        if not done:
            fd = self.open(path, False)
            if PIECE > 0:                                       # all pieces in flight at once
                for off in range(0, n, PIECE):
                    os.posix_fadvise(fd, pos + off, min(PIECE, n - off), os.POSIX_FADV_WILLNEED)
            _read_all(fd, view, pos, path)
        if TEST_DELAY > 0 and self.timed:
            time.sleep(TEST_DELAY)
        if self.timed:
            dt = time.perf_counter() - t0
            with _LOCK:
                _ACC["n"] += 1
                _ACC["s"] += dt
                _ACC["bytes"] += n
                if len(_ACC["times"]) < 100000:
                    _ACC["times"].append(dt)
        return n

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _read_all(fd: int, view, pos: int, path: str) -> None:
    while view:
        k = os.preadv(fd, [view], pos)
        if k <= 0:
            raise IOError(f"short read in {path} at {pos}")
        view, pos = view[k:], pos + k


def mode() -> str:
    if ODIRECT and _DIRECT_OK[0]:
        return "O_DIRECT"
    return "buffered" + (" + FADV_RANDOM" if FADV_RANDOM else "") + (f" + WILLNEED {PIECE >> 10} KiB" if PIECE else "")


def window() -> dict:
    """The record reads since the last call: count, read seconds summed, bytes, median / p90 / max per record."""
    with _LOCK:
        acc = dict(_ACC)
        _ACC.update(n=0, s=0.0, bytes=0, times=[])
    ts = sorted(acc["times"])
    if ts:
        acc["p50"], acc["p90"], acc["max"] = ts[len(ts) // 2], ts[min(len(ts) - 1, int(len(ts) * 0.9))], ts[-1]
    return acc


# ------------------------------------------------------------------------------------------------- the drive
def _sys(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _fs_type(path: str) -> str:
    """The file system type of the mount holding `path` (longest matching mount point in /proc/mounts)."""
    best, fst = "", ""
    real = os.path.realpath(path)
    try:
        with open("/proc/mounts") as f:
            for line in f:
                p = line.split()
                if len(p) >= 3:
                    mp = p[1].replace("\\040", " ")
                    if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best):
                        best, fst = mp, p[2]
    except OSError:
        pass
    return fst


def drive_info(path: str) -> dict:
    """The block device under `path`, its queue settings and temperature (sysfs; read-only in a container too)."""
    info = {"fs": _fs_type(path)}
    try:
        st = os.stat(path)
    except OSError:
        return info
    dev = os.path.realpath(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    if not os.path.isdir(dev):
        return info
    if os.path.exists(os.path.join(dev, "partition")):
        dev = os.path.dirname(dev)
    info["name"] = os.path.basename(dev)
    info["model"] = _sys(os.path.join(dev, "device", "model"))
    q = os.path.join(dev, "queue")
    for k in ("read_ahead_kb", "max_sectors_kb", "rotational"):
        info[k] = _sys(os.path.join(q, k))
    if info["name"].startswith("nvme"):
        ctrl = os.path.realpath(os.path.join(dev, "device"))
        info["apst_us"] = _sys(os.path.join(ctrl, "power", "pm_qos_latency_tolerance_us"))
        try:
            for h in sorted(os.listdir(ctrl)):
                if h.startswith("hwmon"):
                    t = _sys(os.path.join(ctrl, h, "temp1_input"))
                    if t:
                        info["temp_c"] = int(t) // 1000
                    break
        except OSError:
            pass
    return info


def describe(di: dict) -> str:
    if not di.get("name"):
        return f"file system {di.get('fs') or '?'}, device unknown"
    s = f"{di['name']}" + (f" ({di['model']})" if di.get("model") else "") + f", {di.get('fs') or '?'}"
    s += f", read_ahead_kb {di.get('read_ahead_kb', '?')}, max_sectors_kb {di.get('max_sectors_kb', '?')}"
    if "temp_c" in di:
        s += f", {di['temp_c']} C"
    return s


def advice(di: dict, gbs: float, first_ms: float = 0.0) -> str:
    """Plain-language causes and fixes for a slow model drive, most likely first."""
    tips = ["another program reading or writing the same drive (copies, downloads, backups) takes its bandwidth"]
    if di.get("fs") in ("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs") or (di.get("fs") or "").startswith("fuse"):
        tips.insert(0, f"the store is on a network or FUSE file system ({di['fs']}): put it on a local NVMe drive")
    elif not di.get("name") or not di["name"].startswith("nvme"):
        tips.insert(0, "the store does not sit directly on an NVMe drive (RAID, SATA, USB or a virtual disk): a "
                       "single local PCIe Gen4/Gen5 NVMe drive works best")
    if "temp_c" in di and di["temp_c"] >= 70:
        tips.insert(0, f"the drive is hot ({di['temp_c']} C) and may be slowing itself down: improve its cooling")
    tips.append("on some consumer drives, data that has been read very many times reads slower until it is "
                "rewritten: copying the record store to a new folder (or re-running the store build) restores the speed")
    if gbs < NOTE_GBS:
        tips.append("a PCIe Gen4 drive reads about half as fast as Gen5 here, a SATA drive far slower")
    if first_ms > 20 and di.get("apst_us") not in (None, "", "0"):
        tips.append("NVMe power saving delays the first reads after an idle period (harmless for steady serving; "
                    "writing 0 to /sys/class/nvme/<drive>/power/pm_qos_latency_tolerance_us turns it off)")
    return "; ".join(f"({i + 1}) {t}" for i, t in enumerate(tips))


# ------------------------------------------------------------------------------------------------- start-up check
def check(store: str, hot: list, cold: list, workers: int) -> dict:
    """Time-boxed check of the model drive. `hot`: (path, offset, size) of records the engine will read often (the
    banked experts' records); `cold`: (path, record size) for random records. Runs on a daemon thread; returns what
    finished within K31_STORE_CHECK_S seconds (never raises)."""
    if os.environ.get("K31_STORE_CHECK", "1") != "1":
        return {}
    budget = float(os.environ.get("K31_STORE_CHECK_S", "3.5"))
    out: dict = {"done": False}
    th = threading.Thread(target=_check_body, args=(store, hot, cold, workers, budget, out), daemon=True,
                          name="k31-store-check")
    t0 = time.perf_counter()
    th.start()
    th.join(budget + 0.5)
    try:
        _report(store, out, workers, time.perf_counter() - t0, budget)
    except Exception as ex:                                     # noqa: BLE001
        _log(f"store check: report failed ({type(ex).__name__}: {ex})")
    return out


def _check_body(store, hot, cold, workers, budget, out):
    try:
        import numpy as np
        rng = random.Random()
        picks = []
        nh = min(len(hot), 72)
        picks += rng.sample(hot, nh) if nh else []
        for _ in range(109 - len(picks)):                       # a third or more from anywhere in the store
            if not cold:
                break
            f, R = rng.choice(cold)
            size = os.path.getsize(f)
            if size >= R:
                picks.append((f, rng.randrange(size // R) * R, R))
        rng.shuffle(picks)
        if len(picks) < 2:
            out["error"] = "no records to read"
            return
        for f, pos, R in picks:                                 # cold: drop these records' cached pages (no root)
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, pos, R, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        Rmax = max(R for _, _, R in picks)
        nb = max(1, workers)
        bufs = [np.frombuffer(mmap.mmap(-1, Rmax), dtype=np.uint8) for _ in range(nb)]
        for b in bufs:                                          # first touch outside the timing
            b.fill(0)
        rd = Reader(timed=False)
        for f, _, _ in picks:                                   # opens outside the timing
            rd.open(f, rd.direct and _DIRECT_OK[0])
        f, pos, R = picks.pop()
        t = time.perf_counter()
        rd.read(f, bufs[0][:R], pos)                            # wakes the drive from a power-saving state
        out["first_ms"] = 1e3 * (time.perf_counter() - t)
        t_end = time.perf_counter() + budget * 0.35             # one record at a time: latency
        lat = []
        while picks and len(lat) < 12 and time.perf_counter() < t_end:
            f, pos, R = picks.pop()
            t = time.perf_counter()
            rd.read(f, bufs[0][:R], pos)
            lat.append(1e3 * (time.perf_counter() - t))
            out["lat_ms"] = sorted(lat)
        out["mode"] = mode()
        lock, sums = threading.Lock(), {"bytes": 0, "n": 0}
        t_end = time.perf_counter() + budget * 0.4              # all workers: rate

        def work(buf, r):
            try:
                while time.perf_counter() < t_end:
                    with lock:
                        if not picks:
                            return
                        f, pos, R = picks.pop()
                    r.read(f, buf[:R], pos)
                    with lock:
                        sums["bytes"] += R
                        sums["n"] += 1
            except Exception as ex:                             # noqa: BLE001
                out["error"] = f"{type(ex).__name__}: {ex}"
        readers = [Reader(timed=False) for _ in range(nb)]
        for r in readers:
            r.fds = dict(rd.fds) if r is readers[0] else {}
        t0 = time.perf_counter()
        ths = [threading.Thread(target=work, args=(b, r), daemon=True) for b, r in zip(bufs, readers)]
        for x in ths:
            x.start()
        for x in ths:
            x.join(max(0.0, t_end - time.perf_counter()) + budget * 0.2)
        dt = time.perf_counter() - t0
        with lock:
            out["par_n"], out["par_bytes"], out["par_s"], out["threads"] = sums["n"], sums["bytes"], dt, nb
        out["done"] = all(not x.is_alive() for x in ths)
        if out["done"]:                                         # a reader still inside a read keeps its files
            rd.close()
            for r in readers[1:]:
                r.close()
    except Exception as ex:                                     # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"


def _report(store, out, workers, wall, budget):
    di = drive_info(store)
    out["drive"] = di
    if out.get("error") and not out.get("lat_ms"):
        _log(f"store check: could not run ({out['error']}); {describe(di)}")
        return
    lat = out.get("lat_ms") or []
    p50 = lat[len(lat) // 2] if lat else float("inf")
    gbs = out.get("par_bytes", 0) / out["par_s"] / 1e9 if out.get("par_s") else 0.0
    out["gbs"], out["p50_ms"] = gbs, p50
    msg = (f"store check ({wall:.1f} s): {out.get('mode', mode())} reads; one record at a time {p50:.1f} ms median "
           f"over {len(lat)}; {out.get('par_n', 0)} records with {out.get('threads', workers)} readers "
           f"{gbs:.1f} GB/s; {describe(di)}")
    if out.get("first_ms", 0) > 20:
        msg += f"; first read {out['first_ms']:.0f} ms"
    _log(msg)
    if not lat:
        _log("WARNING store check: not one record was read in the check's time: the model drive is extremely slow or "
             "stuck. The engine will start, but expect very low speed. Possible causes: " + advice(di, 0.0))
        return
    if gbs >= NOTE_GBS and p50 <= WARN_MS:
        return
    if gbs >= WARN_GBS and p50 <= WARN_MS:
        _log(f"store check: the model drive reads {gbs:.1f} GB/s here (a fast Gen5 NVMe drive gives about 10-14). "
             "The engine works, but experts follow the conversation more slowly than on a Gen5 drive.")
        return
    how = "far too slowly" if gbs < BAD_GBS else "slowly"
    why = []
    if gbs < WARN_GBS:
        why.append(f"{gbs:.1f} GB/s with {out.get('threads', workers)} readers (a fast NVMe drive gives about 10-14)")
    if p50 > WARN_MS:
        why.append(f"{p50:.0f} ms to read one record (a healthy NVMe drive takes 1-3)")
    _log(f"WARNING store check: the model drive reads {how}: " + "; ".join(why) + ". The model will still answer "
         "correctly, but it will run slower and its answers will drift further from the official model, because "
         "the experts it needs arrive late. Possible causes and fixes: " + advice(di, gbs, out.get("first_ms", 0)))


# ------------------------------------------------------------------------------------------------- while serving
class Watch:
    """The runtime warning: the readers are busy most of the time and the drive delivers less than a usable rate,
    two 30 s windows in a row (or under half this run's best while busy); at most one warning per 10 minutes."""

    def __init__(self, store: str, workers: int):
        self.store, self.workers = store, max(1, workers)
        self.best = 0.0
        self.bad = 0
        self.t_warn = 0.0

    def line(self, w: dict, dt: float) -> str:
        """The 30 s line's text for window `w` (from window()), and a warning when due."""
        if not w.get("n"):
            return ""
        gbs = w["bytes"] / dt / 1e9
        busy = w["s"] / (self.workers * dt)
        txt = (f"store reads {w['n']}: {1e3 * w['p50']:.1f} / {1e3 * w['p90']:.1f} / {1e3 * w['max']:.0f} ms per "
               f"record (median / p90 / max), readers {100 * min(busy, 1.0):.0f}% busy")
        slow = busy > 0.7 and (gbs < WARN_GBS or (self.best > 0 and gbs < 0.5 * self.best))
        self.best = max(self.best, gbs) if busy > 0.7 else self.best
        self.bad = self.bad + 1 if slow else 0
        if self.bad >= 2 and time.time() - self.t_warn > 600:
            self.t_warn = time.time()
            di = drive_info(self.store)
            _log(f"WARNING the model drive is holding the engine back: it delivered {gbs:.1f} GB/s over the last "
                 f"{dt:.0f} s with the readers busy {100 * min(busy, 1.0):.0f}% of the time "
                 f"(best this run {self.best:.1f} GB/s; a fast NVMe drive gives about 10-14), "
                 f"{1e3 * w['p50']:.0f} ms per record. Answers stay correct but come more slowly and drift further "
                 f"from the official model. {describe(di)}. Possible causes and fixes: " + advice(di, gbs))
        return txt
