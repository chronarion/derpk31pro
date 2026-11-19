"""RTX band sidecar of derpk31pro: the band store on the RTX PRO 6000 (the 2-bit records an engine's static start puts
in the band), the persistent service kernel that serves the engine's calls, and the two files the engine waits for.

  CUDA_MODULE_LOADING=EAGER python3 -m band.sidecar --build /work/build --hi2 /hi2 --rank /rank/rank-pass1.json \\
      --hbm 105 --n 118 --seg /dev/shm/k31band

Band of layer L: the rank file's experts HBM .. HBM + N - 1 (the engine's K31_RANK, K31_HBM, K31_BAND_N), in store slots
(L's place among the layers) * N + j. Writes <seg>.map.npy (int32 [93, 896]: store slot, -1 = not in the band) and,
once the store is loaded and the kernel runs, <seg>.ready (JSON). SIGTERM / SIGINT: the kernel leaves, the ready file
goes. After the start this process launches nothing on the RTX (the service holds every SM): it only reads the
mailbox's state words.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch


def _store_format() -> None:
    """The record store's format decides the band's record sizes and libraries: rtxband.py reads K31_REC_FMT when it is
    imported, so it is set here from --hi2's layout.json (hi2-k1, or hi2-k2: packed scales)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--hi2" and i + 1 < len(sys.argv):
            try:
                os.environ["K31_REC_FMT"] = json.load(open(os.path.join(sys.argv[i + 1], "layout.json"))).get("format", "hi2-k1")
            except OSError:
                pass


_store_format()
from .rtxband import HI2_REC, LIB_SUFFIX, REC, Band  # noqa: E402
from .service import Mailbox, Service  # noqa: E402


def log(msg: str) -> None:
    sys.stdout.write(f"[band {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


FADV_RANDOM = os.environ.get("K31_FADV_RANDOM", "0") == "1"
# K31_BAND_RING=R (> 0): a /dev/shm ring of R hi2 records (<seg>.rec), pinned here; the engine copies an entrant's record
# from its Grace bank into a row and posts the entry with rec = row + 1, so the entry reads nothing from the model drive
RING = int(os.environ.get("K31_BAND_RING", "0") or 0)


def read_records(path: str, experts, out: np.ndarray) -> None:
    fd = os.open(path, os.O_RDONLY)
    if FADV_RANDOM:                              # no read-ahead past the record (wasted drive bandwidth)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    try:
        for j, e in enumerate(experts):
            view, pos = memoryview(out[j]), int(e) * HI2_REC
            while view:
                k = os.preadv(fd, [view], pos)
                if k <= 0:
                    raise IOError(f"short read in {path} at {pos}")
                view, pos = view[k:], pos + k
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--hi2", required=True)
    ap.add_argument("--rank", required=True)
    ap.add_argument("--hbm", type=int, required=True)
    # K31_HBM_MAP (the engine's resident rows per layer): each layer's band starts after its own resident cut
    ap.add_argument("--hbm-map", default="")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--band-map", default="", help="records per layer ({\"layers\": {L: n}}; the engine's "
                    "K31_BAND_MAP): layer L's store slots start after the earlier layers' records")
    ap.add_argument("--seg", required=True)
    ap.add_argument("--layers", default="", help="lo-hi: only these layers (tests)")
    ap.add_argument("--idle-limit-s", type=float, default=0.0, help="0: the kernel never parks")
    ap.add_argument("--status-s", type=float, default=300.0)
    ap.add_argument("--engine", default="/engine", help="the engine tree (k31_engine.band: the command ring)")
    a = ap.parse_args()
    sys.path.insert(0, a.engine)
    from k31_engine.band import BandCmd
    for f in (a.seg + ".ready", a.seg + ".map.npy", a.seg + ".cmd", a.seg + ".rec", a.seg + ".bulk"):
        if os.path.exists(f):
            os.remove(f)
    layout = json.load(open(os.path.join(a.hi2, "layout.json")))
    assert layout.get("format") in ("hi2-k1", "hi2-k2") and layout["record_bytes"] == HI2_REC, \
        f"{a.hi2}: {layout.get('format')} records of {layout.get('record_bytes')} B, this sidecar reads {HI2_REC} B"
    log(f"record store {a.hi2}: {layout.get('format')}, {HI2_REC:,} B per record; band records {REC:,} B")
    rank = json.load(open(a.rank))["layers"]
    hmap = {}
    if a.hbm_map:
        raw = json.load(open(a.hbm_map))
        hmap = {str(k): int(v) for k, v in raw.get("layers", raw).items()}
        log(f"resident rows per layer from {a.hbm_map}: {min(hmap.values())}-{max(hmap.values())}, "
            f"{sum(hmap.values())} in all")
    layers = sorted(int(k) for k in rank)
    if a.layers:
        lo, hi = (int(v) for v in a.layers.split("-"))
        layers = [L for L in layers if lo <= L <= hi]
    n = a.n
    nmap = {}
    if a.band_map:
        raw = json.load(open(a.band_map))
        nmap = {str(k): int(v) for k, v in raw.get("layers", raw).items()}
    counts = [int(nmap.get(str(L), n)) for L in layers]
    bases = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64).tolist()
    total = int(sum(counts))
    nmax = max(counts)
    if a.band_map:
        log(f"band records per layer from {a.band_map}: {min(counts)}-{nmax}, {total} in all")
    stop = {"now": False}

    def on_signal(sig, frame):
        stop["now"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    t0 = time.time()
    torch.zeros(1, device="cuda")
    free, total_mem = torch.cuda.mem_get_info()
    need = total * REC
    log(f"{torch.cuda.get_device_name()}: {free / 2**30:.1f} of {total_mem / 2**30:.1f} GiB free; store {total} "
        f"records over {len(layers)} layers = {need / 2**30:.1f} GiB")
    band = Band(os.path.join(a.build, f"librtxband{LIB_SUFFIX}.so"), total)
    m = np.full((93, 896), -1, dtype=np.int32)
    buf = torch.empty((nmax, HI2_REC), dtype=torch.uint8).pin_memory()
    bnp = buf.numpy()
    nbytes = 0
    for li, L in enumerate(layers):
        if stop["now"]:
            log("stopped while loading")
            return 1
        h = int(hmap.get(str(L), a.hbm))
        c, b0 = counts[li], bases[li]
        ex = [int(e) for e in rank[str(L)][h:h + c]]
        assert len(ex) == c and len(set(ex)) == c, f"layer {L}: the rank file has no {c} experts after {h}"
        read_records(os.path.join(a.hi2, f"L{L:02d}.hi2"), ex, bnp[:c])
        band.load(b0, buf[:c])
        m[L, ex] = b0 + np.arange(c, dtype=np.int32)
        nbytes += c * HI2_REC
        if li % 10 == 9 or li == len(layers) - 1:
            torch.cuda.synchronize()
            log(f"layer {L}: {li + 1} of {len(layers)} layers loaded, {nbytes / 1e9:.0f} GB in {time.time() - t0:.0f} s")
    torch.cuda.synchronize()
    tmp = a.seg + ".map.tmp.npy"
    np.save(tmp, m)
    os.replace(tmp, a.seg + ".map.npy")
    mb = Mailbox(os.path.join(a.build, "libbandhost.so"), a.seg, create=True)
    os.chmod(a.seg, 0o666)
    # record entries the engine asks for (a band expert promoted to a native row hands its slot to the evicted one)
    cmd = BandCmd(a.seg + ".cmd", os.path.join(a.build, "libbandhost.so"), create=True)
    os.chmod(a.seg + ".cmd", 0o666)
    ring = None
    if RING > 0:
        import mmap as _mmap
        fd = os.open(a.seg + ".rec", os.O_RDWR | os.O_CREAT, 0o666)
        try:
            os.ftruncate(fd, RING * HI2_REC)
            ringbuf = _mmap.mmap(fd, RING * HI2_REC)
        finally:
            os.close(fd)
        os.chmod(a.seg + ".rec", 0o666)
        ring = np.frombuffer(ringbuf, dtype=np.uint8).reshape(RING, HI2_REC)
        ring[:, :1] = 0                                # touch every row before pinning
        rc = int(torch.cuda.cudart().cudaHostRegister(ring.ctypes.data, ring.nbytes, 0))
        if rc != 0:
            raise RuntimeError(f"could not pin the record ring ({ring.nbytes / 2**20:.0f} MiB): cuda error {rc}")
        log(f"record ring: {RING} rows ({ring.nbytes / 2**20:.0f} MiB) in {a.seg}.rec, pinned")
    n_ring = 0
    staging = bnp[0]                                   # pinned: the copy engine's source
    svc = Service(band, os.path.join(a.build, f"librtxband_svc{LIB_SUFFIX}.so"), mb)
    svc.start(idle_limit_s=a.idle_limit_s)
    # K31_BAND_BULK=1: prompt chunks send their band picks in one request per layer (band/bulk.py); the service
    # kernel is parked while the engine reads a prompt, and records that enter the band meanwhile take the launched
    # repack
    bulk = None
    if os.environ.get("K31_BAND_BULK", "0") == "1":
        from .bulk import BulkServer
        bulk = BulkServer(band, os.path.join(a.build, "libbandhost.so"), a.seg + ".bulk")
        log(f"bulk requests: {a.seg}.bulk ({bulk.L['bytes'] / 2**20:.0f} MiB, up to {bulk.max_rows} rows)")
    landing = torch.empty(HI2_REC, dtype=torch.uint8, device=band.dev) if bulk is not None else None

    def enter(rec_np, slot: int) -> bool:
        """A record enters the band: through the running service kernel, or with the launched repack when it is
        parked for bulk requests."""
        if bulk is not None and bulk.mode == 1:
            landing.copy_(torch.from_numpy(rec_np))
            return band.load_device(slot, landing)
        fits, _, _, _ = svc.enter(rec_np, slot)
        return fits
    info = dict(seg=a.seg, n=n, band_map=a.band_map, hbm=a.hbm, rank=a.rank, layers=[layers[0], layers[-1]], records=int((m >= 0).sum()),
                grid=svc.grid, regs=svc.regs, per_sm=svc.per_sm, sms=svc.sms, pid=os.getpid(),
                loaded_s=round(time.time() - t0, 1))
    with open(a.seg + ".ready.tmp", "w") as f:
        json.dump(info, f)
    os.replace(a.seg + ".ready.tmp", a.seg + ".ready")
    log(f"ready: {info}")
    last = check = time.time()
    rc = 0
    n_slots = total
    entries = 0
    # BAND_READAHEAD=k (> 0): k reader threads pull the next posted entries' records into rows of the pinned load
    # buffer while the current one is entered (the score-fed band, K31_BAND_FEED, posts ~1,000+ entries per second;
    # one at a time the file read is ~half of the ~3 ms per entry); entries are still entered and completed in order
    ra = max(0, min(int(os.environ.get("BAND_READAHEAD", "0") or 0), nmax))
    pool = queue = free = None
    if ra:
        import collections
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(ra)
        queue = collections.deque()                    # (seq, L, e, slot, row, future, t_start), in seq order
        free = list(range(ra))
        log(f"read-ahead: {ra} reader threads")
    try:
        while not stop["now"]:
            if bulk is not None:
                want = bulk.wanted()
                if want != bulk.mode:                  # park the service kernel for a prompt, or bring it back
                    t_sw = time.perf_counter()
                    if want:
                        svc.stop()
                    else:
                        svc.start(idle_limit_s=a.idle_limit_s)
                    bulk.ack(want)
                    if bulk.n < 400 or bulk.n % 2000 == 0:
                        log(f"bulk mode {want}: the service kernel {'parked' if want else 'running'} "
                            f"({1e3 * (time.perf_counter() - t_sw):.1f} ms)")
                if bulk.mode == 1 and bulk.pending():
                    bulk.serve()
                    continue
            if ra:
                while free:                            # read the next posted entries ahead
                    nxt = cmd.peek(cmd.get(64) + 1 + len(queue))
                    if nxt is None:
                        break
                    seq, L, e, slot = nxt
                    rec = cmd.rec(seq) if ring is not None else 0
                    row = free.pop()
                    fut = None
                    if 1 <= L <= 92 and 0 <= e < 896 and 0 <= slot < n_slots:
                        if 0 < rec <= RING:
                            fut = "ring"
                        else:
                            fut = pool.submit(read_records, os.path.join(a.hi2, f"L{L:02d}.hi2"), [e],
                                              bnp[row:row + 1])
                    queue.append((seq, L, e, slot, row, fut, time.perf_counter(), rec))
                if not queue:
                    time.sleep(0.0002)
                else:
                    seq, L, e, slot, row, fut, t1, rec = queue.popleft()
                    status = 3
                    try:
                        if fut == "ring":
                            fits = enter(ring[rec - 1], slot)
                            status = 1 if fits else 2
                            n_ring += 1
                        elif fut is not None:
                            fut.result()
                            fits = enter(bnp[row], slot)
                            status = 1 if fits else 2
                    except Exception as ex:            # noqa: BLE001  (reported through the ring)
                        log(f"entry {seq} (layer {L}, expert {e}, slot {slot}) failed: {ex}")
                    cmd.complete(seq, status, int(1e6 * (time.perf_counter() - t1)))
                    free.append(row)
                    entries += 1
            else:
                job = cmd.next_job()
                if job is None:
                    time.sleep(0.0002)
                else:
                    seq, L, e, slot = job
                    rec = cmd.rec(seq) if ring is not None else 0
                    t1 = time.perf_counter()
                    status = 3
                    try:
                        if 1 <= L <= 92 and 0 <= e < 896 and 0 <= slot < n_slots:
                            if 0 < rec <= RING:
                                fits = enter(ring[rec - 1], slot)
                                n_ring += 1
                            else:
                                read_records(os.path.join(a.hi2, f"L{L:02d}.hi2"), [e], bnp[0:1])
                                fits = enter(staging, slot)
                            status = 1 if fits else 2
                    except Exception as ex:            # noqa: BLE001  (reported through the ring; the service runs on)
                        log(f"entry {seq} (layer {L}, expert {e}, slot {slot}) failed: {ex}")
                    cmd.complete(seq, status, int(1e6 * (time.perf_counter() - t1)))
                    entries += 1
            now = time.time()
            if now - check >= 0.5:
                check = now
                if (bulk is None or bulk.mode == 0) and not svc.running():
                    log(f"the service kernel left on its own: {mb.state()}")
                    rc = 2
                    break
            if now - last >= a.status_s:
                last = now
                log(f"state {mb.state()}; entries {entries} (ok {cmd.get(72)}, unfit {cmd.get(80)}, errors "
                    f"{cmd.get(88)}, {cmd.get(96) / max(entries, 1):.0f} us each"
                    + (f"; {n_ring} from the record ring)" if ring is not None else ")")
                    + (f"; {bulk.stats()}" if bulk is not None else ""))
    finally:
        for f in (a.seg + ".ready", a.seg + ".cmd", a.seg + ".rec", a.seg + ".bulk"):
            if os.path.exists(f):
                os.remove(f)
        try:
            if bulk is None or bulk.mode == 0:
                svc.stop()
        except AssertionError as e:
            log(f"stop: {e}")
            rc = rc or 3
        log(f"stopped: {mb.state()}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
