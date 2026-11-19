"""The Grace pool (K31_POOL=1): pinned threads of the engine process that compute 2-bit expert records where they lie
in the layers' pinned banks (st.bank_cpu), for jobs the GB300 posts into a pinned segment: the job ring of
engine/pool/cpustage_seg2.h, served by engine/band/libcpustage.so (engine/pool/build.sh, cpustage.cpp).

A job (one MoE call of T <= 8 rows, up to 8 records): the GB300 writes the header (CpuJob2: its tag, the records' bank
indices, the routing weights) and the call's rows quantized to int8 with a power-of-two scale per 32 (row-planar)
into slot seq % 4, then the door (the newest tag) with a system-scope release store. The pool runs the jobs in
sequence order and publishes float32 rows [T, 3584] in the slot's result area, then the slot's done tag (and a status
word: 1 = refused). The GB300's landing (late job) or finish (on-time job) waits for the tag it sent, with a bound.
The sender's sequence numbers start at POOL.next_seq() (1: the bind's self-check leaves a fresh ring) and grow by one
per job; the pool keeps the newest 3 jobs wanted (an older one is skipped or abandoned and publishes nothing), and a
state word (+0 of the state block) reads 1 while the threads run.
Arithmetic and its tests: engine/pool/cpustage.h, cpustage_test.cpp (bit-exact against a scalar reference),
engine/tests/test_cpustage_ref.py (against band.reference on real records).

Switches (nothing here runs unless the hook calls `from_env`, and that returns None unless K31_POOL=1):
  K31_POOL=1                start the pool at bind (the late / on-time tiers post its jobs)
  K31_POOL_THREADS=48       one thread per core on K31_POOL_CORE0 .. K31_POOL_CORE0 + THREADS - 1
  K31_POOL_CORE0=24
  K31_POOL_NICE=0           nice value of the pool threads (< 0 needs CAP_SYS_NICE, which the engine container has)
  K31_POOL_HOST_CPUS        where every other thread of the engine process goes before the pool starts (new threads
                            inherit it): default the process's CPU set without the pool's cores; "-" = leave them
  K31_POOL_SELFCHECK=1      one job through the pool against the single-thread kernel at start (bit-equal, a few
                            ms), then a fresh ring
  K31_POOL_LIB              the library (default band/libcpustage.so beside this package)
The engine container needs the pool's cores in its CPU set (CPUS=0-71; the band sidecar off them: BAND_CPUS=22-23,
K31_POOL_HOST_CPUS=0-21).
"""
import ctypes
import mmap
import os

import numpy as np

H, KB, MAXT, PMAX, NSLOT = 3584, 112, 8, 8, 4
REC_BYTES = 9289728
# cpustage_seg2.h
SEG_BYTES = 655360
OFF_DOOR, OFF_STATE, OFF_DONE, OFF_JOB, JOB_BYTES = 128, 256, 512, 4096, 512
OFF_X, X_STRIDE, X_OFF_E = 8192, 32768, MAXT * H
OFF_RES, RES_BYTES = 139264, MAXT * H * 4
JOB2_BYTES = 328
LAYOUT = (SEG_BYTES, OFF_DOOR, OFF_STATE, OFF_DONE, OFF_JOB, JOB_BYTES, OFF_X, X_STRIDE, X_OFF_E, OFF_RES, RES_BYTES,
          NSLOT, PMAX, MAXT, JOB2_BYTES)
MAGIC = 0x3253555043313343                     # "C31CPUS2"
ST_ERROR = 1
FLAG_OUT_BF16, FLAG_NO_BF16, FLAG_ONTIME = 1, 2, 4
SITU_ALPHA, SITU_BETA = 4.0, 25.0
STATS = ("running", "done", "abandoned", "refused", "job_ns", "picks", "max_job_ns", "stale", "resets", "ontime",
         "queued", "queue_max")


def tag(seq: int, layer: int, rows: int) -> int:
    """A job's tag (cpuseg_tag): seq << 16 | layer << 8 | rows; never 0."""
    return (seq << 16) | ((layer & 0xFF) << 8) | (rows & 0xFF)


def job_header(tag_: int, slots, w: np.ndarray, T: int, layer: int, flags: int = FLAG_OUT_BF16,
               alpha: float = SITU_ALPHA, beta: float = SITU_BETA, e_min: int = -100) -> np.ndarray:
    """A job's header (CpuJob2, little-endian, 328 B) as uint8: tag u64; n_pick, n_rows, layer, flags u32; alpha,
    beta f32; e_min i32; pad; slot i32 [8]; w f32 [8][8] (w: [n, T], rows >= T and picks >= n zero)."""
    h = np.zeros(JOB2_BYTES, dtype=np.uint8)
    n = len(slots)
    h[0:8].view(np.uint64)[0] = tag_
    h[8:24].view(np.uint32)[:] = (n, T, layer, flags)
    h[24:32].view(np.float32)[:] = (alpha, beta)
    h[32:36].view(np.int32)[0] = e_min
    h[40:72].view(np.int32)[:min(n, PMAX)] = np.asarray(slots, dtype=np.int32)[:PMAX]
    wa = np.asarray(w, dtype=np.float32)
    w8 = h[72:328].view(np.float32).reshape(PMAX, MAXT)
    w8[:min(n, PMAX), :min(T, wa.shape[1])] = wa[:PMAX, :T]
    return h


def planar_index() -> np.ndarray:
    """Row-planar order (cpuseg_planar): input k of a row is byte planar_index()[k] of its slot row."""
    k = np.arange(H)
    return (k & ~31) + 8 * (k & 3) + ((k & 31) >> 2)


def parse_cpus(spec: str) -> set:
    """"0-21,30" -> {0, ..., 21, 30}."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def pin_threads(cpus) -> int:
    """Every thread this process has now goes to `cpus` (threads created later inherit their creator's set).
    Returns the number of threads moved."""
    n = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
            n += 1
        except (ProcessLookupError, PermissionError):
            pass                                  # a thread that ended meanwhile
    return n


def _default_lib() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "band", "libcpustage.so")


def _ptr(a) -> int:
    """Address of a host buffer: an int, a numpy array or a CPU torch tensor."""
    if isinstance(a, int):
        return a
    if isinstance(a, np.ndarray):
        return a.ctypes.data
    return a.data_ptr()


class CpuPool:
    """The segment (anonymous, page-aligned, zeroed; `map_device` pins it and gives the GB300's pointer), the
    library's pool (one per process) and the host-side helpers that tests and a host-driven engine use."""

    def __init__(self, lib: str = None, threads: int = 48, core0: int = 24, nice: int = 0):
        self.lib = ctypes.CDLL(lib or _default_lib())
        L, V, I, F, U, ULL = (self.lib, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_uint,
                              ctypes.c_ulonglong)
        L.cpustage_init.argtypes = [V, I, I, I]
        L.cpustage_set_layer.argtypes = [I, V, ctypes.c_longlong]
        L.cpustage_stats_n.argtypes = [ctypes.POINTER(ULL), I]
        L.cpustage_stop.restype = None
        L.cpustage_record_bytes.restype = ctypes.c_longlong
        L.cpustage_layout.argtypes = [ctypes.POINTER(ctypes.c_longlong), I]
        L.cpustage_run_single.argtypes = [V, I, V, V, I, F, F, U, I, V, V, V]
        L.cpustage_test_post.argtypes = [V, ULL, I, V, I, V, V, I, F, F, U, I]
        L.cpustage_test_post.restype = ULL
        L.cpustage_test_wait.argtypes = [V, ULL, ULL, ctypes.POINTER(ULL)]
        L.cpustage_test_land.argtypes = [V, ULL, V, I, ULL]
        L.cpustage_test_land.restype = None
        L.cpustage_test_hold.argtypes = [I]
        L.cpustage_test_hold.restype = None
        if L.cpustage_record_bytes() != REC_BYTES:
            raise RuntimeError("K31 pool: the library's record size is not hi2-k1's")
        lay = (ctypes.c_longlong * 16)()
        n = L.cpustage_layout(lay, 16)
        if tuple(lay[:len(LAYOUT)]) != LAYOUT or n < len(LAYOUT):
            raise RuntimeError(f"K31 pool: the library's segment {tuple(lay[:n])} is not this module's {LAYOUT}")
        self.threads, self.core0, self.nice = threads, core0, nice
        self.buf = mmap.mmap(-1, SEG_BYTES)                   # anonymous: zeroed, page-aligned
        self.u8 = np.frombuffer(self.buf, dtype=np.uint8)
        self.u64 = np.frombuffer(self.buf, dtype=np.uint64)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self.seg_t = self.seg_dev = None
        self.started = False
        self.layers = {}
        rc = L.cpustage_init(self.addr, threads, core0, nice)
        if rc != 0:
            raise RuntimeError(f"K31 pool: init failed ({rc}: threads {threads}, core0 {core0})")

    # ------------------------------------------------------------------------------------------------ set-up
    def map_device(self) -> int:
        """Pin the segment and map it for the GPU (as hook._load_bank pins a bank): returns the device pointer the
        GB300 kernels use (`seg_dev`); the pool threads keep using the host address."""
        import torch
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        try:                        # registers torch.ops._C.get_cuda_view_from_cpu_tensor (vLLM's CUDA platform does
            import vllm._C_stable_libtorch  # noqa: F401   too, when it is first resolved)
        except ImportError:
            pass
        if self.seg_dev is None:
            t = torch.frombuffer(self.buf, dtype=torch.uint8)
            if int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0x3)) != 0:
                raise RuntimeError("K31 pool: could not pin the segment")
            self._host_t = t
            self.seg_t = get_accelerator_view_from_cpu_tensor(t)
            self.seg_dev = self.seg_t.data_ptr()
        return self.seg_dev

    def set_layer(self, idx: int, bank, nrec: int = None) -> None:
        """Layer idx's bank: a host uint8 tensor / array [G, REC_BYTES] (st.bank_cpu), or an address with nrec."""
        if nrec is None:
            nrec = int(bank.shape[0])
            if int(bank.shape[1]) != REC_BYTES:
                raise ValueError(f"K31 pool: layer {idx}'s bank rows are {bank.shape[1]} B")
        if self.lib.cpustage_set_layer(int(idx), _ptr(bank), int(nrec)) != 0:
            raise RuntimeError(f"K31 pool: set_layer({idx}) refused (pool running?)")
        self.layers[int(idx)] = (bank, int(nrec))           # keeps the memory referenced

    def start(self, host_cpus=None) -> None:
        """Move the process's other threads to host_cpus (None: leave them), then start the pool's threads (each pins
        itself to its core; a core outside the process's CPU set fails the start)."""
        if host_cpus:
            pin_threads(set(host_cpus))
        rc = self.lib.cpustage_start()
        if rc == -3:
            raise RuntimeError(f"K31 pool: cores {self.core0}-{self.core0 + self.threads - 1} are not all in this "
                               f"process's CPU set (container CPUS?)")
        if rc != 0:
            raise RuntimeError(f"K31 pool: start failed ({rc})")
        self.started = True

    def stop(self) -> None:
        """Stop the threads (no job may be in flight: the GPU idle)."""
        if self.started:
            self.lib.cpustage_stop()
            self.started = False

    def close(self) -> None:
        self.stop()
        if self.seg_dev is not None:
            import torch
            torch.cuda.cudart().cudaHostUnregister(self._host_t.data_ptr())
            self.seg_t = self.seg_dev = None

    # ------------------------------------------------------------------------------------------------ reading
    def stats(self) -> dict:
        v = (ctypes.c_ulonglong * len(STATS))()
        self.lib.cpustage_stats_n(v, len(STATS))
        d = dict(zip(STATS, (int(x) for x in v)))
        d["mean_job_us"] = d["job_ns"] / d["done"] / 1000.0 if d["done"] else 0.0
        return d

    def word(self, off: int) -> int:
        return int(self.u64[off // 8])

    def door(self) -> int:
        return self.word(OFF_DOOR)

    def next_seq(self) -> int:
        """The first sequence number a GB300 sender should use: above the door and every done tag (a sender that
        restarts from a number the pool has already published could match a stale done tag)."""
        top = self.door() >> 16
        for s in range(NSLOT):
            top = max(top, self.word(OFF_DONE + 128 * s) >> 16)
        return top + 1

    def done(self, slot: int):
        """(done tag, status) of a result slot."""
        return self.word(OFF_DONE + 128 * slot), self.word(OFF_DONE + 128 * slot + 8)

    def result(self, tag_or_slot: int, T: int = MAXT) -> np.ndarray:
        """The result rows of a slot (or of a tag's slot), float32 [T, 3584], a view into the segment."""
        s = tag_or_slot if tag_or_slot < NSLOT else (tag_or_slot >> 16) % NSLOT
        o = OFF_RES + s * RES_BYTES
        return self.u8[o:o + T * H * 4].view(np.float32).reshape(T, H)

    # ------------------------------------------------------------------------------------------------ host-side jobs
    def run_single(self, recs, x: np.ndarray, w: np.ndarray, alpha: float = SITU_ALPHA, beta: float = SITU_BETA,
                   flags: int = FLAG_OUT_BF16, e_min: int = -100, want_q: bool = False):
        """One job on the calling thread: recs (addresses / host buffers of n records), x float32 [T, 3584], w
        float32 [n, T] (or [n, 8]) routing weights. Returns out float32 [T, 3584] (and with want_q the quantized rows
        int8 [T, 3584] in input order and their exponent bytes uint8 [T, 112])."""
        n, T = len(recs), int(x.shape[0])
        xs = np.ascontiguousarray(x, dtype=np.float32)
        wa = np.asarray(w, dtype=np.float32)
        w8 = np.zeros((n, MAXT), dtype=np.float32)
        w8[:, :min(T, wa.shape[1])] = wa[:, :T]
        ptrs = (ctypes.c_void_p * PMAX)(*[_ptr(r) for r in recs])
        out = np.zeros((T, H), dtype=np.float32)
        xq = np.zeros((T, H), dtype=np.int8) if want_q else None
        xe = np.zeros((T, KB), dtype=np.uint8) if want_q else None
        rc = self.lib.cpustage_run_single(ptrs, n, xs.ctypes.data, w8.ctypes.data, T, alpha, beta, flags, e_min,
                                          out.ctypes.data, xq.ctypes.data if want_q else None,
                                          xe.ctypes.data if want_q else None)
        if rc != 0:
            raise ValueError(f"K31 pool: run_single refused the job ({rc})")
        return (out, xq, xe) if want_q else out

    def post(self, seq: int, layer: int, slots, x: np.ndarray, w: np.ndarray, flags: int = FLAG_OUT_BF16,
             alpha: float = SITU_ALPHA, beta: float = SITU_BETA, e_min: int = -100) -> int:
        """The GB300's half of a job, from the host: header, quantized rows, door. slots: bank indices of the records,
        w float32 [n, T]. Returns the tag."""
        n, T = len(slots), int(x.shape[0])
        sl = np.zeros(PMAX, dtype=np.int32)
        sl[:min(n, PMAX)] = np.asarray(slots, dtype=np.int32)[:PMAX]
        wa = np.asarray(w, dtype=np.float32)
        w8 = np.zeros((PMAX, MAXT), dtype=np.float32)
        w8[:min(n, PMAX), :min(T, wa.shape[1])] = wa[:PMAX, :T]
        xs = np.ascontiguousarray(x, dtype=np.float32)
        return int(self.lib.cpustage_test_post(self.addr, seq, layer, sl.ctypes.data, n, xs.ctypes.data,
                                               w8.ctypes.data, T, alpha, beta, flags, e_min))

    def wait(self, tag_: int, timeout_s: float = 10.0):
        """Spin until the tag's slot shows it done: its status (0 rows, 1 refused), or None on timeout."""
        st = ctypes.c_ulonglong(0)
        if self.lib.cpustage_test_wait(self.addr, tag_, int(timeout_s * 1e9), ctypes.byref(st)) != 0:
            return None
        return int(st.value)

    def land(self, tag_: int, rows: np.ndarray = None, status: int = 0) -> None:
        """The pool's half of one job without the pool (tests of the GB300's landing): rows float32 [T, 3584] into
        the tag's result slot, the status, then the done tag."""
        r = None if rows is None else np.ascontiguousarray(rows, dtype=np.float32)
        self.lib.cpustage_test_land(self.addr, tag_, None if r is None else r.ctypes.data,
                                    0 if r is None else int(r.shape[0]), status)

    def hold(self, on: bool) -> None:
        """Tests: while on, the pool takes no job (they queue up as behind a busy pool)."""
        self.lib.cpustage_test_hold(1 if on else 0)

    def reset_ring(self, timeout_s: float = 1.0) -> None:
        """A fresh ring: every done line and the door to 0 (no GB300 job may be in flight). A running pool follows the
        door down (stats()["resets"]); this waits until it has, so that a sender starting at sequence number 1
        (next_seq()) has its first job served and can never match a done tag of an earlier job."""
        import time
        before, door_was = self.stats()["resets"], self.door()
        for s in range(NSLOT):
            self.u64[(OFF_DONE + 128 * s) // 8] = 0
            self.u64[(OFF_DONE + 128 * s + 8) // 8] = 0
        self.u64[OFF_DOOR // 8] = 0
        if self.started and door_was != 0:
            t0 = time.perf_counter()
            while self.stats()["resets"] == before:
                if time.perf_counter() - t0 > timeout_s:
                    raise RuntimeError("K31 pool: the pool did not follow the door back to 0")
                time.sleep(0.0002)

    def selfcheck(self, layer: int, seed: int = 0, T: int = 5, n: int = 2) -> float:
        """One job of n records of `layer`'s bank through the running pool against run_single on the same records:
        must be bit-equal. Run it before the GB300 sends anything: it posts at next_seq() and then resets the ring
        (reset_ring), so the sender's first job is sequence number 1 again. Returns the job's microseconds as the host
        saw them."""
        import time
        bank, nrec = self.layers[layer]
        if nrec < 1:
            raise RuntimeError(f"K31 pool: layer {layer} has no bank records")
        rng = np.random.default_rng(seed)
        slots = [int(s) for s in rng.integers(0, nrec, size=n)]
        x = (rng.standard_normal((T, H)) * 0.5).astype(np.float32)
        w = np.zeros((n, T), dtype=np.float32)
        for p in range(n):
            w[p, rng.integers(0, T, size=2)] = rng.uniform(0.05, 0.3, size=2)
        base = _ptr(bank)
        want = self.run_single([base + s * REC_BYTES for s in slots], x, w)
        try:
            t0 = time.perf_counter()
            tg = self.post(self.next_seq(), layer, slots, x, w)
            st = self.wait(tg, 10.0)
            us = (time.perf_counter() - t0) * 1e6
            if st != 0:
                raise RuntimeError(f"K31 pool: self-check job {'timed out' if st is None else 'refused'} (layer "
                                   f"{layer})")
            got = self.result(tg, T)
            if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
                raise RuntimeError(f"K31 pool: self-check rows differ from the single-thread kernel ({bad} elements)")
        finally:
            self.reset_ring()
        return us


POOL = None            # the engine's pool once from_env started it (the GB300 side's send / landing take it from here)


def from_env(layers, log=print):
    """The hook's call at bind (after the banks are loaded and pinned): None unless K31_POOL=1, else a started pool
    with every layer's bank registered, kept in POOL for the process's life (its segment and the banks must outlive
    the threads). layers: iterable of (idx, bank_cpu, nrec): bank_cpu a host uint8 [>= nrec, REC_BYTES] (st.bank_cpu,
    nrec = len(st.bank)), or None."""
    global POOL
    if os.environ.get("K31_POOL", "0") != "1":
        return None
    if POOL is not None:
        return POOL
    threads = int(os.environ.get("K31_POOL_THREADS", "48"))
    core0 = int(os.environ.get("K31_POOL_CORE0", "24"))
    nice = int(os.environ.get("K31_POOL_NICE", "0"))
    pool = CpuPool(os.environ.get("K31_POOL_LIB") or None, threads, core0, nice)
    first = None
    for idx, bank, nrec in layers:
        if bank is not None and nrec > 0:
            if int(bank.shape[0]) < nrec or int(bank.shape[1]) != REC_BYTES:
                raise ValueError(f"K31 pool: layer {idx}'s bank {tuple(bank.shape)} does not hold {nrec} records")
            pool.set_layer(idx, bank, nrec)
            first = idx if first is None else first
    spec = os.environ.get("K31_POOL_HOST_CPUS", "")
    if spec == "-":
        host = None
    elif spec:
        host = parse_cpus(spec)
    else:
        host = os.sched_getaffinity(0) - set(range(core0, core0 + threads))
    if host is not None and not host:
        raise RuntimeError("K31 pool: no CPU left for the engine's own threads")
    pool.start(host)
    POOL = pool
    import atexit
    atexit.register(pool.stop)
    msg = (f"Grace pool: {threads} threads on cores {core0}-{core0 + threads - 1}, {len(pool.layers)} layers' banks, "
           f"the process's other threads on {_fmt_cpus(host) if host else 'their own sets'}")
    if os.environ.get("K31_POOL_SELFCHECK", "1") == "1" and first is not None:
        msg += f"; self-check on layer {first} bit-equal ({pool.selfcheck(first):.0f} us)"
    log(msg)
    return pool


def _fmt_cpus(cpus) -> str:
    c = sorted(cpus)
    out, i = [], 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[j] + 1:
            j += 1
        out.append(f"{c[i]}-{c[j]}" if j > i else f"{c[i]}")
        i = j + 1
    return ",".join(out)
