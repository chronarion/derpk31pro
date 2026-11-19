"""K31_LATE=1: the late tier on the Grace pool: a MoE call's bank picks that nothing serves on time are
computed by the Grace cores from their 2-bit records where they lie, and land half a layer late.

Per call of T <= 8 rows (decode-size calls; FULL CUDA graphs or eager, never inside a breakable capture), in a layer
with late slots:
  after unitk (late stream, forked from main): late.plan (the call's late records: distinct bank experts not served on
      time, by summed weight^2) -> late.mean (the share of the on-time cut-mean those picks got, when the layer has
      mean tables) -> late_send (job header, int8 rows, door tag). The pool computes the job. Nothing on the main stream
      reads any of it: the on-time path (unitk, stager, cut-mean, MegaMoE) is unchanged, so the late picks keep their
      mean on time ("mean now, correction later").
  the routed output transform (main): vLLM's own three steps (RMSNorm, up GEMV, residual add), keeping u (the routed
      latent sum) and y (the up GEMV's output) by reference; then on the late stream late_land (a bounded wait for the
      pool, xl = bf16(ra g v) and the coefficients) and z = up(xl) through the layer's own int8 weights, beside the
      next layer's attention.
  the next layer's _post_attn_norm (main): join the late stream; z + cb y (late.add) goes into its attention output,
      or into the stored AttnRes block state when layer L ended a block, before its MoE input mix reads it.
up(norm(u + v)) - up(norm(u)) = up(ra g v) + (ra / rb - 1) y exactly (check_identity.py), so the completion is one
GEMV. A job the pool does not finish within the landing's bound (K31_LATE_TIMEOUT_US: next layer KDA, MLA) adds
nothing. The last MoE layer sends no job (no next layer to land in).

Switches: K31_LATE=1 (or pool); K31_LATE_SLOTS records per layer-call (default 5, at most 8), K31_LATE_FROM /
K31_LATE_TO the rank range (default 0..15), K31_LATE_MAP=FILE per-layer record counts (JSON {"<layer>": n},
{"layers": {...}}, or a round-13 plan file {"<layer>": {"late": n, ...}}); a K31_POLICY file's `late` / `late_slots`
fields win for the layers that set them. K31_LATE_SELFCHECK=1 (one job posted by the GB300 through the pool at bind:
its rows bit-equal to the pool's single-thread kernel, the landing that collects them bit-equal to the device-row
landing), K31_LATE_LOG_S (counters every n seconds).
The pool is the engine's one Grace pool, cpupool.POOL (K31_POOL=1 starts it at bind, with its cores and threads:
K31_POOL_THREADS / K31_POOL_CORE0 / K31_POOL_HOST_CPUS; engine/k31.sh sets K31_POOL=1 and the core layout whenever
K31_LATE=1). The late tier maps the pool's segment for the GB300 and owns its job sequence from POOL.next_seq().
Needs: mode staged, K31_DEFER=0, K31_FUSE_UP=0, the int8 routed up (K31_DENSE_FMT routed_expert_up_proj:w8i).
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
import sys
import time

import numpy as np
import torch

from . import late

MODE = os.environ.get("K31_LATE", "0")
ON = MODE in ("1", "pool")
SLOTS = int(os.environ.get("K31_LATE_SLOTS", "5"))
LFROM = int(os.environ.get("K31_LATE_FROM", "0"))
LTO = int(os.environ.get("K31_LATE_TO", "16"))
MAP = os.environ.get("K31_LATE_MAP", "")
TIMEOUT_US = [float(v) for v in os.environ.get("K31_LATE_TIMEOUT_US", "50,105").split(",") if v.strip()]
LIB = os.environ.get("K31_LATE_LIB", "/engine/band/liblate.so")
SELFCHECK = os.environ.get("K31_LATE_SELFCHECK", "1") == "1"
LOG_S = float(os.environ.get("K31_LATE_LOG_S", "60"))
PDL = os.environ.get("K31_PDL", "0") == "1"
FLAGS = 1                     # CPUJOB_OUT_BF16: each record's output rounded to bfloat16 before the sum (MegaMoE's order)
E_MIN = -100                  # lowest block exponent of the int8 activations
SITU_ALPHA, SITU_BETA = 4.0, 25.0          # config.json: activation_situ_beta, activation_situ_linear_beta (band.py)
ATTN_RES_BLOCK = 12
STATE_WORDS = ("seq", "tag", "n_jobs", "n_recs", "n_added", "n_timeout", "n_error", "n_empty", "n_down", "wait_ns",
               "wait_max", "t_pub", "n_rows")
POOL_STATE = ("running", "done", "abandoned", "refused", "job_ns", "picks", "max_ns", "stale")
LAYOUT = ("bytes", "door", "state", "done", "job", "job_bytes", "x", "x_stride", "res", "res_bytes", "nslot", "pmax",
          "maxt", "h", "kb", "magic", "job_size", "off_slot", "off_w", "state_bytes", "threads")
V = ctypes.c_void_p
_LOG = lambda m: sys.stderr.write(f"K31 {m}\n")      # noqa: E731


class _Send(ctypes.Structure):
    _fields_ = [("seg", V), ("st", V), ("lslot", V), ("lw", V), ("x", V), ("T", ctypes.c_int),
                ("layer", ctypes.c_int), ("flags", ctypes.c_int), ("e_min", ctypes.c_int), ("alpha", ctypes.c_float),
                ("beta", ctypes.c_float)]


class _Land(ctypes.Structure):
    _fields_ = [("seg", V), ("st", V), ("u", V), ("v_dev", V), ("mfix", V), ("g", V), ("xl", V), ("coef", V),
                ("T", ctypes.c_int), ("v_bf16", ctypes.c_int), ("eps", ctypes.c_float), ("layer", ctypes.c_int),
                ("timeout_ns", ctypes.c_ulonglong)]


class Lib:
    """liblate.so (csrc/late.cu): the two kernels, the segment's mapping, its layout, and the host twins."""

    def __init__(self, path: str = LIB):
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.late_send.argtypes = [ctypes.POINTER(_Send), V, ctypes.c_int]
        L.late_land.argtypes = [ctypes.POINTER(_Land), V, ctypes.c_int]
        L.late_map.argtypes = [V, ctypes.c_longlong, ctypes.POINTER(V)]
        L.late_unmap.argtypes = [V]
        L.late_layout.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        L.late_host_quant_rows.argtypes = [V, ctypes.c_int, ctypes.c_int, V, V]
        L.late_host_header.argtypes = [ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int, V, V, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float, V]
        L.late_host_coef.argtypes = [V, ctypes.c_int, ctypes.c_float, V]
        L.late_host_get.argtypes = [V, ctypes.c_longlong]
        L.late_host_get.restype = ctypes.c_ulonglong
        L.late_host_set.argtypes = [V, ctypes.c_longlong, ctypes.c_ulonglong]
        L.late_host_wait.argtypes = [V, ctypes.c_longlong, ctypes.c_ulonglong, ctypes.c_longlong]
        out = (ctypes.c_longlong * 32)()
        L.late_layout(out)
        self.layout = {k: int(out[i]) for i, k in enumerate(LAYOUT)}
        lo = self.layout
        assert lo["maxt"] == late.MAXT and lo["pmax"] == late.PMAX and lo["h"] == late.H, "cpustage_seg2.h constants"
        assert lo["state_bytes"] == 8 * 16, "late state block"

    def send(self, seg, st, lslot, lw, x, T: int, layer: int, stream=None, flags: int = FLAGS, e_min: int = E_MIN,
             alpha: float = SITU_ALPHA, beta: float = SITU_BETA, pdl: bool = PDL) -> None:
        a = _Send(seg, st.data_ptr(), lslot.data_ptr(), lw.data_ptr(), x.data_ptr(), int(T), int(layer), int(flags),
                  int(e_min), float(alpha), float(beta))
        s = torch.cuda.current_stream() if stream is None else stream
        rc = self.lib.late_send(ctypes.byref(a), V(s.cuda_stream), 1 if pdl else 0)
        if rc != 0:
            raise RuntimeError(f"K31 late: send launch failed ({rc})")

    def land(self, seg, st, u, v_dev, mfix, g, xl, coef, T: int, eps: float, layer: int, timeout_us: float,
             stream=None, pdl: bool = PDL) -> None:
        """seg None: the rows come from v_dev (float32 or bf16 [T, 3584]) instead of the pool."""
        a = _Land(seg, st.data_ptr() if st is not None else None, u.data_ptr(),
                  v_dev.data_ptr() if v_dev is not None else None, mfix.data_ptr() if mfix is not None else None,
                  g.data_ptr(), xl.data_ptr(), coef.data_ptr(), int(T),
                  1 if (v_dev is not None and v_dev.dtype == torch.bfloat16) else 0, float(eps), int(layer),
                  int(timeout_us * 1000))
        s = torch.cuda.current_stream() if stream is None else stream
        rc = self.lib.late_land(ctypes.byref(a), V(s.cuda_stream), 1 if pdl else 0)
        if rc != 0:
            raise RuntimeError(f"K31 late: landing launch failed ({rc})")

    # host twins (CPU tests)
    def host_quant_rows(self, x: torch.Tensor, e_min: int = E_MIN):
        xf = np.ascontiguousarray(x.float().cpu().numpy(), dtype=np.float32)
        T = xf.shape[0]
        xq = np.zeros((T, late.H), dtype=np.uint8)
        xe = np.zeros((T, late.H // 32), dtype=np.uint8)
        self.lib.late_host_quant_rows(xf.ctypes.data, T, e_min, xq.ctypes.data, xe.ctypes.data)
        return xq, xe

    def host_header(self, seq: int, layer: int, T: int, lslot, lw, flags: int = FLAGS, e_min: int = E_MIN,
                    alpha: float = SITU_ALPHA, beta: float = SITU_BETA) -> bytes:
        ls = np.ascontiguousarray(np.asarray(lslot, dtype=np.int32).reshape(-1)[:late.PMAX])
        w = np.ascontiguousarray(np.asarray(lw, dtype=np.float32).reshape(late.PMAX, late.MAXT))
        out = np.zeros(self.layout["job_bytes"], dtype=np.uint8)
        self.lib.late_host_header(seq, layer, T, ls.ctypes.data, w.ctypes.data, flags, e_min, alpha, beta,
                                  out.ctypes.data)
        return out.tobytes()

    def host_coef(self, sums, rows: bool, eps: float):
        s = np.ascontiguousarray(np.asarray(sums, dtype=np.float32))
        c = np.zeros(4, dtype=np.float32)
        self.lib.late_host_coef(s.ctypes.data, 1 if rows else 0, eps, c.ctypes.data)
        return c


class Segment:
    """The pool's mailbox (cpustage_seg2.h): the pool uses the host address, the kernels the device pointer of the
    same pinned pages. Segment(lib): new anonymous zeroed pages pinned with late_map (tests, with a stand-in pool);
    Segment.attach(lib, pool): the pages of the engine's pool (cpupool.CpuPool), mapped by pool.map_device()."""

    def __init__(self, lib: Lib):
        self.lib = lib
        lo = lib.layout
        self.lo = lo
        self.nbytes = lo["bytes"]
        self.buf = mmap.mmap(-1, self.nbytes)
        self.u8 = np.frombuffer(self.buf, dtype=np.uint8)
        self.host = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        dptr = V()
        rc = lib.lib.late_map(V(self.host), self.nbytes, ctypes.byref(dptr))
        if rc != 0:
            raise RuntimeError(f"K31 late: pinning the pool segment failed (CUDA error {rc})")
        self.dev = dptr.value

    @classmethod
    def attach(cls, lib: Lib, pool, dev: int = None) -> "Segment":
        """The segment of a cpupool.CpuPool (its buffer and host address); dev: its device pointer (default
        pool.map_device(), which pins the pages; the CPU tests' emulated kernels pass the host address)."""
        seg = cls.__new__(cls)
        seg.lib, seg.lo, seg.nbytes = lib, lib.layout, lib.layout["bytes"]
        seg.buf, seg.u8, seg.host = pool.buf, pool.u8, pool.addr
        seg.dev = int(dev) if dev is not None else int(pool.map_device())
        return seg

    def get(self, off: int) -> int:
        return int(self.lib.lib.late_host_get(V(self.host), off))

    def set(self, off: int, v: int) -> None:
        self.lib.lib.late_host_set(V(self.host), off, v)

    def wait(self, off: int, v: int, timeout_s: float) -> bool:
        return self.lib.lib.late_host_wait(V(self.host), off, v, int(timeout_s * 1e9)) == 0

    def done_off(self, slot: int) -> int:
        return self.lo["done"] + 128 * slot

    def job_off(self, slot: int) -> int:
        return self.lo["job"] + slot * self.lo["job_bytes"]

    def x_off(self, slot: int) -> int:
        return self.lo["x"] + slot * self.lo["x_stride"]

    def res_off(self, slot: int) -> int:
        return self.lo["res"] + slot * self.lo["res_bytes"]

    def rows(self, slot: int, T: int) -> np.ndarray:
        """The result slot's first T rows (float32 [T, 3584], a view of the host pages)."""
        return np.frombuffer(self.buf, dtype=np.float32, count=T * late.H, offset=self.res_off(slot)).reshape(T, late.H)

    def pool_state(self) -> dict:
        return {k: self.get(self.lo["state"] + 8 * i) for i, k in enumerate(POOL_STATE)}


def _late_map(path: str) -> dict:
    """K31_LATE_MAP: {layer: records} from {"<layer>": n}, {"layers": {...}} or a round-13 plan {"<layer>": {"late": n}}."""
    raw = json.load(open(path))
    raw = raw.get("layers", raw)
    out = {}
    for k, v in raw.items():
        n = v.get("late", 0) if isinstance(v, dict) else v
        out[int(k)] = int(n)
    return out


class Tier:
    """The engine's late tier: the segment, the pool, the device state and buffers, and the per-call steps."""

    def __init__(self, dev, lib: Lib, seg: Segment, pool=None):
        self.dev = dev
        self.lib, self.seg, self.pool = lib, seg, pool
        self.st = torch.zeros(16, dtype=torch.int64, device=dev)          # LateState (late.cu)
        # the job sequence continues above the door and every done tag (cpupool.CpuPool.next_seq): a sender that
        # restarted below a published tag could match a stale done line
        self.st[0] = (pool.next_seq() - 1) if hasattr(pool, "next_seq") else self._door_seq()
        self.lslot = torch.full((late.PMAX,), -1, dtype=torch.int32, device=dev)
        self.lw = torch.zeros((late.PMAX, late.MAXT), dtype=torch.float32, device=dev)
        self.lmask = torch.full((late.MAXT, late.K), -1, dtype=torch.int32, device=dev)
        self.count = torch.zeros(1, dtype=torch.int64, device=dev)       # late picks planned
        self.mfix = torch.zeros((late.MAXT, late.H), dtype=torch.float32, device=dev)
        self.xl = torch.zeros((late.MAXT, late.H), dtype=torch.bfloat16, device=dev)
        self.coef = torch.zeros((late.MAXT, 4), dtype=torch.float32, device=dev)
        self.z = torch.zeros((late.MAXT, 7168), dtype=torch.float32, device=dev)
        self.stream = torch.cuda.Stream(device=dev)
        self.cfg: dict = {}          # layer -> (records, rank from, rank to, timeout us)
        # The late stream reads tensors of the main stream (the router's picks, unitk's rows, the latent rows, u, y):
        # they stay referenced until main has joined the late stream, or the caching allocator could give their memory
        # to a later main-stream tensor while the late stream still reads it (as hook's _FIN["keep"])
        self.sent = None             # (layer, T, event, mfix or None, kept tensors) of the job this call's layer sent
        self.pend = None             # (layer, T, event, y, kept tensors) of a landing the next layer joins
        self.n_join = self.n_drop = 0
        self.t_log = time.time()
        self.host_st = torch.zeros(16, dtype=torch.int64).pin_memory()
        self.host_cnt = torch.zeros(1, dtype=torch.int64).pin_memory()
        self.log_ev = None

    def _door_seq(self) -> int:
        return self.seg.get(self.seg.lo["door"]) >> 16

    # ---------------------------------------------------------------------------------------------- per call
    def flush(self, main) -> None:
        """A send or a landing nobody consumed (its layer took another path): join it on main and drop it, so that every
        fork of the late stream is joined in the same forward (a CUDA graph capture must end with none open)."""
        for k in ("sent", "pend"):
            p = getattr(self, k)
            if p is not None:
                main.wait_event(p[2])
                setattr(self, k, None)
                self.n_drop += 1

    def post(self, st, x, topk_ids, topk_weights, rows, main, ld_ev=None, breakable: bool = False) -> None:
        """After unitk (rows = its row ids): this call's late job, on the late stream."""
        self.flush(main)
        cfg = self.cfg.get(st.idx)
        T = topk_ids.shape[0]
        if cfg is None or cfg[0] <= 0 or T > late.MAXT or breakable or x.dim() != 2 or not x.is_contiguous() or \
                x.dtype != torch.bfloat16 or rows.dtype != torch.int32:
            return
        n, lfrom, lto, _ = cfg
        ls = self.stream
        ls.wait_stream(main)                              # the router's picks and unitk's rows
        with torch.cuda.stream(ls):
            late.plan(topk_ids, topk_weights, st.row_map, st.bank_map, rows, lfrom, lto, n, self.lslot, self.lw,
                      self.lmask, self.count, band_map=getattr(st, "band_map", None))
            mfix = None
            if getattr(st, "cut_mean", None) is not None:    # the late picks' share of the on-time mean
                late.mean(self.mfix, self.lmask, topk_ids, topk_weights, st.cut_mean)
                mfix = self.mfix
            if ld_ev is not None:                         # the latent rows (K31_SHARED_SIDE=3's pending down-proj)
                ls.wait_event(ld_ev)
            self.lib.send(self.seg.dev, self.st, self.lslot, self.lw, x, T, st.idx, stream=ls)
            ev = torch.cuda.Event()
            ev.record(ls)
        self.sent = (st.idx, T, ev, mfix, (x, topk_ids, topk_weights, rows))

    def rot(self, mod, hidden_states, residual, fin_take):
        """KimiRoutedOutputTransform.forward for a call whose layer sent a job: vLLM's three steps keeping u and y, then
        the landing and the completion GEMV on the late stream. None: not such a call (the caller runs its own path)."""
        s = self.sent
        if s is None:
            return None
        self.sent = None
        idx, T, sev, mfix, keep = s
        main = torch.cuda.current_stream()
        up = getattr(mod, "up_proj", None)
        i8 = getattr(up, "_k31_i8", None)
        norm = getattr(mod, "norm", None)
        if hidden_states.dim() != 2 or hidden_states.shape[0] != T or hidden_states.dtype != torch.bfloat16 or \
                i8 is None or norm is None or getattr(norm, "weight", None) is None:
            main.wait_event(sev)                          # not this call's shape: the job's rows are never read
            self.n_drop += 1
            return None
        fin_take()                                        # the band finish (K31_TAIL1) has written u
        u = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        xn = norm(u)                                      # vLLM's own steps (KimiRoutedOutputTransform.forward with a
        y, _ = up(xn)                                     # quantized up): RMSNorm, up GEMV, residual add
        out = residual.add_(y) if residual is not None else y
        ls = self.stream
        ls.wait_stream(main)                              # u is final (and y exists)
        with torch.cuda.stream(ls):
            self.lib.land(self.seg.dev, self.st, u, None, mfix, norm.weight, self.xl, self.coef, T,
                          float(norm.variance_epsilon), idx, self.cfg[idx][3], stream=ls)
            i8[0](self.xl[:T], self.z[:T])                # z = up(xl): the layer's int8 GEMV, float32 out
            ev = torch.cuda.Event()
            ev.record(ls)
        self.pend = (idx, T, ev, y, (u, keep))
        return out

    def join(self, layer_idx: int, hidden_states, residual):
        """KimiDecoderLayer._post_attn_norm of layer L + 1: the late share of layer L into its attention output, or into
        the stored block state L ended. Returns the hidden states to pass on."""
        p = self.pend
        if p is None or p[0] != layer_idx - 1:
            return hidden_states
        self.pend = None
        L, T, ev, y, _keep = p
        main = torch.cuda.current_stream()
        main.wait_event(ev)
        if hidden_states.dim() != 2 or hidden_states.shape[0] != T:
            self.n_drop += 1
            return hidden_states
        self.n_join += 1
        z = self.z[:T]
        if (L + 1) % ATTN_RES_BLOCK == 0 and residual is not None and residual.dim() == 3:
            late.add(residual[:, (L + 1) // ATTN_RES_BLOCK, :], z, y, self.coef)      # in place, row-strided
            return hidden_states
        out = torch.empty_like(hidden_states)
        late.add(hidden_states, z, y, self.coef, out=out)
        return out

    # ---------------------------------------------------------------------------------------------- reporting
    def counters(self) -> dict:
        v = self.st.tolist()
        return {k: int(v[i]) for i, k in enumerate(STATE_WORDS)}

    def tick(self) -> None:
        """Once per engine step (host only, no GPU wait): every LOG_S seconds one line of counters."""
        if self.log_ev is not None and self.log_ev.query():
            v = self.host_st.tolist()
            c = {k: int(v[i]) for i, k in enumerate(STATE_WORDS)}
            ps = self.pool.stats() if hasattr(self.pool, "stats") else self.seg.pool_state()
            nl = max(1, c["n_added"] + c["n_timeout"] + c["n_error"] + c["n_down"])
            _LOG(f"late tier: {c['n_jobs']} jobs ({c['n_recs']} records, {int(self.host_cnt[0])} picks), landed "
                 f"{c['n_added']} / timeout {c['n_timeout']} / refused {c['n_error']} / pool down {c['n_down']} / no job "
                 f"{c['n_empty']}, mean wait {c['wait_ns'] / nl / 1e3:.1f} us (max {c['wait_max'] / 1e3:.1f}), rows "
                 f"with a share {c['n_rows']}; pool: done {ps['done']}, abandoned {ps['abandoned']}, stale "
                 f"{ps['stale']}, mean job {ps['job_ns'] / max(1, ps['done']) / 1e3:.1f} us (max "
                 f"{ps.get('max_job_ns', ps.get('max_ns', 0)) / 1e3:.1f}); dropped calls {self.n_drop}")
            self.log_ev = None
        if self.log_ev is None and time.time() - self.t_log > LOG_S:
            self.t_log = time.time()
            self.host_st.copy_(self.st, non_blocking=True)
            self.host_cnt.copy_(self.count, non_blocking=True)
            self.log_ev = torch.cuda.Event()
            self.log_ev.record()


TIER: list = [None]


def _moe_modules(model) -> dict:
    """{layer: the KimiMoE module} of the served model, by its experts' K31 layer state."""
    out = {}
    for _name, m in model.named_modules():
        if type(m).__name__ == "KimiMoE":
            st = getattr(getattr(m, "experts", None), "_k31", None)
            if st is not None:
                out[st.idx] = m
    return out


def bind(layers: dict, dev, cfg, model, last_moe: int, defer: bool, fuse_up: bool, mode: str) -> Tier:
    """At the end of hook._bind (after the band and the residency, whose pooled maps the plan reads, and after
    cpupool.from_env started the Grace pool): each layer's records per call, the pool's segment mapped for the GB300,
    the self-check. Refuses what the tier cannot run."""
    if MODE == "gpu":
        raise RuntimeError("K31_LATE=gpu (late shares as MegaMoE pseudo rows) is not built; use K31_LATE=1")
    if not ON:
        raise RuntimeError(f"K31_LATE={MODE!r}: expected 1 (or pool)")
    if mode != "staged" or defer or fuse_up:
        raise RuntimeError("K31_LATE needs K31_MODE=staged, K31_DEFER=0 (the old deferral owns the late rows) and "
                           "K31_FUSE_UP=0 (the fused up leaves no y)")
    if not 1 <= len(TIMEOUT_US) <= 2:
        raise RuntimeError(f"K31_LATE_TIMEOUT_US={TIMEOUT_US}: one or two bounds (next layer KDA, MLA)")
    from . import cpupool
    pool = cpupool.POOL
    if pool is None:
        raise RuntimeError("K31_LATE=1 needs the Grace pool started at bind: K31_POOL=1 (engine/k31.sh sets it, with "
                           "the pool's core layout, whenever K31_LATE=1)")
    t0 = time.time()
    moes = _moe_modules(model)
    lmap = _late_map(MAP) if MAP else {}
    is_kda = getattr(cfg, "is_kda_layer", None)
    conf = {}
    for idx, st in sorted(layers.items()):
        n, lf, lt = lmap.get(idx, SLOTS), LFROM, LTO
        p = getattr(st, "policy", None)
        if p is not None and getattr(p, "late", None) is not None:
            (lf, lt), n = p.late, int(p.late_n)
        if idx == last_moe or not st.bank or getattr(st, "bank_cpu", None) is None:
            n = 0
        if not 0 <= n <= late.PMAX or not 0 <= lf <= lt <= 16:
            raise RuntimeError(f"K31 late: layer {idx}: {n} records per call (0..{late.PMAX}), ranks {lf}..{lt - 1}")
        if n > 0:
            m = moes.get(idx)
            rot = getattr(m, "routed_output_transform", None)
            if rot is None or getattr(getattr(rot, "up_proj", None), "_k31_i8", None) is None or \
                    getattr(rot, "norm", None) is None:
                raise RuntimeError(f"K31 late: layer {idx}'s routed up-projection is not on the int8 GEMV "
                                   f"(K31_DENSE_FMT routed_expert_up_proj:w8i) or has no RMSNorm")
            if pool.layers.get(idx, (None, 0))[1] != len(st.bank):
                raise RuntimeError(f"K31 late: the pool holds no bank of {len(st.bank)} records for layer {idx}")
        kda = bool(is_kda(idx + 1)) if callable(is_kda) else True
        conf[idx] = (n, lf, lt, TIMEOUT_US[0] if (kda or len(TIMEOUT_US) == 1) else TIMEOUT_US[1])
    if not any(c[0] > 0 for c in conf.values()):
        raise RuntimeError("K31_LATE=1 but no layer has late records (K31_LATE_SLOTS / K31_LATE_MAP / the policy)")
    lib = Lib(LIB)
    lo = lib.layout
    mine = (lo["bytes"], lo["door"], lo["state"], lo["done"], lo["job"], lo["job_bytes"], lo["x"], lo["x_stride"],
            late.MAXT * late.H, lo["res"], lo["res_bytes"], lo["nslot"], lo["pmax"], lo["maxt"], lo["job_size"])
    if mine != tuple(cpupool.LAYOUT):
        raise RuntimeError(f"K31 late: liblate.so's segment {mine} is not the pool's {tuple(cpupool.LAYOUT)}")
    seg = Segment.attach(lib, pool)
    if seg.get(lo["state"]) == 0:
        raise RuntimeError("K31 late: the pool does not report running (state word 0 of the segment)")
    tier = Tier(dev, lib, seg, pool)
    tier.cfg = conf
    TIER[0] = tier
    if SELFCHECK:
        _selfcheck(tier, layers, conf, moes)
    on = [i for i, c in conf.items() if c[0] > 0]
    ns = sorted({c[0] for c in conf.values() if c[0] > 0})
    _LOG(f"late tier (K31_LATE): {len(on)} layers ({on[0]}-{on[-1]}), {ns[0] if len(ns) == 1 else f'{ns[0]}-{ns[-1]}'} "
         f"records per call, ranks from the policy / {LFROM}..{LTO - 1}; landing bounds {TIMEOUT_US} us; on the Grace "
         f"pool ({pool.threads} threads from core {pool.core0}), jobs from sequence {int(tier.st[0]) + 1}; bound in "
         f"{time.time() - t0:.1f} s")
    return tier


@torch.no_grad()
def _selfcheck(tier: Tier, layers: dict, conf: dict, moes: dict) -> None:
    """One job posted by the GB300 (late_send) on a late layer's real records (3 picks, 5 rows): the pool's rows
    against its own single-thread kernel on the same records and rows (cpupool run_single: bit-equal, so the send's
    header and int8 rows are what the pool expects) and against the float64 reference; the landing that collects
    them bit-equal to the device-row landing of the same rows; a landing with no job adds nothing. Stops the bind on
    any difference."""
    idx = next((i for i, c in sorted(conf.items()) if c[0] > 0 and len(layers[i].bank) >= 3), None)
    if idx is None:
        return
    st = layers[idx]
    dev = tier.dev
    rot = moes[idx].routed_output_transform
    gw, eps = rot.norm.weight, float(rot.norm.variance_epsilon)
    g = torch.Generator(device="cpu").manual_seed(0)
    T, P = 5, 3
    x = (torch.randn((T, late.H), generator=g) * 0.5).to(dev, torch.bfloat16)
    u = (torch.randn((T, late.H), generator=g) * 0.5).to(dev, torch.bfloat16)
    lw = torch.zeros(late.PMAX, late.MAXT)
    lw[0, 0], lw[0, 3] = 0.31, 0.12
    lw[1, 1], lw[1, 2], lw[1, 4] = 0.22, 0.05, 0.40
    lw[2, 0] = 0.02
    tier.lslot.copy_(torch.tensor([0, 1, 2] + [-1] * (late.PMAX - P), dtype=torch.int32))
    tier.lw.copy_(lw)
    s = torch.cuda.Stream(device=dev)
    c0 = tier.counters()
    with torch.cuda.stream(s):
        tier.lib.send(tier.seg.dev, tier.st, tier.lslot, tier.lw, x, T, idx, stream=s)
    s.synchronize()
    tag = tier.counters()["tag"]
    slot = (tag >> 16) % tier.seg.lo["nslot"]
    if tag == 0 or not tier.seg.wait(tier.seg.done_off(slot), tag, 10.0):
        raise RuntimeError(f"K31 late selfcheck: the pool did not finish a 3-record job in 10 s (layer {idx}, tag {tag})")
    if tier.seg.get(tier.seg.done_off(slot) + 8) & 1:
        raise RuntimeError(f"K31 late selfcheck: the pool refused the job (layer {idx}, records 0-2)")
    got = tier.seg.rows(slot, T).copy()
    base = st.bank_cpu.data_ptr()
    single = tier.pool.run_single([base + p * late.REC_BYTES for p in range(P)], x.float().cpu().numpy(),
                                  lw[:P, :T].numpy())
    bit = bool(np.array_equal(got.view(np.uint32), single.view(np.uint32)))
    rows = torch.from_numpy(got).to(dev)
    experts = [late.decode_hi2(st.bank_cpu[p].to(dev)) for p in range(P)]
    _, _, xd = late.quant_rows_reference(x)
    want = late.pool_reference(experts, xd.to(dev), lw[:P].to(dev), FLAGS, SITU_ALPHA, SITU_BETA)
    rel = max(float((rows[t].double() - want[t]).norm() / want[t].norm().clamp_min(1e-30)) for t in range(T))
    with torch.cuda.stream(s):                    # the landing that collects the job (rows in host memory)
        tier.lib.land(tier.seg.dev, tier.st, u, None, None, gw, tier.xl, tier.coef, T, eps, idx, 1e6, stream=s)
    s.synchronize()
    xl1, cf1 = tier.xl[:T].clone(), tier.coef[:T].clone()
    with torch.cuda.stream(s):                    # a landing without a job
        tier.lib.land(tier.seg.dev, tier.st, u, None, None, gw, tier.xl, tier.coef, T, eps, idx, 1e6, stream=s)
    s.synchronize()
    cf2 = tier.coef[:T].clone()
    with torch.cuda.stream(s):                    # the same rows from device memory
        tier.lib.land(None, None, u, rows, None, gw, tier.xl, tier.coef, T, eps, idx, 0.0, stream=s)
    s.synchronize()
    same = bool(torch.equal(xl1, tier.xl[:T]) and torch.equal(cf1, tier.coef[:T]))
    cref = late.coef_reference(u, rows, eps)
    crel = float((cf1[:, 1:3].double() - cref[:, 1:3]).abs().max() / cref[:, 2].abs().max())
    xl_ok = bool(torch.equal(xl1, late.land_reference(u, rows, gw, cf1)))
    c1 = tier.counters()
    msg = (f"late selfcheck layer {idx}: a GB300-posted job of 3 records x 5 rows: the pool's rows "
           f"{'bit-equal to' if bit else 'DIFFERENT from'} its single-thread kernel, {rel:.2e} from the float64 "
           f"reference; landing from host memory = from device memory: {same}; coefficients against float64 "
           f"{crel:.1e}, xl {'bit-equal' if xl_ok else 'DIFFERENT'} to its reference; a landing with no job: has "
           f"{cf2[:, 0].tolist()} (counted added {c1['n_added'] - c0['n_added']}, empty {c1['n_empty'] - c0['n_empty']})")
    _LOG(msg)
    if not (bit and rel < 2e-2 and same and crel < 1e-4 and xl_ok and float(cf2[:, 0].abs().sum()) == 0.0 and
            c1["n_added"] - c0["n_added"] == 1 and c1["n_empty"] - c0["n_empty"] == 1):
        raise RuntimeError(f"K31 {msg}: stop")
    tier.lslot.fill_(-1)
    tier.lw.zero_()


def policy_from_plan(plan: dict, base_params: dict, base_env: dict = None) -> tuple:
    """A K31_POLICY file (dict) for a round-13 plan file ({"<layer>": {"mode": "A" | "N" | "S", "stage": s,
    "stage_cap": c, "late": n}}): per layer the plan's staged ranks (0..s-1; mode N stages nothing), late ranks 0-15 with
    n records per call (none when n = 0), and the base policy's hit / band / band_cap / mean (policy.load of the base,
    base_params). Returns (policy, notes): mode A needs K31_STAGE_ALL=1 (unitk all rows), and stage caps below 99 have
    no knob yet (listed in the notes; staging then takes up to 16 records per call)."""
    from . import policy as P
    per, capped, modes = {}, [], set()
    for L in P.LAYERS:
        b = base_params[L]
        q = plan.get(str(L), plan.get(L, {}))
        mode, s, cap, n = q.get("mode", "N"), int(q.get("stage", 0)), int(q.get("stage_cap", 99)), int(q.get("late", 0))
        modes.add(mode)
        if mode == "N":
            s = 0
        if s and cap < 99:
            capped.append(L)
        f = {"hit": f"0-{b.resmax - 1}" if b.resmax > 0 else "none", "band": "0-15" if b.band else "none",
             "band_cap": int(b.band_cap), "stage": f"0-{s - 1}" if s > 0 else "none",
             "mean": "0-15" if b.mean else "none"}
        if n > 0:
            f["late"], f["late_slots"] = "0-15", n
        per[L] = f
    names, layers = {}, {}
    for L in P.LAYERS:
        key = json.dumps(per[L], sort_keys=True)
        nm = names.setdefault(key, f"p{len(names)}")
        layers.setdefault(nm, []).append(L)

    def spans(ls):
        out, i = [], 0
        while i < len(ls):
            j = i
            while j + 1 < len(ls) and ls[j + 1] == ls[j] + 1:
                j += 1
            out.append(f"{ls[i]}-{ls[j]}" if j > i else f"{ls[i]}")
            i = j + 1
        return ",".join(out)
    pol = {"version": 1, "note": "plan: staged ranks and late records per layer (latetier.policy_from_plan)",
           "env": dict(base_env or {}), "policies": {nm: json.loads(k) for k, nm in names.items()},
           "layers": {spans(ls): nm for nm, ls in layers.items()}}
    notes = []
    if "A" in modes:
        notes.append("mode A (all rows): run with K31_STAGE_ALL=1")
    if capped:
        notes.append(f"stage caps below 99 in layers {spans(capped)}: no knob yet, those layers stage up to 16 records")
    return pol, notes


if __name__ == "__main__":            # python3 -m k31_engine.latetier PLAN BASE_POLICY OUT: a policy file for a plan
    from . import policy as _P
    _plan = json.load(open(sys.argv[1]))
    _pol, _notes = policy_from_plan(_plan, _P.load(sys.argv[2]), _P.env_section(sys.argv[2]))
    with open(sys.argv[3], "w") as _f:
        json.dump(_pol, _f, indent=1)
    print(f"wrote {sys.argv[3]}: {len(_pol['policies'])} policies; " + "; ".join(_notes))
    print(_P.describe(_P.load(sys.argv[3])))
