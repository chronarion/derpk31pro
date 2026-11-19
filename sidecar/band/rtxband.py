"""RTX band of derpk31pro: Python side of band/rtxband.cu.

- `repack(records)`: hi2 records (engine/store/build_store.py, format hi2-k1) -> band records: the same codes,
  permuted into the order the kernel's MMA fragments consume them, and the scale codes as 4-bit offsets from the
  record's lowest one (a one-time step when a record enters the band; lossless).
- `Band`: a device store of band records, the scratch buffers of a call, and `run()`.
- `mxfp8_encode()`: MXFP8 (E4M3 codes + UE8M0 scale codes) of a float tensor, the "ceil" scale rule of
  runner/k31/formats.py `mx_fake_quant` (what the GB300 side sends, and the reference's quantizer).
- `pack_x()`: row-major MXFP8 rows -> the band's input order.

Record layouts
  hi2 record (9,289,728 B): w13 codes [6144 rows, 896 B] (2 bits per weight, k ascending from bit 0), w2 codes
  [3584, 768 B], w13 scales int32 [28 K-groups][6144 scale rows], w2 scales int32 [24][3584]; rows in MegaMoE's kernel
  layout (w13 row 16 j + m = gate neuron 8 j + m, row 16 j + 8 + m = up neuron 8 j + m; scale row of weight row r =
  128 (r // 128) + 4 (r % 32) + (r % 128) // 32), one scale byte per 32 inputs, four per int32, stored code = native - 1.
  band record (8,773,632 B) + one base byte per record:
    w13 codes  [384 row tiles][28 steps][512 B]   a step = one 16-row tile x 128 inputs: for lane = 4 g + t a uint4 =
               four uint32 (K-blocks 4 step .. + 3), byte j of each holding the pairs of (row g, k), (row g + 8, k),
               (row g, k + 16), (row g + 8, k + 16) with k = 4 t + j inside the K-block, from bit 0 up
    w13 scales [384][28][16 rows x uint16]        four 4-bit codes per row and step, K-block j in bits 4j..4j+3:
               stored scale code - base, natural row order
    w2 codes   [224][24][512 B], w2 scales [224][24][32 B]   the same

K31_REC_FMT=hi2-k2 (read at import; the sidecar sets it from the record store's layout.json): the records are hi2-k2
(engine/k31_engine/recfmt.py: 8,527,872 B, the scales as a base byte per scale row and a 2-bit offset per scale, in
units of 512 scale rows with exact exception entries) and the band keeps that code, in its own order:
  band record (8,526,528 B) + one flag byte per record (1 = it has exception entries):
    w13 codes  as above
    w13 scales [384 row tiles]{16 row bases, [28 steps][16 rows x 1 B: four 2-bit codes, K-block j in bits 2j..2j+1]}
    w2 codes, w2 scales [224]{16, [24][16]}
    exception entries [76 blocks of 128 rows][4 x uint32]: row in the block | K-block << 7 | scale byte << 14,
               unused 0xFFFFFFFF (the hi2-k2 record's entries, one for one; a row may have two)
The libraries are built per format (build.sh: librtxband.so / librtxband_k2.so, and the service's).
"""
import ctypes
import os

import torch

H, I = 3584, 3072
MAXT, MAXE = 8, 16
K2 = os.environ.get("K31_REC_FMT", "hi2-k1") == "hi2-k2"
CODES13, CODES2 = 6144 * 896, 3584 * 768
SF13, SF2 = 28 * 6144 * 4, 24 * 3584 * 4      # scale bytes of the kernel tensors (a hi2-k1 record's)
BREC = 1056
KQ1, KQ2, RT1, RT2 = 28, 24, 384, 224
W13C, W2C = RT1 * KQ1 * 512, RT2 * KQ2 * 512
if K2:
    UROWS, UHEAD, XENT, XTILES = 512, 512 + 64, 4, (6144 + 3584) // 128      # hi2-k2 scale units; exception entries
    U13, U2 = UHEAD + KQ1 * UROWS, UHEAD + KQ2 * UROWS                       # 14,912 / 12,864 B per unit
    P13, P2 = (6144 // UROWS) * U13, (3584 // UROWS) * U2
    HI2_STATUS = CODES13 + CODES2 + P13 + P2
    HI2_REC = (HI2_STATUS + 4 * XTILES + 4095) // 4096 * 4096                # 8,527,872
    W13S, W2S = RT1 * (16 + KQ1 * 16), RT2 * (16 + KQ2 * 16)
    XOFF = W13C + W13S + W2C + W2S
    REC = XOFF + XTILES * XENT * 4            # 8,526,528
    LIB_SUFFIX = "_k2"
else:
    HI2_REC = 9289728
    W13S, W2S = RT1 * KQ1 * 32, RT2 * KQ2 * 32
    REC = W13C + W13S + W2C + W2S             # 8,773,632
    LIB_SUFFIX = ""
EMPTY = 0xFFFFFFFF
FLAG_WEIGHT_AFTER, FLAG_NO_BF16, FLAG_ACT_BF16, FLAG_PRECISE, FLAG_OUT_BF16 = 1, 2, 4, 8, 16
# The two arithmetics the band can mirror (both: gate / up rounded to bfloat16, MXFP8 intermediate, fp32 sum):
#   MEGAMOE  what DeepGEMM's MegaMoE SiTU kernel computes on the GB300: routing weight before the FP8 cast, fast-math
#            SiTU, block scale code floored at 105, each expert's output rounded to bfloat16 before the sum
#   RUNNER   runner/k31/stream.py: torch SiTU, activation rounded to bfloat16, no scale floor, each expert's output
#            rounded to bfloat16, then weighted and summed
FLAGS_MEGAMOE, SF_MIN_MEGAMOE = FLAG_OUT_BF16, 105
FLAGS_RUNNER, SF_MIN_RUNNER = FLAG_WEIGHT_AFTER | FLAG_ACT_BF16 | FLAG_PRECISE | FLAG_OUT_BF16, 0
XF_BYTES = KQ1 * BREC
SITU_ALPHA, SITU_BETA = 4.0, 25.0          # config.json: activation_situ_beta, activation_situ_linear_beta


def scale_row(rows: int, device=None) -> torch.Tensor:
    """Scale-tensor row of each weight row in the kernel layout (the UTCCP order of _kernel_layout_shard)."""
    r = torch.arange(rows, device=device)
    return 128 * (r // 128) + 4 * (r % 32) + (r % 128) // 32


def _codes_to_steps(c: torch.Tensor, kq: int) -> torch.Tensor:
    """[N, R, K/4] code bytes (row-major, 4 weights per byte) -> [N, R/16, kq, 512] step code bytes."""
    n, rows, _ = c.shape
    rt = rows // 16
    p = torch.stack([(c >> (2 * j)) & 3 for j in range(4)], dim=-1)          # [N, R, K/4, 4]: k = 4 byte + j
    # row = 16 rt + 8 h + g ; k = 128 kq + 32 kap + 16 half + 4 t + j
    p = p.view(n, rt, 2, 8, kq, 4, 2, 4, 4)                                    # rt, h, g, kq, kap, half, t, j
    p = p.permute(0, 1, 4, 3, 7, 5, 8, 6, 2)                                   # rt, kq, g, t, kap, j, half, h
    b = p[..., 0, 0] | (p[..., 0, 1] << 2) | (p[..., 1, 0] << 4) | (p[..., 1, 1] << 6)
    return b.reshape(n, rt, kq, 512)


def _scales_to_steps(s: torch.Tensor, rows: int, base: torch.Tensor) -> torch.Tensor:
    """[N, kq, R, 4] scale bytes (kernel scale-row order), base [N] -> [N, R/16, kq, 32]: per natural row a uint16 of
    four 4-bit codes (scale - base)."""
    n, kq = s.shape[0], s.shape[1]
    nib = s[:, :, scale_row(rows, s.device), :] - base.view(n, 1, 1, 1)        # [N, kq, R natural, 4], 0..15
    b = torch.stack((nib[..., 0] | (nib[..., 1] << 4), nib[..., 2] | (nib[..., 3] << 4)), dim=-1)
    return b.view(n, kq, rows // 16, 16, 2).permute(0, 2, 1, 3, 4).reshape(n, rows // 16, kq, 32)


def _k2_parts(r: torch.Tensor, off: int, rows: int, kq: int):
    """One tensor's hi2-k2 scale units of records r [n, HI2_REC] from byte `off` -> (bases uint8 [n, rows] and code
    bytes uint8 [n, kq, rows], both in scale-row order, exception entries int64 [n, rows / 128, 4])."""
    n, units = r.shape[0], rows // UROWS
    u = r[:, off:off + units * (UHEAD + kq * UROWS)].view(n, units, UHEAD + kq * UROWS)
    base = u[:, :, :UROWS].reshape(n, rows)
    codes = u[:, :, UHEAD:].view(n, units, kq, UROWS).permute(0, 2, 1, 3).reshape(n, kq, rows)
    e = u[:, :, UROWS:UHEAD].reshape(n, units * 16, 4).to(torch.int64)
    e = e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)
    return base, codes, e.view(n, rows // 128, XENT)


def scales_of(rec: torch.Tensor):
    """The scale bytes of hi2 records [n, HI2_REC] of either format as the kernel tensors hold them:
    (w13 uint8 [n, 28, 6144, 4], w2 uint8 [n, 24, 3584, 4]) (scale-row order, as stored: native - 1)."""
    n = rec.shape[0]
    o = CODES13 + CODES2
    if not K2:
        return rec[:, o:o + SF13].view(n, KQ1, 6144, 4), rec[:, o + SF13:o + SF13 + SF2].view(n, KQ2, 3584, 4)
    out = []
    for off, rows, kq in ((o, 6144, KQ1), (o + P13, 3584, KQ2)):
        base, codes, e = _k2_parts(rec, off, rows, kq)
        sh = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=rec.device)
        v = base.view(n, 1, rows, 1) + ((codes.unsqueeze(-1) >> sh) & 3)       # [n, kq, rows, 4]
        for i, t, j in (e != EMPTY).nonzero().tolist():
            w = int(e[i, t, j])
            v[i, ((w >> 9) & 127) // 4, (t // 4) * UROWS + (w & 511), ((w >> 9) & 127) % 4] = (w >> 16) & 255
        out.append(v)
    return out[0], out[1]


def _repack_k2(r: torch.Tensor):
    """hi2-k2 records [n, HI2_REC] -> (band records [n, REC], flag uint8 [n]: has exception entries, ok bool [n])."""
    n = r.shape[0]
    dev = r.device
    parts, ents = [], []
    o = CODES13 + CODES2
    for coff, soff, rows, row_b, kq in ((0, o, 6144, 896, KQ1), (CODES13, o + P13, 3584, 768, KQ2)):
        base, codes, e = _k2_parts(r, soff, rows, kq)
        sr = scale_row(rows, dev)                                              # natural row -> scale row
        tiles = torch.empty((n, rows // 16, 16 + kq * 16), dtype=torch.uint8, device=dev)
        tiles[:, :, :16] = base[:, sr].view(n, rows // 16, 16)
        tiles[:, :, 16:] = codes[:, :, sr].view(n, kq, rows // 16, 16).permute(0, 2, 1, 3).reshape(n, rows // 16, kq * 16)
        parts += [_codes_to_steps(r[:, coff:coff + rows * row_b].view(n, rows, row_b), kq).reshape(n, -1),
                  tiles.reshape(n, -1)]
        ents.append(e)
    e = torch.cat(ents, dim=1)                                                 # [n, 76, 4] hi2-k2 entries
    used = e != EMPTY
    p = e & 511                                                                # scale row in the unit: 128 b + 4 i + j
    x = (32 * (p & 3) + ((p & 127) >> 2)) | (((e >> 9) & 127) << 7) | (((e >> 16) & 255) << 14)
    x = torch.where(used, x, torch.full_like(x, EMPTY))
    xb = torch.stack([(x >> (8 * b)) & 0xFF for b in range(4)], dim=-1).to(torch.uint8).view(n, -1)
    # fits: no status word set, no row with more than two entries (a lane of the kernel patches two)
    st = r[:, HI2_STATUS:HI2_STATUS + 4 * XTILES]
    same = ((p.unsqueeze(-1) == p.unsqueeze(-2)) & used.unsqueeze(-1) & used.unsqueeze(-2)).sum(-1)   # [n, 76, 4]
    ok = (st == 0).all(1) & (same <= 2).all(-1).all(-1)
    return torch.cat(parts + [xb], dim=1), used.any(-1).any(-1).to(torch.uint8), ok


def repack(rec: torch.Tensor, chunk: int = 4):
    """hi2 records uint8 [N, HI2_REC] -> (band records uint8 [N, REC], base uint8 [N], ok bool [N]) on the same
    device. ok is False for a record whose scale codes do not fit 4 bits above its lowest one (or that has a zero
    scale code): such a record cannot enter the band (none of the 1,136 K3.1 records looked at is one).
    K31_REC_FMT=hi2-k2: `base` is 1 for a record with exception entries, else 0; ok is False for a record with a
    status word set or a row with more than two exception entries (none of the checkpoint's 82,432 is one)."""
    assert rec.dtype == torch.uint8 and rec.shape[-1] == HI2_REC
    if K2:
        out = torch.empty((rec.shape[0], REC), dtype=torch.uint8, device=rec.device)
        base = torch.empty(rec.shape[0], dtype=torch.uint8, device=rec.device)
        ok = torch.empty(rec.shape[0], dtype=torch.bool, device=rec.device)
        for a in range(0, rec.shape[0], chunk):
            out[a:a + chunk], base[a:a + chunk], ok[a:a + chunk] = _repack_k2(rec[a:a + chunk])
        return out, base, ok
    out = torch.empty((rec.shape[0], REC), dtype=torch.uint8, device=rec.device)
    base = torch.empty(rec.shape[0], dtype=torch.uint8, device=rec.device)
    ok = torch.empty(rec.shape[0], dtype=torch.bool, device=rec.device)
    for a in range(0, rec.shape[0], chunk):
        r = rec[a:a + chunk]
        n = r.shape[0]
        c13 = r[:, :CODES13].view(n, 6144, 896)
        c2 = r[:, CODES13:CODES13 + CODES2].view(n, 3584, 768)
        o = CODES13 + CODES2
        sf = r[:, o:o + SF13 + SF2]
        lo, hi = sf.amin(1), sf.amax(1)
        base[a:a + n], ok[a:a + n] = lo, (lo > 0) & (hi - lo <= 15)
        s13 = sf[:, :SF13].view(n, KQ1, 6144, 4)
        s2 = sf[:, SF13:].view(n, KQ2, 3584, 4)
        out[a:a + n] = torch.cat((_codes_to_steps(c13, KQ1).reshape(n, -1), _scales_to_steps(s13, 6144, lo).reshape(n, -1),
                                  _codes_to_steps(c2, KQ2).reshape(n, -1), _scales_to_steps(s2, 3584, lo).reshape(n, -1)),
                                 dim=1)
    return out, base, ok


def mxfp8_encode(x: torch.Tensor, sf_min: int = 0):
    """x [..., K] (K % 32 == 0) -> (E4M3 codes uint8 [..., K], UE8M0 scale codes uint8 [..., K/32], dequantized
    float32 [..., K]). Scale = the smallest power of two with amax / scale <= 448 ("ceil" rule), code floored at
    sf_min; elements round to nearest even."""
    shape = x.shape
    xf = x.float().reshape(-1, 32)
    amax = xf.abs().amax(-1, keepdim=True)
    bits = amax.view(torch.int32)
    code = (((bits + 0x1FFFFF) >> 23) - 8).clamp_min(sf_min)                   # DeepGEMM's get_ue8m0_sf_exp
    scale = torch.exp2(code.float() - 127.0)
    q = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    vals = (q.float() * scale).reshape(shape)
    return (q.view(torch.uint8).reshape(shape), code.to(torch.uint8).reshape(*shape[:-1], shape[-1] // 32), vals)


def pack_x(xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Row-major MXFP8 rows (xq uint8 [T, 3584] E4M3 codes, xs uint8 [T, 112] UE8M0 codes, T <= 8) -> the band's input
    uint8 [29568]: 28 activation step records, each [lane = 4 token + t][K-block 0..3][8 B: k = 4 t + 0..3 and
    16 + 4 t + 0..3 of the K-block] + [token][4 scale codes]; rows >= T are zero. This is the order the sender should
    write (the kernel `rtxband_pack_x` does the same on the device)."""
    T = xq.shape[0]
    x8 = torch.zeros(MAXT, H, dtype=torch.uint8, device=xq.device)
    s8 = torch.zeros(MAXT, H // 32, dtype=torch.uint8, device=xq.device)
    x8[:T], s8[:T] = xq, xs
    x = x8.view(MAXT, KQ1, 4, 2, 4, 4).permute(1, 0, 4, 2, 3, 5).reshape(KQ1, 1024)    # kq, token, t, kap, half, j
    s = s8.view(MAXT, KQ1, 4).permute(1, 0, 2).reshape(KQ1, 32)
    return torch.cat((x, s), dim=1).reshape(-1).contiguous()


def aq_index(device=None) -> torch.Tensor:
    """Byte offset inside one expert's activation records (24 x 1056 B) of (token c, intermediate i): [8, 3072]."""
    c = torch.arange(MAXT, device=device).view(-1, 1)
    i = torch.arange(I, device=device).view(1, -1)
    kap, n32 = i // 32, i % 32
    lane = 4 * c + (n32 % 16) // 4
    return (kap // 4) * BREC + lane * 32 + (kap % 4) * 8 + 4 * (n32 // 16) + n32 % 4


def aq_scale_index(device=None) -> torch.Tensor:
    """Byte offset of the scale code of (token c, K-block kap): [8, 96]."""
    c = torch.arange(MAXT, device=device).view(-1, 1)
    kap = torch.arange(I // 32, device=device).view(1, -1)
    return (kap // 4) * BREC + 1024 + 4 * c + kap % 4


class _RtxBand(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("store", "sbase", "aq", "ypart", "y", "sync")]


class Band:
    """A band store on one GPU and the buffers of a call. Records are addressed by slot."""

    def __init__(self, lib_path: str, slots: int, device="cuda"):
        self.lib = ctypes.CDLL(lib_path)
        self.dev = torch.device(device)
        sizes = (ctypes.c_longlong * 12)()
        self.lib.rtxband_sizes(sizes)
        self.sizes = list(sizes)
        assert self.sizes[0] == REC, (f"{lib_path} is built for band records of {self.sizes[0]} B, K31_REC_FMT "
                                      f"({'hi2-k2' if K2 else 'hi2-k1'}) asks for {REC}: use librtxband{LIB_SUFFIX}.so")
        assert self.sizes[0] == REC and self.sizes[1] == 4 * (8 + MAXE + MAXE * MAXT) and self.sizes[2] == XF_BYTES
        self.store = torch.empty((slots, REC), dtype=torch.uint8, device=self.dev)
        self.sbase = torch.zeros(slots, dtype=torch.uint8, device=self.dev)     # lowest scale code of each slot
        #                                                                         (hi2-k2: 1 = has exception entries)
        self.xf = torch.zeros(XF_BYTES, dtype=torch.uint8, device=self.dev)
        self.aq = torch.zeros(self.sizes[3], dtype=torch.uint8, device=self.dev)
        self.ypart = torch.zeros(self.sizes[4] // 4, dtype=torch.float32, device=self.dev)
        self.y = torch.zeros((MAXT, H), dtype=torch.float32, device=self.dev)
        self.sync = torch.zeros(self.sizes[6] // 4, dtype=torch.int32, device=self.dev)
        self.c = _RtxBand(self.store.data_ptr(), self.sbase.data_ptr(), self.aq.data_ptr(), self.ypart.data_ptr(),
                          self.y.data_ptr(), self.sync.data_ptr())
        V, B = ctypes.c_void_p, ctypes.POINTER(_RtxBand)
        self.lib.rtxband_run.argtypes = [B, V, V, ctypes.c_int, V]
        self.lib.rtxband_pack_x.argtypes = [V, V, ctypes.c_int, V, V]
        self.lib.rtxband_prefetch.argtypes = [B, V, V]
        self.lib.rtxband_repack.argtypes = [V] * 5
        self.lib.rtxband_bench_aux.argtypes = [ctypes.c_int, B, V, ctypes.c_int, V, V, V, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_double)]
        self._range = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._meta = torch.zeros(2, dtype=torch.uint8, device=self.dev)
        self.lib.rtxband_bench.argtypes = [B, V, ctypes.c_int, V, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

    def load(self, slot0: int, hi2: torch.Tensor):
        """Repack hi2 records [N, HI2_REC] (any device) into slots slot0 .. slot0 + N - 1."""
        for a in range(0, hi2.shape[0], 4):
            part = hi2[a:a + 4].to(self.dev)
            rec, base, ok = repack(part)
            assert bool(ok.all()), "a record's scale codes do not fit the band's code"
            self.store[slot0 + a:slot0 + a + part.shape[0]] = rec
            self.sbase[slot0 + a:slot0 + a + part.shape[0]] = base

    def load_device(self, slot: int, hi2: torch.Tensor) -> bool:
        """Repack one hi2 record [HI2_REC] already in device memory into `slot` with the device kernels (the path a
        record takes when it enters the band). Returns False if the record does not fit the band's scale code."""
        assert hi2.is_cuda and hi2.dtype == torch.uint8 and hi2.numel() == HI2_REC and hi2.is_contiguous()
        st = torch.cuda.current_stream(self.dev).cuda_stream
        rc = self.lib.rtxband_repack(hi2.data_ptr(), self.store[slot].data_ptr(), self._range.data_ptr(),
                                     self._meta.data_ptr(), st)
        assert rc == 0, f"launch failed: {rc}"
        base, ok = self._meta.tolist()
        self.sbase[slot] = base
        return bool(ok)

    def make_call(self, slots, weights, n_rows: int, flags: int = FLAGS_MEGAMOE, sf_min: int = SF_MIN_MEGAMOE,
                  alpha: float = SITU_ALPHA, beta: float = SITU_BETA) -> torch.Tensor:
        """Work list of one call as a device int32 tensor. slots: n band slots; weights [n, 8] routing weights (0 =
        the row does not use the expert)."""
        n = len(slots)
        assert 0 <= n <= MAXE and 1 <= n_rows <= MAXT
        call = torch.zeros(8 + MAXE + MAXE * MAXT, dtype=torch.int32)
        call[0], call[1], call[2], call[3] = n, n_rows, flags, sf_min
        call[4:6] = torch.tensor([alpha, beta], dtype=torch.float32).view(torch.int32)
        call[8:8 + n] = torch.as_tensor(slots, dtype=torch.int32)
        w = torch.zeros(MAXE, MAXT, dtype=torch.float32)
        if n:
            w[:n] = torch.as_tensor(weights, dtype=torch.float32).reshape(n, MAXT).cpu()
            w[:n, n_rows:] = 0
        call[8 + MAXE:] = w.view(-1).view(torch.int32)
        return call.to(self.dev)

    def pack_x(self, xq: torch.Tensor, xs: torch.Tensor, sync: bool = True) -> torch.Tensor:
        """Row-major MXFP8 rows on the device (xq uint8 [T, 3584], xs uint8 [T, 112]) -> the band's input buffer,
        by the device kernel."""
        assert xq.dtype == torch.uint8 and xs.dtype == torch.uint8 and xq.is_cuda and xs.is_cuda
        assert xq.is_contiguous() and xs.is_contiguous() and xq.shape[0] == xs.shape[0] <= MAXT
        st = torch.cuda.current_stream(self.dev).cuda_stream
        rc = self.lib.rtxband_pack_x(xq.data_ptr(), xs.data_ptr(), xq.shape[0], self.xf.data_ptr(), st)
        assert rc == 0, f"launch failed: {rc}"
        if sync:
            torch.cuda.synchronize(self.dev)
        return self.xf

    def prefetch(self, call: torch.Tensor):
        """Hint: pull the records of a work list's slots into the L2 cache (asynchronous; never changes results)."""
        st = torch.cuda.current_stream(self.dev).cuda_stream
        rc = self.lib.rtxband_prefetch(ctypes.byref(self.c), call.data_ptr(), st)
        assert rc == 0, f"launch failed: {rc}"

    def run(self, call: torch.Tensor, xf: torch.Tensor = None, fused: bool = True, sync: bool = True) -> torch.Tensor:
        """xf uint8 [29568] = the input in fragment order (`pack_x`), device memory; default the band's own buffer.
        fused: one launch (default) or two. Returns y float32 [8, 3584], the band's output buffer (rows the call
        does not use are zero)."""
        xf = self.xf if xf is None else xf
        assert xf.dtype == torch.uint8 and xf.is_cuda and xf.numel() == XF_BYTES and xf.is_contiguous()
        st = torch.cuda.current_stream(self.dev).cuda_stream
        rc = self.lib.rtxband_run(ctypes.byref(self.c), call.data_ptr(), xf.data_ptr(), int(fused), st)
        assert rc == 0, f"launch failed: {rc}"
        if sync:
            torch.cuda.synchronize(self.dev)
        return self.y

    def bench_repack(self, hi2: torch.Tensor, slot0: int = 0, iters: int = 5):
        """Seconds per record of the device repack of hi2 [n, HI2_REC] (device) into slots slot0 .. (mean, best)."""
        out = (ctypes.c_double * 2)()
        rc = self.lib.rtxband_bench_aux(0, ctypes.byref(self.c), hi2.data_ptr(), hi2.shape[0],
                                        self.store[slot0].data_ptr(), self._range.data_ptr(), self._meta.data_ptr(),
                                        iters, out)
        assert rc == 0
        return out[0], out[1]

    def bench_prefetch(self, calls: torch.Tensor, iters: int = 200):
        """Seconds per L2 prefetch hint, launched and synchronized (median, p90)."""
        out = (ctypes.c_double * 2)()
        rc = self.lib.rtxband_bench_aux(1, ctypes.byref(self.c), calls.data_ptr(), calls.shape[0], None, None, None,
                                        iters, out)
        assert rc == 0
        return out[0], out[1]

    def error(self) -> bool:
        """True if a wait inside a call gave up (its output is invalid); clears the flag and the counters."""
        bad = bool(self.sync[-1].item())
        if bad:
            self.sync.zero_()
        return bad

    def bench(self, calls: torch.Tensor, xf: torch.Tensor, mode: int, which: int = 4, iters: int = 200,
              pause_us: int = 0):
        """calls int32 [n, 152] on the device. which: 4 fused, 3 two launches, 1 / 2 one linear. Returns (seconds per
        call, p90 or best)."""
        out = (ctypes.c_double * 2)()
        rc = self.lib.rtxband_bench(ctypes.byref(self.c), calls.data_ptr(), calls.shape[0], xf.data_ptr(), mode, which,
                                    iters, pause_us, out)
        assert rc == 0, f"bench failed: {rc}"
        return out[0], out[1]


def build(src_dir: str, out: str, defines=()) -> str:
    """Compile rtxband.cu into a shared library (needs nvcc 13+; sm_120a through -gencode) for this module's record
    format (K31_REC_FMT)."""
    import subprocess
    cmd = ["nvcc", "-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-shared", "-Xcompiler",
           "-fPIC", *[f"-D{d}" for d in list(defines) + (["BAND_K2=1"] if K2 else [])], "-o", out,
           os.path.join(src_dir, "rtxband.cu")]
    subprocess.run(cmd, check=True)
    return out
