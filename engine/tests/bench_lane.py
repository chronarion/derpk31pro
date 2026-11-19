"""Copy-engine lane microbenchmark: can copy engines move 2-bit records
(9,289,728 B each) from pinned Grace memory into HBM beside a bandwidth-bound decode-like workload without slowing it?

Main stream: a CUDA graph of 92 x (a KDA-in_proj-sized BF16 GEMV [3 x 7168] @ [7168 x 49376]) ~ the step's dense
traffic. Copy stream (a host thread, no GPU sync on the main stream): records at a paced rate (records per main
"layer" period), cudaMemcpyAsync H2D into a ring of HBM landing slots. Reported per rate: main graph time vs alone,
copy bandwidth achieved. Then the mailbox round trip: a kernel writes an epoch to pinned memory, a host thread
spinning on it issues one record copy plus a tag copy, a device kernel spins on the tag; time from the epoch write to
the tag (GPU globaltimer), p50 / p99 over 200 trials, alone and under the main workload.
Run on the GB300 with nothing else on the GPU."""
import statistics
import threading
import time

import torch
import triton
import triton.language as tl

dev = torch.device("cuda")
REC = 9289728
NREC = 48
host = torch.empty((NREC, REC), dtype=torch.uint8).pin_memory()
host.random_(0, 255)
land = torch.empty((16, REC), dtype=torch.uint8, device=dev)

W = (torch.randn(49376, 7168, device=dev) * 0.02).to(torch.bfloat16)
X = torch.randn(3, 7168, device=dev, dtype=torch.bfloat16)
Y = torch.empty(3, 49376, device=dev, dtype=torch.bfloat16)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    torch.matmul(X, W.T, out=Y)
    with torch.cuda.graph(g):
        for _ in range(92):
            torch.matmul(X, W.T, out=Y)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()


def main_time(n=20):
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


alone = main_time()
layer_us = alone * 1e3 / 92
print(f"main graph alone: {alone:.2f} ms ({layer_us:.0f} us per 'layer', {92 * W.numel() * 2 / alone / 1e9:.0f} GB/s)")

cs = torch.cuda.Stream()


def copier(stop, per_layer, out):
    """Issue `per_layer` record copies every layer period (host-paced), no sync with the main stream."""
    n, t0 = 0, time.perf_counter()
    period = layer_us * 1e-6
    nxt = t0
    with torch.cuda.stream(cs):
        while not stop.is_set():
            for _ in range(per_layer):
                land[n % 16].copy_(host[n % NREC], non_blocking=True)
                n += 1
            nxt += period
            while time.perf_counter() < nxt and not stop.is_set():
                pass
    cs.synchronize()
    out.append((n, time.perf_counter() - t0))


for per_layer in (1, 2, 4, 6, 8):
    stop, out = threading.Event(), []
    th = threading.Thread(target=copier, args=(stop, per_layer, out))
    th.start()
    time.sleep(0.05)
    t = main_time()
    stop.set()
    th.join()
    n, dt = out[0]
    print(f"{per_layer} records per layer period: main {t:.2f} ms ({100 * (t / alone - 1):+.1f}%), copies "
          f"{n * REC / dt / 1e9:.0f} GB/s ({n} records in {dt:.2f} s)")


# ---- mailbox round trip
@triton.jit
def _post(mail_ptr, epoch, t_ptr):
    t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(t_ptr, t)
    tl.store(mail_ptr, epoch, cache_modifier=".wt")      # write-through to the pinned (system) mailbox


@triton.jit
def _wait(tag_ptr, epoch, t_ptr):
    v = tl.load(tag_ptr, volatile=True)
    while v != epoch:
        v = tl.load(tag_ptr, volatile=True)
    t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(t_ptr, t)


mail = torch.zeros(16, dtype=torch.int64).pin_memory()


class _Arr:                                                 # a CUDA view of pinned memory (UVA: same address)
    def __init__(self, t):
        self.__cuda_array_interface__ = {"shape": tuple(t.shape), "typestr": "<i8", "data": (t.data_ptr(), False),
                                         "strides": None, "version": 3}


mail_dev = torch.as_tensor(_Arr(mail), device=dev)
tag_host = torch.zeros(16, dtype=torch.int64).pin_memory()
tag = torch.zeros(16, dtype=torch.int64, device=dev)
tpost = torch.zeros(1, dtype=torch.int64, device=dev)
twait = torch.zeros(1, dtype=torch.int64, device=dev)
ls = torch.cuda.Stream()


def responder(stop):
    seen = 0
    with torch.cuda.stream(ls):
        while not stop.is_set():
            e = int(mail[0])
            if e != seen:
                seen = e
                land[e % 16].copy_(host[e % NREC], non_blocking=True)
                tag_host[0] = e
                tag[:1].copy_(tag_host[:1], non_blocking=True)


def roundtrip(trials, busy):
    stop = threading.Event()
    th = threading.Thread(target=responder, args=(stop,), daemon=True)
    th.start()
    ws = torch.cuda.Stream()
    lat = []
    for i in range(1, trials + 1):
        if busy:
            g.replay()                                     # main workload on the default stream
        with torch.cuda.stream(ws):
            _post[(1,)](mail_dev, i, tpost)
            _wait[(1,)](tag, i, twait)
        ws.synchronize()
        lat.append((int(twait) - int(tpost)) / 1e3)
    stop.set()
    torch.cuda.synchronize()
    lat.sort()
    return lat[len(lat) // 2], lat[int(len(lat) * 0.99) - 1]


for busy in (False, True):
    p50, p99 = roundtrip(200, busy)
    print(f"mailbox -> host thread -> 1 record + tag -> device ({'under the main workload' if busy else 'alone'}): "
          f"p50 {p50:.0f} us, p99 {p99:.0f} us")
