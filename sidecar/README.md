# The RTX band

The RTX PRO 6000 holds about 11,000 experts as 2-bit records and computes them while the GB300 runs the rest of the
layer. This directory is the RTX side: a sidecar process in its own container (`engine/k31.sh` starts it when the
profile sets `K31_BAND_N`).

- **The store** (`band/rtxband.py`, `band/rtxband.cu`): each layer's band records, repacked once into the order the
  kernel's matrix instructions read them (lossless).
- **The service kernel** (`band/rtxband_svc.cu`, `band/service.py`): one persistent kernel that holds every SM of the
  RTX and serves the GB300's calls through a pinned mailbox in shared memory (`band/band_mailbox.h`): the GB300
  writes a layer's activations and work list, the RTX computes its experts and writes the rows back, and the GB300's
  finish kernel adds them in. Nothing else launches on the RTX while it runs.
- **Entries** (`band/sidecar.py`): the engine asks for experts to enter the band (a promoted expert's slot taken by
  a better candidate) through a command ring; the sidecar reads the record from the record store (or, with
  `K31_BAND_RING`, from a ring in shared memory that the engine fills from its CPU-memory copies) and repacks it into
  the slot while the kernel keeps serving.
- **The GB300 side** (`band/rtxband_gb300.cu`, `band/rtxband_fused.cu`, `band/band_gb300.cuh`): the send and finish
  kernels the engine runs inside its CUDA graphs, built into `engine/band/` by `../build.sh`.

<details>
<summary>Tests</summary>

`tests/test_band.py` (the band kernel against the torch reference), `tests/test_native.py` (records and arithmetic
against Moonshot's MXFP4 experts), `tests/test_service.py` (the service kernel, driven from a CPU thread),
`tests/test_roundtrip.py` (the GB300 side against the service kernel). The engine's end-to-end band test is
`engine/tests/test_band_e2e.py`.
</details>
