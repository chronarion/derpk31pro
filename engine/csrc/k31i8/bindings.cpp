// derpk31pro: Hecate's int8 decode GEMVs (dense_i8.cu, copied unchanged from hecate-overlay
// docker/patches/exllamav3/fold) as a standalone torch extension for the K3.1 engine.
// Only the non-graph entry points: the engine captures its own CUDA graphs.
#include <torch/extension.h>
#include "dense_i8.cuh"

// dense_i8.cu records graph parameters only when a Graph is passed (never here); this keeps the linker happy without
// exllamav3's graph.cu and its device-context dependencies.
void Graph::record_param(void*, int, int, int) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_i8", &dense_i8, "y [M, N] (fp16 or fp32) = x [M, K] fp16 @ dequant(w8, scale)^T, 1..8 rows");
  m.def("dense_i8_dual", &dense_i8_dual, "two int8 GEMVs on one x, one launch");
  m.def("dense_i8_gu_act", &dense_i8_gu_act, "gate + up int8 GEMVs and the SiLU product, one launch");
}
