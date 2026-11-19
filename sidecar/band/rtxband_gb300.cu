// RTX band of derpk31pro: the GB300 side of the mailbox as kernels (the device functions are in band_gb300.cuh).
// Each kernel is one CTA with fixed launch parameters and everything per call in device memory, so a CUDA graph
// replays them: band_send, band_finish, band_late, band_hint. The state block (bandgb::GbState, 160 B of device
// memory) carries the seq between the kernels of a call, the outcome of the last finish and the late call.
//
// Also here: the replay of a trace through these kernels (the round-trip benchmark) and a kernel that streams host
// memory, to measure the send and the finish while the GB300 reads Grace memory at full rate.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC -o librtxband_gb300.so
//        rtxband_gb300.cu
#include <cuda_runtime.h>

#include "band_gb300.cuh"

#include <ctime>

namespace {
using namespace bandgb;

constexpr int SEND_THREADS = 512, FIN_THREADS = 512;

struct GbArgs {                        // what the kernels work on; strides are per call (0 = always the same buffer)
  uint8_t* seg;                        // the mailbox as the GB300 sees it
  GbState* st;
  const uint8_t* xq; long long xq_stride;        // [T][3584] E4M3 bytes, row-major
  const uint8_t* xs; long long xs_stride;        // [T][112] UE8M0 codes
  const uint8_t* calls; long long call_stride;   // the work list (band::BandCall, 608 B)
  const int* hints; long long hint_stride;       // int32: count, 16 store slots
  float* dst; long long dst_stride;              // [T][3584] float32: the result rows are added here
  float* late; long long late_stride;            // [T][3584] float32: a late call's rows are added here
  unsigned long long* log; long long log_stride; // LOG_WORDS per call (may be null)
  int layers, pad;                               // layer field of the tag = idx % layers
  unsigned long long timeout_ns;                 // the finish's spin bound
  unsigned long long work_ns;                    // benchmark: the finish starts this long after the tag was published
                                                 // (the layer's own kernels between send and finish)
  unsigned long long lead_ns;                    // benchmark: the send starts this long after the hint's tag was
                                                 // published (the attention between the pre-gated hint and the send)
};

__global__ void band_send(const GbArgs a) {
  const unsigned long long idx = a.st->idx;
  if (a.lead_ns != 0) {
    if (threadIdx.x == 0) {
      const unsigned long long t = a.st->t_hint;
      while (globaltimer() - t < a.lead_ns) {}
    }
    __syncthreads();
  }
  send_body(a.seg, a.st, a.xq + idx * a.xq_stride, a.xs + idx * a.xs_stride, a.calls + idx * a.call_stride,
            (uint32_t)(idx % (unsigned long long)max(a.layers, 1)), a.log != nullptr ? a.log + idx * a.log_stride : nullptr);
}

__global__ void band_finish(const GbArgs a) {
  __shared__ unsigned long long sh[4];
  const unsigned long long idx = a.st->idx;
  if (a.work_ns != 0 && threadIdx.x == 0) {
    const unsigned long long t = a.st->t_pub;
    while (globaltimer() - t < a.work_ns) {}
  }
  finish_body(a.seg, a.st, a.dst + idx * a.dst_stride, a.timeout_ns, sh,
              a.log != nullptr ? a.log + idx * a.log_stride : nullptr);
}

__global__ void band_late(const GbArgs a) {
  __shared__ unsigned long long sh[2];
  late_body(a.seg, a.st, a.late, a.late_stride, sh, a.log, a.log_stride);
}

__global__ void band_hint(const GbArgs a) { hint_body(a.seg, a.st, a.hints + a.st->idx * a.hint_stride); }

// Reads `bytes` of host memory over and over, 16 B per thread and turn, the grid sweeping it front to back, until
// duration_ns have passed or stats[2] is set. stats: [0] bytes read, [1] ns, [2] stop, [3] unused.
__global__ void band_stream(const uint8_t* host, unsigned long long bytes, unsigned long long duration_ns,
                            unsigned long long* stats) {
  const unsigned long long t0 = globaltimer();
  const unsigned long long n16 = bytes / 16, step = (unsigned long long)gridDim.x * blockDim.x;
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x, count = 0;
  unsigned acc = 0;
  for (;;) {
#pragma unroll 8
    for (int k = 0; k < 32; k++) {
      const uint4 v = ld_volatile_u4(host + 16 * i);
      acc ^= v.x ^ v.w;
      i += step;
      if (i >= n16) i -= n16;
    }
    count += 32;
    if (globaltimer() - t0 > duration_ns || ld_volatile_u32(&stats[2]) != 0) break;
  }
  atomicAdd(&stats[0], count * 16);
  if (blockIdx.x == 0 && threadIdx.x == 0) stats[1] = globaltimer() - t0;
  if (acc == 0x9E3779B9u) atomicAdd(&stats[3], 1ull);
}

long long now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

cudaStream_t g_load = nullptr;
unsigned long long* g_stats = nullptr;

}  // namespace

extern "C" {

// The caller's view of GbArgs (same layout).
struct RtxBandGb {
  void* seg; void* st;
  const void* xq; long long xq_stride;
  const void* xs; long long xs_stride;
  const void* calls; long long call_stride;
  const void* hints; long long hint_stride;
  void* dst; long long dst_stride;
  void* late; long long late_stride;
  void* log; long long log_stride;
  int layers, pad;
  unsigned long long timeout_ns;
  unsigned long long work_ns;
  unsigned long long lead_ns;
};
static_assert(sizeof(RtxBandGb) == sizeof(GbArgs), "argument block");

// [state block bytes, log words per call]
void rtxband_gb_sizes(long long* out) { out[0] = sizeof(GbState); out[1] = LOG_WORDS; }

// Register a host mapping (the mailbox, or a buffer for rtxband_gb_load_start) with CUDA for the current device;
// *dev = the pointer kernels use.
int rtxband_gb_map(void* host, long long bytes, void** dev) {
  cudaError_t e = cudaHostRegister(host, (size_t)bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaHostGetDevicePointer(dev, host, 0);
}

// One launch each; inside a stream capture they become nodes of the caller's graph.
int rtxband_gb_send(const RtxBandGb* a, void* stream) {
  band_send<<<1, SEND_THREADS, 0, (cudaStream_t)stream>>>(*reinterpret_cast<const GbArgs*>(a));
  return (int)cudaGetLastError();
}
int rtxband_gb_finish(const RtxBandGb* a, void* stream) {
  band_finish<<<1, FIN_THREADS, 0, (cudaStream_t)stream>>>(*reinterpret_cast<const GbArgs*>(a));
  return (int)cudaGetLastError();
}
int rtxband_gb_late(const RtxBandGb* a, void* stream) {
  band_late<<<1, FIN_THREADS, 0, (cudaStream_t)stream>>>(*reinterpret_cast<const GbArgs*>(a));
  return (int)cudaGetLastError();
}
int rtxband_gb_hint(const RtxBandGb* a, void* stream) {
  band_hint<<<1, 32, 0, (cudaStream_t)stream>>>(*reinterpret_cast<const GbArgs*>(a));
  return (int)cudaGetLastError();
}

// Replay n calls of a trace: a CUDA graph of [late,] [hint,] send, finish launched once per period_ns (0 = back to
// back) and synchronized. With hints (a->hints set) the call's hint list goes out first and the send waits on the
// device until a->lead_ns after it (the lead is then exact whatever the host or another process on the GPU does).
// with_late = 1 puts band_late in front (it collects the previous call's rows if that call was late: the engine's
// order); with_late = 2 puts it right behind the finish instead (a call that was late cannot be there yet: every
// late call is lost; for tests). host_ns[2 i] = graph launch -> synchronized, [2 i + 1] = how late the launch was
// against its slot. The device log has the times that matter. Returns a CUDA error code.
int rtxband_gb_trace(const RtxBandGb* a, int n, long long period_ns, int with_late, long long* host_ns) {
  cudaStream_t st;
  cudaError_t e = cudaStreamCreate(&st);
  if (e != cudaSuccess) return (int)e;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
  if (with_late == 1) rtxband_gb_late(a, st);
  if (a->hints != nullptr) rtxband_gb_hint(a, st);
  rtxband_gb_send(a, st);
  rtxband_gb_finish(a, st);
  if (with_late == 2) rtxband_gb_late(a, st);
  cudaStreamEndCapture(st, &graph);
  e = cudaGraphInstantiate(&exec, graph, 0);
  if (e != cudaSuccess) return (int)e;
  const long long t_begin = now_ns() + 500000;
  for (int i = 0; i < n; i++) {
    const long long due = t_begin + (long long)i * period_ns;
    if (period_ns > 0) while (now_ns() < due) {}
    const long long t0 = now_ns();
    cudaGraphLaunch(exec, st);
    cudaStreamSynchronize(st);
    host_ns[2 * i] = now_ns() - t0;
    host_ns[2 * i + 1] = t0 - due;
  }
  if (with_late == 1) {                                  // the last call's rows, if it was late
    if (period_ns > 0) while (now_ns() < t_begin + (long long)n * period_ns) {}
    rtxband_gb_late(a, st);
  }
  e = cudaStreamSynchronize(st);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(st);
  return (int)e;
}

// A stand-in for a kernel that streams Grace memory (the engine's stager): `grid` CTAs of 256 threads read the
// mapped host buffer for duration_ns on their own stream. rtxband_gb_load_stop ends it early and returns
// [bytes read, ns] in out.
int rtxband_gb_load_start(const void* host_dev, long long bytes, int grid, long long duration_ns) {
  cudaError_t e = cudaSuccess;
  if (g_load == nullptr) e = cudaStreamCreateWithFlags(&g_load, cudaStreamNonBlocking);
  if (e == cudaSuccess && g_stats == nullptr) e = cudaMalloc(&g_stats, 64);
  if (e != cudaSuccess) return (int)e;
  cudaMemsetAsync(g_stats, 0, 64, g_load);
  band_stream<<<grid, 256, 0, g_load>>>((const uint8_t*)host_dev, (unsigned long long)bytes,
                                         (unsigned long long)duration_ns, g_stats);
  return (int)cudaGetLastError();
}
int rtxband_gb_load_stop(long long* out) {
  if (g_load == nullptr) return 0;
  cudaStream_t s;                                        // the stop word goes in beside the running kernel
  cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (e != cudaSuccess) return (int)e;
  unsigned long long one = 1;
  cudaMemcpyAsync(g_stats + 2, &one, 8, cudaMemcpyHostToDevice, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);
  e = cudaStreamSynchronize(g_load);
  unsigned long long stats[4] = {0, 0, 0, 0};
  cudaMemcpy(stats, g_stats, 32, cudaMemcpyDeviceToHost);
  out[0] = (long long)stats[0]; out[1] = (long long)stats[1];
  return (int)e;
}

}  // extern "C"
