// The bulk band request, GB300 side (k31_engine/band.py Bulk, K31_BAND_BULK=1): a prompt chunk's band picks go to the
// RTX in one request per MoE layer, and their summed rows come back in one result (the band sidecar's band/bulk.py
// serves it; the segment's layout is in that file and in the segment itself).
//
//   bandbulk_post    the chunk's MXFP8 rows, each pick's band store slot and weight -> the segment, then the header and
//                    the request's tag in the door word. Every block fences its writes (system scope) before it counts
//                    itself; the last block writes the header and stores the tag with release semantics.
//   bandbulk_finish  waits (bounded) until the done word holds the tag, then y [T][3584] bf16 = bf16(y + result).
//                    A timeout or an error status leaves y as it is and counts in err[0] / err[1].
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC \
//          -o libbandbulk.so bandbulk.cu
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int H = 3584, K = 16;
constexpr uint64_t OFF_DOOR = 128, OFF_DONE = 512, OFF_STATUS = 520, OFF_HDR = 640;

__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void copy16(uint4* dst, const uint4* src, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

struct Post {
  uint8_t* seg;
  long long off_xq, off_xs, off_slot, off_w;
  const uint8_t* xq;      // [T][3584] E4M3
  const uint8_t* xs;      // [T][112] UE8M0
  const int16_t* slot;    // [T][16]
  const float* w;         // [T][16]
  int T, layer, flags, sf_min;
  float alpha, beta;
  uint64_t tag;
  unsigned int* counter;  // device word, 0 between requests
};

__global__ void bulk_post(Post p) {
  const long long T = p.T;                      // per row: 3,584 + 112 code bytes, 32 B of slots, 64 B of weights
  copy16(reinterpret_cast<uint4*>(p.seg + p.off_xq), reinterpret_cast<const uint4*>(p.xq), T * H / 16);
  copy16(reinterpret_cast<uint4*>(p.seg + p.off_xs), reinterpret_cast<const uint4*>(p.xs), T * (H / 32) / 16);
  copy16(reinterpret_cast<uint4*>(p.seg + p.off_slot), reinterpret_cast<const uint4*>(p.slot), T * K * 2 / 16);
  copy16(reinterpret_cast<uint4*>(p.seg + p.off_w), reinterpret_cast<const uint4*>(p.w), T * K * 4 / 16);
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) return;
  if (atomicAdd(p.counter, 1u) != gridDim.x - 1) return;
  __threadfence_system();                       // the last block: every block's rows are out before the header
  uint32_t* h = reinterpret_cast<uint32_t*>(p.seg + OFF_HDR);
  h[0] = (uint32_t)p.T; h[1] = (uint32_t)p.layer; h[2] = (uint32_t)p.flags; h[3] = (uint32_t)p.sf_min;
  h[4] = __float_as_uint(p.alpha); h[5] = __float_as_uint(p.beta); h[6] = 0u;
  __threadfence_system();
  st_release_sys(reinterpret_cast<uint64_t*>(p.seg + OFF_DOOR), p.tag);
  *p.counter = 0u;                              // the next request's launch follows this one on the stream
}

__global__ void bulk_finish(const uint8_t* seg, long long off_res, __nv_bfloat16* y, long long y_stride, int T,
                            uint64_t tag, unsigned long long timeout_ns, int* err) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    const uint64_t* done = reinterpret_cast<const uint64_t*>(seg + OFF_DONE);
    const uint64_t t0 = globaltimer();
    int o;
    while (true) {
      if (ld_acquire_sys(done) == tag) {
        o = ld_acquire_sys(reinterpret_cast<const uint64_t*>(seg + OFF_STATUS)) == 0 ? 1 : 0;  // 2: no band pick
        break;
      }
      if (globaltimer() - t0 > timeout_ns) { o = -1; break; }
      __nanosleep(256);
    }
    ok = o;
    if (o < 0 && blockIdx.x == 0) atomicAdd(err, 1);
    if (o == 0 && blockIdx.x == 0 && ld_acquire_sys(reinterpret_cast<const uint64_t*>(seg + OFF_STATUS)) == 1)
      atomicAdd(err + 1, 1);
  }
  __syncthreads();
  if (ok <= 0) return;
  const float* res = reinterpret_cast<const float*>(seg + off_res);
  const long long n4 = (long long)T * (H / 4);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    const long long t = i / (H / 4), c = (i % (H / 4)) * 4;
    const float4 r = __ldcv(reinterpret_cast<const float4*>(res + t * H + c));
    __nv_bfloat162* yp = reinterpret_cast<__nv_bfloat162*>(y + t * y_stride + c);
    const float2 a = __bfloat1622float2(yp[0]), b = __bfloat1622float2(yp[1]);
    yp[0] = __floats2bfloat162_rn(a.x + r.x, a.y + r.y);
    yp[1] = __floats2bfloat162_rn(b.x + r.z, b.y + r.w);
  }
}

}  // namespace

extern "C" {

// The segment (host memory, this process's mapping) registered with CUDA; *dptr = its device address.
int bandbulk_map(void* addr, long long bytes, void** dptr) {
  cudaError_t e = cudaHostRegister(addr, (size_t)bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (e != cudaSuccess && e != cudaErrorHostMemoryAlreadyRegistered) return (int)e;
  cudaGetLastError();
  return (int)cudaHostGetDevicePointer(dptr, addr, 0);
}

int bandbulk_post(void* seg, const long long* offs, const void* xq, const void* xs, const void* slot, const void* w,
                  int T, int layer, int flags, int sf_min, float alpha, float beta, unsigned long long tag,
                  void* counter, void* stream) {
  Post p{(uint8_t*)seg, offs[0], offs[1], offs[2], offs[3], (const uint8_t*)xq, (const uint8_t*)xs,
         (const int16_t*)slot, (const float*)w, T, layer, flags, sf_min, alpha, beta, tag, (unsigned int*)counter};
  bulk_post<<<296, 256, 0, (cudaStream_t)stream>>>(p);
  return (int)cudaGetLastError();
}

int bandbulk_finish(const void* seg, long long off_res, void* y, long long y_stride, int T, unsigned long long tag,
                    unsigned long long timeout_ns, void* err, void* stream) {
  bulk_finish<<<1184, 256, 0, (cudaStream_t)stream>>>((const uint8_t*)seg, off_res, (__nv_bfloat16*)y, y_stride, T,
                                                      tag, timeout_ns, (int*)err);
  return (int)cudaGetLastError();
}

}  // extern "C"
