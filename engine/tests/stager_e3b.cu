// derpk31pro E3-b / E3-d, GB300 (sm_103a).
// E3-b, the stager: 2-bit expert records in pinned Grace memory -> native FP4 rows in an HBM staging slot. A record
// is one expert's codes (2 bits per weight, four per byte, already in the kernel's tile order) followed by its 6-bit
// scale codes; staging expands each code byte to two nibble bytes through a 256-entry table (the depth-2 code map,
// pair -> (pair << 2) | 3) and copies the scale bytes. Reads are 1-D TMA bulk copies (cp.async.bulk) into a ring of
// shared-memory stages; writes are 16-byte stores. Measured alone on NSM SMs, and beside a "busy" kernel streaming
// HBM on the other SMs (a stand-in for the native MegaMoE call): record bytes per second and the busy kernel's
// slowdown. Gate: >= 280 GB/s of record bytes beside the call, busy kernel slowed <= 15%, bytes equal to a CPU
// expansion.
// E3-d, the mailbox hazard (Hecate: hand-off writes 6 -> 100+ us while the GB300 streams host memory): the duration
// of a one-CTA kernel that writes 4 KB to pinned memory, alone and while the stager streams. Gate <= 30 us during.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

constexpr size_t W_PER_EXPERT = 3ull * 3072 * 3584;        // weights per expert (w1, w3: 3072 x 3584; w2: 3584 x 3072)
constexpr size_t CODE_BYTES = W_PER_EXPERT / 4;             // 2 bits per weight: 8.26 MB
constexpr size_t SCALE_BYTES = (W_PER_EXPERT / 32) * 6 / 8; // 6-bit scale codes: 0.77 MB
constexpr size_t REC_BYTES = CODE_BYTES + SCALE_BYTES;      // one record
constexpr size_t OUT_BYTES = 2 * CODE_BYTES + SCALE_BYTES;  // one staged native row
constexpr int CHUNK = 16384, STAGES = 6, THREADS = 256;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(b))); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t n, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(n), "r"(sa(b)) : "memory"); }

// Work unit = one CHUNK of one record. Units of the code part expand; units of the scale part copy.
__global__ void __launch_bounds__(THREADS) stager(const uint8_t* __restrict__ rec, uint8_t* __restrict__ out, int nrec,
                                                  const uint16_t* __restrict__ lut_g) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint64_t bars[STAGES];
  __shared__ uint16_t lut[256];
  for (int i = threadIdx.x; i < 256; i += THREADS) lut[i] = lut_g[i];
  const int per_rec = (int)((REC_BYTES + CHUNK - 1) / CHUNK);
  const int total = per_rec * nrec;
  const int n = blockIdx.x < total ? (total - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  if (threadIdx.x == 0) { for (int s = 0; s < STAGES; s++) mbar_init(&bars[s]); asm volatile("fence.mbarrier_init.release.cluster;"); }
  __syncthreads();
  auto issue = [&](int i) {
    int u = blockIdx.x + i * gridDim.x, r = u / per_rec, c = u % per_rec;
    size_t off = (size_t)c * CHUNK, len = REC_BYTES - off < (size_t)CHUNK ? REC_BYTES - off : (size_t)CHUNK;
    mbar_expect(&bars[i % STAGES], (uint32_t)len);
    bulk_g2s(smem + (i % STAGES) * CHUNK, rec + (size_t)r * REC_BYTES + off, (uint32_t)len, &bars[i % STAGES]);
  };
  if (threadIdx.x == 0) for (int i = 0; i < STAGES && i < n; i++) issue(i);
  for (int i = 0; i < n; i++) {
    int s = i % STAGES;
    mbar_wait(&bars[s], (i / STAGES) & 1);
    int u = blockIdx.x + i * gridDim.x, r = u / per_rec, c = u % per_rec;
    size_t off = (size_t)c * CHUNK, len = REC_BYTES - off < (size_t)CHUNK ? REC_BYTES - off : (size_t)CHUNK;
    const uint8_t* src = smem + s * CHUNK;
    uint8_t* dst_rec = out + (size_t)r * OUT_BYTES;
    if (off + len <= CODE_BYTES) {                         // codes: 16 bytes in -> 32 bytes out per thread step
      for (size_t k = threadIdx.x * 16; k < len; k += THREADS * 16) {
        uint4 v = *reinterpret_cast<const uint4*>(src + k);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        uint4 o0, o1;
        uint16_t* p0 = reinterpret_cast<uint16_t*>(&o0);
        uint16_t* p1 = reinterpret_cast<uint16_t*>(&o1);
#pragma unroll
        for (int j = 0; j < 8; j++) { p0[j] = lut[b[j]]; p1[j] = lut[b[8 + j]]; }
        uint4* d = reinterpret_cast<uint4*>(dst_rec + 2 * (off + k));
        d[0] = o0; d[1] = o1;
      }
    } else {                                               // scale codes (the code part is a whole number of chunks)
      size_t so = off - CODE_BYTES;
      for (size_t k = threadIdx.x * 16; k < len; k += THREADS * 16)
        *reinterpret_cast<uint4*>(dst_rec + 2 * CODE_BYTES + so + k) = *reinterpret_cast<const uint4*>(src + k);
    }
    __syncthreads();
    if (threadIdx.x == 0 && i + STAGES < n) issue(i + STAGES);
  }
}

// Stand-in for the native call: streams an HBM buffer (reads) on its own SMs; large shared memory keeps one CTA per SM
// so it never shares an SM with the stager.
__global__ void __launch_bounds__(1024) busy(const uint4* __restrict__ src, size_t n16, int reps, unsigned long long* sink) {
  extern __shared__ uint8_t pad[];
  unsigned long long acc = 0;
  for (int r = 0; r < reps; r++)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
      uint4 v = __ldcs(src + i); acc += (unsigned long long)v.x + v.w; }
  if (pad[0] == 255 && acc == 42) atomicAdd(sink, acc);    // keep the loads alive
}

__global__ void mailbox(volatile uint32_t* box, uint32_t tag) {   // one CTA writes 4 KB to pinned memory
  box[threadIdx.x] = tag + threadIdx.x;
  __threadfence_system();
}

static float elapsed(cudaEvent_t a, cudaEvent_t b) { float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }

int main(int argc, char** argv) {
  int dev = 0; CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  const int SMS = prop.multiProcessorCount;
  const int NREC = 5;
  printf("%s, %d SMs; record %.2f MB (codes %.2f + scales %.2f), staged row %.2f MB\n", prop.name, SMS, REC_BYTES / 1e6,
         CODE_BYTES / 1e6, SCALE_BYTES / 1e6, OUT_BYTES / 1e6);
  uint8_t *h_rec, *d_rec_view, *d_out; uint16_t *d_lut; uint32_t* h_box;
  CK(cudaHostAlloc(&h_rec, NREC * REC_BYTES, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)&d_rec_view, h_rec, 0));
  CK(cudaMalloc(&d_out, NREC * OUT_BYTES));
  CK(cudaHostAlloc(&h_box, 4096, cudaHostAllocMapped));
  uint32_t* d_box; CK(cudaHostGetDevicePointer((void**)&d_box, h_box, 0));
  srand(1);
  for (size_t i = 0; i < NREC * REC_BYTES; i++) h_rec[i] = (uint8_t)rand();
  uint16_t lut[256];                                          // byte of four 2-bit pairs -> two bytes of four nibbles
  for (int b = 0; b < 256; b++) {
    uint16_t v = 0;
    for (int j = 0; j < 4; j++) v |= (uint16_t)((((b >> (2 * j)) & 3) << 2) | 3) << (4 * j);
    lut[b] = v;
  }
  CK(cudaMalloc(&d_lut, sizeof(lut))); CK(cudaMemcpy(d_lut, lut, sizeof(lut), cudaMemcpyHostToDevice));
  const size_t smem = (size_t)STAGES * CHUNK;
  CK(cudaFuncSetAttribute(stager, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  // busy buffer: 2 GB in HBM
  const size_t BUSY = 2ull << 30; uint4* d_busy; CK(cudaMalloc(&d_busy, BUSY)); CK(cudaMemset(d_busy, 1, BUSY));
  unsigned long long* d_sink; CK(cudaMalloc(&d_sink, 8));
  const int busy_smem = 160 * 1024;
  CK(cudaFuncSetAttribute(busy, cudaFuncAttributeMaxDynamicSharedMemorySize, busy_smem));
  cudaStream_t s1, s2, s3; CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));
  cudaEvent_t a, b, c, d, e, f; for (cudaEvent_t* x : {&a, &b, &c, &d, &e, &f}) CK(cudaEventCreate(x));

  // correctness: stage NREC records, compare with a CPU expansion
  stager<<<16, THREADS, smem, s1>>>(d_rec_view, d_out, NREC, d_lut); CK(cudaStreamSynchronize(s1));
  std::vector<uint8_t> got(NREC * OUT_BYTES); CK(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (int r = 0; r < NREC; r++) {
    const uint8_t* src = h_rec + r * REC_BYTES; const uint8_t* g = got.data() + r * OUT_BYTES;
    for (size_t i = 0; i < CODE_BYTES; i++) { uint16_t v = lut[src[i]]; if (g[2 * i] != (v & 255) || g[2 * i + 1] != (v >> 8)) bad++; }
    if (memcmp(g + 2 * CODE_BYTES, src + CODE_BYTES, SCALE_BYTES)) bad++;
  }
  printf("E3-b bytes equal to a CPU expansion: %s (%zu bad)\n", bad ? "FAIL" : "PASS", bad);

  auto time_stager = [&](int nsm, int nrec, cudaStream_t st) {
    float best = 1e9;
    for (int rep = 0; rep < 7; rep++) {
      CK(cudaEventRecord(a, st)); stager<<<nsm, THREADS, smem, st>>>(d_rec_view, d_out, nrec, d_lut); CK(cudaEventRecord(b, st));
      CK(cudaEventSynchronize(b)); best = std::min(best, elapsed(a, b));
    }
    return best;
  };
  printf("E3-b alone (best of 7):\n");
  for (int nsm : {8, 16, 24, 32}) for (int k : {1, 2, 4}) {
    float ms = time_stager(nsm, k, s1);
    printf("  %2d SMs, %d record(s): %7.1f us = %5.1f us per record, %6.1f GB/s of record bytes\n", nsm, k, ms * 1e3,
           ms * 1e3 / k, k * REC_BYTES / (ms * 1e-3) / 1e9);
  }
  // busy alone
  const int reps = 6; const size_t n16 = BUSY / 16;
  auto time_busy = [&](int grid) {
    CK(cudaEventRecord(c, s2)); busy<<<grid, 1024, busy_smem, s2>>>(d_busy, n16, reps, d_sink); CK(cudaEventRecord(d, s2));
    CK(cudaEventSynchronize(d)); return elapsed(c, d);
  };
  printf("E3-b beside a busy HBM stream on the other SMs:\n");
  for (int nsm : {16, 24, 32}) {
    int grid = SMS - nsm;
    time_busy(grid); float base = time_busy(grid);
    // stager on s1 repeatedly while busy runs on s2
    CK(cudaEventRecord(c, s2)); busy<<<grid, 1024, busy_smem, s2>>>(d_busy, n16, reps, d_sink); CK(cudaEventRecord(d, s2));
    float stage_total = 0; int nst = 0;
    while (cudaEventQuery(d) == cudaErrorNotReady && nst < 400) {
      CK(cudaEventRecord(a, s1)); stager<<<nsm, THREADS, smem, s1>>>(d_rec_view, d_out, 4, d_lut); CK(cudaEventRecord(b, s1));
      CK(cudaEventSynchronize(b)); stage_total += elapsed(a, b); nst++;
    }
    CK(cudaEventSynchronize(d)); float with = elapsed(c, d);
    printf("  stager %2d SMs, busy on %3d SMs (%.0f GB/s alone): busy %.2f -> %.2f ms (+%.1f%%), stager %.1f us per "
           "record = %.1f GB/s of record bytes (%d calls)\n", nsm, grid, reps * BUSY / (base * 1e-3) / 1e9, base, with,
           100 * (with / base - 1), stage_total * 1e3 / (4 * nst), 4.0 * nst * REC_BYTES / (stage_total * 1e-3) / 1e9, nst);
  }
  // E3-d: mailbox write alone and while the stager streams
  auto time_box = [&]() {
    float best = 1e9, worst = 0;
    for (int rep = 0; rep < 20; rep++) {
      CK(cudaEventRecord(e, s3)); mailbox<<<1, 1024, 0, s3>>>(d_box, rep); CK(cudaEventRecord(f, s3)); CK(cudaEventSynchronize(f));
      float t = elapsed(e, f) * 1e3; best = std::min(best, t); worst = std::max(worst, t);
    }
    return std::make_pair(best, worst);
  };
  auto alone = time_box();
  printf("E3-d mailbox 4 KB write alone: best %.1f us, worst %.1f us\n", alone.first, alone.second);
  float best = 1e9, worst = 0; int n = 0;
  for (int rep = 0; rep < 20; rep++) {
    stager<<<32, THREADS, smem, s1>>>(d_rec_view, d_out, NREC, d_lut);              // ~150-200 us of streaming
    CK(cudaEventRecord(e, s3)); mailbox<<<1, 1024, 0, s3>>>(d_box, 100 + rep); CK(cudaEventRecord(f, s3));
    CK(cudaEventSynchronize(f)); CK(cudaStreamSynchronize(s1));
    float t = elapsed(e, f) * 1e3; best = std::min(best, t); worst = std::max(worst, t); n++;
  }
  printf("E3-d mailbox 4 KB write while the stager streams Grace memory: best %.1f us, worst %.1f us (%d tries)\n",
         best, worst, n);
  return 0;
}
