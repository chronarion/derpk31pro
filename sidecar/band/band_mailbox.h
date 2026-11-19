// RTX band of derpk31pro: the mailbox between the GB300 (sender of a layer's call, reader of its result) and the
// persistent service kernel on the RTX PRO 6000. One pinned host segment (a /dev/shm file) mapped by both processes
// and registered with CUDA in each (cudaHostRegisterMapped): the GB300 reads and writes it through ATS, the RTX over
// PCIe. There is no path between the two GPUs that does not go through host memory.
//
// The protocol has the shape of dsv41's peer tier v3 (peer3, from garnermccloud/dgx-station-recipes, MIT, copyright
// 2026 Garner McCloud): a tag published last with a system-scope release store, a poller CTA that reads the tags
// over PCIe, a result tag the sender waits for, idle parking. No code is taken from it; the PTX memory-order
// instructions below are the same ones it uses.
//
// A call:  GB300 writes request slot s = seq % 4 (work list, T rows of MXFP8 in the band's byte order), then the
//          slot's tag in the doorbell line. The RTX poller sees the tag, copies the slot to device memory, releases the
//          band CTAs; the CTAs that finish the sum write T rows of float32 into result slot s; the last of them writes
//          the slot's done tag (= the call's tag). The GB300 waits for it (bounded) and adds the rows.
//          The poller serves the oldest request whose seq is above the last one served: a sender may skip seqs (calls
//          sent while the kernel was away are passed over) but never reuses one. A result stays in its slot until
//          call seq + 4 is served, so a finish that gave up can still collect it one call later.
// A hint:  GB300 writes up to 16 store slots and the hint tag; the RTX pulls those records into L2 while it idles.
// Entry:   the service's host thread copies an hi2 record into a device landing buffer and posts a repack word; the
//          RTX repacks it into its store slot (right after a call has ended, or when calls are rare) and acknowledges.
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define BANDMB_FN __host__ __device__
#else
#define BANDMB_FN
#endif

namespace bandmb {

constexpr uint64_t MAGIC = 0x31444E4142585452ull;      // "RTXBAND1"
constexpr uint64_t VERSION = 1;
constexpr int NSLOT = 4;                               // request / result slots, by seq % NSLOT
constexpr int MAX_ROWS = 8;
constexpr int ROW_IN = 3584 + 112;                     // one row of MXFP8: 3,584 E4M3 bytes (band order) + 112 UE8M0
constexpr int ROW_OUT = 3584 * 4;                      // one result row: 3,584 float32
constexpr int CALL_BYTES = 608;                        // band::BandCall
constexpr int HINT_MAX = 16;

// Segment layout (bytes); control words own whole 128 B lines.
constexpr uint64_t OFF_MAGIC = 0;                      // u64 magic, u64 version
// The doorbell line: everything the RTX poller looks at, read in one access.
constexpr uint64_t OFF_DOOR = 128;
constexpr uint64_t DOOR_CTRL = 0;                      // u64, service host -> RTX: 1 = leave the kernel
constexpr uint64_t DOOR_REQ = 8;                       // u64[NSLOT], GB300 -> RTX: tag of the call in request slot s
constexpr uint64_t DOOR_HINT = 40;                     // u64, GB300 -> RTX: tag of the hint list (count in bits 0..7)
constexpr uint64_t DOOR_REPACK = 48;                   // u64, service host -> RTX: seq << 40 | landing << 32 | slot
// What the RTX reports to the hosts.
constexpr uint64_t OFF_STATE = 256;
constexpr uint64_t STATE_RUNNING = 0;                  // u64: launcher sets 1, the kernel 0 when it leaves
constexpr uint64_t STATE_ERROR = 8;                    // u64: nonzero = a bounded wait gave up (where << 32 | detail)
constexpr uint64_t STATE_REPACK_ACK = 16;              // u64: seq << 40 | fits << 8 | scale base
constexpr uint64_t STATE_SERVED = 24;                  // u64: seq of the last call served (a relaunch continues there)
constexpr uint64_t STATE_CALLS = 32;                   // u64: calls served since the segment was created
constexpr uint64_t STATE_HINTS = 40;                   // u64: hints taken
constexpr uint64_t STATE_POLL_TURNS = 48;              // u64: doorbell reads while idle, and
constexpr uint64_t STATE_POLL_NS = 56;                 // u64: the idle time they took (their ratio = one poll turn)
constexpr uint64_t OFF_HINT = 384;                     // int32 n, int32 slot[HINT_MAX]
constexpr uint64_t OFF_REQ = 4096;                     // NSLOT request slots
constexpr uint64_t REQ_BYTES = 32768;
constexpr uint64_t REQ_CALL = 0;                       //   the work list (CALL_BYTES)
constexpr uint64_t REQ_ROWS = 1024;                    //   rows [T][ROW_IN]
constexpr uint64_t OFF_RES = OFF_REQ + NSLOT * REQ_BYTES;
constexpr uint64_t RES_BYTES = 131072;                 // NSLOT result slots
constexpr uint64_t RES_TAG = 0;                        //   u64 done tag (written last)
constexpr uint64_t RES_STATUS = 8;                     //   u64: bit 0 = error inside the call, bit 1 = no rows (empty call)
constexpr uint64_t RES_TIMES = 16;                     //   u64[7], RTX timer ns: tag seen, call released, done (before the
                                                       //   tag's store), first worker at work, last unit's sum stored, and
                                                       //   fenced; [6], written after the tag: the tag's store returned
constexpr uint64_t RES_ROWS = 128;                     //   rows [T][ROW_OUT]
constexpr uint64_t SEG_BYTES = OFF_RES + NSLOT * RES_BYTES;      // 659,456
static_assert(REQ_ROWS + MAX_ROWS * ROW_IN <= REQ_BYTES && RES_ROWS + MAX_ROWS * ROW_OUT <= RES_BYTES, "slot sizes");
static_assert(SEG_BYTES % 4096 == 0, "whole pages");

constexpr uint64_t RES_ERROR = 1, RES_EMPTY = 2;

// A call's tag: seq (from 1, +1 per call) << 16 | layer << 8 | rows. Never zero.
BANDMB_FN constexpr uint64_t make_tag(uint64_t seq, uint32_t layer, uint32_t rows) { return (seq << 16) | (uint64_t(layer & 0xFF) << 8) | (rows & 0xFF); }
BANDMB_FN constexpr uint64_t tag_seq(uint64_t tag) { return tag >> 16; }
BANDMB_FN constexpr uint32_t tag_layer(uint64_t tag) { return uint32_t(tag >> 8) & 0xFF; }
BANDMB_FN constexpr uint32_t tag_rows(uint64_t tag) { return uint32_t(tag) & 0xFF; }

// Byte offset, inside a row's 3,584 E4M3 bytes as sent, of input k: the band's fragment order. Bytes k .. k + 3 of
// k % 4 == 0 stay together; a row's 128 inputs of one step are one 128 B run.
BANDMB_FN constexpr int row_in_offset(int k) {
  return (k / 128) * 128 + ((k % 16) / 4) * 32 + ((k % 128) / 32) * 8 + ((k % 32) / 16) * 4 + k % 4;
}

#if defined(__CUDACC__)
// PTX memory-order helpers. sys = visible to the host and the other GPU, gpu = within this device.
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const void* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys_u64(void* p, uint64_t v) {
  asm volatile("st.release.sys.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_u64(void* p, uint64_t v) {
  asm volatile("st.relaxed.sys.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t ld_acquire_gpu_u64(const void* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu_u64(void* p, uint64_t v) {
  asm volatile("st.release.gpu.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu_u32(void* p, uint32_t v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
// Loads that are never served from a cache line this SM read earlier (host memory, or another kernel's writes).
__device__ __forceinline__ uint4 ld_volatile_u4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
#endif

}  // namespace bandmb
