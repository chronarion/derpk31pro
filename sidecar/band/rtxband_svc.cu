// RTX band of derpk31pro: the persistent service kernel. One cooperative launch stays on the RTX and serves the
// mailbox (band_mailbox.h) without the host: calls (the two linears of band_core.cuh), L2 prefetch hints, and the
// repack of an arriving hi2 record into its store slot.
//
//   CTA 0, the poller: warp 0 reads the doorbell line of the mailbox over PCIe in a loop (one turn = 1.0 us). On a
//     call's tag the whole CTA copies the request slot (work list, T rows) to device memory, publishes an item in a
//     small ring of device memory and waits for the call's completion word; then it polls again. Hints and repack
//     requests are items too. A work list that names a slot outside the store is refused (error bit, no rows).
//   CTAs 1..G-1, the workers: thread 0 spins on the ring's head word (device memory), the others wait at a barrier.
//     For a call every worker walks its share of the units: linear-1 unit u of the call (expert u / 48, part u % 48)
//     belongs to worker u mod Wk, linear-2 unit v (expert v / 28, part v % 28) to worker Wk - 1 - (v mod Wk), so
//     workers without linear-1 work start on linear 2 at once. Units wait on each other only through the counters of
//     band_core.cuh (an expert's linear 2 for its linear 1, the last expert's sum for the others' partial rows), and
//     a worker's list is ordered so that no wait can be circular.
//     The units that finish the sum store their part of the T result rows straight into the mailbox, as whole
//     128 B lines (band_core.cuh: scattered 4 B stores of the same rows reach the host up to 20 us later); the last
//     of them publishes the done tag (system-scope release) and the completion word for the poller.
//   Nothing in a call depends on the grid size or on timing: results are bit-identical to the launched kernels.
//
// A kernel that never ends never gets its L1 flushed, so every global load of data that changes between calls goes
// around L1 (band_core.cuh, SVC = true) and host memory is read with volatile loads.
// Every wait is bounded; a wait that gives up sets the error word in the mailbox and the kernel leaves.
// The launcher keeps the kernel on its own non-blocking stream; after an idle limit the kernel leaves by itself and
// the host relaunches it when a call is pending (an idle RTX should not spin).
// A repack occupies every worker for ~15 us, so it starts right after a call has ended (the next call is a layer
// away then) or when no call has ended for a while; a call that is pending always goes first.
#include "band_core.cuh"
#include "band_mailbox.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>

namespace {
using namespace band;
using namespace bandmb;

// An idle worker looks at the ring's head once per BAND_SVC_IDLE_NS (0 = as fast as it can).
#ifndef BAND_SVC_IDLE_NS
#define BAND_SVC_IDLE_NS 0
#endif

constexpr int RING = 8;
constexpr int ITEM_CALL = 1, ITEM_HINT = 2, ITEM_RANGE = 3, ITEM_PACK = 4, ITEM_EXIT = 5;
constexpr int HINT_ISSUERS = 8;        // workers whose thread 0 issues prefetch pieces (more lose pieces)

struct Item {                          // 64 B, written by the poller before it publishes the generation
  int kind;
  int mslot;                           // call: mailbox slot
  unsigned long long tag;              // call: its tag
  int a, b, c, pad;                    // call: rows; hint: count; range: landing; pack: landing, store slot, base
  unsigned long long t_seen, t_rel;    // RTX timer: tag seen, item released
  unsigned long long pad2[2];
};
static_assert(sizeof(Item) == 64, "item size");

struct Ctl {                           // device memory, zero at every launch
  unsigned long long head;             // generation of the newest item (poller: release, workers: acquire)
  unsigned long long pad0[15];
  Item items[RING];
  int fin;                             // units of the running call that have stored their result rows
  int fin_gen;                         // generation of the last call whose done tag is out
  int phase;                           // workers that finished the running repack phase
  int err;
  int range[2];                        // scale range of the record being repacked (as band_scale_range)
  int pad1[2];
  unsigned long long t_work;           // RTX timer: worker 0 starts on the running call
  int pad3[22];
  int sync[64];                        // the linears' counters (band_core.cuh)
  BandCall call;                       // the running call's work list
  int hint[4 + HINT_MAX];              // count, slots (80 B copied from the mailbox)
  int pad2[12];
};
static_assert(SYNC_WORDS <= 64, "counters");
static_assert(sizeof(Ctl) % 16 == 0, "ctl size");

struct SvcParams {
  uint8_t* seg;                        // the mailbox as this device sees it
  uint8_t* store;                      // band records by slot
  uint8_t* sbase;                      // lowest scale code by slot
  uint8_t* xf;                         // the running call's input in fragment order (XF_BYTES)
  uint8_t* aq;                         // [MAXE][AQ_BYTES]
  float* ypart;                        // [MAXE - 1][MAXT][H]
  Ctl* ctl;
  const uint8_t* landing;              // [n][HI2_BYTES]: hi2 records waiting for their repack
  unsigned long long idle_limit_ns;    // leave after this long without an item (0 = never)
  unsigned long long wait_limit_ns;    // a wait of the poller that takes longer is an error
  long long n_slots;                   // slots of the store: a work list that names another one is refused
  unsigned long long repack_after_ns;  // a repack starts within this long after a call's end (the next call is
  unsigned long long repack_idle_ns;   // then a layer away), or once no call has ended for this long
};

// A worker's shared memory beyond the linears' work area: its copy of the work list, the item, the hint's slot
// list, and the buffer pointers of the running call. They are read where they are used: nothing of them has to stay
// in registers across a tile's GEMM (the service must fit three CTAs per SM without spilling in that loop).
constexpr int SVC_CALL_OFF = SMEM_BYTES;
constexpr int SVC_ITEM_OFF = SVC_CALL_OFF + (int)sizeof(BandCall);
constexpr int SVC_HINT_OFF = SVC_ITEM_OFF + 64;
constexpr int SVC_BUFS_OFF = SVC_HINT_OFF + 64;
constexpr int SVC_SMEM = SVC_BUFS_OFF + 64;
static_assert(SVC_SMEM % 16 == 0 && sizeof(Bufs) <= 64, "shared memory layout");

__device__ __forceinline__ void put_item(Ctl* C, unsigned gen, int kind, int mslot, unsigned long long tag, int a, int b,
                                         int c, unsigned long long t_seen) {
  Item* it = &C->items[gen % RING];
  it->kind = kind; it->mslot = mslot; it->tag = tag; it->a = a; it->b = b; it->c = c;
  it->t_seen = t_seen; it->t_rel = globaltimer();
  __threadfence();
  st_release_gpu_u64(&C->head, gen);
}

// thread 0 of the poller: wait until *word == want; false after the limit
__device__ __forceinline__ bool poller_wait(const int* word, int want, unsigned long long limit) {
  const unsigned long long t0 = globaltimer();
  for (unsigned n = 1;; n++) {
    if ((int)ld_acquire_gpu_u32(word) == want) return true;
    if ((n & 255) == 0 && limit != 0 && globaltimer() - t0 > limit) return false;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// CTA 0
// ---------------------------------------------------------------------------------------------------------------
struct PState {
  int act, n_exp;
  unsigned long long word, t_seen, turns, idle_ns;
  int bad, pad;                        // the work list names a slot outside the store
};

__device__ __forceinline__ void poller(const SvcParams& p, uint8_t* smem) {
  Ctl* C = p.ctl;
  PState* ps = reinterpret_cast<PState*>(smem);
  const int tid = threadIdx.x, lane = tid & 31;
  const int Wk = gridDim.x - 1;
  uint8_t* door = p.seg + OFF_DOOR;
  uint8_t* state = p.seg + OFF_STATE;
  unsigned long long next_seq = 0, hint_seen = 0, rep_seen = 0, calls = 0, hints = 0, turns = 0, idle_ns = 0;
  unsigned long long t_done = 0;       // warp 0: when the last call ended
  unsigned gen = 0;
  if (tid < 32) {                      // where the previous launch stopped; requests older than this launch's start
    unsigned long long a = 0, b = 0, c = 0, d = 0, e = 0;
    if (lane == 0) {
      a = ld_acquire_sys_u64(state + STATE_SERVED); b = ld_acquire_sys_u64(door + DOOR_HINT);
      c = ld_acquire_sys_u64(door + DOOR_REPACK); d = ld_acquire_sys_u64(state + STATE_CALLS);
      e = ld_acquire_sys_u64(state + STATE_HINTS);
    }
    next_seq = __shfl_sync(0xffffffffu, a, 0) + 1; hint_seen = __shfl_sync(0xffffffffu, b, 0);
    rep_seen = __shfl_sync(0xffffffffu, c, 0); calls = __shfl_sync(0xffffffffu, d, 0);
    hints = __shfl_sync(0xffffffffu, e, 0);
  }
  for (;;) {
    if (tid < 32) {                    // warp 0: the doorbell line, one PCIe read per turn
      const unsigned long long idle0 = __shfl_sync(0xffffffffu, globaltimer(), 0);
      int act = 0;
      unsigned long long word = 0;
      unsigned n = 1;
      for (;; n++) {
        const unsigned long long w = lane < 8 ? ld_acquire_sys_u64(door + 8 * lane) : 0ull;
        const unsigned long long ctrl = __shfl_sync(0xffffffffu, w, 0);
        unsigned long long req = 0;    // the oldest call not served yet: normally the only one, and seq = next_seq
#pragma unroll
        for (int s = 1; s <= NSLOT; s++) {
          const unsigned long long r = __shfl_sync(0xffffffffu, w, s);
          if (tag_seq(r) >= next_seq && (req == 0 || r < req)) req = r;
        }
        const unsigned long long hint = __shfl_sync(0xffffffffu, w, 5), rep = __shfl_sync(0xffffffffu, w, 6);
        if (ctrl != 0) { act = ITEM_EXIT; break; }
        if (req != 0) { act = ITEM_CALL; word = req; break; }
        if (rep != rep_seen) {         // a record waits for its repack: right after a call, or when calls are rare
          const unsigned long long since = __shfl_sync(0xffffffffu, globaltimer(), 0) - t_done;
          if (since <= p.repack_after_ns || since >= p.repack_idle_ns) { act = ITEM_PACK; word = rep; rep_seen = rep; break; }
        }
        if (hint != hint_seen) { act = ITEM_HINT; word = hint; hint_seen = hint; break; }
        if ((n & 63) == 0 && p.idle_limit_ns != 0) {
          const unsigned long long now = __shfl_sync(0xffffffffu, globaltimer(), 0);
          if (now - idle0 > p.idle_limit_ns) { act = ITEM_EXIT; break; }
        }
      }
      if (lane == 0) {
        const unsigned long long now = globaltimer();
        ps->act = act; ps->word = word; ps->t_seen = now; ps->n_exp = 1; ps->bad = 0;
        ps->turns = n; ps->idle_ns = now - idle0;
      }
    }
    __syncthreads();
    const int act = ps->act;
    const unsigned long long word = ps->word;
    if (act == ITEM_EXIT) {
      if (tid == 0) {
        put_item(C, ++gen, ITEM_EXIT, 0, 0, 0, 0, 0, 0);
        st_release_sys_u64(state + STATE_RUNNING, 0);
      }
      return;
    }
    if (act == ITEM_CALL) {
      const int T = (int)tag_rows(word), ms = (int)(tag_seq(word) % NSLOT);
      const uint8_t* rq = p.seg + OFF_REQ + (unsigned long long)ms * REQ_BYTES;
      uint8_t* res = p.seg + OFF_RES + (unsigned long long)ms * RES_BYTES;
      if (tid < CALL_BYTES / 16) {     // the work list
        const uint4 v = ld_volatile_u4(rq + REQ_CALL + 16 * tid);
        *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(&C->call) + 16 * tid) = v;
        if (tid == 0) ps->n_exp = (int)v.x;
        if (tid >= 2 && tid < 2 + MAXE / 4) {            // its slots (ints 8 .. 8 + MAXE): all inside the store,
          const unsigned long long ns = (unsigned long long)p.n_slots;       // whether the call uses them or not
          if (v.x >= ns || v.y >= ns || v.z >= ns || v.w >= ns) ps->bad = 1;
        }
      }
      {                                // rows: token g's 128 B of step kq -> lanes 4g .. 4g + 3 of the step record
        constexpr int PER = MAXT * KQ1 * 8 / THREADS;
        static_assert(PER * THREADS == MAXT * KQ1 * 8, "row copy shares");
        uint4 v[PER];
#pragma unroll
        for (int j = 0; j < PER; j++) {
          const int idx = tid + j * THREADS, g = idx / (KQ1 * 8), r = idx % (KQ1 * 8);
          v[j] = make_uint4(0, 0, 0, 0);
          if (g < T) v[j] = ld_volatile_u4(rq + REQ_ROWS + g * ROW_IN + r * 16);
        }
#pragma unroll
        for (int j = 0; j < PER; j++) {
          const int idx = tid + j * THREADS, g = idx / (KQ1 * 8), r = idx % (KQ1 * 8);
          *reinterpret_cast<uint4*>(p.xf + (r >> 3) * BREC + g * 128 + (r & 7) * 16) = v[j];
        }
      }
      if (tid < MAXT * KQ1) {          // scale codes
        const int g = tid / KQ1, kq = tid % KQ1;
        *reinterpret_cast<uint32_t*>(p.xf + kq * BREC + 1024 + 4 * g) =
            g < T ? ld_volatile_u32(rq + REQ_ROWS + g * ROW_IN + H + 4 * kq) : 0u;
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        calls++;
        turns += ps->turns; idle_ns += ps->idle_ns;
        st_relaxed_sys_u64(state + STATE_POLL_TURNS, turns);
        st_relaxed_sys_u64(state + STATE_POLL_NS, idle_ns);
        const bool refused = ps->n_exp > MAXE || ps->bad != 0 || T < 1 || T > MAXT;
        if (ps->n_exp <= 0 || refused) {                 // nothing to compute, or not a work list: acknowledge alone
          st_relaxed_sys_u64(res + RES_STATUS, refused ? (RES_EMPTY | RES_ERROR) : RES_EMPTY);
          st_release_sys_u64(res + RES_TAG, word);
        } else {
          put_item(C, ++gen, ITEM_CALL, ms, word, T, 0, 0, ps->t_seen);
          if (!poller_wait(&C->fin_gen, (int)gen, p.wait_limit_ns))
            st_release_sys_u64(state + STATE_ERROR, (1ull << 32) | gen);
        }
        st_relaxed_sys_u64(state + STATE_SERVED, tag_seq(word));
        st_relaxed_sys_u64(state + STATE_CALLS, calls);
      }
      if (tid < 32) {
        next_seq = tag_seq(word) + 1;                    // calls the sender gave up on (a gap) are not waited for
        t_done = __shfl_sync(0xffffffffu, globaltimer(), 0);
      }
    } else if (act == ITEM_HINT) {
      if (tid < (4 + HINT_MAX) / 4) {
        const uint4 v = ld_volatile_u4(p.seg + OFF_HINT + 16 * tid);
        *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(C->hint) + 16 * tid) = v;
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        put_item(C, ++gen, ITEM_HINT, 0, word, (int)(word & 0xFF), 0, 0, ps->t_seen);
        st_relaxed_sys_u64(state + STATE_HINTS, ++hints);
      }
    } else if (act == ITEM_PACK) {     // scale range, then the permutation: two items for every worker
      if (tid == 0) {
        const int landing = (int)((word >> 32) & 0xFF), dst = (int)(word & 0xFFFFFFFFull);
        bool fine = true;
        put_item(C, ++gen, ITEM_RANGE, 0, 0, landing, 0, 0, ps->t_seen);
        fine &= poller_wait(&C->phase, Wk, p.wait_limit_ns);
        atomicExch(&C->phase, 0);
        const int lo = 255 - atomicExch(&C->range[0], 0), hi = atomicExch(&C->range[1], 0);
        const bool fits = lo > 0 && hi - lo <= 15;
        put_item(C, ++gen, ITEM_PACK, 0, 0, landing, dst, lo, ps->t_seen);
        fine &= poller_wait(&C->phase, Wk, p.wait_limit_ns);
        atomicExch(&C->phase, 0);
        if (fits) { p.sbase[dst] = (uint8_t)lo; __threadfence(); }
        if (!fine) st_release_sys_u64(state + STATE_ERROR, (2ull << 32) | gen);
        st_release_sys_u64(state + STATE_REPACK_ACK, ((word >> 40) << 40) | ((unsigned long long)fits << 8) | (unsigned)lo);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// CTAs 1 .. G-1
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void worker(const SvcParams& p, uint8_t* smem) {
  Ctl* C = p.ctl;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wi = blockIdx.x - 1, Wk = gridDim.x - 1;
  BandCall* call = reinterpret_cast<BandCall*>(smem + SVC_CALL_OFF);
  Item* it = reinterpret_cast<Item*>(smem + SVC_ITEM_OFF);
  int* hslot = reinterpret_cast<int*>(smem + SVC_HINT_OFF);
  Bufs* bufs = reinterpret_cast<Bufs*>(smem + SVC_BUFS_OFF);
  unsigned seen = 0;
  for (;;) {
    if (tid == 0) {                    // the next item of the ring
      while ((unsigned)ld_acquire_gpu_u64(&C->head) == seen) {
#if BAND_SVC_IDLE_NS > 0
        const unsigned long long t = globaltimer();
        while (globaltimer() - t < BAND_SVC_IDLE_NS) {}
#endif
      }
      const uint8_t* src = reinterpret_cast<const uint8_t*>(&C->items[(seen + 1) % RING]);
#pragma unroll
      for (int k = 0; k < 4; k++) reinterpret_cast<uint4*>(it)[k] = ld_u4<true>(src + 16 * k);
    }
    __syncthreads();
    seen++;
    const int kind = it->kind;
    if (kind == ITEM_EXIT) return;
    if (kind == ITEM_CALL) {
      if (tid < CALL_BYTES / 16)
        reinterpret_cast<uint4*>(call)[tid] = ld_u4<true>(reinterpret_cast<const uint8_t*>(&C->call) + 16 * tid);
      if (tid == 0)
        *bufs = Bufs{p.store, p.sbase, p.aq, p.ypart,
                     reinterpret_cast<float*>(p.seg + OFF_RES + (unsigned long long)it->mslot * RES_BYTES + RES_ROWS),
                     C->sync};
      __syncthreads();
      if (wi == 0 && tid == 0) C->t_work = globaltimer();
      const int n_exp = call->n_exp;
      for (int u = wi; u < n_exp * CTAS1; u += Wk) l1_body<true>(call, *bufs, p.xf, u / CTAS1, u % CTAS1, smem, true);
      for (int v = Wk - 1 - wi; v < n_exp * CTAS2; v += Wk) {
        if (!l2_body<true>(call, *bufs, v / CTAS2, v % CTAS2, n_exp, it->a, smem, true)) continue;
        const unsigned long long t_sum = tid == 0 ? globaltimer() : 0ull;
        if (warp < it->a || tid == 0) fence_sys();       // a thread's result rows are in host memory before anyone says done
        __syncthreads();
        if (tid == 0 && atomicAdd(&C->fin, 1) == CTAS2 - 1) {
          uint8_t* res = p.seg + OFF_RES + (unsigned long long)it->mslot * RES_BYTES;
          atomicExch(&C->fin, 0);
          st_relaxed_sys_u64(res + RES_TIMES, it->t_seen);
          st_relaxed_sys_u64(res + RES_TIMES + 8, it->t_rel);
          st_relaxed_sys_u64(res + RES_TIMES + 24, C->t_work);
          st_relaxed_sys_u64(res + RES_TIMES + 32, t_sum);
          st_relaxed_sys_u64(res + RES_TIMES + 40, globaltimer());
          st_relaxed_sys_u64(res + RES_TIMES + 16, globaltimer());
          st_relaxed_sys_u64(res + RES_STATUS, atomicExch(&C->sync[SYNC_ERR], 0) != 0 ? RES_ERROR : 0ull);
          st_release_sys_u64(res + RES_TAG, it->tag);
          st_relaxed_sys_u64(res + RES_TIMES + 48, globaltimer());
          st_release_gpu_u32(&C->fin_gen, seen);
        }
      }
    } else if (kind == ITEM_HINT) {
      if (wi < HINT_ISSUERS && tid == 0) {
        const int n = min(it->a, HINT_MAX);
        for (int e = 0; e < n; e++) hslot[e] = (int)ld_u32<true>(reinterpret_cast<const uint8_t*>(&C->hint[1 + e]));
        for (int i = wi; i < n * PIECES; i += HINT_ISSUERS) {
          if ((unsigned)ld_acquire_gpu_u64(&C->head) != seen) break;      // something newer: the hint is over
          prefetch_piece(p.store, hslot[i / PIECES], i % PIECES);
        }
      }
    } else if (kind == ITEM_RANGE) {
      uint32_t lo = 0xFFFFFFFFu, hi = 0;
      scale_range_share<true>(p.landing + (long long)it->a * HI2_BYTES, wi * THREADS + tid, Wk * THREADS, lo, hi);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
      }
      if (lane == 0 && lo <= 255u) {
        atomicMax(&C->range[0], 255 - (int)lo);
        atomicMax(&C->range[1], (int)hi);
      }
      __syncthreads();
      if (tid == 0) atomicAdd(&C->phase, 1);
    } else if (kind == ITEM_PACK) {
      repack_share<true>(p.landing + (long long)it->a * HI2_BYTES, p.store + (long long)it->b * REC_BYTES,
                         wi * W + warp, Wk * W, (uint32_t)it->c * 0x01010101u);
      __threadfence();
      __syncthreads();
      if (tid == 0) atomicAdd(&C->phase, 1);
    }
    __syncthreads();
  }
}

// BAND_SVC_PER_SM CTAs must fit on an SM at once: the compiler keeps the registers per thread under
// 65536 / (BAND_SVC_PER_SM * THREADS). Two is what the linears' code needs without spilling (125 registers); three
// (80 registers, a stack frame in the GEMM loop) gives 563 workers instead of 375 and is slower: unhinted p99 +3 us,
// hinted p50 +3 us on the RTX PRO 6000.
#ifndef BAND_SVC_PER_SM
#define BAND_SVC_PER_SM 2
#endif
__global__ void __launch_bounds__(THREADS, BAND_SVC_PER_SM) band_service(const SvcParams p) {
  extern __shared__ __align__(16) uint8_t smem[];
  if (blockIdx.x == 0) poller(p, smem);
  else worker(p, smem);
}

cudaStream_t g_stream = nullptr, g_copy = nullptr;

// A host thread that lets records enter the band while the kernel serves calls: copy-engine transfers of an hi2
// record from pinned host memory into one of two landing buffers, then the repack word, then its acknowledgement
// (awaited before the next repack word; the other landing buffer is filled meanwhile).
// The transfers share the PCIe link with the poller's reads of the mailbox, and a transfer in flight while a call
// arrives slows the call (its staging read above all; whole records at 2,300 per second cost a call +10 to +26 us).
// So the record goes in chunks. gate 1: a chunk is started only right after a call has ended (the next call is then
// a layer away), or after 1 ms without a call. gate 2: chunks are started as long as no call is in flight (no
// request tag above the last call served), so only the chunk on the wire when a call arrives overlaps it.
struct Feed {
  std::thread th;
  std::atomic<int> stop{0};
  long long done = 0, copy_ns = 0, pack_ns = 0, pack_max_ns = 0, unfit = 0, errors = 0, behind = 0, chunks = 0;
  long long chunk_ns = 0, chunk_max_ns = 0;
  unsigned long long rseq = 0;
};
Feed* g_feed = nullptr;

long long host_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

struct FeedArgs {
  uint8_t* seg; uint8_t* landing; const uint8_t* records; const int* recs; const int* slots; int m;
  long long period_ns, chunk_bytes; int gate, mode;      // mode bit 0: no copies after the first two, bit 1: no repacks
};

void feed_loop(Feed* f, FeedArgs a) {
  uint64_t* door = reinterpret_cast<uint64_t*>(a.seg + OFF_DOOR + DOOR_REPACK);
  const uint64_t* ack = reinterpret_cast<const uint64_t*>(a.seg + OFF_STATE + STATE_REPACK_ACK);
  const uint64_t* ended = reinterpret_cast<const uint64_t*>(a.seg + OFF_STATE + STATE_CALLS);
  const uint64_t* served = reinterpret_cast<const uint64_t*>(a.seg + OFF_STATE + STATE_SERVED);
  const uint64_t* req = reinterpret_cast<const uint64_t*>(a.seg + OFF_DOOR + DOOR_REQ);
  auto in_flight = [&] {               // a call has been sent and is not served yet
    const uint64_t s = __atomic_load_n(served, __ATOMIC_ACQUIRE);
    for (int k = 0; k < NSLOT; k++)
      if (tag_seq(__atomic_load_n(req + k, __ATOMIC_ACQUIRE)) > s) return true;
    return false;
  };
  const long long chunk = a.chunk_bytes > 0 ? a.chunk_bytes : HI2_BYTES;
  auto stopped = [&] { return f->stop.load(std::memory_order_relaxed) != 0; };
  auto wait_ack = [&](uint64_t rseq, long long t_post) {   // false: stop or error
    for (unsigned n = 1;; n++) {
      const uint64_t got = __atomic_load_n(ack, __ATOMIC_ACQUIRE);
      if ((got >> 40) == rseq) {
        const long long d = host_ns() - t_post;
        f->done++; f->pack_ns += d;
        if (d > f->pack_max_ns) f->pack_max_ns = d;
        if (((got >> 8) & 1) == 0) f->unfit++;
        return true;
      }
      if ((n & 255) == 0 && host_ns() - t_post > 100000000ll) { f->errors++; return false; }
    }
  };
  const long long t0 = host_ns();
  bool posted = false;
  uint64_t posted_seq = 0;
  long long t_post = 0;
  uint64_t seen = __atomic_load_n(ended, __ATOMIC_ACQUIRE);
  for (long long i = 0; !stopped(); i++) {
    const long long due = t0 + i * a.period_ns;
    if (a.period_ns > 0) {
      if (host_ns() > due + a.period_ns) f->behind++;
      while (host_ns() < due)
        if (stopped()) break;
    }
    uint8_t* land = a.landing + (i & 1) * (long long)HI2_BYTES;
    const uint8_t* src = a.records + (long long)a.recs[i % a.m] * HI2_BYTES;
    const long long c0 = host_ns();
    if (!(a.mode & 1) || i < 2) {
      for (long long o = 0; o < HI2_BYTES && !stopped(); o += chunk) {
        if (a.gate == 1) {             // the end of a call, or a quiet millisecond
          const long long w0 = host_ns();
          for (unsigned n = 1;; n++) {
            const uint64_t now = __atomic_load_n(ended, __ATOMIC_ACQUIRE);
            if (now != seen) { seen = now; break; }
            if ((n & 63) == 0 && (host_ns() - w0 > 1000000ll || stopped())) break;
          }
        } else if (a.gate == 2) {      // no call in flight
          while (in_flight() && !stopped()) {}
        }
        const long long nb = o + chunk <= HI2_BYTES ? chunk : HI2_BYTES - o;
        const long long k0 = host_ns();
        if (cudaMemcpyAsync(land + o, src + o, (size_t)nb, cudaMemcpyHostToDevice, g_copy) != cudaSuccess) { f->errors++; return; }
        cudaError_t e;
        while ((e = cudaStreamQuery(g_copy)) == cudaErrorNotReady) {}
        if (e != cudaSuccess) { f->errors++; return; }
        const long long kd = host_ns() - k0;
        f->chunks++; f->chunk_ns += kd;
        if (kd > f->chunk_max_ns) f->chunk_max_ns = kd;
        if (a.gate == 1) seen = __atomic_load_n(ended, __ATOMIC_ACQUIRE);   // calls that ended meanwhile are gone
      }
    }
    if (stopped()) break;              // a record that is not whole is not repacked
    f->copy_ns += host_ns() - c0;
    if (a.mode & 2) { f->done++; continue; }
    if (posted && !wait_ack(posted_seq, t_post)) return;
    posted_seq = f->rseq++;
    t_post = host_ns();
    __atomic_store_n(door, (posted_seq << 40) | ((uint64_t)(i & 1) << 32) | (uint64_t)(uint32_t)a.slots[i % a.m],
                     __ATOMIC_RELEASE);
    posted = true;
  }
  if (posted) wait_ack(posted_seq, t_post);
}

}  // namespace

extern "C" {

// What the service's host side passes at launch: device pointers (seg = the device pointer of the registered
// mailbox mapping) and the two limits.
struct RtxSvc {
  void* seg; void* store; void* sbase; void* xf; void* aq; void* ypart; void* ctl; const void* landing;
  unsigned long long idle_limit_ns, wait_limit_ns;
  long long n_slots;
  unsigned long long repack_after_ns, repack_idle_ns;
};

// [ctl bytes, shared memory per CTA, mailbox bytes, hi2 record bytes, band record bytes, xf bytes, aq bytes,
//  ypart bytes, threads per CTA]
void rtxband_svc_sizes(long long* out) {
  out[0] = sizeof(Ctl); out[1] = SVC_SMEM; out[2] = SEG_BYTES; out[3] = HI2_BYTES; out[4] = REC_BYTES;
  out[5] = XF_BYTES; out[6] = (long long)MAXE * AQ_BYTES; out[7] = (long long)(MAXE - 1) * MAXT * H * 4;
  out[8] = THREADS;
}

// Load the module now (nothing may be loaded lazily once the kernel spins) and report [registers per thread,
// CTAs that fit on one SM, SMs]. Returns a CUDA error code.
int rtxband_svc_preload(int* out) {
  cudaError_t e = cudaFuncSetAttribute(band_service, cudaFuncAttributeMaxDynamicSharedMemorySize, SVC_SMEM);
  if (e != cudaSuccess) return (int)e;
  cudaFuncAttributes a;
  e = cudaFuncGetAttributes(&a, band_service);
  if (e != cudaSuccess) return (int)e;
  int per_sm = 0, dev = 0, sms = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, band_service, THREADS, SVC_SMEM);
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  out[0] = a.numRegs; out[1] = per_sm; out[2] = sms;
  if (g_stream == nullptr) e = cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
  if (e == cudaSuccess && g_copy == nullptr) e = cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking);
  return (int)e;
}

// Launch the service with `grid` CTAs (all co-resident: a cooperative launch) on its own stream. The control block
// must be zero and the mailbox's running word 1. Returns a CUDA error code; the kernel then runs until the mailbox's
// control word or its idle limit makes it leave.
int rtxband_svc_launch(const RtxSvc* h, int grid) {
  SvcParams p{(uint8_t*)h->seg, (uint8_t*)h->store, (uint8_t*)h->sbase, (uint8_t*)h->xf, (uint8_t*)h->aq,
              (float*)h->ypart, (Ctl*)h->ctl, (const uint8_t*)h->landing, h->idle_limit_ns, h->wait_limit_ns,
              h->n_slots, h->repack_after_ns, h->repack_idle_ns};
  void* args[] = {&p};
  return (int)cudaLaunchCooperativeKernel((void*)band_service, dim3(grid), dim3(THREADS), args, SVC_SMEM, g_stream);
}

// 1 while the kernel runs, 0 once it has left, negative = a CUDA error (the kernel died).
int rtxband_svc_running() {
  const cudaError_t e = cudaStreamQuery(g_stream);
  if (e == cudaErrorNotReady) return 1;
  return e == cudaSuccess ? 0 : -(int)e;
}

int rtxband_svc_wait_exit() { return (int)cudaStreamSynchronize(g_stream); }

// Register a host mapping of the mailbox with CUDA for the current device and return the pointer kernels use.
int rtxband_svc_map(void* host, long long bytes, void** dev) {
  cudaError_t e = cudaHostRegister(host, (size_t)bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaHostGetDevicePointer(dev, host, 0);
}

// Pin a host buffer (the source of rtxband_svc_copy).
int rtxband_svc_pin(void* host, long long bytes) { return (int)cudaHostRegister(host, (size_t)bytes, cudaHostRegisterDefault); }

// Zero the control block (between launches; the kernel must not be running).
int rtxband_svc_reset(void* ctl) { return (int)cudaMemset(ctl, 0, sizeof(Ctl)); }

// An hi2 record from pinned host memory into a landing buffer by the copy engine, on its own stream, while the
// kernel runs. rtxband_svc_copy_done: 1 = landed, 0 = in flight, negative = error.
int rtxband_svc_copy(void* landing, const void* host, long long bytes) {
  return (int)cudaMemcpyAsync(landing, host, (size_t)bytes, cudaMemcpyHostToDevice, g_copy);
}
int rtxband_svc_copy_done() {
  const cudaError_t e = cudaStreamQuery(g_copy);
  if (e == cudaErrorNotReady) return 0;
  return e == cudaSuccess ? 1 : -(int)e;
}

// Records enter the band from a host thread: entry i copies record recs[i % m] of `records` (pinned host memory,
// [..][hi2 bytes]) into landing buffer i % 2 (`landing` = two device buffers of hi2 bytes) and has it repacked into
// store slot slots[i % m]. period_ns: one entry per period (0 = as fast as the rest allows). chunk_bytes: size of
// one copy-engine transfer (0 = the whole record). gate = 1: a transfer starts only right after a call has ended
// or after a quiet millisecond; gate = 2: transfers start as long as no call is in flight.
// mode: 0, or 1 = the two landing buffers are filled once and repacked over and over,
// 2 = transfers only, no repack (to price the two halves apart). rseq0: the first repack seq (above every seq used
// before). The arrays must stay alive until the stop.
int rtxband_svc_feed_start(void* seg_host, void* landing, const void* records, const int* recs, const int* slots, int m,
                           long long period_ns, long long chunk_bytes, int gate, int mode, unsigned long long rseq0) {
  if (g_feed != nullptr) return -1;
  g_feed = new Feed;
  g_feed->rseq = rseq0;
  FeedArgs a{(uint8_t*)seg_host, (uint8_t*)landing, (const uint8_t*)records, recs, slots, m, period_ns, chunk_bytes,
             gate, mode};
  g_feed->th = std::thread(feed_loop, g_feed, a);
  return 0;
}
// Stop the feeder. out: [entries, copy ns (sum, first chunk requested to last landed), repack ns (sum, request to
// acknowledgement), longest repack ns, records that did not fit the band format, errors, periods it was behind,
// next repack seq, transfers, transfer ns (sum, request to landed), longest transfer ns].
int rtxband_svc_feed_stop(long long* out) {
  if (g_feed == nullptr) return -1;
  g_feed->stop.store(1);
  g_feed->th.join();
  out[0] = g_feed->done; out[1] = g_feed->copy_ns; out[2] = g_feed->pack_ns; out[3] = g_feed->pack_max_ns;
  out[4] = g_feed->unfit; out[5] = g_feed->errors; out[6] = g_feed->behind; out[7] = (long long)g_feed->rseq;
  out[8] = g_feed->chunks; out[9] = g_feed->chunk_ns; out[10] = g_feed->chunk_max_ns;
  delete g_feed;
  g_feed = nullptr;
  return 0;
}

}  // extern "C"
