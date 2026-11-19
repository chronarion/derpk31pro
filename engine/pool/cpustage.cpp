// derpk31pro Grace pool: a persistent set of pinned threads that serves the job ring of cpustage_seg2.h. The GB300
// writes a job into slot seq % NSLOT (header with its own tag, the quantized rows) and then the door (the newest tag,
// st.release.sys); the pool runs the jobs in sequence order, computes their records where they lie in the layer's
// pinned bank (cpustage.h), writes the float32 rows into the job's result slot and releases that slot's done tag,
// which the GB300's landing (late job) or finish (on-time job) waits for with a bound. The shape of Hecate's cpuband
// (glm53-exl3/tier/cpuband.cpp) with its gather barrier.
//
// A job in the pool:
//   thread 0   the only thread that watches the door. It takes the next sequence number it has not run (FIFO), reads
//              that slot's header, checks it (its tag must carry that sequence number, the fields must be sane),
//              copies the quantized rows into pair operands                                            (~1-2 us)
//   barrier    (the workers wait here between jobs)
//   all        linear 1 + SiTU + routing weight + quantization: thread t owns units [96 t / n, 96 (t + 1) / n),
//              the same weight rows of every pick (64 rows = 57 KB of codes per unit and pick, read in place)
//   barrier
//   all        linear 2 + the sum over picks: thread t owns row pairs [1792 t / n, 1792 (t + 1) / n) and writes its
//              slice of every result row
//   barrier
//   thread 0   status, then the slot's done tag (release)
// With n = 48, 32 or 24 threads every thread has the same number of linear-1 units.
//
// Staleness (a job whose landing has given up): when thread 0 finds the door NSLOT - 1 or more jobs ahead of the
// next one, it skips to the oldest of the newest NSLOT - 1 (the others are counted stale, nothing is published); a
// running job is abandoned when the door gets NSLOT - 1 jobs ahead of it (the GB300 may be about to rewrite its slot);
// a job whose header does not carry the expected sequence number (its slot was rewritten) is skipped. A door that goes
// backwards (the engine restarted its sequence with this pool kept) is followed. The GB300 never reads rows behind a
// done tag it did not expect, so a skipped or abandoned job costs nothing but its time.
//
// Build:  g++ -O3 -mcpu=neoverse-v2 -std=c++17 -ffp-contract=off -fPIC -shared -pthread cpustage.cpp -o libcpustage.so
// C ABI (ctypes): cpustage_init, cpustage_set_layer, cpustage_start, cpustage_stop, cpustage_stats (7
// words), cpustage_stats_n, cpustage_layout, cpustage_run_single, and for tests and host-side engines
// cpustage_test_post, cpustage_test_wait, cpustage_test_land, cpustage_test_hold.
#include "cpustage.h"

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace {
using namespace cpustage;

static_assert(sizeof(CpuJob2) == 328, "CpuJob2 layout");
static_assert(offsetof(CpuJob2, n_pick) == 8 && offsetof(CpuJob2, alpha) == 24 && offsetof(CpuJob2, e_min) == 32 &&
                  offsetof(CpuJob2, slot) == 40 && offsetof(CpuJob2, w) == 72,
              "CpuJob2 field offsets");

constexpr int MAX_THREADS = 96;
constexpr int MAX_LAYERS = 256;

// Pool counters: u64 words at CPUSEG2_OFF_STATE + 8 i, written by thread 0 only (RUNNING by start / stop).
enum {
  ST_RUNNING = 0,     // 1 while the threads run
  ST_DONE = 1,        // jobs whose rows were published
  ST_ABANDONED = 2,   // jobs given up while running (the door got NSLOT - 1 ahead): nothing published
  ST_REFUSED = 3,     // jobs with a bad header: published with CPUSEG_ST_ERROR, no rows
  ST_JOB_NS = 4,      // sum over done jobs of ns from thread 0 taking the job to its done tag
  ST_PICKS = 5,       // records computed (done jobs)
  ST_MAX_NS = 6,      // the longest done job, ns
  ST_STALE = 7,       // jobs skipped without running (behind the door, rewritten slot): nothing published
  ST_RESETS = 8,      // times the door went backwards
  ST_ONTIME = 9,      // done jobs flagged CPUJOB2_ONTIME
  ST_QUEUED = 10,     // sum over taken jobs of the jobs behind them at the door when taken
  ST_QMAX = 11,       // the most jobs seen behind one
  ST_WORDS = 12,
};
static_assert(CPUSEG2_OFF_STATE + 8 * ST_WORDS <= CPUSEG2_OFF_DONE, "state block");

struct alignas(128) Line {
  std::atomic<uint32_t> v{0};
};

struct Bank {
  const uint8_t* base = nullptr;
  long long nrec = 0;
};

struct Pool {
  uint8_t* seg = nullptr;
  int nt = 0, core0 = 0, nice_val = 0;
  Bank bank[MAX_LAYERS];
  std::vector<std::thread> th;
  std::atomic<bool> stop{false};
  bool started = false;
  uint64_t next0 = 1;                    // the first sequence number thread 0 expects
  std::atomic<int> ready{0}, pin_fail{0};
  std::atomic<int> hold{0};              // tests: thread 0 leaves the door alone while set
  // gather barrier: each worker releases its own line, thread 0 collects them and releases `go`
  Line go;
  Line arrived[MAX_THREADS];
  // the job (written by thread 0 before the first barrier, read-only afterwards)
  Job J;
  int rc = 0, slot = 0;                  // rc: 0 run, 1 refuse
  uint64_t seq = 0, tag = 0, t0 = 0;
  std::atomic<uint64_t> t_seen{0};       // when thread 0 last took a job (ns): the idle clock of every thread
  alignas(128) std::atomic<uint32_t> abandon{0};
  alignas(64) JobBufs buf;
};

Pool* g_pool = nullptr;

inline uint64_t now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
inline void relax() { __asm__ __volatile__("yield"); }

// An idle server: spin for 50 ms after the last job, then 20 us sleeps, 1 ms sleeps after 2 s. (A decode step has a
// job every ~0.25 ms and 2-3 ms between forwards: the pool spins through it.) The first job after a pause is late
// for the threads that sleep; its landing falls back.
inline void idle_backoff(uint64_t idle_ns) {
  if (idle_ns > 2000000000ull) {
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, nullptr);
  } else if (idle_ns > 50000000ull) {
    struct timespec ts = {0, 20000};
    nanosleep(&ts, nullptr);
  }
}

inline uint64_t ld_acq(const uint8_t* p) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE);
}
inline uint64_t ld_rlx(const uint8_t* p) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(p), __ATOMIC_RELAXED);
}
inline void st_rel(uint8_t* p, uint64_t v) { __atomic_store_n(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELEASE); }
inline void st_rlx(uint8_t* p, uint64_t v) { __atomic_store_n(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELAXED); }
inline uint8_t* stw(Pool& P, int i) { return P.seg + CPUSEG2_OFF_STATE + 8 * i; }
inline void st_add(Pool& P, int i, uint64_t v) { st_rlx(stw(P, i), ld_rlx(stw(P, i)) + v); }
inline void st_max(Pool& P, int i, uint64_t v) { if (v > ld_rlx(stw(P, i))) st_rlx(stw(P, i), v); }

// Every thread calls the barriers of a job in the same order; `gen` is the caller's count of them. A worker's
// first barrier of a job is also its wait for the job: only thread 0 watches the door, so every thread runs the same
// sequence of jobs whatever tags it could have seen itself (two tags in quick succession, a thread asleep in its idle
// back-off). `idle`: the wait may be long (no job): back off as thread 0 does. False: the pool is stopping.
bool barrier(Pool& P, int t, uint32_t& gen, bool idle = false) {
  ++gen;
  uint32_t spins = 0;
  if (t == 0) {
    for (int i = 1; i < P.nt; ++i)
      while (P.arrived[i].v.load(std::memory_order_acquire) != gen) {
        relax();
        if ((++spins & 0xFFFF) == 0 && P.stop.load(std::memory_order_relaxed)) return false;
      }
    P.go.v.store(gen, std::memory_order_release);
  } else {
    P.arrived[t].v.store(gen, std::memory_order_release);
    while (P.go.v.load(std::memory_order_acquire) != gen) {
      relax();
      if ((++spins & 1023) == 0) {
        if (P.stop.load(std::memory_order_relaxed)) return false;
        if (idle) {
          const uint64_t seen = P.t_seen.load(std::memory_order_relaxed), now = now_ns();
          idle_backoff(now > seen ? now - seen : 0);
        }
      }
    }
  }
  return true;
}

// Thread 0: the job of sequence number `seq` in slot s into P.J. 0: run it; 1: refuse it (a bad header: publish an
// error under its tag); 2: stale (the header is not that job's: publish nothing).
int parse(Pool& P, int s, uint64_t seq) {
  CpuJob2 h;
  std::memcpy(&h, seg_job(P.seg, s), sizeof(h));
  Job& J = P.J;
  J.T = 0;
  J.P = 0;
  P.tag = h.tag;
  if (cpuseg_seq(h.tag) != seq) return 2;
  if (h.n_pick < 1 || h.n_pick > (uint32_t)PMAX || h.n_rows < 1 || h.n_rows > (uint32_t)MAXT) return 1;
  if (h.n_rows != (uint32_t)(h.tag & 0xFF) || (h.layer & 0xFF) != (uint32_t)((h.tag >> 8) & 0xFF)) return 1;
  if (h.layer >= (uint32_t)MAX_LAYERS || P.bank[h.layer].base == nullptr) return 1;
  if (!(h.alpha > 0.0f) || !(h.beta >= 0.0f) || !std::isfinite(h.alpha) || !std::isfinite(h.beta)) return 1;
  const Bank& B = P.bank[h.layer];
  for (uint32_t p = 0; p < h.n_pick; ++p) {
    if (h.slot[p] < 0 || h.slot[p] >= B.nrec) return 1;
    J.rec[p] = B.base + (size_t)h.slot[p] * (size_t)REC_BYTES;
    for (int t = 0; t < MAXT; ++t) {
      const float wt = t < (int)h.n_rows ? h.w[p][t] : 0.0f;
      if (!std::isfinite(wt)) return 1;
      J.w[p][t] = wt;
    }
  }
  J.flags = h.flags;
  J.emin = h.e_min < -120 ? -120 : (h.e_min > 0 ? 0 : h.e_min);
  J.situ = make_situ(h.alpha, h.beta);
  J.buf = &P.buf;
  J.out = reinterpret_cast<float*>(seg_res(P.seg, s));
  J.T = (int)h.n_rows;
  J.P = (int)h.n_pick;
  prepare(J, reinterpret_cast<const int8_t*>(seg_xq(P.seg, s)), seg_xe(P.seg, s));
  return 0;
}

// The door's sequence number says the running job `seq` is no longer wanted: NSLOT - 1 newer jobs (its slot may be
// rewritten next), or a door that went back (the engine restarted its sequence).
inline bool unwanted(uint64_t door_seq, uint64_t seq) { return door_seq >= seq + NSLOT - 1 || door_seq < seq; }

void worker(Pool* pool, int t) {
  Pool& P = *pool;
  {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(P.core0 + t, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) P.pin_fail.fetch_add(1);
    if (P.nice_val != 0) setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), P.nice_val);
    P.ready.fetch_add(1);
  }
  const uint8_t* door = P.seg + CPUSEG2_OFF_DOOR;
  uint64_t next = P.next0, spins = 0;               // thread 0 only
  uint32_t gen = 0;
  const int nt = P.nt;
  const int u0 = L1_BLOCKS * t / nt, u1 = L1_BLOCKS * (t + 1) / nt;
  const int p0 = (int)((long long)L2_PAIRS * t / nt), p1 = (int)((long long)L2_PAIRS * (t + 1) / nt);
  while (!P.stop.load(std::memory_order_relaxed)) {
    if (t == 0) {                                   // thread 0: the queue
      uint64_t newest = 0;
      const bool held = P.hold.load(std::memory_order_relaxed) != 0;
      if (!held) {
        newest = cpuseg_seq(ld_acq(door));
        if (newest + 1 < next) {                    // the door went back: the engine restarted its sequence
          st_add(P, ST_RESETS, 1);
          next = newest > (uint64_t)(NSLOT - 2) ? newest - (NSLOT - 2) : 1;
        }
      }
      if (held || newest < next) {
        relax();
        if ((++spins & 1023) == 0) idle_backoff(now_ns() - P.t_seen.load(std::memory_order_relaxed));
        continue;
      }
      spins = 0;
      if (newest - next >= (uint64_t)(NSLOT - 1)) {  // too far behind: only the newest NSLOT - 1 can still be wanted
        const uint64_t keep = newest - (NSLOT - 2);
        st_add(P, ST_STALE, keep - next);
        next = keep;
      }
      const uint64_t seq = next++;
      const int s = (int)(seq % NSLOT);
      const uint64_t t0 = now_ns();
      P.t_seen.store(t0, std::memory_order_relaxed);
      P.abandon.store(0, std::memory_order_relaxed);
      P.slot = s;
      P.seq = seq;
      P.t0 = t0;
      int rc = parse(P, s, seq);
      if (rc != 2) {                                // a slot the GB300 may have rewritten while it was read: drop it
        std::atomic_thread_fence(std::memory_order_acquire);
        if (unwanted(cpuseg_seq(ld_acq(door)), seq)) rc = 2;
      }
      if (rc == 2) {
        st_add(P, ST_STALE, 1);
        continue;                                   // no barrier: the workers keep waiting for a job
      }
      P.rc = rc;
      st_add(P, ST_QUEUED, newest - seq);
      st_max(P, ST_QMAX, newest - seq);
    }
    if (!barrier(P, t, gen, true)) break;           // the job starts: the workers were waiting here
    const bool run = P.rc == 0;
    const uint64_t seq = P.seq;
    const Job& J = P.J;
    if (run) {
      for (int p = 0; p < J.P; ++p) prefetch_scales(lin1(J.rec[p]), 64 * u0, 64 * u1);
      for (int u = u0; u < u1; ++u) {
        if (P.abandon.load(std::memory_order_relaxed)) break;
        l1_block(J, u);
        if (unwanted(cpuseg_seq(ld_rlx(door)), seq)) P.abandon.store(1, std::memory_order_relaxed);
      }
    }
    if (!barrier(P, t, gen)) break;
    if (run) {
      for (int p = 0; p < J.P; ++p) prefetch_scales(lin2(J.rec[p]), 2 * p0, 2 * p1);
      for (int rp = p0; rp < p1; rp += 8) {
        if (P.abandon.load(std::memory_order_relaxed)) break;
        l2_pairs(J, rp, rp + 8 < p1 ? rp + 8 : p1);
        if (unwanted(cpuseg_seq(ld_rlx(door)), seq)) P.abandon.store(1, std::memory_order_relaxed);
      }
    }
    if (!barrier(P, t, gen)) break;
    if (t == 0) {
      const uint64_t dt = now_ns() - P.t0;
      uint8_t* done = seg_done(P.seg, P.slot);
      if (run && P.abandon.load(std::memory_order_relaxed)) {
        st_add(P, ST_ABANDONED, 1);                                    // nothing is published
      } else {
        if (run) {                                                     // the counters first: a reader that saw the
          st_add(P, ST_DONE, 1);                                       // done tag sees them too
          st_add(P, ST_JOB_NS, dt);
          st_add(P, ST_PICKS, (uint64_t)J.P);
          st_max(P, ST_MAX_NS, dt);
          if (J.flags & CPUJOB2_ONTIME) st_add(P, ST_ONTIME, 1);
        } else {
          st_add(P, ST_REFUSED, 1);
        }
        st_rlx(done + 8, run ? 0ull : CPUSEG_ST_ERROR);
        st_rel(done, P.tag);                                           // the rows and the status are visible first
      }
    }
  }
}

}  // namespace

extern "C" {

// seg: CPUSEG2_BYTES of zeroed memory (pinned and mapped for the GB300 by the caller). threads on cores core0 ..
// core0 + threads - 1 (the caller keeps every other thread of the process off them); nice_val < 0 needs CAP_SYS_NICE
// (0 = leave).
int cpustage_init(void* seg, int threads, int core0, int nice_val) {
  if (g_pool != nullptr && g_pool->started) return -1;
  if (seg == nullptr || threads < 1 || threads > MAX_THREADS || core0 < 0 || core0 + threads > CPU_SETSIZE) return -2;
  if (g_pool == nullptr) g_pool = new Pool();
  Pool& P = *g_pool;
  P.seg = static_cast<uint8_t*>(seg);
  P.nt = threads;
  P.core0 = core0;
  P.nice_val = nice_val;
  const uint64_t magic = CPUSEG2_MAGIC, bytes = CPUSEG2_BYTES;
  std::memcpy(P.seg + CPUSEG2_OFF_MAGIC, &magic, 8);
  std::memcpy(P.seg + CPUSEG2_OFF_MAGIC + 8, &bytes, 8);
  return 0;
}

// The layer's bank: nrec records of REC_BYTES at `bank` (the address st.bank_cpu has in this process).
int cpustage_set_layer(int layer, const void* bank, long long nrec) {
  if (g_pool == nullptr || g_pool->started || layer < 0 || layer >= MAX_LAYERS || nrec < 0) return -1;
  g_pool->bank[layer].base = static_cast<const uint8_t*>(bank);
  g_pool->bank[layer].nrec = bank != nullptr ? nrec : 0;
  return 0;
}

// Starts the threads; returns once every thread has pinned itself. -3: a thread could not be pinned to its core (the
// process's CPU set does not hold core0 .. core0 + threads - 1): nothing runs.
int cpustage_start() {
  if (g_pool == nullptr || g_pool->started || g_pool->seg == nullptr) return -1;
  Pool& P = *g_pool;
  P.stop.store(false);
  P.next0 = cpuseg_seq(ld_acq(P.seg + CPUSEG2_OFF_DOOR)) + 1;
  P.t_seen.store(now_ns());
  P.go.v.store(0);
  P.ready.store(0);
  P.pin_fail.store(0);
  for (int i = 0; i < MAX_THREADS; ++i) P.arrived[i].v.store(0);
  for (int t = 0; t < P.nt; ++t) P.th.emplace_back(worker, &P, t);
  const uint64_t t0 = now_ns();
  while (P.ready.load() < P.nt && now_ns() - t0 < 10000000000ull) {
    struct timespec ts = {0, 100000};
    nanosleep(&ts, nullptr);
  }
  if (P.ready.load() < P.nt || P.pin_fail.load() > 0) {
    P.stop.store(true);
    for (auto& th : P.th) th.join();
    P.th.clear();
    return -3;
  }
  P.started = true;
  st_rel(P.seg + CPUSEG2_OFF_STATE + 8 * ST_RUNNING, 1);
  return 0;
}

// Call with no job in flight (the GPU idle).
void cpustage_stop() {
  if (g_pool == nullptr || !g_pool->started) return;
  Pool& P = *g_pool;
  P.stop.store(true);
  for (auto& t : P.th) t.join();
  P.th.clear();
  P.started = false;
  st_rel(P.seg + CPUSEG2_OFF_STATE + 8 * ST_RUNNING, 0);
}

// The pool's counters (the state block's words, also readable in the segment): out[0 .. n - 1] of running, jobs
// done, abandoned, refused, sum of job ns, picks computed, longest job ns, stale, door resets, on-time jobs done, sum
// of jobs queued behind a taken job, most jobs queued behind one. Returns the number of words written.
int cpustage_stats_n(unsigned long long* out, int n) {
  const int m = n < ST_WORDS ? n : ST_WORDS;
  for (int i = 0; i < m; ++i)
    out[i] = g_pool != nullptr && g_pool->seg != nullptr ? ld_rlx(g_pool->seg + CPUSEG2_OFF_STATE + 8 * i) : 0ull;
  return m;
}

// The original call: the first 7 words (running .. longest job ns).
void cpustage_stats(unsigned long long* out) { cpustage_stats_n(out, 7); }

long long cpustage_record_bytes() { return REC_BYTES; }

// The segment as this library reads it, for the callers' own constants: bytes, door, state, done, job, job bytes,
// activation slots, their stride, exponent bytes inside a slot, results, result bytes, slots, records per job, rows,
// sizeof(CpuJob2), state words.
int cpustage_layout(long long* out, int n) {
  const long long v[] = {CPUSEG2_BYTES, CPUSEG2_OFF_DOOR, CPUSEG2_OFF_STATE, CPUSEG2_OFF_DONE, CPUSEG2_OFF_JOB,
                         CPUSEG2_JOB_BYTES, CPUSEG2_OFF_X, CPUSEG2_X_STRIDE, (long long)X_OFF_E, CPUSEG2_OFF_RES,
                         CPUSEG2_RES_BYTES, NSLOT, PMAX, MAXT, (long long)sizeof(CpuJob2), ST_WORDS};
  const int m = (int)(sizeof(v) / sizeof(v[0]));
  for (int i = 0; i < n && i < m; ++i) out[i] = v[i];
  return m;
}

// One job on the calling thread, from float rows: x [T][3584] (the values of the call's bfloat16 rows), w
// [n_pick][8], recs[n_pick] record addresses. out [T][3584]. Optional outputs: xq [T][3584] int8 in natural input
// order and xe [T][112] exponent bytes (what the GB300 kernel would have written, for references).
int cpustage_run_single(const void* const* recs, int n_pick, const float* x, const float* w, int T, float alpha,
                        float beta, unsigned flags, int e_min, float* out, signed char* xq_out,
                        unsigned char* xe_out) {
  if (n_pick < 1 || n_pick > PMAX || T < 1 || T > MAXT || !(alpha > 0.0f) || !(beta >= 0.0f)) return -1;
  JobBufs* buf = new JobBufs();
  std::vector<int8_t> xq((size_t)T * H);
  std::vector<uint8_t> xe((size_t)T * KB1);
  for (int t = 0; t < T; ++t)
    quant_row(x + (size_t)t * H, H, e_min, xq.data() + (size_t)t * H, xe.data() + (size_t)t * KB1);
  Job J;
  J.T = T;
  J.P = n_pick;
  J.flags = flags;
  J.emin = e_min;
  J.situ = make_situ(alpha, beta);
  J.buf = buf;
  J.out = out;
  for (int p = 0; p < n_pick; ++p) {
    J.rec[p] = static_cast<const uint8_t*>(recs[p]);
    for (int t = 0; t < MAXT; ++t) J.w[p][t] = t < T ? w[p * MAXT + t] : 0.0f;
  }
  prepare(J, xq.data(), xe.data());
  run_serial(J);
  if (xq_out != nullptr)
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < H; ++k) xq_out[(size_t)t * H + k] = xq[(size_t)t * H + cpuseg_planar(k)];
  if (xe_out != nullptr) std::memcpy(xe_out, xe.data(), xe.size());
  delete buf;
  return 0;
}

// The GB300's half of a job on the host (tests, an engine without a GPU send): the header of slot seq % NSLOT, the
// quantized rows, then the door. n_pick may exceed the slots' 8 (a job the pool must refuse); only 8 are written.
// Returns the tag.
unsigned long long cpustage_test_post(void* segv, unsigned long long seq, int layer, const int* slots, int n_pick,
                                      const float* x, const float* w, int T, float alpha, float beta, unsigned flags,
                                      int e_min) {
  uint8_t* seg = static_cast<uint8_t*>(segv);
  const int s = (int)(seq % NSLOT);
  CpuJob2 h;
  std::memset(&h, 0, sizeof(h));
  h.tag = cpuseg_tag(seq, (uint32_t)layer, (uint32_t)T);
  h.n_pick = (uint32_t)n_pick;
  h.n_rows = (uint32_t)T;
  h.layer = (uint32_t)layer;
  h.flags = flags;
  h.alpha = alpha;
  h.beta = beta;
  h.e_min = e_min;
  for (int p = 0; p < n_pick && p < PMAX; ++p) {
    h.slot[p] = slots[p];
    for (int t = 0; t < MAXT; ++t) h.w[p][t] = t < T ? w[p * MAXT + t] : 0.0f;
  }
  std::memcpy(seg_job(seg, s), &h, sizeof(h));
  for (int t = 0; t < T && t < MAXT; ++t)
    quant_row(x + (size_t)t * H, H, e_min, reinterpret_cast<int8_t*>(seg_xq(seg, s)) + (size_t)t * H,
              seg_xe(seg, s) + (size_t)t * KB1);
  st_rel(seg + CPUSEG2_OFF_DOOR, h.tag);
  return h.tag;
}

// Spin until the done tag of the tag's slot is `tag` (0) or timeout_ns passed (1). *status = the done status word.
int cpustage_test_wait(void* segv, unsigned long long tag, unsigned long long timeout_ns, unsigned long long* status) {
  uint8_t* seg = static_cast<uint8_t*>(segv);
  uint8_t* done = seg_done(seg, (int)(cpuseg_seq(tag) % NSLOT));
  const uint64_t t0 = now_ns();
  for (;;) {
    if (ld_acq(done) == tag) {
      if (status != nullptr) *status = ld_rlx(done + 8);
      return 0;
    }
    if (now_ns() - t0 > timeout_ns) return 1;
    relax();
  }
}

// The pool's half of one job without a pool (tests of the GB300's landing): rows [T][3584] float32 (null: none) into
// the tag's result slot, the status, then the done tag (release).
void cpustage_test_land(void* segv, unsigned long long tag, const float* rows, int T, unsigned long long status) {
  uint8_t* seg = static_cast<uint8_t*>(segv);
  const int s = (int)(cpuseg_seq(tag) % NSLOT);
  if (rows != nullptr && T > 0) std::memcpy(seg_res(seg, s), rows, (size_t)(T < MAXT ? T : MAXT) * H * sizeof(float));
  st_rlx(seg_done(seg, s) + 8, status);
  st_rel(seg_done(seg, s), tag);
}

// Tests: while on, thread 0 does not read the door (jobs posted meanwhile queue up as if the pool were busy).
void cpustage_test_hold(int on) {
  if (g_pool != nullptr) g_pool->hold.store(on != 0 ? 1 : 0);
}

}  // extern "C"
