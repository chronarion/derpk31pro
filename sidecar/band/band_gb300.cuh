// RTX band of derpk31pro: the GB300 side of the mailbox (band_mailbox.h) as device functions, so that an engine
// kernel can inline them (the work list straight out of `unitk`, the add straight into the routed sum) or launch the
// kernels of rtxband_gb300.cu, which only wrap these.
//
//   send    T rows of MXFP8 (row-major E4M3 bytes and UE8M0 codes, as MegaMoE's input) go into request slot
//           seq % 4 in the band's byte order, with the 608 B work list; then the slot's tag is published. A work
//           list without experts sends nothing (the finish then returns at once, status "empty").
//   finish  waits for the call's done tag with a bounded spin and adds the T x 3,584 float32 result rows into a
//           device buffer. If the tag is not there in time nothing is added and the state block says so; the call
//           is then "late": its rows stay in result slot seq % 4 until call seq + 4 is served.
//   late    collects a late call's rows (one look, no spin) into the buffer of the engine's late completion.
//   hint    up to 16 store slots a coming call is expected to use, and the hint tag.
//
// Host memory is written and read through ATS; loads of it are volatile. Every function is written for one CTA of
// any size >= 32 threads (the kernels use 512).
#pragma once
#include "band_mailbox.h"

namespace bandgb {
using namespace bandmb;

constexpr int H = 3584;
constexpr unsigned long long ST_ADDED = 0, ST_TIMEOUT = 1, ST_ERROR = 2, ST_EMPTY = 3, ST_DOWN = 4;
constexpr unsigned long long LATE_NONE = 0, LATE_ADDED = 1, LATE_LOST = 2, LATE_ERROR = 3;

struct GbState {                       // device memory, 160 B, zero at start
  unsigned long long seq;              // seq of the last call sent (start above anything the mailbox has seen)
  unsigned long long tag;              // that call's tag
  unsigned long long idx;              // calls finished: the index a trace's arrays are read at
  unsigned long long hint_seq;
  unsigned long long status;           // the last finish: ST_*
  unsigned long long late_tag;         // a call whose finish gave up and whose rows are not collected yet (0 = none)
  unsigned long long late_idx;         // its index
  unsigned long long late_status;      // the last late: LATE_*
  unsigned long long n_added, n_timeout, n_error, n_empty, n_down, n_late_added, n_lost;
  unsigned long long t_pub;            // GB300 timer: the last call's tag was published
  unsigned long long t_hint;           // GB300 timer: the last hint's tag was published
  unsigned long long pad[3];
};
static_assert(sizeof(GbState) == 160, "state block");

// Words of a call's log line (GB300 timer, ns), LOG_WORDS per call:
//   0 send start, 1 tag published, 2 finish start, 3 done tag seen (or gave up), 4 finish end (rows added),
//   5 longest gap between two polls of the finish (a preemption of the context shows here), 6 status (ST_*),
//   7 RTX: (tag seen -> done) | (tag seen -> released) << 32, 8 outcome of the late collection (LATE_*),
//   9 when the late collection ran.
constexpr int LOG_WORDS = 16;

// --- send -----------------------------------------------------------------------------------------------------------
// xq [T][3584] E4M3 bytes row-major, xs [T][112] UE8M0 codes, call = band::BandCall (608 B; its second int is T).
__device__ __forceinline__ void send_body(uint8_t* seg, GbState* st, const uint8_t* xq, const uint8_t* xs,
                                          const uint8_t* call, uint32_t layer, unsigned long long* log) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
  const unsigned long long seq = st->seq + 1;
  const unsigned long long t0 = globaltimer();
  if (reinterpret_cast<const int*>(call)[0] <= 0) {      // no band expert in this layer: nothing goes out
    if (tid == 0) {
      st->tag = 0;
      if (log != nullptr) { log[0] = t0; log[1] = globaltimer(); }
    }
    return;
  }
  const int T = min(max(reinterpret_cast<const int*>(call)[1], 1), MAX_ROWS);
  uint8_t* rq = seg + OFF_REQ + (seq % NSLOT) * REQ_BYTES;
  // one warp per (row, step): the row's 128 inputs of the step become one 128 B line in fragment order
  for (int r = warp; r < T * 28; r += warps) {
    const int g = r / 28, kq = r % 28;
    const int t = lane >> 3, kap = (lane >> 1) & 3, half = lane & 1;
    const uint32_t v = *reinterpret_cast<const uint32_t*>(xq + g * H + kq * 128 + kap * 32 + half * 16 + t * 4);
    *reinterpret_cast<uint32_t*>(rq + REQ_ROWS + g * ROW_IN + kq * 128 + lane * 4) = v;
  }
  if (tid < T * 28) {
    const int g = tid / 28, kq = tid % 28;
    *reinterpret_cast<uint32_t*>(rq + REQ_ROWS + g * ROW_IN + H + 4 * kq) =
        *reinterpret_cast<const uint32_t*>(xs + g * 112 + 4 * kq);
  }
  if (tid < CALL_BYTES / 16)
    reinterpret_cast<uint4*>(rq + REQ_CALL)[tid] = reinterpret_cast<const uint4*>(call)[tid];
  fence_sys();
  __syncthreads();
  if (tid == 0) {
    const unsigned long long tag = make_tag(seq, layer, (uint32_t)T);
    st_release_sys_u64(seg + OFF_DOOR + DOOR_REQ + 8 * (seq % NSLOT), tag);
    const unsigned long long t1 = globaltimer();
    st->seq = seq; st->tag = tag; st->t_pub = t1;
    if (log != nullptr) { log[0] = t0; log[1] = t1; }
  }
}

// --- finish ---------------------------------------------------------------------------------------------------------
// Adds the call's rows into dst [T][3584] float32. `sh`: 4 words of shared memory. Returns the status to every
// thread; thread 0 has updated the state block.
__device__ __forceinline__ unsigned long long finish_body(uint8_t* seg, GbState* st, float* dst,
                                                          unsigned long long timeout_ns, unsigned long long* sh,
                                                          unsigned long long* log) {
  const int tid = threadIdx.x;
  const unsigned long long tag = st->tag;
  uint8_t* res = seg + OFF_RES + (tag_seq(tag) % NSLOT) * RES_BYTES;
  if (tid == 0) {
    const unsigned long long t0 = globaltimer();
    unsigned long long last = t0, gap = 0, status = ST_TIMEOUT;
    if (tag == 0) {
      status = ST_EMPTY;               // the send had nothing to send
    } else if (ld_acquire_sys_u64(seg + OFF_STATE + STATE_RUNNING) == 0 && ld_acquire_sys_u64(res + RES_TAG) != tag) {
      status = ST_DOWN;                // no service kernel: nothing to wait for
    } else {
      for (;;) {
        if (ld_acquire_sys_u64(res + RES_TAG) == tag) {
          const unsigned long long rs = ld_acquire_sys_u64(res + RES_STATUS);
          status = (rs & RES_ERROR) ? ST_ERROR : (rs & RES_EMPTY) ? ST_EMPTY : ST_ADDED;
          break;
        }
        const unsigned long long now = globaltimer();
        gap = max(gap, now - last);
        last = now;
        if (now - t0 > timeout_ns) break;
      }
    }
    sh[0] = status; sh[1] = t0; sh[2] = globaltimer(); sh[3] = gap;
  }
  __syncthreads();
  const unsigned long long status = sh[0];
  if (status == ST_ADDED) {
    const int n4 = (int)tag_rows(tag) * (H / 4);
    for (int i = tid; i < n4; i += blockDim.x) {
      const uint4 v = ld_volatile_u4(res + RES_ROWS + 16 * i);
      float4* d = reinterpret_cast<float4*>(dst + 4 * i);
      float4 o = *d;
      o.x += __uint_as_float(v.x); o.y += __uint_as_float(v.y); o.z += __uint_as_float(v.z); o.w += __uint_as_float(v.w);
      *d = o;
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (log != nullptr) {
      log[2] = sh[1]; log[3] = sh[2]; log[4] = globaltimer(); log[5] = sh[3]; log[6] = status;
      const unsigned long long* tm = reinterpret_cast<const unsigned long long*>(res + RES_TIMES);
      log[7] = status != ST_ADDED ? 0ull : ((tm[2] - tm[0]) & 0xFFFFFFFFull) | ((tm[1] - tm[0]) << 32);
    }
    st->status = status;
    if (status == ST_ADDED) st->n_added += 1;
    else if (status == ST_ERROR) st->n_error += 1;
    else if (status == ST_EMPTY) st->n_empty += 1;
    else if (status == ST_DOWN) st->n_down += 1;
    else {                             // late: the rows may still come; `late` collects them
      st->n_timeout += 1;
      if (st->late_tag != 0) st->n_lost += 1;            // the call before was late too and nobody collected it
      st->late_tag = tag;
      st->late_idx = st->idx;
    }
    st->idx += 1;
  }
  return status;
}

// --- late -----------------------------------------------------------------------------------------------------------
// If a call is late and its rows have arrived meanwhile, add them into late_dst [T][3584] float32 (the buffer of the
// engine's late completion). One look at the result slot, no spin: a call that is still not there is lost.
// `late_base` + late_idx * late_stride is where the rows go. `sh`: 2 words of shared memory.
__device__ __forceinline__ unsigned long long late_body(uint8_t* seg, GbState* st, float* late_base, long long late_stride,
                                                        unsigned long long* sh, unsigned long long* log_base,
                                                        long long log_stride) {
  const int tid = threadIdx.x;
  const unsigned long long tag = st->late_tag;
  uint8_t* res = seg + OFF_RES + (tag_seq(tag) % NSLOT) * RES_BYTES;
  if (tid == 0) {
    unsigned long long status = LATE_NONE;
    if (tag != 0) {
      status = LATE_LOST;
      if (ld_acquire_sys_u64(res + RES_TAG) == tag)
        status = ld_acquire_sys_u64(res + RES_STATUS) == 0 ? LATE_ADDED : LATE_ERROR;
    }
    sh[0] = status;
  }
  __syncthreads();
  const unsigned long long status = sh[0];
  if (status == LATE_ADDED) {
    const int n4 = (int)tag_rows(tag) * (H / 4);
    float* dst = late_base + (long long)st->late_idx * late_stride;
    for (int i = tid; i < n4; i += blockDim.x) {
      const uint4 v = ld_volatile_u4(res + RES_ROWS + 16 * i);
      float4* d = reinterpret_cast<float4*>(dst + 4 * i);
      float4 o = *d;
      o.x += __uint_as_float(v.x); o.y += __uint_as_float(v.y); o.z += __uint_as_float(v.z); o.w += __uint_as_float(v.w);
      *d = o;
    }
  }
  __syncthreads();
  if (tid == 0) {
    st->late_status = status;
    if (status != LATE_NONE) {
      if (log_base != nullptr) {
        unsigned long long* lg = log_base + (long long)st->late_idx * log_stride;
        lg[8] = status; lg[9] = globaltimer();
      }
      if (status == LATE_ADDED) st->n_late_added += 1;
      else st->n_lost += 1;
      st->late_tag = 0;
    }
  }
  return status;
}

// --- hint -----------------------------------------------------------------------------------------------------------
// h: int32 count, then up to 16 store slots.
__device__ __forceinline__ void hint_body(uint8_t* seg, GbState* st, const int* h) {
  const int tid = threadIdx.x;
  if (tid < 1 + HINT_MAX) reinterpret_cast<int*>(seg + OFF_HINT)[tid] = h[tid];
  fence_sys();
  __syncthreads();
  if (tid == 0) {
    const unsigned long long hs = st->hint_seq + 1;
    st->hint_seq = hs;
    st_release_sys_u64(seg + OFF_DOOR + DOOR_HINT, (hs << 8) | (unsigned long long)(h[0] & 0xFF));
    st->t_hint = globaltimer();
  }
}

}  // namespace bandgb
