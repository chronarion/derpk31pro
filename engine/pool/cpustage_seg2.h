// derpk31pro r13: the Grace pool's mailbox with a ring of job slots (UNCOMPILED; replaces r12/cpustage_seg.h when the
// pool serves late jobs). r12's segment has one request and one result because a layer's finish ended before the next
// layer's job was written. A late job (deadline: the next layer's attention end, ~145 us after its tag in a KDA layer,
// ~200 us before an MLA layer) is still running when the next layer publishes its own, so jobs and results get
// CPUSEG2_NSLOT slots, used round-robin by sequence number, and the pool runs them in order instead of abandoning the
// running one when the door changes.
//
//   GB300 (send kernel, one CTA):  seq = st->seq + 1; s = seq % NSLOT; writes JOB[s], XQ[s], XE[s], fence, then
//                                  DOOR = tag(seq, layer, rows) with st.release.sys; st->tag[s] = tag
//   pool thread 0:                 next = last_seq + 1 while next <= seq(DOOR): the job in slot next % NSLOT, checked
//                                  against its own header tag (JOB[s].tag == tag(next, ..): a slot overwritten by a job
//                                  NSLOT later is skipped), barriers as r12, rows into RES[s], DONE[s] = its tag (release)
//   GB300 (late_land, one CTA):    bounded wait for DONE[slot] == st->tag[slot]; rows from RES[slot]
//
// A job is dropped without rows when the door is already >= NSLOT - 1 jobs ahead of it (its landing has timed out).
// One class of job per layer and call in this version: `flags & CPUJOB2_ONTIME` marks a job whose rows the same
// layer's finish adds on time (r12's tier; at most CPUSEG_PMAX_ONTIME records), otherwise the rows are a late share.
// A layer that wants both sends two jobs (the on-time one first).
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define CPUSEG_FN __host__ __device__
#else
#define CPUSEG_FN
#endif

#define CPUSEG2_MAGIC 0x3253555043313343ull /* "C31CPUS2" */
#define CPUSEG_ST_ERROR 1ull

enum {
  CPUSEG_MAXT = 8,            // rows of a call
  CPUSEG2_PMAX = 8,           // records of a job (late jobs: up to 7 before an MLA layer)
  CPUSEG_PMAX_ONTIME = 4,
  CPUSEG_H = 3584,
  CPUSEG_KB = 112,
  CPUSEG2_NSLOT = 4,
};

enum {
  CPUSEG2_JOB_BYTES = 512,                                              // CpuJob2, padded
  CPUSEG2_X_BYTES = CPUSEG_MAXT * CPUSEG_H + CPUSEG_MAXT * CPUSEG_KB,   // int8 rows (row-planar) + exponent bytes: 29,568
  CPUSEG2_RES_BYTES = CPUSEG_MAXT * CPUSEG_H * 4,                       // float32 rows: 114,688
};

// Byte offsets; control words own whole 128 B lines (one writer per line).
enum {
  CPUSEG2_OFF_MAGIC = 0,
  CPUSEG2_OFF_DOOR = 128,                                   // u64 tag of the newest job (GB300 -> pool)
  CPUSEG2_OFF_STATE = 256,                                  // pool counters, as r12 (+56 u64 jobs dropped as stale)
  CPUSEG2_OFF_DONE = 512,                                   // NSLOT x 128 B: u64 done tag, u64 status (pool -> GB300)
  CPUSEG2_OFF_JOB = 4096,                                   // NSLOT x CPUSEG2_JOB_BYTES
  CPUSEG2_OFF_X = 8192,                                     // NSLOT x CPUSEG2_X_BYTES (rounded up to 32,768 each)
  CPUSEG2_X_STRIDE = 32768,
  CPUSEG2_OFF_RES = 8192 + CPUSEG2_NSLOT * 32768,           // 139,264: NSLOT x CPUSEG2_RES_BYTES
  CPUSEG2_BYTES = 8192 + CPUSEG2_NSLOT * 32768 + CPUSEG2_NSLOT * 114688 + 57344,   // 655,360 = 10 x 64 KiB pages
};

CPUSEG_FN static inline uint64_t cpuseg_tag(uint64_t seq, uint32_t layer, uint32_t rows) {
  return (seq << 16) | ((uint64_t)(layer & 0xFF) << 8) | (rows & 0xFF);
}
CPUSEG_FN static inline uint64_t cpuseg_seq(uint64_t tag) { return tag >> 16; }

struct CpuJob2 {
  uint64_t tag;                             // this job's own tag (the pool checks it against the sequence it expects)
  uint32_t n_pick;                          // 1..CPUSEG2_PMAX records
  uint32_t n_rows;                          // T
  uint32_t layer;
  uint32_t flags;                           // CPUJOB_OUT_BF16 | CPUJOB_NO_BF16 (r12) | CPUJOB2_ONTIME
  float alpha, beta;
  int32_t e_min;
  uint32_t pad;
  int32_t slot[CPUSEG2_PMAX];               // bank record index of pick p
  float w[CPUSEG2_PMAX][CPUSEG_MAXT];       // routing weight of row t for pick p; 0 = the row does not use it
};                                           // 8 + 16 + 8 + 8 + 32 + 256 = 328 B
#define CPUJOB_OUT_BF16 1u
#define CPUJOB_NO_BF16 2u
#define CPUJOB2_ONTIME 4u

CPUSEG_FN static inline int cpuseg_planar(int k) { return (k & ~31) + 8 * (k & 3) + ((k & 31) >> 2); }

#ifdef __cplusplus
#define CPUSEG2_SA static_assert
#else
#define CPUSEG2_SA _Static_assert
#endif
CPUSEG2_SA(sizeof(struct CpuJob2) <= CPUSEG2_JOB_BYTES, "job header");
CPUSEG2_SA(CPUSEG2_X_BYTES <= CPUSEG2_X_STRIDE, "activation slot");
CPUSEG2_SA(CPUSEG2_OFF_RES + CPUSEG2_NSLOT * CPUSEG2_RES_BYTES <= CPUSEG2_BYTES, "segment size");
CPUSEG2_SA(CPUSEG2_BYTES % 65536 == 0, "whole 64 KiB pages");
