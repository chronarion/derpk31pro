// RTX band of derpk31pro: the mailbox from a CPU thread. Used by the service's host side (control words, the repack
// request) and as a stand-in sender for tests that do not use the GB300: it writes a call the way the GB300's send
// kernel does and waits for the done tag the way its finish kernel does.
//
// Build: g++ -O2 -std=c++17 -shared -fPIC -o libbandhost.so band_host.cpp
#include "band_mailbox.h"

#include <cstring>
#include <ctime>

namespace {
using namespace bandmb;

inline long long now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
inline uint64_t* word(uint8_t* seg, uint64_t off) { return reinterpret_cast<uint64_t*>(seg + off); }
inline void spin_until(long long t) { while (now_ns() < t) {} }

}  // namespace

extern "C" {

long long bandhost_now_ns() { return now_ns(); }

uint64_t bandhost_get(uint8_t* seg, uint64_t off) { return __atomic_load_n(word(seg, off), __ATOMIC_ACQUIRE); }
void bandhost_set(uint8_t* seg, uint64_t off, uint64_t v) { __atomic_store_n(word(seg, off), v, __ATOMIC_RELEASE); }

uint64_t bandhost_tag(uint64_t seq, int layer, int rows) { return make_tag(seq, layer, rows); }

// Offsets of the mailbox a Python caller needs: [segment bytes, doorbell, state, hint list, request slots, request
// bytes, result slots, result bytes, result rows, slots, bytes of a row in, bytes of a row out]
void bandhost_layout(long long* out) {
  out[0] = SEG_BYTES; out[1] = OFF_DOOR; out[2] = OFF_STATE; out[3] = OFF_HINT; out[4] = OFF_REQ; out[5] = REQ_BYTES;
  out[6] = OFF_RES; out[7] = RES_BYTES; out[8] = RES_ROWS; out[9] = NSLOT; out[10] = ROW_IN; out[11] = ROW_OUT;
}

// Row-major MXFP8 rows (xq [T][3584] E4M3 bytes, xs [T][112] scale codes) -> rows as sent ([T][3696], band order).
void bandhost_pack_rows(const uint8_t* xq, const uint8_t* xs, int T, uint8_t* rows) {
  for (int g = 0; g < T; g++) {
    for (int k = 0; k < 3584; k++) rows[g * ROW_IN + row_in_offset(k)] = xq[g * 3584 + k];
    memcpy(rows + g * ROW_IN + 3584, xs + g * 112, 112);
  }
}

// A seq above everything the mailbox has seen (served or pending): where a new sender starts.
uint64_t bandhost_next_seq(uint8_t* seg) {
  uint64_t top = bandhost_get(seg, OFF_STATE + STATE_SERVED);
  for (int s = 0; s < NSLOT; s++) {
    const uint64_t q = tag_seq(bandhost_get(seg, OFF_DOOR + DOOR_REQ + 8 * s));
    if (q > top) top = q;
  }
  return top + 1;
}

// One call: request slot, then its tag.
void bandhost_send(uint8_t* seg, uint64_t tag, const uint8_t* call, const uint8_t* rows, int T) {
  const int s = (int)(tag_seq(tag) % NSLOT);
  uint8_t* rq = seg + OFF_REQ + (uint64_t)s * REQ_BYTES;
  memcpy(rq + REQ_CALL, call, CALL_BYTES);
  memcpy(rq + REQ_ROWS, rows, (size_t)T * ROW_IN);
  bandhost_set(seg, OFF_DOOR + DOOR_REQ + 8 * s, tag);
}

// Wait for the call's done tag. Returns its status word (>= 0), or -1 after timeout_ns.
long long bandhost_wait(uint8_t* seg, uint64_t tag, long long timeout_ns) {
  const int s = (int)(tag_seq(tag) % NSLOT);
  uint8_t* res = seg + OFF_RES + (uint64_t)s * RES_BYTES;
  const long long t0 = now_ns();
  for (unsigned n = 1;; n++) {
    if (__atomic_load_n(word(res, RES_TAG), __ATOMIC_ACQUIRE) == tag) return (long long)*word(res, RES_STATUS);
    if ((n & 63) == 0 && now_ns() - t0 > timeout_ns) return -1;
  }
}

// A hint: up to 16 store slots, then the hint tag (hint_seq must change with every hint).
void bandhost_hint(uint8_t* seg, uint64_t hint_seq, const int* slots, int n) {
  int* h = reinterpret_cast<int*>(seg + OFF_HINT);
  h[0] = n;
  for (int i = 0; i < n && i < HINT_MAX; i++) h[1 + i] = slots[i];
  bandhost_set(seg, OFF_DOOR + DOOR_HINT, (hint_seq << 8) | (uint64_t)(n & 0xFF));
}

// Post a repack of landing buffer `landing` into store slot `slot` and wait for the acknowledgement.
// Returns fits << 8 | scale base, or -1 after timeout_ns.
long long bandhost_repack(uint8_t* seg, uint64_t rseq, int landing, int slot, long long timeout_ns) {
  bandhost_set(seg, OFF_DOOR + DOOR_REPACK, (rseq << 40) | ((uint64_t)(landing & 0xFF) << 32) | (uint32_t)slot);
  const long long t0 = now_ns();
  for (unsigned n = 1;; n++) {
    const uint64_t a = bandhost_get(seg, OFF_STATE + STATE_REPACK_ACK);
    if ((a >> 40) == rseq) return (long long)(a & 0xFFFF);
    if ((n & 63) == 0 && now_ns() - t0 > timeout_ns) return -1;
  }
}

// Replay a trace of n calls from this thread, one every period_ns (0 = back to back). Call i: calls + 608 i, rows
// + i * T * 3696, seq = seq0 + i, layer = i % layers. hints (may be null): [n][17] = count, 16 slots; the hint of
// call i is posted hint_lead_ns before call i is due, but never before call i - 1 is done. Per call:
//   out_ns[i][0] = send start -> done tag seen, [1] = how late the send started against its slot in the schedule,
//   [2] = status (0, RES_ERROR, or -1 for a timeout), then RTX ns from the tag seen to: [3] done published, [4] the
//   call released to the band CTAs, [5] the first worker at work, [6] the last unit's sum stored, [7] and fenced;
//   [8] = RTX ns the store of the previous call's done tag took (read one call later: it is written after the tag).
// out_rows (may be null): [n][T][3584] float32, the result rows. Returns the number of timeouts.
int bandhost_trace(uint8_t* seg, const uint8_t* calls, const uint8_t* rows, const int* hints, int n, int T,
                   uint64_t seq0, int layers, long long period_ns, long long hint_lead_ns, long long timeout_ns,
                   long long* out_ns, float* out_rows, uint64_t hint_seq0) {
  int late = 0;
  const long long t_begin = now_ns() + 200000;
  for (int i = 0; i < n; i++) {
    const long long due = t_begin + (long long)i * period_ns;
    if (hints != nullptr) {
      if (period_ns > 0) spin_until(due - hint_lead_ns);
      bandhost_hint(seg, hint_seq0 + i, hints + 17 * i + 1, hints[17 * i]);
    }
    if (period_ns > 0) spin_until(due);
    const uint64_t tag = make_tag(seq0 + i, i % layers, T);
    const long long t0 = now_ns();
    bandhost_send(seg, tag, calls + (size_t)CALL_BYTES * i, rows + (size_t)i * T * ROW_IN, T);
    const long long st = bandhost_wait(seg, tag, timeout_ns);
    const long long t1 = now_ns();
    uint8_t* res = seg + OFF_RES + (tag_seq(tag) % NSLOT) * RES_BYTES;
    const uint64_t* tm = word(res, RES_TIMES);
    long long* o = out_ns + 9 * i;
    o[0] = t1 - t0; o[1] = t0 - due; o[2] = st;
    const bool timed = st >= 0 && !(st & RES_EMPTY);
    o[3] = st >= 0 ? (long long)(tm[2] - tm[0]) : 0;
    o[4] = timed ? (long long)(tm[1] - tm[0]) : 0;
    o[5] = timed ? (long long)(tm[3] - tm[0]) : 0;
    o[6] = timed ? (long long)(tm[4] - tm[0]) : 0;
    o[7] = timed ? (long long)(tm[5] - tm[0]) : 0;
    o[8] = 0;
    if (i > 0) {
      const uint64_t* pm = word(seg + OFF_RES + ((tag_seq(tag) - 1) % NSLOT) * RES_BYTES, RES_TIMES);
      if (pm[6] > pm[2] && pm[6] - pm[2] < 1000000000ull) o[8] = (long long)(pm[6] - pm[2]);
    }
    if (st < 0) late++;
    if (out_rows != nullptr) {
      if (st >= 0 && !(st & RES_EMPTY)) memcpy(out_rows + (size_t)i * T * 3584, res + RES_ROWS, (size_t)T * ROW_OUT);
      else memset(out_rows + (size_t)i * T * 3584, 0, (size_t)T * ROW_OUT);
    }
  }
  return late;
}

}  // extern "C"
