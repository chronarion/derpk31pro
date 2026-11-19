// The band send's plan (sidecar/band/rtxband_fused.cu: band_qsend / band_plan_hint) built for the host, for
// tests/test_band_prior.py on the CPU: the kernel's phases as loops over the pick index in the same order, the B1
// phase through the same device functions (sidecar/band/band_prior.cuh).
//   g++ -O2 -std=c++17 -shared -fPIC -I<repo>/sidecar/band -o libbandprior_host.so band_prior_host.cpp
#include <math.h>

#include "band_prior.cuh"

namespace {
constexpr int NE = 16, MT = 8, NPMAX = MT * 16;
}

// One call: ids / w [T][16] (T <= 8), row_map / band_map [896], cap (as the send's flags bits 16-23: 0 or >= 16 =
// all 16), prior [8] or null (today's plan). Out: work [16 + 16 * 8] ints = the request's work list after its 8-int
// header (band slot per place, then each place's weight per row as float bits; unused entries 0), served [T * 16],
// stats [3] (served, past the cap, all picks), hint [1 + 16] (n, slots). Returns the call's expert count n_exp.
extern "C" int bandprior_plan_host(const int* ids, const float* w, const int* row_map, const int* band_map, int T,
                                   int cap, const float* prior, int* work, int* served, int* stats, int* hint) {
  int e_s[NPMAX], b_s[NPMAX], f_s[NPMAX], p_s[NPMAX], kp_s[NPMAX];
  float w_s[NPMAX], sc_s[NPMAX], k_s[NPMAX];
  T = T < 1 ? 1 : (T > MT ? MT : T);
  const int NP = T * 16;
  for (int tid = 0; tid < NPMAX; tid++) {
    int e = -1, b = -1;
    float wv = 0.f;
    if (tid < NP) {
      e = ids[tid];
      wv = w[tid];
      if (e >= 0 && e < 896 && row_map[e] < 0) b = band_map[e];
    }
    e_s[tid] = e; w_s[tid] = wv; b_s[tid] = b >= 0 ? b : -1;
  }
  for (int tid = 0; tid < NPMAX; tid++) {      // first occurrence of the expert in the call, its largest weight
    int first = 0;
    float sc = -INFINITY;
    if (b_s[tid] >= 0) {
      first = 1;
      const int e = e_s[tid];
      for (int j = 0; j < NP; j++)
        if (b_s[j] >= 0 && e_s[j] == e) {
          if (j < tid) first = 0;
          sc = fmaxf(sc, w_s[j]);
        }
    }
    f_s[tid] = first; sc_s[tid] = sc;
  }
  for (int tid = 0; tid < NPMAX; tid++) {      // place among the distinct experts: by largest weight, ties lower id
    int pos = -1;
    if (b_s[tid] >= 0) {
      pos = 0;
      const float sc = sc_s[tid];
      const int e = e_s[tid];
      for (int j = 0; j < NP; j++)
        if (f_s[j] && (sc_s[j] > sc || (sc_s[j] == sc && e_s[j] < e))) pos++;
    }
    p_s[tid] = pos;
  }
  int n_first = 0;
  for (int j = 0; j < NP; j++) n_first += f_s[j];
  const int ne = (cap > 0 && cap < NE) ? cap : NE;
  if (prior != nullptr) {                      // prior_places
    for (int tid = 0; tid < NPMAX; tid++) k_s[tid] = bandprior::key(tid, NP, e_s, b_s, w_s, prior);
    for (int tid = 0; tid < NPMAX; tid++) kp_s[tid] = bandprior::kept(tid, NP, e_s, b_s, f_s, k_s, ne);
    for (int tid = 0; tid < NPMAX; tid++)
      if (b_s[tid] >= 0) p_s[tid] = bandprior::place(tid, NP, e_s, f_s, kp_s, sc_s, NPMAX);
  }
  const int n_exp = n_first < ne ? n_first : ne;
  for (int tid = 0; tid < NP; tid++) served[tid] = (b_s[tid] >= 0 && p_s[tid] < ne) ? 1 : 0;
  stats[0] = stats[1] = stats[2] = 0;
  for (int j = 0; j < NP; j++) {
    stats[2] += e_s[j] >= 0;
    if (b_s[j] >= 0) { if (p_s[j] < ne) stats[0]++; else stats[1]++; }
  }
  for (int i = 0; i < NE + NE * MT; i++) work[i] = 0;
  for (int i = 0; i < 1 + NE; i++) hint[i] = 0;
  hint[0] = n_exp;
  for (int tid = 0; tid < NP; tid++) {
    if (b_s[tid] >= 0 && p_s[tid] < ne) {
      const int p = p_s[tid];
      if (f_s[tid]) work[p] = b_s[tid];
      float wv = w_s[tid];
      int bits;
      __builtin_memcpy(&bits, &wv, 4);
      work[NE + p * MT + tid / 16] = bits;
    }
    if (b_s[tid] >= 0 && p_s[tid] < n_exp && f_s[tid]) hint[1 + p_s[tid]] = b_s[tid];
  }
  return n_exp;
}
