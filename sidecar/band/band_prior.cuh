// RTX band of derpk31pro: B1 (K31_BAND_PRIOR), which band experts a call keeps when it has more than the cap.
//
// Today a call keeps its first `ne` distinct band experts by their largest routing weight (ties: lower id). In a verify
// call the rows are not equally likely to matter: row 0 of a request's block is always accepted, draft row i only if
// drafts 1..i all are. With a prior per row (prior[t], the chance that row t is accepted), the kept set is chosen by
// each expert's largest w x prior[row] over its picks (ties: lower id). Only the kept set changes: the kept experts
// keep today's work-list order (largest weight, ties by lower id), so a call under its cap sends the same bytes as
// without a prior, and with every prior 1 every call does.
//
// The functions take the send kernel's shared arrays, one entry per pick i = row * 16 + k of the call's NP picks:
// e (expert), b (band slot or -1), w (weight), f (1 at the expert's first pick in the call), sc (the expert's largest
// weight). Each is evaluated for one pick: per thread in rtxband_fused.cu (between barriers), per pick in a loop in
// the host test (engine/tests/band_prior_host.cpp).
#pragma once
#include <math.h>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace bandprior {

// The pick's expert's key: the largest w x prior[row] over the expert's picks in the call (-inf without a band slot).
__host__ __device__ inline float key(int i, int NP, const int* e, const int* b, const float* w, const float* prior) {
  if (b[i] < 0) return -INFINITY;
  float k = -INFINITY;
  for (int j = 0; j < NP; j++)
    if (b[j] >= 0 && e[j] == e[i]) k = fmaxf(k, w[j] * prior[j >> 4]);
  return k;
}

// 1 if the pick's expert is among the call's first `ne` distinct band experts by key (ties: lower id), else 0.
__host__ __device__ inline int kept(int i, int NP, const int* e, const int* b, const int* f, const float* k, int ne) {
  if (b[i] < 0) return 0;
  int pos = 0;
  for (int j = 0; j < NP; j++)
    if (f[j] && (k[j] > k[i] || (k[j] == k[i] && e[j] < e[i]))) pos++;
  return pos < ne ? 1 : 0;
}

// The kept pick's place in the work list: its expert's rank among the kept experts by largest weight, ties by lower
// id (today's order); `past` for a pick whose expert is not kept.
__host__ __device__ inline int place(int i, int NP, const int* e, const int* f, const int* kp, const float* sc,
                                     int past) {
  if (!kp[i]) return past;
  int pos = 0;
  for (int j = 0; j < NP; j++)
    if (f[j] && kp[j] && (sc[j] > sc[i] || (sc[j] == sc[i] && e[j] < e[i]))) pos++;
  return pos;
}

}  // namespace bandprior
