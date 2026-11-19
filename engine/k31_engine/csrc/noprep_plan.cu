// K31_MEGAMOE_NOPREP: the plan kernel of a decode-size MoE call (T <= 16 rows), one CTA. It writes MegaMoE's routing
// inputs itself, so vLLM's _prepare_megamoe_inputs_kernel no longer runs before MegaMoE (index half):
//   1. TOPK: vLLM's sigmoid grouped top-k for one group, 896 experts, top 16, with vLLM's arithmetic. The scoring
//      (sigmoid_accurate, preprocess_score) and the routing weights (write_outputs: the warp sum, the IEEE division,
//      the product) are copied from vLLM's csrc/moe/grouped_topk_kernels.cu (image commit 266e88c05; Apache-2.0,
//      LICENSE.vllm here; adapted there from TensorRT-LLM and FlashInfer). vLLM's selection orders the keys (twiddled
//      biased score << 32 | 65535 - id), a strict total order, so any exact top 16 of it is vLLM's: here the 16th
//      largest lane maximum bounds a candidate set that is ranked directly (or, past 64 candidates, 16 rounds of the
//      warp maximum).
//      Without TOPK the call's ids / weights come from vLLM's top-k kernel.
//   2. unitk's plan (k31_engine/unitk.py, non-PRE mode), the same function of the same inputs: ranks within each row
//      (by weight, ties by position), candidates (no resident row, a bank record, rank < strong), distinct candidate
//      experts ordered by their largest weight (ties: lower id) into SLOTS slots, the stage list, the call's row ids
//      (resident row, S + slot or -1), and the residency inputs (decayed scores + w^2, or the pending rows; pick
//      counts).
//      ALLROWS (K31_STAGE_ALL, unitk's ALLROWS): every non-resident pick of an expert that holds a slot uses the slot,
//      at any rank and in any row, not only the candidate picks that staged it.
//   3. MegaMoE's index half in its symmetric buffer: the row ids as int64 and the routing weights as float32, padding
//      rows unrouted (-1 and 0.0), as vLLM's prep kernel writes them.
// Compiled without fast math (vLLM's _moe_C is: IEEE division, no flush to zero; FMA contraction on).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int E = 896;          // experts
constexpr int K = 16;           // picks per row
constexpr int MAXT = 16;        // rows per call
constexpr int SP = 4;           // unitk: candidate ranks considered (strong <= SP)
constexpr int SLOTS = 16;       // staging slots per call
constexpr int THREADS = 512;    // 16 warps: one per row in the selection
constexpr int CHUNKS = E / 32;  // experts per lane in the selection
constexpr int CAND = 64;        // candidate capacity of the ranked path
constexpr unsigned FULL = 0xffffffffu;
constexpr int DYN_PER_ROW = CAND * 8 + K * 8 + E * 4 + E * 4;   // cand keys, top keys, selection bits, unbiased scores

struct Args {
  const float* logits;      // TOPK: [T, E] float32, contiguous
  const float* bias;        // TOPK: [E] float32 (e_score_correction_bias)
  const int* in_ids;        // !TOPK: [T, K] int32 (vLLM's top-k)
  const float* in_w;        // !TOPK: [T, K] float32
  const int* row_map;       // [E] resident row or -1
  const int* bank_map;      // [E] bank index or -1
  float* score;             // [E] or null (no residency inputs)
  int* stats;               // [4] (with score)
  int* pid;                 // pending rows (K31_ACCEPTED, with score): [16, K] ids
  float* pw;                // [16, K] w^2
  int* pt;                  // [1] rows
  const uint8_t* pad;       // [T] bool (vLLM's is_padding) or null
  float* out_w;             // TOPK: [T, K] routing weights
  int* out_ids;             // TOPK: [T, K] expert ids
  int* rows;                // [T, K] the call's row ids (unitk's out)
  int* stage;               // [SLOTS] bank index per slot or -1
  long long* mm_idx;        // MegaMoE's topk_idx (int64), row stride mm_idx_sm
  float* mm_w;              // MegaMoE's topk_weights (float32), row stride mm_w_sm
  long long mm_idx_sm, mm_w_sm, pad_sm;
  int T, strong, S, resmax, renorm, topk, allrows;
  float scale, decay;
};

// ------------------------------------------------------------------------------- vLLM's scoring (copied, see top)
__device__ inline float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

__device__ __forceinline__ void preprocess_score(float input_float, float bias, float& unbiased_score,
                                                 float& selection_score) {
  unbiased_score = 0.0F;
  selection_score = -INFINITY;
  if (!cuda::std::isfinite(input_float) || !cuda::std::isfinite(bias)) {
    return;
  }
  float const unbiased = sigmoid_accurate(input_float);
  float const biased = unbiased + bias;
  unbiased_score = unbiased;
  selection_score = biased == 0.0F ? 0.0F : biased;
}

// cub::Traits<float>::TwiddleIn / TwiddleOut: order-preserving unsigned bits of a float
__device__ __forceinline__ uint32_t twiddle_in(float v) {
  const uint32_t b = __float_as_uint(v);
  return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ __forceinline__ float twiddle_out(uint32_t b) {
  return __uint_as_float(b ^ ((b & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu));
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t v) {
  const uint32_t hi = static_cast<uint32_t>(v >> 32), lo = static_cast<uint32_t>(v);
  const uint32_t mh = __reduce_max_sync(FULL, hi);
  const uint32_t ml = __reduce_max_sync(FULL, hi == mh ? lo : 0u);
  return (static_cast<uint64_t>(mh) << 32) | ml;
}

__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

template <bool TOPK>
__global__ void __launch_bounds__(THREADS) plan_kernel(const Args a) {
  pdl_enter();                                   // the router's logits / ids and every earlier write are visible
  extern __shared__ __align__(16) uint8_t dyn[];
  __shared__ int ids_s[MAXT * K];
  __shared__ float w_s[MAXT * K];
  __shared__ int pk_native[MAXT * K], pk_bank[MAXT * K], pk_rank[MAXT * K], pk_e[MAXT * K];
  __shared__ uint8_t pk_valid[MAXT * K], pk_cand[MAXT * K];
  __shared__ int ent_e[MAXT * SP], ent_b[MAXT * SP], ent_slot[MAXT * SP], ent_full[MAXT * SP];
  __shared__ float ent_w[MAXT * SP], ent_score[MAXT * SP];
  __shared__ uint8_t ent_c[MAXT * SP], ent_first[MAXT * SP], ent_ok[MAXT * SP];
  __shared__ int stage_s[SLOTS], hold_s[SLOTS];       // per slot: bank index, expert
  __shared__ int cnt_s[3];                       // resident, staged, valid picks
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int T = a.T;
  const int NP = T * K, NE = T * SP;

  if constexpr (TOPK) {
    uint64_t* cand_s = reinterpret_cast<uint64_t*>(dyn);                 // [T][CAND]
    uint64_t* top_s = cand_s + T * CAND;                                 // [T][K]
    uint32_t* sel_s = reinterpret_cast<uint32_t*>(top_s + T * K);        // [T][E] twiddled selection scores
    float* unb_s = reinterpret_cast<float*>(sel_s + T * E);              // [T][E] unbiased scores
    // 1a. scores of every (row, expert), spread over the CTA
    for (int i = tid; i < T * E; i += THREADS) {
      const int e = i % E;
      float un, se;
      preprocess_score(a.logits[i], a.bias[e], un, se);
      unb_s[i] = un;
      sel_s[i] = twiddle_in(se);
    }
    __syncthreads();
    // 1b. one warp per row: the top 16 keys in descending order. Lane l owns experts l, l + 32, ...; a key is
    // (twiddled selection score << 32) | (65535 - id): unique, never 0
    if (warp < T) {
      const int t = warp;
      const uint32_t* sv = sel_s + t * E;
      auto key_of = [&](int i) -> uint64_t {
        const int e = lane + 32 * i;
        return (static_cast<uint64_t>(sv[e]) << 32) | static_cast<uint64_t>(65535 - e);
      };
      uint64_t m = 0;
#pragma unroll 4
      for (int i = 0; i < CHUNKS; ++i) {
        const uint64_t kk = key_of(i);
        m = kk > m ? kk : m;
      }
      // tau = the 16th largest of the 32 lane maxima (distinct, as the keys are): at least 16 keys are >= tau, so every
      // key of the top 16 is >= tau
      int r = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) r += __shfl_sync(FULL, m, j) > m;
      const unsigned holder = __ballot_sync(FULL, r == K - 1);
      const uint64_t tau = __shfl_sync(FULL, m, __ffs(holder) - 1);
      int c = 0;
#pragma unroll 4
      for (int i = 0; i < CHUNKS; ++i) c += key_of(i) >= tau;
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(FULL, incl, o);
        if (lane >= o) incl += v;
      }
      const int C = __shfl_sync(FULL, incl, 31);
      uint64_t* cand = cand_s + t * CAND;
      uint64_t* top = top_s + t * K;
      if (C <= CAND) {                           // the usual case: rank the candidates against each other
        int off = incl - c;
#pragma unroll 4
        for (int i = 0; i < CHUNKS; ++i) {
          const uint64_t kk = key_of(i);
          if (kk >= tau) cand[off++] = kk;
        }
        __syncwarp();
        for (int j = lane; j < C; j += 32) {
          const uint64_t kj = cand[j];
          int rank = 0;
          for (int q = 0; q < C; ++q) rank += cand[q] > kj;
          if (rank < K) top[rank] = kj;
        }
      } else {                                   // any input: 16 rounds of the warp maximum over the remaining keys
        uint32_t taken = 0;                      // bit i: this lane's key i is already selected
        for (int k = 0; k < K; ++k) {
          uint64_t lm = 0;
          int li = -1;
#pragma unroll 4
          for (int i = 0; i < CHUNKS; ++i) {
            const uint64_t kk = key_of(i);
            if (!((taken >> i) & 1u) && kk > lm) {
              lm = kk;
              li = i;
            }
          }
          const uint64_t gm = warp_max_u64(lm);
          if (li >= 0 && lm == gm) taken |= 1u << li;    // the one owner (keys are unique, and gm > 0)
          if (lane == 0) top[k] = gm;
        }
      }
      __syncwarp();
      // vLLM's write_outputs for lane k = the k-th key
      float lane_score = -INFINITY;
      int lane_expert = -1;
      float lane_unbiased = 0.0F;
      if (lane < a.topk) {
        const uint64_t kk = top[lane];
        lane_expert = 65535 - static_cast<int>(kk & 0xFFFF);
        lane_score = twiddle_out(static_cast<uint32_t>(kk >> 32));
        lane_unbiased = lane_expert >= 0 && lane_expert < E ? unb_s[t * E + lane_expert] : 0.0F;
      }
      auto tile = cg::tiled_partition<32>(cg::this_thread_block());
      bool const finite_selection = lane < a.topk && lane_score != -INFINITY;
      lane_unbiased = finite_selection ? lane_unbiased : 0.0F;
      unsigned const finite_mask = __ballot_sync(FULL, finite_selection);
      float const sum = cg::reduce(tile, lane_unbiased, cg::plus<float>{});
      if (lane < a.topk) {
        float output = 0.0F;
        if (finite_mask == 0) {
          if (a.renorm) {
            output = 1.0F / static_cast<float>(a.topk);
          }
        } else if (finite_selection) {
          float scale = a.scale;
          if (a.renorm) {
            scale /= sum + 1e-20F;
          }
          output = lane_unbiased * scale;
        }
        ids_s[t * K + lane] = lane_expert;
        w_s[t * K + lane] = output;
        a.out_ids[t * K + lane] = lane_expert;
        a.out_w[t * K + lane] = output;
      }
    }
  } else {
    for (int p = tid; p < NP; p += THREADS) {
      ids_s[p] = a.in_ids[p];
      w_s[p] = a.in_w[p];
    }
  }
  if (tid < SLOTS) stage_s[tid] = hold_s[tid] = -1;
  if (tid < 3) cnt_s[tid] = 0;
  if (a.score != nullptr && a.pid == nullptr) {  // residency scores: decay first (the adds follow after barriers)
    for (int e = tid; e < E; e += THREADS) a.score[e] = a.score[e] * a.decay;
  }
  __syncthreads();

  // 2a. picks: validity, rows, rank within the row (larger weight first, ties: lower position), candidates
  for (int p = tid; p < NP; p += THREADS) {
    const int t = p / K, k = p % K;
    const int id = ids_s[p];
    const float w = w_s[p];
    const bool valid = id >= 0 && id < E;
    const int e = valid ? id : 0;
    const int native = valid ? a.row_map[e] : -1;
    const int bank = valid ? a.bank_map[e] : -1;
    int rank = 0;
    for (int j = 0; j < K; ++j) {
      const float wj = w_s[t * K + j];
      rank += (wj > w) || (wj == w && j < k);
    }
    pk_native[p] = native;
    pk_bank[p] = bank;
    pk_rank[p] = rank;
    pk_e[p] = e;
    pk_valid[p] = valid;
    pk_cand[p] = valid && native < 0 && bank >= 0 && rank < a.strong;
  }
  __syncthreads();
  // 2b. entries: per row, the pick of rank r < SP (ranks are a permutation of 0..K-1 within a row)
  for (int n = tid; n < NE; n += THREADS) {
    const int t = n / SP, r = n % SP;
    int ce = 0, cb = 0;
    float cw = 0.0F;
    bool cc = false;
    for (int k = 0; k < K; ++k) {
      const int p = t * K + k;
      if (pk_rank[p] == r) {
        ce = pk_e[p];
        cw = w_s[p];
        cc = pk_cand[p];
        cb = pk_bank[p];
      }
    }
    ent_e[n] = ce;
    ent_w[n] = cw;
    ent_c[n] = cc;
    ent_b[n] = cb;
  }
  __syncthreads();
  // 2c. the first entry of each candidate expert and the expert's largest weight
  for (int n = tid; n < NE; n += THREADS) {
    bool first = false;
    float sc = -INFINITY;
    if (ent_c[n]) {
      first = true;
      for (int m = 0; m < NE; ++m) {
        if (ent_c[m] && ent_e[m] == ent_e[n]) {
          if (m < n) first = false;
          sc = ent_w[m] > sc ? ent_w[m] : sc;
        }
      }
    }
    ent_first[n] = first;
    ent_score[n] = sc;
  }
  __syncthreads();
  // 2d. slots: place among the distinct candidate experts (larger weight first, ties: lower id)
  for (int n = tid; n < NE; n += THREADS) {
    int slot = 0;
    for (int m = 0; m < NE; ++m) {
      slot += ent_first[m] && (ent_score[m] > ent_score[n] || (ent_score[m] == ent_score[n] && ent_e[m] < ent_e[n]));
    }
    ent_slot[n] = slot;
    ent_ok[n] = ent_first[n] && slot < SLOTS;
  }
  __syncthreads();
  // 2e. every entry's slot = its expert's first entry's slot (or -1); the stage list
  for (int n = tid; n < NE; n += THREADS) {
    int full = -1;
    if (ent_c[n]) {
      for (int m = 0; m < NE; ++m) {
        if (ent_ok[m] && ent_c[m] && ent_e[m] == ent_e[n]) full = ent_slot[m];
      }
    }
    ent_full[n] = full;
    if (ent_ok[n]) {
      stage_s[ent_slot[n]] = ent_b[n];
      hold_s[ent_slot[n]] = ent_e[n];
    }
  }
  __syncthreads();
  // 3. the call's rows, MegaMoE's index half, the residency inputs
  int n_res = 0, n_stg = 0, n_all = 0;
  for (int p = tid; p < NP; p += THREADS) {
    const int t = p / K, k = p % K;
    const bool valid = pk_valid[p], cand = pk_cand[p];
    const int native = pk_native[p], rank = pk_rank[p];
    const bool served = valid && native >= 0 && rank < a.resmax;
    int out = served ? native : -1;
    if (a.allrows) {                             // the slot that holds this pick's expert (at most one does)
      int sidx = -1;
#pragma unroll
      for (int s = 0; s < SLOTS; ++s) {
        if (hold_s[s] >= 0 && hold_s[s] == pk_e[p]) sidx = s;
      }
      if (valid && native < 0 && sidx >= 0) out = a.S + sidx;
    } else {
#pragma unroll
      for (int r = 0; r < SP; ++r) {
        const int col = ent_full[t * SP + r];
        if (cand && rank == r && col >= 0) out = a.S + col;
      }
    }
    a.rows[p] = out;
    const bool is_pad = a.pad != nullptr && a.pad[static_cast<long long>(t) * a.pad_sm] != 0;
    a.mm_idx[t * a.mm_idx_sm + k] = is_pad ? -1LL : static_cast<long long>(out);
    a.mm_w[t * a.mm_w_sm + k] = is_pad ? 0.0F : w_s[p];
    if (a.score != nullptr) {
      const float w = w_s[p];
      if (a.pid != nullptr) {
        a.pid[p] = valid ? ids_s[p] : -1;
        a.pw[p] = valid ? w * w : 0.0F;
      } else if (valid) {
        atomicAdd(a.score + pk_e[p], w * w);
      }
      n_all += valid;
      n_res += served;
      n_stg += out >= a.S && out < a.S + SLOTS && native < 0 && valid;
    }
  }
  if (tid < SLOTS) a.stage[tid] = stage_s[tid];
  if (a.score != nullptr) {
    if (n_res) atomicAdd(&cnt_s[0], n_res);
    if (n_stg) atomicAdd(&cnt_s[1], n_stg);
    if (n_all) atomicAdd(&cnt_s[2], n_all);
    __syncthreads();
    if (tid == 0) {
      atomicAdd(a.stats + 0, cnt_s[0]);
      atomicAdd(a.stats + 1, cnt_s[1]);
      atomicAdd(a.stats + 2, cnt_s[2] - cnt_s[0] - cnt_s[1]);
      atomicAdd(a.stats + 3, cnt_s[2]);
      if (a.pid != nullptr) *a.pt = T;
    }
  }
}

int g_attr = 0;

}  // namespace

// ptrs: logits, bias, in_ids, in_w, row_map, bank_map, score, stats, pid, pw, pt, pad, out_w, out_ids, rows, stage,
//       mm_idx, mm_w (0 = null); ints: T, strong, S, resmax, renorm, topk, mm_idx_sm, mm_w_sm, pad_sm, use_topk, pdl,
//       allrows
void plan(std::vector<int64_t> ptrs, std::vector<int64_t> ints, double scale, double decay) {
  TORCH_CHECK(ptrs.size() == 18 && ints.size() == 12, "noprep plan: argument lists");
  Args a;
  a.logits = reinterpret_cast<const float*>(ptrs[0]);
  a.bias = reinterpret_cast<const float*>(ptrs[1]);
  a.in_ids = reinterpret_cast<const int*>(ptrs[2]);
  a.in_w = reinterpret_cast<const float*>(ptrs[3]);
  a.row_map = reinterpret_cast<const int*>(ptrs[4]);
  a.bank_map = reinterpret_cast<const int*>(ptrs[5]);
  a.score = reinterpret_cast<float*>(ptrs[6]);
  a.stats = reinterpret_cast<int*>(ptrs[7]);
  a.pid = reinterpret_cast<int*>(ptrs[8]);
  a.pw = reinterpret_cast<float*>(ptrs[9]);
  a.pt = reinterpret_cast<int*>(ptrs[10]);
  a.pad = reinterpret_cast<const uint8_t*>(ptrs[11]);
  a.out_w = reinterpret_cast<float*>(ptrs[12]);
  a.out_ids = reinterpret_cast<int*>(ptrs[13]);
  a.rows = reinterpret_cast<int*>(ptrs[14]);
  a.stage = reinterpret_cast<int*>(ptrs[15]);
  a.mm_idx = reinterpret_cast<long long*>(ptrs[16]);
  a.mm_w = reinterpret_cast<float*>(ptrs[17]);
  a.T = static_cast<int>(ints[0]);
  a.strong = static_cast<int>(ints[1]);
  a.S = static_cast<int>(ints[2]);
  a.resmax = static_cast<int>(ints[3]);
  a.renorm = static_cast<int>(ints[4]);
  a.topk = static_cast<int>(ints[5]);
  a.mm_idx_sm = ints[6];
  a.mm_w_sm = ints[7];
  a.pad_sm = ints[8];
  const bool use_topk = ints[9] != 0;
  const bool pdl = ints[10] != 0;
  a.allrows = static_cast<int>(ints[11] != 0);
  a.scale = static_cast<float>(scale);          // vLLM: static_cast<float>(routed_scaling_factor)
  a.decay = static_cast<float>(decay);          // unitk: the Python float as a float32 argument
  TORCH_CHECK(a.T >= 1 && a.T <= MAXT && a.topk == K && a.strong <= SP, "noprep plan: T, topk or strong out of range");
  TORCH_CHECK(a.rows && a.stage && a.mm_idx && a.mm_w && a.row_map && a.bank_map, "noprep plan: missing buffer");
  TORCH_CHECK(use_topk ? (a.logits && a.bias && a.out_w && a.out_ids) : (a.in_ids && a.in_w), "noprep plan: inputs");
  TORCH_CHECK(!a.score || (a.stats && (!a.pid || (a.pw && a.pt))), "noprep plan: residency buffers");
  const size_t dyn = use_topk ? static_cast<size_t>(a.T) * DYN_PER_ROW : 0;
  if (!g_attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(plan_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        MAXT * DYN_PER_ROW));
    g_attr = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  if (use_topk) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, plan_kernel<true>, a));
  } else {
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, plan_kernel<false>, a));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("plan", &plan); }
