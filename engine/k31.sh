#!/bin/bash
# derpk31pro: serve Kimi K3.1 Pro on a DGX Station GB300 (+ RTX PRO 6000) with vLLM (kimi-k31 image) and the
# k31_engine MegaMoE hook.
#   k31.sh up [PROFILE]     start (PROFILE = profiles/PROFILE.env, default: default)
#   k31.sh down | status | logs [N]
# Paths and GPUs come from ../config.env (copy config.example.env and edit it) or the environment. The served model is
# a stripped checkpoint (MODEL, no routed experts); the engine reads the resident experts from the official checkpoint
# (NATIVE) and the 2-bit records from the record store (HI2_DIR, store/build_store.py).
set -u
D=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$D/.." && pwd)
# Settings passed in the environment win over config.env and the profile (every variable can be overridden from the
# environment): snapshot them before sourcing, re-apply after
_caller=$(export -p | grep -E '^declare -x (K31_[A-Z0-9_]*|DRAFT|DRAFT_FINAL|DRAFTS|DRAFT_REJECT|SPEC_EXTRA|KV_BYTES|MAXLEN|MAX_SEQS|MAX_BATCHED|GPU_UTIL|CPUS|BAND_CPUS|MEM_LIMIT|MEGAMOE_FORK|GEN_DEFAULTS|PREFIX_UNIT|ALLOC_CONF|HI2_DIR)=' || true)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
PROFILE=${2:-default}
[ -f "$D/profiles/$PROFILE.env" ] && . "$D/profiles/$PROFILE.env"
eval "$_caller"
# K31_POLICY (a container path, /engine/...; from the environment or the profile): its optional "env" object sets global
# knobs that are not set yet (the environment and the profile win)
if [ -n "${K31_POLICY:-}" ] && [ "${1:-}" = up ]; then
  _pf=${K31_POLICY/#\/engine/$D}
  [ -f "$_pf" ] || { echo "K31_POLICY $K31_POLICY: no file $_pf"; exit 1; }
  _pe=$(python3 -I -c 'import json, os, shlex, sys
cfg = json.load(open(sys.argv[1]))
for k, v in cfg.get("env", {}).items():
    if k not in os.environ:
        print(f"export {k}={shlex.quote(str(v))}")' "$_pf") || { echo "K31_POLICY $_pf: unreadable"; exit 1; }
  eval "$_pe"
fi
# K31_LATE=1 (the late tier, k31_engine/latetier.py; off by default) posts its jobs to the Grace pool: it implies
# K31_POOL=1 and the pool's core layout below
case "${K31_LATE:-0}" in ""|0) ;; *) export K31_POOL=1 ;; esac
# K31_POOL=1 (the Grace pool, k31_engine/cpupool.py; off by default): the engine container also holds the pool's
# cores (POOL_CPUS, default 0-71; the pool takes K31_POOL_CORE0.. = 24-71), the engine's own threads move to
# K31_POOL_HOST_CPUS (0-21) and the band sidecar to BAND_CPUS (22-23)
if [ "${K31_POOL:-0}" = 1 ]; then
  CPUS=${POOL_CPUS:-0-71}
  BAND_CPUS=${BAND_CPUS:-22-23}
  export K31_POOL_HOST_CPUS=${K31_POOL_HOST_CPUS:-0-21}
fi
IMG=${IMG:-vllm/vllm-openai:kimi-k31}
MODEL=${MODEL:?set MODEL (the stripped checkpoint, store/run_kstrip.sh) in config.env}
NATIVE=${NATIVE:?set NATIVE (the official moonshotai/Kimi-K3.1-Pro checkpoint) in config.env}
DRAFT=${DRAFT:-}
HI2_DIR=${HI2_DIR:?set HI2_DIR (the record store, store/build_store.py) in config.env}
STATE=${STATE:-$REPO/state}
KLD_DIR=${KLD_DIR:-$STATE/kld}                     # the KLD harness's references and exact-KL outputs (/kl)
MEANS_DIR=${MEANS_DIR:-}                           # the cut picks' mean tables (/means; K31_CUT_MEAN=/means)
RUN_AS=${RUN_AS:-$(id -u):$(id -g)}
gpu_uuid() { nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>/dev/null | awk -F', ' -v pat="$1" 'index($2, pat) {print $1; exit}'; }
GB=${GB:-$(gpu_uuid GB300)}
[ -n "$GB" ] || { echo "no GB300 found (nvidia-smi); set GB to its UUID"; exit 1; }
PORT=${PORT:-8113}
NAME=${NAME:-derpk31pro}
MAXLEN=${MAXLEN:-65536}
log() { echo "[k31 $(date +%T)] $*"; }
# RTX band (K31_BAND_N > 0): a sidecar container on the RTX PRO 6000 holds K31_BAND_N 2-bit records per layer (the
# experts ranked K31_HBM .. K31_HBM + N - 1 by K31_RANK) and serves the engine's band calls through a /dev/shm mailbox
RTX=${RTX:-$(gpu_uuid "RTX PRO 6000")}
SIDECAR_BUILD=${SIDECAR_BUILD:-$REPO/build/sidecar}   # build.sh writes the sidecar's libraries here
BAND_SEG=${BAND_SEG:-/dev/shm/k31band}
[ "${K31_BAND_N:-0}" -gt 0 ] && export K31_BAND=$BAND_SEG

band_down() {
  if docker inspect $NAME-band >/dev/null 2>&1; then
    docker stop -t 60 $NAME-band >/dev/null 2>&1
    mkdir -p $STATE/logs; docker logs $NAME-band > $STATE/logs/$NAME-band-$(date +%Y%m%d-%H%M%S).log 2>&1
    docker rm -f $NAME-band >/dev/null 2>&1
    log "band sidecar down"
  fi
}

band_up() {
  [ "${K31_BAND_N:-0}" -gt 0 ] || return 0
  band_down
  [ -n "$RTX" ] || { log "no RTX PRO 6000 found (nvidia-smi); set RTX to its UUID or K31_BAND_N=0"; return 1; }
  mkdir -p $STATE/band-home
  log "starting the RTX band sidecar: $K31_BAND_N records per layer"
  docker run -d --name $NAME-band --restart no --entrypoint python3 --ipc host --cpuset-cpus ${BAND_CPUS:-48-49} \
    --memory 24g --ulimit core=0 --ulimit memlock=-1:-1 --user $RUN_AS -e HOME=/work/home -e CUDA_MODULE_LOADING=EAGER \
    -e BAND_READAHEAD=${BAND_READAHEAD:-0} -e K31_FADV_RANDOM=${K31_FADV_RANDOM:-0} -e K31_ODIRECT=${K31_ODIRECT:-1} \
    -e K31_BAND_RING=${K31_BAND_RING:-0} \
    -e K31_BAND_BULK=${K31_BAND_BULK:-0} -e K31_BAND_BULK_ROWS=${K31_BAND_BULK_ROWS:-4096} \
    --device nvidia.com/gpu=$RTX -v $REPO/sidecar:/work/code:ro -v $SIDECAR_BUILD:/work/build:ro \
    -v $STATE/band-home:/work/home -v $D:/engine:ro -v $HI2_DIR:/hi2:ro \
    -v ${RANK_DIR:-$REPO/tools/runner/universes}:/rank:ro -w /work/code ${BAND_IMG:-$IMG} -m band.sidecar \
    --build /work/build --hi2 /hi2 --rank ${K31_RANK:-/rank/rank-native.json} --hbm ${K31_HBM:-115} ${K31_HBM_MAP:+--hbm-map $K31_HBM_MAP} ${K31_BAND_MAP:+--band-map $K31_BAND_MAP} --n $K31_BAND_N \
    --seg $BAND_SEG >/dev/null
}

down() {
  docker stop -t 30 $NAME >/dev/null 2>&1
  if docker inspect $NAME >/dev/null 2>&1; then
    mkdir -p $STATE/logs; docker logs $NAME > $STATE/logs/$NAME-$(date +%Y%m%d-%H%M%S).log 2>&1
  fi
  docker rm -f $NAME >/dev/null 2>&1
  band_down
  log "down"
}

up() {
  mkdir -p $STATE/{vllm,flashinfer,triton,dg,nv,torch_ext}
  SPEC_ARGS=()
  [ -n "${SPEC:-}" ] && SPEC_ARGS=(--speculative-config "$SPEC")
  # DRAFT_REJECT=block: the drafter card's rejection sampler (vLLM rejection_sample_method; default standard)
  # DRAFTS=n: the DSpark drafter (DRAFT dir, mounted at /draft) with n draft tokens per step, as tomcupr's recipe
  # DRAFT_SAMPLE=greedy: vLLM's argmax drafts, verified against a one-hot q (default probabilistic: drafts sampled at
  # the request's temperature, their logits cached for the verifier)
  # SPEC_EXTRA: more fields for the speculative config, e.g. '"enable_adaptive_verification":true' (vLLM sizes each
  # step's verify block from the drafter's confidence and a measured cost curve)
  [ -n "${DRAFTS:-}" ] && SPEC_ARGS=(--speculative-config "{\"model\":\"/draft\",\"num_speculative_tokens\":$DRAFTS,\"method\":\"dspark\",\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"${DRAFT_REJECT:+,\"rejection_sample_method\":\"$DRAFT_REJECT\"}${SPEC_EXTRA:+,$SPEC_EXTRA}}")
  # HARNESS=1 (harness/sweep.sh): the KLD harness's forced-token logits processor and 64 log-probs per token; serving
  # runs leave both out
  HARNESS_ARGS=()
  [ "${HARNESS:-0}" = 1 ] && HARNESS_ARGS=(--max-logprobs ${MAX_LOGPROBS:-64} --logits-processors ${LOGITS_PROCS:-k31_engine.force:ForceTokens})
  # PREFIX_UNIT: prefix-cache hits land on multiples of this many tokens (vLLM --prefix-match-unit); without it they land
  # on the hybrid KV blocks (16,384 tokens here), so a repeated long prompt recomputes up to 16K tokens
  PREFIX_UNIT=${PREFIX_UNIT-64}
  # ALLOC_CONF: PyTorch's CUDA allocator settings for the engine (PYTORCH_CUDA_ALLOC_CONF; unset = PyTorch's defaults).
  # ALLOC_CONF=expandable_segments:True lets freed blocks of any size be merged and reused: the default allocator
  # keeps blocks cached for one step shape that other shapes cannot use (3.4 GiB at a prompt chunk's peak)
  # GEN_DEFAULTS: the sampling a request gets when it sets none (vLLM --override-generation-config): temperature 1.0,
  # top-p 0.95, the model card's evaluation setting and the benchmarks' (requests may still send any value); empty =
  # vLLM's own defaults (top-p 1.0)
  GEN_DEFAULTS=${GEN_DEFAULTS-'{"temperature":1.0,"top_p":0.95}'}
  GEN_ARGS=()
  [ -n "$GEN_DEFAULTS" ] && GEN_ARGS=(--override-generation-config "$GEN_DEFAULTS")
  mkdir -p $KLD_DIR
  [ -z "${DRAFTS:-}" ] || [ -n "$DRAFT" ] || { log "DRAFTS needs DRAFT (the DSpark drafter checkpoint) in config.env"; return 1; }
  for p in $MODEL $NATIVE ${DRAFTS:+$DRAFT} $HI2_DIR ${MEANS_DIR:+$MEANS_DIR}; do   # docker would create missing paths as root
    [ -e "$p" ] || { log "missing $p"; return 1; }
  done
  band_up || { log "the band sidecar did not start"; return 1; }
  log "starting: profile $PROFILE, model $MODEL, mode ${K31_MODE:-hbm}, port $PORT"
  docker run -d --name $NAME --restart no --device nvidia.com/gpu=$GB --ipc host --network host \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
    --security-opt label=disable ${MEM_LIMIT:+--memory $MEM_LIMIT} ${CPUS:+--cpuset-cpus $CPUS} \
    -v $MODEL:/model:ro -v $NATIVE:/native:ro -v $D:/engine:ro ${DRAFTS:+-v $DRAFT:/draft:ro} -v ${RANK_DIR:-$REPO/tools/runner/universes}:/rank:ro \
    -v $STATE/vllm:/root/.cache/vllm -v $STATE/flashinfer:/root/.cache/flashinfer -v $STATE/triton:/root/.triton \
    -v $STATE/dg:/root/.deep_gemm -v $STATE/nv:/root/.nv -v $STATE/torch_ext:/torch_ext -e TORCH_EXTENSIONS_DIR=/torch_ext \
    -v $KLD_DIR:/kl ${MEANS_DIR:+-v $MEANS_DIR:/means:ro} -v $HI2_DIR:/hi2:ro -e K31_HI2=/hi2 -e K31_BANK=${K31_BANK:-300} -e K31_STRONG=${K31_STRONG:-3} \
    -e K31_PROMOTE=${K31_PROMOTE:-0} -e K31_HALFLIFE=${K31_HALFLIFE:-32} -e K31_DG_SMS=${K31_DG_SMS:-0} \
    -e K31_MERGE_BLOCKS=${K31_MERGE_BLOCKS:-152} -e K31_TIMING=${K31_TIMING:-1} \
    ${OVERLAY:+-v $OVERLAY:/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh:ro} \
    ${MEGAMOE_FORK:+-v $D/overlay/megamoe_k31/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh:/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe_situ.cuh:ro -v $D/overlay/megamoe_k31/include/deep_gemm/scheduler/mega_moe.cuh:/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:ro} \
    -e PYTHONPATH=/engine/hook:/engine -e K31_ENGINE=1 -e K31_MODE=${K31_MODE:-hbm} -e K31_NATIVE=/native \
    -e K31_RANK=${K31_RANK:-/rank/rank-native.json} -e K31_HBM=${K31_HBM:-115} ${K31_ROWMAP:+-e K31_ROWMAP=$K31_ROWMAP} \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO -e PYTHONUNBUFFERED=1 ${EXTRA:-} \
    ${ALLOC_CONF:+-e PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF} \
    $(env | grep -o "^K31_[A-Z0-9_]*" | grep -vxE "K31_(MODE|NATIVE|RANK|HBM|ROWMAP|HI2|BANK|STRONG|PROMOTE|HALFLIFE|DG_SMS|MERGE_BLOCKS|TIMING)" | sed "s/^/-e /" | tr "\n" " ") \
    --entrypoint /usr/bin/numactl $IMG --membind=0 \
    vllm serve /model --served-model-name kimi-k3.1-pro derpk31pro --trust-remote-code --language-model-only \
    --tensor-parallel-size 1 --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
    --quantization ${QUANT:-fp8_per_block} ${QUANT_CONFIG:+--quantization-config "$QUANT_CONFIG"} --kv-cache-dtype ${KV_DTYPE:-fp8} \
    --gpu-memory-utilization ${GPU_UTIL:-0.92} ${KV_BYTES:+--kv-cache-memory-bytes $KV_BYTES} \
    --max-model-len $MAXLEN --max-num-seqs ${MAX_SEQS:-4} --max-num-batched-tokens ${MAX_BATCHED:-2048} \
    --disable-flashinfer-autotune ${EAGER:+--enforce-eager} "${SPEC_ARGS[@]}" ${PREFIX_UNIT:+--prefix-match-unit $PREFIX_UNIT} \
    "${HARNESS_ARGS[@]}" "${GEN_ARGS[@]}" \
    --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 --enable-auto-tool-choice \
    --host ${HOST:-127.0.0.1} --port $PORT ${SERVE_EXTRA:+"$SERVE_EXTRA_FLAG"} ${SERVE_EXTRA:+"$SERVE_EXTRA"} >/dev/null || return 1
  for i in $(seq 1 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
      log "ready on :$PORT after $((i * 10)) s"; return 0
    fi
    if [ "$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)" != running ]; then
      log "server died"; docker logs $NAME 2>&1 | grep -vE "^\s*$" | tail -40; return 1
    fi
    sleep 10
  done
  log "not ready after 60 min"; return 1
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'; curl -s http://127.0.0.1:$PORT/v1/models | head -c 300; echo ;;
  logs) docker logs --tail "${2:-60}" $NAME 2>&1 ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
