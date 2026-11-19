#!/bin/bash
# Engine sweep on brandonmusic's KLD protocol: for each configuration (a tag and the
# environment for k31.sh), start the server, score the 512 cuts of corpus/k31-bm-eval.pt against the runner's
# reference (REF, default the official W4A8 one: w4ref0 + w4ref1; KLREF = a full-logits dump as the server sees it
# for exact KL in decode mode, e.g. /kl/ref-w4a8-rot32.npy), run the decode benchmark (2 drafts, chat prompts), stop.
#   sweep.sh PROFILE "tag1 VAR=x VAR=y" "tag2 ..." ...      (NOKLD=1: decode benchmark only; NOBENCH=1: KLD only)
# Data (KLD_DATA, default $STATE/kld-data, mounted at /s): corpus/k31-bm-eval.pt (tools/runner/tools/bm_eval.py) and the
# reference pass's files (REF, KLREF). Results: $STATE/results/engine-bm-<tag>.json (+ .tokens.pt) and sweep.log.
# The harness runs in the runner image (docker build -t derpk31pro-runner tools/runner/docker; RUNNER_IMG overrides).
# L32=1: the long-context set instead (corpus/k31-l32.pt, harness/kld_long.py: 32 x 32,768 tokens, the first 32,768 -
# tail prefilled, the rest forced, exact KL on every row from the reference's hidden states); L32=both: the 512 cuts,
# then L32 with the ratio to them (--cuts). L32_HIDDEN / L32_REF: the reference pass's hidden states and top-64 files
# as both containers see them (default /kl/l32/ref{0,1}/{hidden.npy,kld.pt}); L32_ARGS: more harness arguments (e.g.
# --windows 1 for a smoke run, --kl-rows=-4097:). Results: $STATE/results/engine-l32-<tag>.json (+ .tokens.pt).
set -u
E=$(cd "$(dirname "$0")/.." && pwd); REPO=$(cd "$E/.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
STATE=${STATE:-$REPO/state}; S=${KLD_DATA:-$STATE/kld-data}; L=$STATE/results/sweep.log
mkdir -p $STATE/results $S
RUNNER_IMG=${RUNNER_IMG:-derpk31pro-runner}
PROFILE=$1; shift
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
l32_kld() {     # the long-context set for configuration $1 (server up)
  local tag=$1 t0=$(date +%s) cuts=""
  [ "${L32:-}" = both ] && [ -f $STATE/results/engine-bm-$tag.json ] && cuts="--cuts /r/engine-bm-$tag.json"
  docker run --rm --network host --cpus 2 --cpuset-cpus 44-45 --memory 32g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v $S:/s -v $STATE/results:/r -v ${KLD_DIR:-$STATE/kld}:/kl -v $E:/engine:ro --entrypoint python3 $RUNNER_IMG \
    /engine/harness/kld.py http://127.0.0.1:${PORT:-8113} /s/corpus/k31-l32.pt \
    ${L32_REF:-/kl/l32/ref0/kld.pt,/kl/l32/ref1/kld.pt} /r/engine-l32-$tag.json --long --mode decode --prompt tail \
    --kl-hidden ${L32_HIDDEN:-/kl/l32/ref0/hidden.npy,/kl/l32/ref1/hidden.npy} --pipeline --prefetch \
    ${KLD_FAST:+--decode-logprobs 1} $cuts ${L32_ARGS:-} > $STATE/results/sweep-$tag-l32.log 2>&1
  python3 -I -c "
import json, sys
r = json.load(open(sys.argv[1]))
L, t = r['long'], sys.argv[2]
h, gp = L.get('headline') or {}, L['groups'].get('generated_prompt') or {}
f = lambda x: '-' if x is None else f'{x:.4f}'
ci = h.get('ci95') or [None, None]
print(f'{t} L32: GENERATED FORCED ROWS KLD {f(h.get(\"kld\"))} [{f(ci[0])}, {f(ci[1])}] over {h.get(\"rows\")} rows, '
      f'top-1 {100 * (h.get(\"top1\") or 0):.2f}% | generated prompt rows {f(gp.get(\"kld\"))} over {gp.get(\"rows\")} '
      f'| all rows {f(r[\"kld\"])} | by distance ' + ' '.join(f'{k} {f(v.get(\"generated\"))}'
                                                         for k, v in L['by_distance'].items()))
bk = L.get('by_kind') or {}
print(f'{t} L32 by kind (generated forced / generated prompt): ' + ' '.join(
    f'{k} {f(v.get(\"generated_forced\"))}/{f(v.get(\"generated_prompt\"))}' for k, v in bk.items()))
c = L.get('cuts')
if c and c.get('ratio') is not None:
    print(f'{t} L32 vs the 2,048-token cuts (same kinds): {f(c[\"headline_same_kinds\"])} / '
          f'{f(c[\"cuts_generated_same_kinds\"])} = x{c[\"ratio\"]:.2f}')
" $STATE/results/engine-l32-$tag.json $tag 2>&1 | tee -a $L
  log "$tag: L32 scored in $(( $(date +%s) - t0 )) s"
}
for cfg in "$@"; do
  tag=${cfg%% *}; envs=${cfg#* }; [ "$envs" = "$cfg" ] && envs=""
  log "== $tag: $PROFILE $envs"
  (cd $E && ./k31.sh down >/dev/null 2>&1)
  if ! (cd $E && env HARNESS=1 $envs ./k31.sh up $PROFILE > $STATE/results/sweep-$tag-up.log 2>&1); then
    log "$tag: server failed to start (see sweep-$tag-up.log)"; continue
  fi
  docker logs derpk31pro 2>&1 | grep -E "^K31 ACTIVE" | tail -1 | tee -a $L
  t0=$(date +%s)
  if [ -z "${NOKLD:-}" ] && [ "${L32:-}" != 1 ]; then
  docker run --rm --network host --cpus 2 --cpuset-cpus 44-45 --memory 32g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v $S:/s -v $STATE/results:/r -v ${KLD_DIR:-$STATE/kld}:/kl -v $E:/engine:ro --entrypoint python3 $RUNNER_IMG /engine/harness/kld.py \
    http://127.0.0.1:${PORT:-8113} /s/corpus/k31-bm-eval.pt ${REF:-/s/w4ref0/kld.pt,/s/w4ref1/kld.pt} /r/engine-bm-$tag.json ${SUBSET:+--subset $SUBSET} ${HMODE:+--mode $HMODE} ${PROMPT:+--prompt $PROMPT} ${HWIN:+--windows $HWIN} ${KLREF:+--kl-ref $KLREF} ${KLHIDDEN:+--kl-hidden $KLHIDDEN} \
    ${KLD_FAST:+--decode-logprobs 1 --pipeline --prefetch} \
    > $STATE/results/sweep-$tag-kld.log 2>&1
  python3 -I -c "
import json, sys
r = json.load(open('$STATE/results/engine-bm-$tag.json'))
print(f\"$tag: KLD {r['kld']:.4f} [{r['kld_ci95'][0]:.4f}, {r['kld_ci95'][1]:.4f}] median {r['median']:.4f} p90 {r['p90']:.3f} \"
      f\"p99 {r['p99']:.2f} top-1 {100 * r['top1']:.2f}% cover {r['cover']:.3f} | \" + ' '.join(f'{k} {v:.3f}' for k, v in r.get('by_domain', {}).items()))
" 2>&1 | tee -a $L
  python3 -I -c "
import json
r = json.load(open('$STATE/results/engine-bm-$tag.json'))
if r.get('kld_generated') is not None:
    ci = r['kld_generated_ci95']
    print(f\"$tag: GENERATED ROWS (headline) KLD {r['kld_generated']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] over {r['generated_rows']} rows, \"
          f\"top-1 {100 * r['generated_top1']:.2f}% | by role \" + ' '.join(f'{k} {v:.3f}' for k, v in r['by_role'].items())
          + ' | generated by domain ' + ' '.join(f'{k} {v:.3f}' for k, v in (r.get('generated_by_domain') or {}).items() if v is not None))
if r.get('kld_decode_rows') is not None and 'exact' in r.get('note', ''):
    print('$tag: exact KL on the decode rows (all roles)', round(r['kld_decode_rows'], 4), '| top-k estimate', round(r['kld_topk_estimate'], 4))
" 2>&1 | tee -a $L
  log "$tag: scored in $(( $(date +%s) - t0 )) s"
  fi
  [ -z "${NOKLD:-}" ] && [ -n "${L32:-}" ] && l32_kld "$tag"
  [ -z "${NOBENCH:-}" ] && python3 -I $E/harness/bench.py http://127.0.0.1:8113 --runs ${BENCH_RUNS:-6} --max-tokens ${BENCH_TOKENS:-256} 2>&1 | tee $STATE/results/bench-$tag.out | tail -1 | sed "s/^/$tag decode: /" | tee -a $L
  docker logs derpk31pro 2>&1 | grep -E "SpecDecoding metrics" | tail -1 | sed "s/.*Mean acceptance/$tag acceptance: Mean acceptance/" | tee -a $L
  docker logs derpk31pro 2>&1 | grep -E "^K31 residency: [0-9]" | tail -1 | sed "s/^/$tag /" | tee -a $L
  (cd $E && ./k31.sh down >/dev/null 2>&1)
done
log "sweep done"
