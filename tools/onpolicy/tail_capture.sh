#!/bin/bash
# On-policy tail benchmark, step 1 (on the serving machine): sample the frozen prompt set from one server
# configuration at each in-flight count.
#   tail_capture.sh PROFILE OUT_DIR PROMPTS.jsonl "1 4" [VAR=value ...]
# PROMPTS.jsonl is the frozen set rebuilt from its manifest (prompts.py --recipe sets/tail-v1.json ...). For every
# count N the configuration answers every prompt once with N requests in flight (temperature 1, top-p 0.95, the
# prompt's own seed, at most 1,024 tokens: capture.py) -> OUT_DIR/TAG-cN.jsonl. The server (engine/k31.sh up PROFILE
# with the VAR=value overrides) is started once and stopped at the end. Environment:
#   TAG          name of the configuration in the file names (default PROFILE)
#   MAX_TOKENS   tokens per answer (default 1024; the score is defined at 1024)
#   EFFORT       reasoning_effort to send (default: not sent)
#   KEEP_UP=1    leave the server running;  SERVE_E, CAP_S, TEMP, TOP_P as in capture.sh
# Use the highest in-flight count the configuration serves (its request slots) as the last count: damage that only
# appears under load is the reason this benchmark exists.
# Then, where the official model runs: tail_score.sh OUT_DIR/TAG-cN.jsonl ...
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PROFILE=${1:?profile}; OUT=${2:?output directory}; PROMPTS=${3:?prompts.jsonl}; COUNTS=${4:?in-flight counts, e.g. "1 4"}
shift 4
TAG=${TAG:-$PROFILE}; want=$(wc -l < "$PROMPTS"); rc=0; last=${COUNTS##* }
for n in $COUNTS; do
  keep=1; [ "$n" = "$last" ] && keep=${KEEP_UP:-0}
  NAME=$TAG-c$n CONC=$n MAX_TOKENS=${MAX_TOKENS:-1024} KEEP_UP=$keep "$HERE/capture.sh" "$PROFILE" "$OUT" "$PROMPTS" "$@" > /dev/null
  if grep -qE "capture done: $want ok, 0 errors, 0 with failed checks" "$OUT/$TAG-c$n.log"; then
    echo "$TAG-c$n: $(grep -E 'capture done' "$OUT/$TAG-c$n.log" | tail -1 | cut -d' ' -f3-)"
  else
    echo "$TAG-c$n: INCOMPLETE: $(tail -1 "$OUT/$TAG-c$n.log" | cut -c1-300)"; rc=1
  fi
done
exit $rc
