#!/bin/bash
# On-policy capture for one server configuration: start it, check one request, sample the prompts, stop it.
#   capture.sh PROFILE OUT_DIR PROMPTS.jsonl [VAR=value ...]
# PROFILE and the VAR=value overrides go to engine/k31.sh up (as harness/sweep.sh passes them), so any candidate
# configuration can be captured the same way. Environment:
#   SERVE_E      engine directory whose k31.sh starts the server (default: this tree's)
#   NAME         name of the capture file, OUT_DIR/NAME.jsonl (default capture)
#   FIRST / SKIP prompt lines to take / to skip (capture.py's --first / --skip)
#   STOP_FILE    no new request starts once this file exists
#   CAP_S        seconds before the client is stopped (default 10800); finished records are kept
#   CONC, MAX_TOKENS, TEMP, TOP_P   (4, 32768, 1, 0.95)
#   EFFORT       sent as reasoning_effort (default: not sent)
#   KEEP_UP=1    leave the server running at the end
# Then: build_sets.py on OUT_DIR/NAME.jsonl, the reference runner over the buckets, and
#   OP="bucket ..." engine/harness/sweep.sh PROFILE "tag K31_FORCE_DRAFTS=1 K31_FORCE_REJECT=drafter VAR=value ..."
# to read the answers back through the same configuration, and analyze.py for the report.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
[ -f "$REPO/config.env" ] && . "$REPO/config.env"
E=${SERVE_E:-$REPO/engine}
PROFILE=${1:?profile}; OUT=${2:?output directory}; PROMPTS=${3:?prompts.jsonl}; shift 3
NAME=${NAME:-capture}; URL=http://127.0.0.1:${PORT:-8113}
mkdir -p "$OUT"
say() { echo "$(date '+%F %T') capture $PROFILE $*: $NAME: $MSG" | tee -a "$OUT/capture.log"; }
if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" != 200 ]; then
  (cd "$E" && env "$@" ./k31.sh up "$PROFILE" > "$OUT/$NAME-up.log" 2>&1) || { MSG="server did not start"; say "$@"; exit 1; }
fi
STREAM=""
if ! python3 -I "$HERE/capture.py" $URL "$PROMPTS" "$OUT/$NAME.jsonl" --smoke > "$OUT/$NAME-smoke.json" 2>&1; then
  if python3 -I "$HERE/capture.py" $URL "$PROMPTS" "$OUT/$NAME.jsonl" --smoke --no-stream > "$OUT/$NAME-smoke.json" 2>&1; then
    STREAM="--no-stream"
  else
    MSG="smoke request failed its checks: $(cut -c1-300 "$OUT/$NAME-smoke.json")"; say "$@"
  fi
fi
timeout ${CAP_S:-10800} python3 -I "$HERE/capture.py" $URL "$PROMPTS" "$OUT/$NAME.jsonl" --concurrency ${CONC:-4} \
  --max-tokens ${MAX_TOKENS:-32768} --temperature ${TEMP:-1} --top-p ${TOP_P:-0.95} $STREAM ${EFFORT:+--reasoning-effort $EFFORT} \
  ${FIRST:+--first $FIRST} ${SKIP:+--skip $SKIP} ${STOP_FILE:+--stop-file $STOP_FILE} 2>> "$OUT/$NAME.log"
rc=$?
docker logs ${SERVER_NAME:-derpk31pro} > "$OUT/$NAME-server.log" 2>&1
MSG="rc $rc; $(grep -E 'capture done' "$OUT/$NAME.log" | tail -1 | cut -d' ' -f3-)"; say "$@"
[ "${KEEP_UP:-0}" = 1 ] || (cd "$E" && ./k31.sh down > /dev/null 2>&1)
exit $rc
