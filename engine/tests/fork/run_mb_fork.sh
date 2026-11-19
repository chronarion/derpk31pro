#!/bin/bash
# derpk31pro gate (a) and the native part of (d): one arm of mb_fork.py (kimi-k31 image, GB300, fresh container).
# usage: run_mb_fork.sh MODE E ["ENV=VAL ..."]      MODE = stock | fork | copy (see run_fork_test.sh)
# Both arms dump into $OUT/dump (same seeds): run stock first, then fork; the fork run compares with the stock dumps,
# and run_cmp_dumps.sh compares the directory again without a GPU.
MODE=$1; E=$2; EXTRA=${3:-}
exec "$(dirname "$0")"/run_fork_test.sh mb-$MODE-e$E${TAGSFX:-} $MODE "E=$E DUMP=/out/dump $EXTRA" mb_fork.py
