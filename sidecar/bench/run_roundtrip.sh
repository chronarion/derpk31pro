#!/bin/bash
# Host-side launcher of the round-trip benchmark inside the container rtxband-run (both GPUs attached by CDI).
#   bench/run_roundtrip.sh [cpu|gb300] [extra arguments of bench/roundtrip.py ...]
# With gb300 it first checks that no decode speed bench of the engine sweep is running on the GB300 (a "scored"
# line of results/sweep.log without its " decode: median" line yet) and refuses if one is.
RTX=${RTX:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "RTX PRO 6000") {print $1; exit}')}
GB300=${GB300:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " 'index($2, "GB300") {print $1; exit}')}
LOG=${LOG:-$PWD/roundtrip.log}
mode="${1:-cpu}"; shift || true
gb=""
if [ "$mode" = gb300 ]; then
  s=$(grep -n "scored" "$LOG" | tail -1 | cut -d: -f1); d=$(grep -n " decode: median" "$LOG" | tail -1 | cut -d: -f1)
  if [ "${s:-0}" -gt "${d:-0}" ]; then echo "a speed bench is running on the GB300 (sweep.log line $s): not now"; exit 3; fi
  gb="--gb300 $GB300"
fi
docker exec -e CUDA_VISIBLE_DEVICES=$RTX rtxband-run bash -c "cd /work/code && timeout -s KILL 900 python3 bench/roundtrip.py --build /work/build --hi2 /hi2/L01.hi2 $gb $*"
rc=$?
if [ $rc -eq 137 ]; then echo "TIMED OUT: the benchmark was killed"; fi
exit $rc
