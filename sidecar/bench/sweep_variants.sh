#!/bin/bash
# Build and time kernel variants (lookahead x warps per CTA). Usage: sweep_variants.sh "PF:W ..." [experts]
here="$(cd "$(dirname "$0")/.." && pwd)"
exps="${2:-1,2,4,8,16}"
for v in $1; do
  IFS=: read pf w <<< "$v"
  so=/work/build/librtxband_pf${pf}_w${w}.so
  "$here"/build.sh $so -DBAND_PF=$pf -DBAND_WARPS=$w 2>&1 | grep -E " error |Used" | tr '\n' ' '
  echo; echo "== PF=$pf W=$w"
  python "$here"/bench/probe_stages.py --lib $so --hi2 /work/data/L01.hi2 --experts $exps --brief 2>&1 | grep -E "^ *[0-9E]+ \|"
done
