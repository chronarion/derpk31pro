#!/bin/bash
# derpk31pro: the MegaMoE fork's gates on the kimi-k31 image, one fresh GB300 container per arm (each under 1 minute,
# under 12 GB of HBM; 13 GPU containers, about 10 minutes in all). Outputs in $OUT (default
# $PWD/out).
#   (a) fork == the image's stock kernel, bit for bit: 128 and 512 rows, T = 1..64, with and without PDL
#   (b) host rows (D = 8, lane 4) == the same experts in HBM       (c) the same with 6-bit scale codes
#   and the row layout check and the per-call control block test
#   (d) call time: mb lines "fit over" (per call + per expert) and the step lines "each further host row"
# Lane timings need a quiet host link (no other host <-> GPU traffic); repeat run_step_cost.sh and keep the minima.
cd "$(dirname "$0")"
OUT=${OUT:-$PWD/out}
./run_mb_fork.sh stock 128 && ./run_mb_fork.sh fork 128
./run_mb_fork.sh stock 512 && ./run_mb_fork.sh fork 512
TAGSFX=-pdl ./run_mb_fork.sh stock 128 "PDL=1 DUMP=/out/dump-pdl" && TAGSFX=-pdl ./run_mb_fork.sh fork 128 "PDL=1 DUMP=/out/dump-pdl"
./run_step_cost.sh lane4 "LANE=4"
./run_step_cost.sh sf6 "LANE=4 SF6=1"
./run_step_cost.sh lane4-pdl "LANE=4 PDL=1"
./run_step_cost.sh sf6-pdl "LANE=4 SF6=1 PDL=1"
./run_fork_test.sh layout fork "" layout_check.py
./run_fork_test.sh percall fork "" ctl_percall_test.py
./run_fork_test.sh percall-pdl fork "PDL=1" ctl_percall_test.py
./run_cmp_dumps.sh dump; ./run_cmp_dumps.sh dump-pdl
echo "== summary"
grep -h "^done\|exactness part\|layout check\|per-call control block\|pairs compared\|fit over\|each further host row" \
  "$OUT"/mb-{stock,fork}-e{128,512}.out "$OUT"/mb-{stock,fork}-e128-pdl.out "$OUT"/step-{lane4,sf6,lane4-pdl,sf6-pdl}.out \
  "$OUT"/layout.out "$OUT"/percall.out "$OUT"/percall-pdl.out "$OUT"/cmp_dumps-dump.out "$OUT"/cmp_dumps-dump-pdl.out | cut -c1-260
