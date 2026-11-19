#!/bin/bash
# derpk31pro gates (b), (c) and the host part of (d): step_cost_k31.py on the fork (kimi-k31 image, GB300).
# usage: run_step_cost.sh TAG ["ENV=VAL ..."]      e.g. run_step_cost.sh lane4 "LANE=4"; run_step_cost.sh sf6 "LANE=4 SF6=1"
# MODE=copy in the environment runs the fork from a copy of the include tree (for DEFINES=...).
TAG=$1; ENVS=${2:-}
exec "$(dirname "$0")"/run_fork_test.sh step-$TAG ${MODE:-fork} "$ENVS" step_cost_k31.py
