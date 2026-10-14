#!/bin/bash
# queue10 (after queue9): simulator-correlation pool v2 — every candidate x 3 repeats at 224 ranks,
# plus default-calibration sim scores for the v2 pool.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
export MANIFEST=$AP/manifest_v2.json
mkdir -p /home/ubuntu/sweep_v2
(cd $REPO && PYTHONPATH=$REPO SIM_OUT=/home/ubuntu/sweep_v2/sim_default_v2.json $VENV/bin/python $AP/score_candidates.py > /home/ubuntu/sweep_v2/sim_scoring.log 2>&1 &)
for rep in 1 2 3; do
  stage "E2v2 rep$rep: pool v2 sweep ($(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))") candidates)"
  SWEEP_OUT=/home/ubuntu/sweep_v2/rep$rep bash $AP/sweep_candidates.sh >> $Q 2>&1
done
stage "QUEUE10 DONE"
