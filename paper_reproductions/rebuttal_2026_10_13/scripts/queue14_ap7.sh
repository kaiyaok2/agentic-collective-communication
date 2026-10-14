#!/bin/bash
# queue14: AllToAllV seed-library helper variants on the per-primitive harness (sim-vs-HW pool for A-Q2), then queue11 (EXDIM).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE14 a) AllToAllV seed-library helper variants ($(python3 -c "import json;print(len(json.load(open('$AP/manifest_a2av.json'))))") cands) on per-primitive harness"
MANIFEST=$AP/manifest_a2av.json SWEEP_OUT=/home/ubuntu/sweep_a2av bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "QUEUE14 b) resume queue11 (EXDIM sweep)"
bash $AP/queue11_ap7.sh
stage "QUEUE14 DONE"
