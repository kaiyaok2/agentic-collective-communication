#!/bin/bash
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE19 HW check of the fixed-simulator search finds on PP (LICM-hoisted zeros buffer; GA unroll+shared buffer), 3 reps"
for r in 1 2 3; do MANIFEST=$AP/manifest_found2.json SWEEP_OUT=/home/ubuntu/sweep_found2_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE19 DONE"
