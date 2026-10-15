#!/bin/bash
# queue55: second cluster-2 repeat of the 57-candidate pool (small-collective harnesses run fine on the degraded fabric).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE55 cluster-2 repeat 2 of pool v2 (57 candidates)"
MANIFEST=$AP/manifest_v2.json SWEEP_OUT=/home/ubuntu/sweep_c2_v2_r2 bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "QUEUE55 DONE"
