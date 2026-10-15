#!/bin/bash
# queue58: hardware-gate the Ring-KV programs the final-simulator search found (N_PARTS 2 and 1 variants), 3 repeats.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE58 HW check of v16b-search Ring-KV finds, 3 repeats"
for r in 1 2 3; do RUN_TIMEOUT=900 MANIFEST=$AP/manifest_found3.json SWEEP_OUT=/home/ubuntu/sweep_found3_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE58 DONE"
