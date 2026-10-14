#!/bin/bash
# queue18: a clean 4th repeat of the AllToAllV seed-library candidates (repeat 2 ran while CPU-heavy searches were on the workers).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE17 DONE" $Q; do sleep 15; done
stage "QUEUE18 a2av seed-library helper candidates, repeat 4 (clean)"
MANIFEST=$AP/manifest_a2av.json SWEEP_OUT=/home/ubuntu/sweep_a2av_r4 bash $AP/sweep_candidates.sh 'alltoallv' >> $Q 2>&1
stage "QUEUE18 DONE"
