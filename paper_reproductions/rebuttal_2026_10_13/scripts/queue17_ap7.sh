#!/bin/bash
# queue17: two more repeats of the AllToAllV seed-library helper candidates (n=1 -> n=3), after queue16.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE16 DONE" $Q; do sleep 15; done
for r in 2 3; do
  stage "QUEUE17 a2av seed-library helper candidates, repeat $r"
  MANIFEST=$AP/manifest_a2av.json SWEEP_OUT=/home/ubuntu/sweep_a2av_r$r bash $AP/sweep_candidates.sh 'alltoallv' >> $Q 2>&1
done
stage "QUEUE17 DONE"
