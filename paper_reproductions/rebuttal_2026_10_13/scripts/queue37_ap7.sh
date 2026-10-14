#!/bin/bash
# queue37 (after queue30): cross-cluster reproducibility — one repeat of the whole candidate pool on cluster 2
# (57-candidate pool v2 + AllToAllV seed helpers + Uniform-A2A working candidates), 300 steps (ua2a 120).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE30 DONE" $Q; do sleep 20; done
stage "QUEUE37 cluster-2 repeat of pool v2 (57 candidates)"
MANIFEST=$AP/manifest_v2.json SWEEP_OUT=/home/ubuntu/sweep_c2_v2 bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "QUEUE37 cluster-2 repeat of AllToAllV seed helpers"
MANIFEST=$AP/manifest_a2av.json SWEEP_OUT=/home/ubuntu/sweep_c2_a2av bash $AP/sweep_candidates.sh 'alltoallv' >> $Q 2>&1
stage "QUEUE37 cluster-2 repeat of Uniform-A2A working candidates (120 steps)"
STEPS=120 MANIFEST=$AP/manifest_ua2a_ok.json SWEEP_OUT=/home/ubuntu/sweep_c2_ua2a bash $AP/sweep_candidates.sh 'uniform_a2a' >> $Q 2>&1
stage "QUEUE37 DONE"
