#!/bin/bash
# queue56: cluster-2 repeat 2 restricted to problems whose collectives still run on the degraded fabric (no PP: 235 MB masked ARs hang).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE56 cluster-2 repeat 2 (pool v2 minus PP; 20-min per-run cap)"
RUN_TIMEOUT=1200 MANIFEST=$AP/manifest_v2.json SWEEP_OUT=/home/ubuntu/sweep_c2_v2_r2 bash $AP/sweep_candidates.sh 'ring_kv|dxe|llama_block|tp_mlp|fsdp|uniform_a2a|alltoallv' >> $Q 2>&1
stage "QUEUE56 DONE"
