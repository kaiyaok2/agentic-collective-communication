#!/bin/bash
# queue59: cluster-2 repeat 3 of the pool (minus PP: large masked ARs hang on the degraded fabric); 15-min per-run cap; must end before block expiry.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE59 cluster-2 repeat 3 (pool v2 minus PP; 15-min per-run cap)"
RUN_TIMEOUT=900 MANIFEST=$AP/manifest_v2.json SWEEP_OUT=/home/ubuntu/sweep_c2_v2_r3 bash $AP/sweep_candidates.sh 'ring_kv|dxe|llama_block|tp_mlp|fsdp|uniform_a2a|alltoallv' >> $Q 2>&1
stage "QUEUE59 DONE"
