#!/bin/bash
# queue54: small-collective Llama experiments that the degraded fabric still supports: (a) Llama-7B at 3 nodes (D), (b) Llama-7B 6000 steps from a cold cache (C-Q5 wall-clock incl. compile), both backends.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
W=($WORKERS)
stage "QUEUE54 a) Llama-7B 3-node (cluster 2): per_mb, bundled, 300 steps"
for B in per_mb bundled; do WORKERS_OVERRIDE="${W[0]} ${W[1]}" RESULTS_DIR=/home/ubuntu/llama7b_results_3node timeout 2400 bash $AP/run_llama7b.sh $B 300 ${B}_3node_300 >> $Q 2>&1; done
stage "QUEUE54 b) Llama-7B 6000 steps cold cache (cluster 2): per_mb, bundled"
for B in per_mb bundled; do COLD=1 RESULTS_DIR=/home/ubuntu/llama7b_results timeout 3000 bash $AP/run_llama7b.sh $B 6000 ${B}_6000_cold >> $Q 2>&1; done
stage "QUEUE54 DONE"
