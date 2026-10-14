#!/bin/bash
# queue5 (after queue4): E5c OLMoE agent seed3 (lost to the public-runtime OOM), 250 steps
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "E5c OLMoE agent seed3 (250 steps)"
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
T=olmoe_seed3_agent; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
RESULTS_DIR=$D timeout 3600 bash $AP/run_ap7.sh training/train_olmoe10b_seed3.py agent 250 $T --ce agent >> $Q 2>&1
for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
stage "QUEUE5 DONE"
