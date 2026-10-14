#!/bin/bash
# queue42 (after queue41): redo the topk2+ex8192 seed-43 agent run that stalled in bootstrap (one worker's devices were still resetting).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
until grep -q "QUEUE41 DONE" $Q; do sleep 20; done
stage "QUEUE42 redo topk2ex8192 seed 43 agent"
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
D=/home/ubuntu/olmoe_results/olmoe_topk2ex8192_s43_agent; mkdir -p $D
RESULTS_DIR=$D PERCALL_PROBE=0 timeout 2000 bash $AP/run_ap7.sh training/train_olmoe10b_topk2ex8192_s43.py agent 160 olmoe_topk2ex8192_s43_agent --ce agent >> $Q 2>&1; harvest $D
stage "QUEUE42 DONE"
