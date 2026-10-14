#!/bin/bash
# queue35 (after queue34): OLMoE-scope measurement of the live-arm deployed AllToAllV (collective_permute ring, wrapped helper).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log; LC=/home/ubuntu/livecal
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
until grep -q "QUEUE34 DONE" $Q; do sleep 20; done
stage "QUEUE35 OLMoE-scope: live-arm permute-ring AllToAllV (wrapped helper)"
cp $LC/alltoallv_live_runtime_wrapped.py $REPO/runtime/trainium_alltoallv_7node.py
D=/home/ubuntu/olmoe_results/olmoe_livecal_live_a2av; mkdir -p $D
RESULTS_DIR=$D PERCALL_PROBE=0 timeout 1500 bash $AP/run_ap7.sh training/train_olmoe10b_hl_w100.py agent 160 olmoe_livecal_live_a2av --ce agent >> $Q 2>&1; harvest $D
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
stage "QUEUE35 DONE"
