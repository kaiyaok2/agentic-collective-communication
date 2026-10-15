#!/bin/bash
# queue53 (after queue52): retry the OLMoE headline reproduction (baseline, agent) in case the EFA retransmit storm has cleared.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
until grep -q "QUEUE52 DONE" $Q; do sleep 30; done
stage "QUEUE53 OLMoE headline retry on cluster 2 (baseline, agent)"
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
for B in baseline agent; do D=/home/ubuntu/olmoe_results/olmoe_c2_headline_${B}_retry; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 2400 bash $AP/run_ap7.sh training/train_olmoe10b_hl_w100.py $B 160 olmoe_c2_headline_${B}_retry $FL >> $Q 2>&1; harvest $D; done
stage "QUEUE53 DONE"
