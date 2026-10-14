#!/bin/bash
# queue27 (idle hour before expiry): second repeats of the two best collective-share configs (agent + baseline).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 1000 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D; }
stage "QUEUE27 repeat-2 of topk2ex8192 and topk4ex8192 (agent, baseline)"
run olmoe_topk2ex8192_agent_r2    training/train_olmoe10b_topk2ex8192.py agent
run olmoe_topk2ex8192_baseline_r2 training/train_olmoe10b_topk2ex8192.py baseline
run olmoe_topk4ex8192_agent_r2    training/train_olmoe10b_topk4ex8192.py agent
run olmoe_topk4ex8192_baseline_r2 training/train_olmoe10b_topk4ex8192.py baseline
stage "QUEUE27 DONE"
