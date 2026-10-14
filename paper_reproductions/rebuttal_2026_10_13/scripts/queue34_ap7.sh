#!/bin/bash
# queue34 (after queue33): OLMoE-scope measurement of the pipeline-deployed AllToAllV runtimes (wrapped helper):
# v4live (LLM pack-and-gather variant) and snapshot (ag_slice_cat template), headline config (TOPK 8, EXDIM 1024), 160 steps.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log; LC=/home/ubuntu/livecal
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
until grep -q "QUEUE33 DONE" $Q; do sleep 20; done
sed -E 's/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/' $REPO/training/train_olmoe10b.py > $REPO/training/train_olmoe10b_hl_w100.py
run () { T=$1; RT=$2
  cp $RT $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 1500 bash $AP/run_ap7.sh training/train_olmoe10b_hl_w100.py agent 160 $T --ce agent >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py; }
stage "QUEUE34 OLMoE-scope: deployed AllToAllV runtimes (wrapped helper) v4live, snapshot"
[ -f $LC/alltoallv_v4live_runtime_wrapped.py ] && run olmoe_livecal_v4live_a2av $LC/alltoallv_v4live_runtime_wrapped.py
[ -f $LC/alltoallv_snapshot_runtime_wrapped.py ] && run olmoe_livecal_snapshot_a2av $LC/alltoallv_snapshot_runtime_wrapped.py
stage "QUEUE34 DONE"
