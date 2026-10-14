#!/bin/bash
# queue38 (after queue37): collective-share direction DM=4096 (dense compute ~DM^2, payload ~DM): TOPK 2 / EXDIM 8192 and TOPK 4 / EXDIM 8192,
# x agent/stub/baseline; 160 steps.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 2400 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py; }
mk () { sed -E "$2; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/training/train_olmoe10b_$1.py; }
until grep -q "QUEUE37 DONE" $Q; do sleep 20; done
mk dm4096_topk2_ex8192 's/^DM *=.*/DM        = 4096/; s/^TOPK *=.*/TOPK      = 2/; s/^EXDIM *=.*/EXDIM     = 8192/'
mk dm4096_topk4_ex8192 's/^DM *=.*/DM        = 4096/; s/^TOPK *=.*/TOPK      = 4/; s/^EXDIM *=.*/EXDIM     = 8192/'
stage "QUEUE38 OLMoE DM=4096: topk2/ex8192 and topk4/ex8192 x agent/stub/baseline"
for cfg in dm4096_topk2_ex8192 dm4096_topk4_ex8192; do
  run olmoe_${cfg}_agent    training/train_olmoe10b_$cfg.py agent    -
  run olmoe_${cfg}_nocomm   training/train_olmoe10b_$cfg.py agent    $AP/cands/alltoallv__nocomm__olmoe.py
  run olmoe_${cfg}_baseline training/train_olmoe10b_$cfg.py baseline -
done
stage "QUEUE38 DONE"
