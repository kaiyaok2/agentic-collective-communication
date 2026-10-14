#!/bin/bash
# queue41: Overlay vs baseline step latency under the lowest-collective-share feasible configs, 3 data seeds each
# (seed 42 already measured twice on cluster 1): topk2+ex8192 and topk4+ex8192, seeds 43 and 44, agent + baseline.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 2000 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D; }
mk () { sed -E "$2; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/training/train_olmoe10b_$1.py; }
stage "QUEUE41 multi-seed Overlay vs baseline: topk2+ex8192, topk4+ex8192; seeds 43, 44"
for seed in 43 44; do
  mk topk2ex8192_s$seed "s/^TOPK *=.*/TOPK      = 2/; s/^EXDIM *=.*/EXDIM     = 8192/; s/^SEED *=.*/SEED      = $seed/"
  mk topk4ex8192_s$seed "s/^TOPK *=.*/TOPK      = 4/; s/^EXDIM *=.*/EXDIM     = 8192/; s/^SEED *=.*/SEED      = $seed/"
  for cfg in topk2ex8192 topk4ex8192; do
    run olmoe_${cfg}_s${seed}_agent    training/train_olmoe10b_${cfg}_s$seed.py agent
    run olmoe_${cfg}_s${seed}_baseline training/train_olmoe10b_${cfg}_s$seed.py baseline
  done
done
stage "QUEUE41 DONE"
