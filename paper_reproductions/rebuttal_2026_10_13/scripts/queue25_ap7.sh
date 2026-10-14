#!/bin/bash
# queue25 (after queue24): B>1 fix v2 (per-sample top-k keeps every sort at S=256; comparison one-hot, no scatter).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 1800 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py; }
until grep -q "QUEUE24 DONE" $Q; do sleep 10; done
cp $AP/cands/train_olmoe10b_B*fix2*.py $REPO/training/
stage "QUEUE25 B>1 fix v2: B2 DM1024 agent/stub/baseline, then B4 DM1024"
for cfg in B2fix2_DM1024 B4fix2_DM1024; do
  run olmoe_${cfg}_agent    training/train_olmoe10b_$cfg.py agent    -
  run olmoe_${cfg}_nocomm   training/train_olmoe10b_$cfg.py agent    $AP/cands/alltoallv__nocomm__olmoe.py
  run olmoe_${cfg}_baseline training/train_olmoe10b_$cfg.py baseline -
done
stage "QUEUE25 DONE"
