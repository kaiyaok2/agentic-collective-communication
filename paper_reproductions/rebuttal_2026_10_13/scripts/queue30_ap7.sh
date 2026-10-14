#!/bin/bash
# queue30 (new cluster): B>1 OLMoE with the one-hot gather/scatter fix. B2 DM1024, B4 DM1024, B2 DM2048 x agent/stub/baseline.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
[ -f $REPO/runtime/trainium_alltoallv_7node.py.paper.bak ] || cp $REPO/runtime/trainium_alltoallv_7node.py $REPO/runtime/trainium_alltoallv_7node.py.paper.bak
run () { T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 3600 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py; }
stage "QUEUE30 B>1 OLMoE (one-hot gather/scatter fix): B2 DM1024, B4 DM1024, B2 DM2048"
for cfg in B2fix_DM1024 B4fix_DM1024 B2fix; do
  run olmoe_${cfg}_agent    training/train_olmoe10b_$cfg.py agent    -
  run olmoe_${cfg}_nocomm   training/train_olmoe10b_$cfg.py agent    $AP/cands/alltoallv__nocomm__olmoe.py
  run olmoe_${cfg}_baseline training/train_olmoe10b_$cfg.py baseline -
done
stage "QUEUE30 DONE"
