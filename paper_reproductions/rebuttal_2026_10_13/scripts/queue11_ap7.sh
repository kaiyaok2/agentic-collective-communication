#!/bin/bash
# queue11 (after queue10): A-Q1 — OLMoE where AllToAllV is NOT ~100% of the step: raise expert compute
# (EXDIM 1024 -> 4096 -> 8192; A2A payload unchanged), both A2A implementations + no-A2A stub.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { # <tag> <script> <backend> <runtime-file|->
  T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 3600 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
}
stage "E4e OLMoE compute-fraction sweep: EXDIM 4096 / 8192 (B=1,S=256,L=8,DM=2048), baseline / agent / nocomm"
for ex in 4096 8192; do
  V=training/train_olmoe10b_ex$ex.py
  sed -E "s/^EXDIM *=.*/EXDIM     = $ex/; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/$V
  run olmoe_ex${ex}_baseline $V baseline -
  run olmoe_ex${ex}_agent    $V agent    -
  run olmoe_ex${ex}_nocomm   $V agent    $AP/cands/alltoallv__nocomm__olmoe.py
done
stage "QUEUE11 DONE"
