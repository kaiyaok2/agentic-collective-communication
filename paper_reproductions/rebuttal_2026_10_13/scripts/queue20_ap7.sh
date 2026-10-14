#!/bin/bash
# queue20: (a) Uniform-A2A candidates re-measured after fixing the deploy target (harness imports trainium_uniform_a2a.py);
#          (b) OLMoE collective-fraction knobs: TOPK 4 / 2 at EXDIM 1024, and EXDIM 16384 (agent + stub; baseline attempted).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
cp $REPO/runtime/trainium_uniform_a2a.py $REPO/runtime/trainium_uniform_a2a.py.shipped.bak 2>/dev/null || true
stage "QUEUE20 a) uniform_a2a candidates (seed templates + archived variants + runtime files), deploy fixed, 120 steps"
STEPS=120 MANIFEST=$AP/manifest_all.json SWEEP_OUT=/home/ubuntu/sweep_ua2a_fix bash $AP/sweep_candidates.sh 'uniform_a2a' >> $Q 2>&1
STEPS=120 MANIFEST=$AP/manifest_ua2a.json SWEEP_OUT=/home/ubuntu/sweep_ua2a_fix bash $AP/sweep_candidates.sh 'uniform_a2a' >> $Q 2>&1
cp $REPO/runtime/trainium_uniform_a2a.py.shipped.bak $REPO/runtime/trainium_uniform_a2a.py 2>/dev/null || true
run () { # <tag> <script> <backend> <runtime-file|->
  T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 3000 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
}
mk () { # <name> <sed-expr>
  sed -E "$2; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/training/train_olmoe10b_$1.py
}
stage "QUEUE20 b) OLMoE knobs: TOPK=4, TOPK=2 (EXDIM 1024) x baseline/agent/stub; EXDIM=16384 x agent/stub/baseline"
mk topk4 's/^TOPK *=.*/TOPK      = 4/'; mk topk2 's/^TOPK *=.*/TOPK      = 2/'; mk ex16384 's/^EXDIM *=.*/EXDIM     = 16384/'
for cfg in topk4 topk2; do
  run olmoe_${cfg}_baseline training/train_olmoe10b_$cfg.py baseline -
  run olmoe_${cfg}_agent    training/train_olmoe10b_$cfg.py agent    -
  run olmoe_${cfg}_nocomm   training/train_olmoe10b_$cfg.py agent    $AP/cands/alltoallv__nocomm__olmoe.py
done
run olmoe_ex16384_agent  training/train_olmoe10b_ex16384.py agent  -
run olmoe_ex16384_nocomm training/train_olmoe10b_ex16384.py agent  $AP/cands/alltoallv__nocomm__olmoe.py
run olmoe_ex16384_baseline training/train_olmoe10b_ex16384.py baseline -
stage "QUEUE20 DONE"
