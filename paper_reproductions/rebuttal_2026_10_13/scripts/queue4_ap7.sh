#!/bin/bash
# queue4 (after queue3): E3c OLMoE-scope AllToAllV via AG+T+RS helper (algorithm-vs-path test),
#                        E4b realistic shapes that fit (B2_S512_L8, B4_S256_L8), both backends.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
stage "E3c OLMoE-scope: AG+T+RS through the agent path (160 steps)"
V=training/train_olmoe10b_w100.py
sed -E "s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/$V
cp $AP/cands/alltoallv__helperAGRS__olmoe.py $REPO/runtime/trainium_alltoallv_7node.py
D=/home/ubuntu/olmoe_results/olmoe_a2av_helperAGRS; mkdir -p $D
RESULTS_DIR=$D timeout 3600 bash $AP/run_ap7.sh $V agent 160 olmoe_a2av_helperAGRS --ce agent >> $Q 2>&1; harvest $D
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
stage "E4b OLMoE realistic shapes that fit: B2_S512_L8, B4_S256_L8 (160 steps, steady from 100)"
for cfg in "B2_S512_L8:2:512:8" "B4_S256_L8:4:256:8"; do
  IFS=: read -r name bsz seq lay <<< "$cfg"
  VV=training/train_olmoe10b_$name.py
  sed -E "s/^BSZ *=.*/BSZ       = $bsz/; s/^SEQLEN *=.*/SEQLEN    = $seq/; s/^LAYERS *=.*/LAYERS    = $lay/" $REPO/$V > $REPO/$VV
  for b in baseline agent; do
    T=olmoe_${name}_$b; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
    FL=""; [ $b = agent ] && FL="--ce agent"
    RESULTS_DIR=$D timeout 4800 bash $AP/run_ap7.sh $VV $b 160 $T $FL >> $Q 2>&1; harvest $D
  done
done
stage "QUEUE4 DONE"
