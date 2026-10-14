#!/bin/bash
# queue7 (after queue6): A-Q1 at reduced rank count where larger token counts fit:
# 3-node OLMoE (96 experts/ranks, Table 9's 3n config) at S=256 / 512 / 1024, B=1, L=8, both backends,
# 160 steps (steady from step 100).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
W3=$(echo $WORKERS | awk '{print $1" "$2}')
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
stage "E4c 3-node OLMoE token sweep (S=256/512/1024), both backends"
for seq in 256 512 1024; do
  V=training/train_olmoe10b_3node_S${seq}_w100.py
  sed -E "s/^SEQLEN *=.*/SEQLEN    = $seq/; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b_3node.py > $REPO/$V
  for b in baseline agent; do
    T=olmoe3n_S${seq}_$b; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
    FL=""; [ $b = agent ] && FL="--ce agent"
    WORKERS_OVERRIDE="$W3" RESULTS_DIR=$D timeout 3600 bash $AP/run_ap7.sh $V $b 160 $T $FL >> $Q 2>&1
    for ip in $W3; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
  done
done
stage "QUEUE7 DONE"
