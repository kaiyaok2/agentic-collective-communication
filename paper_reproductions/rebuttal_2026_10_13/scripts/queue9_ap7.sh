#!/bin/bash
# queue9 (after queue8): 3-node OLMoE S=256 agent re-run (needs *_3node runtime modules), + baseline repeat.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
W3=$(echo $WORKERS | awk '{print $1" "$2}')
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_3node.py
cp $REPO/runtime/trainium_dxe_7node.py $REPO/runtime/trainium_dxe_3node.py
sed -i -E 's/^_WORLD *=.*/_WORLD = 96/; s/^_NUM_DEVICES *=.*/_NUM_DEVICES = 48/; s/^_NUM_NODES *=.*/_NUM_NODES = 3/' $REPO/runtime/trainium_alltoallv_3node.py $REPO/runtime/trainium_dxe_3node.py
stage "E4c-fix 3-node OLMoE S=256 agent (+ baseline repeat)"
V=training/train_olmoe10b_3node_S256_w100.py
for b in agent baseline; do
  T=olmoe3n_S256_${b}_r2; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  FL=""; [ $b = agent ] && FL="--ce agent"
  WORKERS_OVERRIDE="$W3" RESULTS_DIR=$D timeout 3000 bash $AP/run_ap7.sh $V $b 160 $T $FL >> $Q 2>&1
  for ip in $W3; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
done
stage "QUEUE9 DONE"
