#!/bin/bash
# Pull all result artifacts from the master to /tmp/ap7/pull (idempotent).
set -uo pipefail
source /tmp/ap7/cluster.env
KEY=/Users/kekaiyao/.ssh/Kaiyao.pem
OPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
D=/tmp/ap7/pull; mkdir -p $D
for p in sweep_results sweep_results_pass2 sweep_v2 llama7b_results olmoe_results; do
  rsync -az -e "ssh $OPT" --exclude='raw' --exclude='logs' ubuntu@$MASTER_PUB:/home/ubuntu/$p/ $D/$p/ 2>/dev/null
done
scp -q $OPT ubuntu@$MASTER_PUB:/home/ubuntu/queue.log $D/queue.log 2>/dev/null
scp -q $OPT ubuntu@$MASTER_PUB:/home/ubuntu/sweep_results_sim.log $D/sim_scoring.log 2>/dev/null
echo "pulled at $(date -u +%T): $(find $D -name '*.json' | wc -l) json files"; tail -3 $D/queue.log 2>/dev/null
