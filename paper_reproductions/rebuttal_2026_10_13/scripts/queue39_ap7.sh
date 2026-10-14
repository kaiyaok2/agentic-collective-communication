#!/bin/bash
# queue39 (after queue38): if the DM=4096 configs ran, repeat agent+baseline with two more data seeds each (Overlay vs baseline
# step-latency comparison under the lower-collective-share configuration).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 2400 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D; }
until grep -q "QUEUE38 DONE" $Q; do sleep 20; done
for cfg in dm4096_topk2_ex8192 dm4096_topk4_ex8192; do
  ok=$(ls /home/ubuntu/olmoe_results/olmoe_${cfg}_agent/*ce-agent.json 2>/dev/null | wc -l)
  [ "$ok" -ge 1 ] || { stage "QUEUE39 skip $cfg (no agent result)"; continue; }
  for seed in 2 3; do
    sed -E "s/^SEED *=.*/SEED      = $((42 + seed))/" $REPO/training/train_olmoe10b_$cfg.py > $REPO/training/train_olmoe10b_${cfg}_s$seed.py
    stage "QUEUE39 $cfg seed $seed agent/baseline"
    run olmoe_${cfg}_s${seed}_agent    training/train_olmoe10b_${cfg}_s$seed.py agent
    run olmoe_${cfg}_s${seed}_baseline training/train_olmoe10b_${cfg}_s$seed.py baseline
  done
done
stage "QUEUE39 DONE"
