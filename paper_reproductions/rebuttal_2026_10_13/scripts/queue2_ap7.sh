#!/bin/bash
# Restart queue after harvest fix: E2 sweep (skips done) -> E5 -> E4 -> E6
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
OLMOE_FLAGS_baseline=""; OLMOE_FLAGS_agent="--ce agent"
Q=/home/ubuntu/queue.log; echo "=== queue2 start $(date -u +%FT%TZ) ===" >> $Q
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "E2 sim-vs-measured sweep (resume)"
bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "E5 OLMoE-10B repeats: 3 seeds x {baseline, agent}, 250 steps"
mkdir -p /home/ubuntu/olmoe_results
for seed in 1 2 3; do
  V=training/train_olmoe10b_seed$seed.py
  sed -E "s/^SEED *=.*/SEED = $seed/" $REPO/training/train_olmoe10b.py > $REPO/$V
  grep -q "^SEED" $REPO/$V || sed -i "0,/^LAYERS/s//SEED = $seed\nLAYERS/" $REPO/$V
  for b in baseline agent; do
    T=olmoe_seed${seed}_$b; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
    RESULTS_DIR=$D timeout 3600 bash $AP/run_ap7.sh $V $b 250 $T $(eval echo \$OLMOE_FLAGS_$b) >> $Q 2>&1
    for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
  done
done

stage "E4 OLMoE realistic shapes (steady window shortened to steps>=100; 160 steps)"
for cfg in "B4_S1024_L8:4:1024:8" "B1_S256_L16:1:256:16" "B4_S1024_L16:4:1024:16"; do
  IFS=: read -r name bsz seq lay <<< "$cfg"
  V=training/train_olmoe10b_$name.py
  sed -E "s/^BSZ *=.*/BSZ       = $bsz/; s/^SEQLEN *=.*/SEQLEN    = $seq/; s/^LAYERS *=.*/LAYERS    = $lay/; s/times\[200:\]/times[100:]/; s/len\(times\) - 200/len(times) - 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/$V
  for b in baseline agent; do
    T=olmoe_${name}_$b; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
    RESULTS_DIR=$D timeout 5400 bash $AP/run_ap7.sh $V $b 160 $T $(eval echo \$OLMOE_FLAGS_$b) >> $Q 2>&1
    for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
  done
done
stage "E6 Llama-7B at 3 nodes (model fixed; D's cluster-size question): per_mb vs bundled, 1000 steps"
W3=$(echo $WORKERS | awk '{print $1" "$2}')
WORKERS_OVERRIDE="$W3" SCRIPT=$REPO/experiments/model_extension/train_llama_e2e_7b_3node.py timeout 2400 bash $AP/run_llama7b.sh per_mb  1000 n3_per_mb_1000  >> $Q 2>&1
WORKERS_OVERRIDE="$W3" SCRIPT=$REPO/experiments/model_extension/train_llama_e2e_7b_3node.py timeout 2400 bash $AP/run_llama7b.sh bundled 1000 n3_bundled_1000 >> $Q 2>&1
stage "QUEUE DONE"
