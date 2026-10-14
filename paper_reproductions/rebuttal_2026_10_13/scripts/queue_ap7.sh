#!/bin/bash
# Rebuttal experiment queue. Run ON master under nohup. Order = answer value per cluster-hour.
#  E1  C-Q5   Llama-7B past break-even incl. compile (cold cache), 2 backends x 2 cold repeats + 1 warm
#  E2  A-Q2/C-Q1  sim-vs-measured sweep over the candidate pool (training scope, 300 steps)
#  E5  C-Q5/C7    OLMoE-10B headline repeats under the paper protocol, 3 seeds x 2 backends (250 steps)
#  E4  A-Q1   OLMoE at realistic shapes (B=4,S=1024 / L=16), both backends
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
OLMOE_FLAGS_baseline="${OLMOE_FLAGS_baseline:-}"; OLMOE_FLAGS_agent="${OLMOE_FLAGS_agent:---ce agent}"
Q=/home/ubuntu/queue.log; echo "=== queue start $(date -u +%FT%TZ) ===" >> $Q

stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }

stage "E0 sim scoring (CPU, default calibration)"
(cd $REPO && PYTHONPATH=$REPO $VENV/bin/python $AP/score_candidates.py > /home/ubuntu/sweep_results_sim.log 2>&1 &)

stage "E1 Llama-7B break-even: cold per_mb/bundled x2, warm x1, 6000 steps"
for rep in 1 2; do
  COLD=1 timeout 5400 bash $AP/run_llama7b.sh per_mb  6000 cold_per_mb_6000_r$rep  >> $Q 2>&1
  COLD=1 timeout 5400 bash $AP/run_llama7b.sh bundled 6000 cold_bundled_6000_r$rep >> $Q 2>&1
done
timeout 3600 bash $AP/run_llama7b.sh per_mb  6000 warm_per_mb_6000  >> $Q 2>&1
timeout 3600 bash $AP/run_llama7b.sh bundled 6000 warm_bundled_6000 >> $Q 2>&1

stage "E2 sim-vs-measured sweep (all problems)"
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
