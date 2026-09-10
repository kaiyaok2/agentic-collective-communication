#!/bin/bash
# Complete the 3-way matrix: add strat-enum runs alongside the already-logged
# baseline+sorcar. strat emits DISTINCT source (accumulate loops, per-tensor AR
# loops) but keeps baseline collective SCHEDULE on the divergent anchors -> we
# expect strat ~= baseline wall-clock, sorcar 2.0x+ over BOTH.
set -u
cd /home/ubuntu
OUT=/home/ubuntu/e2e_sweep/ablation_results.jsonl   # append to the existing 8
run() { local SCRIPT=$1 BK=$2 TAG=$3 PORT=$4; shift 4
  echo "=== [$(date -u +%H:%M:%S)] $TAG (backend=$BK port=$PORT extra=$*) ==="
  PORT=$PORT STEPS=${STEPS:-12} SEED=${SEED:-42} \
    bash /home/ubuntu/e2e_launch.sh /home/ubuntu/$SCRIPT $BK "$TAG" "$@" 2>&1 | tail -3
  local RJ=$(grep -h "RESULT_JSON" /home/ubuntu/e2e_sweep/logs/$TAG/*.log 2>/dev/null | head -1 | sed "s/.*RESULT_JSON *//")
  if [ -n "$RJ" ]; then echo "$RJ" >> "$OUT"; echo "  logged: $RJ"; else echo "  NO RESULT_JSON for $TAG"; fi
}
# headline arm2: strat, both archs, nmb16 fused
run train_llama10b_tp_families.py strat arm2_llama_strat_s42 29641 --nmb 16 --fuse
run train_gpt10b_tp_families.py   strat arm2_gpt_strat_s42   29642 --nmb 16 --fuse
# arm1 adversarial: strat unfused + shallow
run train_llama10b_tp_families.py strat arm1_llama_strat_unfused 29643 --nmb 16
run train_llama10b_tp_families.py strat arm1_llama_strat_L24 29644 --nmb 16 --fuse --layers 24
echo "=== 3WAY STRAT DONE $(date -u) ==="
