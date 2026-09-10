#!/bin/bash
# Drive arm-2 primary (controller gap re-confirm) + arm-1 E2E half (adversarial).
# Runs on the master. Each config = one e2e_launch.sh invocation; result grepped
# to /home/ubuntu/e2e_sweep/ablation_results.jsonl.
set -u
cd /home/ubuntu
OUT=/home/ubuntu/e2e_sweep/ablation_results.jsonl
: > "$OUT"
run() {  # arch_script backend tag port extra...
  local SCRIPT=$1 BK=$2 TAG=$3 PORT=$4; shift 4
  echo "=== [$(date -u +%H:%M:%S)] $TAG (backend=$BK port=$PORT extra=$*) ==="
  PORT=$PORT STEPS=${STEPS:-12} SEED=${SEED:-42} \
    bash /home/ubuntu/e2e_launch.sh /home/ubuntu/$SCRIPT $BK "$TAG" "$@" 2>&1 | tail -3
  # harvest RESULT_JSON from any node log
  local RJ=$(grep -h "RESULT_JSON" /home/ubuntu/e2e_sweep/logs/$TAG/*.log 2>/dev/null | head -1 | sed 's/.*RESULT_JSON *//')
  if [ -n "$RJ" ]; then echo "$RJ" >> "$OUT"; echo "  logged: $RJ"; else echo "  NO RESULT_JSON for $TAG"; fi
}

# ---- ARM 2 PRIMARY: controller gap, both archs, nmb16 fused ----
run train_llama10b_tp_families.py baseline arm2_llama_base_s42 29631 --nmb 16 --fuse
run train_llama10b_tp_families.py sorcar   arm2_llama_sorc_s42 29631 --nmb 16 --fuse
run train_gpt10b_tp_families.py   baseline arm2_gpt_base_s42   29632 --nmb 16 --fuse
run train_gpt10b_tp_families.py   sorcar   arm2_gpt_sorc_s42   29632 --nmb 16 --fuse

# ---- ARM 1 E2E HALF: adversarial (unfused; shallow) ----
run train_llama10b_tp_families.py baseline arm1_llama_base_unfused 29633 --nmb 16
run train_llama10b_tp_families.py sorcar   arm1_llama_sorc_unfused 29633 --nmb 16
run train_llama10b_tp_families.py baseline arm1_llama_base_L24 29634 --nmb 16 --fuse --layers 24
run train_llama10b_tp_families.py sorcar   arm1_llama_sorc_L24 29634 --nmb 16 --fuse --layers 24

echo "=== ALL DONE $(date -u) ==="
cat "$OUT"
