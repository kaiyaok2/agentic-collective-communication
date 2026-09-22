#!/bin/zsh
# Robust chain driver: r16 -> r17 -> r18. Greps the CORRECT marker file
# (campaign writes "[rN] DONE" to campaign_rN.stdout, NOT .log). recompute_ci.py
# also reads .stdout since that is where confirm lines land.
cd /private/tmp/fair_diverge

run_round () {  # $1=round  $2=problems  $3=seeds  $4=note
  local tag=$1 out=campaign_$1.stdout
  echo "--- launching $tag ($4) @ $3 seeds ---"
  nohup /private/tmp/ablation_venv/bin/python campaign.py --round $tag \
    --problems $2 --confirm-seeds $3 > $out 2>&1 &
  local pid=$!
  echo "$tag launched pid $pid"
  # wait for DONE in the stdout
  for i in $(seq 1 3600); do
    if grep -q "\[$tag\] DONE" "$out" 2>/dev/null; then break; fi
    sleep 20
  done
  echo "=== ${tag:u} DONE $(date -u +%H:%M:%SZ) ==="
  grep -E '^\[screen |^\[confirm ' "$out" 2>/dev/null
  /private/tmp/ablation_venv/bin/python recompute_ci.py "$out" 2>/dev/null || echo "(no confirm data)"
}

run_round r16 "r16_neutral,r16_deepdoc,r16_hintdoc" 8 "CAUSAL framing test"
run_round r17 "r2_deep8,r9_deep8_big" 16 "best-of-16 robustness"
run_round r18 "r18_zs4,r18_zs6,r18_zs8,r18_multctl6" 8 "reverse-lever additive-vs-mult"
echo "=== chain r16->r17->r18 COMPLETE $(date -u +%H:%M:%SZ) ==="
