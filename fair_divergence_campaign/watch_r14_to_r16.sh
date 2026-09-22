#!/bin/zsh
# After r14 finishes: recompute, launch r15; after r15: recompute, launch r16.
cd /private/tmp/fair_diverge

wait_done () {  # $1 = round tag e.g. r14
  local tag=$1 log=campaign_$1.log
  for i in $(seq 1 1200); do
    if [ -f "$log" ] && grep -q "\[$tag\] DONE" "$log" 2>/dev/null; then break; fi
    sleep 20
  done
  echo "=== ${tag:u} DONE $(date -u +%H:%M:%SZ) ==="
  grep -E '^\[screen ' "$log" 2>/dev/null
  echo "--- fixed-CI ---"
  /private/tmp/ablation_venv/bin/python recompute_ci.py "$log" 2>/dev/null || echo "(no confirm data)"
}

launch () {  # $1 round  $2 problems-csv  $3 note
  echo "--- launching $1 ($3) ---"
  nohup /private/tmp/ablation_venv/bin/python campaign.py --round $1 \
    --problems $2 --confirm-seeds 8 > campaign_$1.stdout 2>&1 &
  echo "$1 launched pid $!"
}

wait_done r14
launch r15 "r15_wrap4,r15_wrap6,r15_wrap8,r15_wrap6_big" "partial-collapse gradient"
wait_done r15
launch r16 "r16_neutral,r16_deepdoc,r16_hintdoc" "CAUSAL framing test"
wait_done r16
echo "=== r14->r15->r16 chain complete $(date -u +%H:%M:%SZ) ==="
