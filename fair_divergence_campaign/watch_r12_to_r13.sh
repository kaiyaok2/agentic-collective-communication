#!/bin/zsh
# Wait for r12 to finish, recompute its CIs, then launch r13 (reverse-direction).
cd /private/tmp/fair_diverge
LOG=campaign_r12.log
for i in $(seq 1 900); do
  if grep -q '\[r12\] DONE' "$LOG" 2>/dev/null; then break; fi
  sleep 20
done
echo "=== R12 DONE $(date -u +%H:%M:%SZ) ==="
grep -E '^\[screen ' "$LOG" 2>/dev/null
echo "--- fixed-CI ---"
/private/tmp/ablation_venv/bin/python recompute_ci.py "$LOG" 2>/dev/null || echo "(no confirm data / recompute skipped)"
echo "--- launching r13 (REVERSE-DIRECTION hunt) ---"
nohup /private/tmp/ablation_venv/bin/python campaign.py --round r13 \
  --problems r13_minimal,r13_minimal_big,r13_single,r13_hetero \
  --confirm-seeds 8 > campaign_r13.stdout 2>&1 &
echo "r13 launched pid $!"
