#!/bin/zsh
# After r16 finishes: recompute, then launch r17 = best-of-16 robustness
# re-confirm of the two headline wins (r2_deep8, r9_deep8_big). Larger N tests
# whether best-of-8 overstated the effect.
cd /private/tmp/fair_diverge
LOG=campaign_r16.log
for i in $(seq 1 1600); do
  if [ -f "$LOG" ] && grep -q '\[r16\] DONE' "$LOG" 2>/dev/null; then break; fi
  sleep 20
done
echo "=== R16 DONE $(date -u +%H:%M:%SZ) ==="
grep -E '^\[screen ' "$LOG" 2>/dev/null
/private/tmp/ablation_venv/bin/python recompute_ci.py "$LOG" 2>/dev/null || echo "(no confirm data)"
echo "--- launching r17 (best-of-16 robustness re-confirm) ---"
nohup /private/tmp/ablation_venv/bin/python campaign.py --round r17 \
  --problems r2_deep8,r9_deep8_big --confirm-seeds 16 \
  > campaign_r17.stdout 2>&1 &
echo "r17 launched pid $!"
