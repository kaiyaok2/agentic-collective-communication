#!/bin/zsh
# Wait for r13 to finish, recompute its CIs, then launch r14 (L8 positive test).
cd /private/tmp/fair_diverge
LOG=campaign_r13.log
# r13 may not exist yet (r12->r13 watcher launches it); wait for the file first.
for i in $(seq 1 1200); do
  if [ -f "$LOG" ] && grep -q '\[r13\] DONE' "$LOG" 2>/dev/null; then break; fi
  sleep 20
done
echo "=== R13 DONE $(date -u +%H:%M:%SZ) ==="
grep -E '^\[screen ' "$LOG" 2>/dev/null
echo "--- fixed-CI ---"
/private/tmp/ablation_venv/bin/python recompute_ci.py "$LOG" 2>/dev/null || echo "(no confirm data)"
echo "--- launching r14 (L8 positive: new distributive families) ---"
nohup /private/tmp/ablation_venv/bin/python campaign.py --round r14 \
  --problems r14_zerosum6,r14_zerosum8,r14_lincomb6,r14_lincomb8 \
  --confirm-seeds 8 > campaign_r14.stdout 2>&1 &
echo "r14 launched pid $!"
