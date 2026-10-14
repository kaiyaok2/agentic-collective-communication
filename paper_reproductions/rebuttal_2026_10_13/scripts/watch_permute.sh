#!/bin/bash
# kill the alltoallv livecal_live (collective_permute ring) measurement if it stalls, and drop it from the manifest so
# repeats 2-3 skip it (its outcome = hardware-infeasible at 224 ranks).
Q=/home/ubuntu/queue.log; source /home/ubuntu/venv.env
until grep -q "alltoallv__livecal_live" $Q; do sleep 10; done
echo "$(date -u +%T) watcher: live permute-ring run started" >> /home/ubuntu/watch_permute.log
python3 - <<'PY'
import json; p='/home/ubuntu/ap7/manifest_livecal.json'; m=json.load(open(p)); m=[c for c in m if not (c['problem']=='alltoallv' and c['cand']=='livecal_live')]; json.dump(m,open(p,'w'),indent=1)
PY
T0=$(date +%s)
while pgrep -f 'train_alltoallv_7nod[e]' >/dev/null; do
  L=$(ls -t /home/ubuntu/sweep_livecal_r1/raw/logs/alltoallv__livecal_live_node0.log 2>/dev/null | head -1)
  if [ -n "$L" ] && { grep -qE 'CCOM WARN|Assertion|ENC_MAX' "$L" || [ $(( $(date +%s) - T0 )) -gt 900 ]; }; then
    echo "$(date -u +%T) watcher: killing stalled permute-ring run" >> /home/ubuntu/watch_permute.log
    pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'
    for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'" 2>/dev/null; done
    break
  fi
  sleep 20
done
echo "$(date -u +%T) watcher done" >> /home/ubuntu/watch_permute.log
