#!/bin/bash
# queue21 (after queue20): two more repeats of the working uniform_a2a candidates (n=1 -> n=3).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE20 DONE" $Q; do sleep 20; done
python3 - <<'PY'
import json, glob, os
ok = {os.path.basename(f)[:-5] for f in glob.glob('/home/ubuntu/sweep_ua2a_fix/uniform_a2a__*.json') if 'percall' not in f and json.load(open(f)).get('measured_ms') is not None}
man = [c for c in json.load(open('/home/ubuntu/ap7/manifest_all2.json')) if c['problem'] == 'uniform_a2a' and f"uniform_a2a__{c['cand']}" in ok]
json.dump(man, open('/home/ubuntu/ap7/manifest_ua2a_ok.json', 'w'), indent=1); print(len(man), 'working ua2a candidates')
PY
for r in 2 3; do
  stage "QUEUE21 uniform_a2a working candidates, repeat $r (120 steps)"
  STEPS=120 MANIFEST=$AP/manifest_ua2a_ok.json SWEEP_OUT=/home/ubuntu/sweep_ua2a_fix_r$r bash $AP/sweep_candidates.sh 'uniform_a2a' >> $Q 2>&1
done
stage "QUEUE21 DONE"
