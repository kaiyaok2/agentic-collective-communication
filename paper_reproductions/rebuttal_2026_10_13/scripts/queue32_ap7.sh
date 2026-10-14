#!/bin/bash
# queue32 (after queue31): third calibration arm = LIVE probe table + the corrected (v4) simulator structure, end to end
# (Phase 1 -> 3 -> 4 -> 5) for dxe and alltoallv, in a separate repo copy whose search/correctness_test.py is the v4 fork.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; R2=/home/ubuntu/OverlayCCL_v4; Q=/home/ubuntu/queue.log; LC=/home/ubuntu/livecal; mkdir -p $LC
source /home/ubuntu/venv.env; source /home/ubuntu/llm.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE31 DONE" $Q; do sleep 20; done
stage "QUEUE32 live table + v4 simulator structure, run_search dxe + alltoallv"
rsync -a --delete --exclude=.git --exclude=__pycache__ --exclude=training/results $REPO/ $R2/
cp $AP/correctness_test_v4.py $R2/search/correctness_test.py
python3 $AP/live_probe_table.py $R2 >> $Q 2>&1 || true
WCSV=$(echo $WORKERS | tr ' ' ',')
cd $R2
for P in dxe alltoallv; do
  RT=runtime/trainium_${P}_7node.py; [ -f $RT.paper.bak ] || cp $RT $RT.paper.bak; cp $RT.paper.bak $RT
  stage "QUEUE32 run_search --problem $P [v4live]"
  env OVERLAY_HW_TABLE=/home/ubuntu/hwprobe/hw_probe_measured.json NEURON_VENV=$VENV PYTHONPATH=$R2 PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH timeout 5400 $VENV/bin/python experiments/run_search.py \
    --problem $P --llm-model sonnet --phase3-style strategy-enumerate --max-rounds 4 --num-nodes 7 --worker-addrs "$WCSV" --hw-eval \
    --output-dir experiments/results_v4live_${P} > $LC/${P}_v4live.log 2>&1
  echo "  rc=$? $(grep -cE 'Generated:' $LC/${P}_v4live.log) generated" | tee -a $Q
  cp $RT $LC/${P}_v4live_runtime.py; cp -r experiments/results_v4live_${P} $LC/ 2>/dev/null || true; cp $RT.paper.bak $RT
done
cd $REPO; for ip in $WORKERS; do rsync -az -e "ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" $REPO/runtime/ ubuntu@$ip:$REPO/runtime/; done
python3 - <<'PY'
import json, os
man=[{"problem":p,"cand":"livecal_v4live","kind":"file","path":f"/home/ubuntu/livecal/{p}_v4live_runtime.py"} for p in ("dxe","alltoallv") if os.path.exists(f"/home/ubuntu/livecal/{p}_v4live_runtime.py")]
json.dump(man, open("/home/ubuntu/ap7/manifest_livecal_v4.json","w"), indent=1); print(len(man))
PY
stage "QUEUE32 measure v4live runtimes, 3 repeats"
for r in 1 2 3; do MANIFEST=$AP/manifest_livecal_v4.json SWEEP_OUT=/home/ubuntu/sweep_livecal_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE32 DONE"
