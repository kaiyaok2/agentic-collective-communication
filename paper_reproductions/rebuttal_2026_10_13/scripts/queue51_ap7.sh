#!/bin/bash
# queue51 (after queue50): live-calibration end-to-end on a THIRD problem (ring_kv): snapshot arm vs live+v4 arm; measure deployed runtimes 3x.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; R2=/home/ubuntu/OverlayCCL_v4; Q=/home/ubuntu/queue.log; LC=/home/ubuntu/livecal
source /home/ubuntu/venv.env; source /home/ubuntu/llm.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE50 DONE" $Q; do sleep 20; done
WCSV=$(echo $WORKERS | tr ' ' ',')
rsync -a --delete --exclude=.git --exclude=__pycache__ --exclude=training/results $REPO/ $R2/; cp $AP/correctness_test_v4.py $R2/search/correctness_test.py; python3 $AP/live_probe_table.py $R2 >> $Q 2>&1 || true
P=ring_kv; RT=runtime/trainium_${P}_7node.py
for ARM in snapshot v4live; do
  if [ $ARM = snapshot ]; then WD=$REPO; ENVX=""; else WD=$R2; ENVX="OVERLAY_HW_TABLE=/home/ubuntu/hwprobe/hw_probe_measured.json"; fi
  cd $WD; [ -f $RT.paper.bak ] || cp $RT $RT.paper.bak; cp $RT.paper.bak $RT
  stage "QUEUE51 run_search --problem $P [$ARM]"
  env $ENVX NEURON_VENV=$VENV PYTHONPATH=$WD PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH timeout 5400 $VENV/bin/python experiments/run_search.py \
    --problem $P --llm-model sonnet --phase3-style strategy-enumerate --max-rounds 4 --num-nodes 7 --worker-addrs "$WCSV" --hw-eval \
    --output-dir experiments/results_${ARM}_${P} > $LC/${P}_${ARM}.log 2>&1
  echo "  rc=$? $(grep -cE 'Generated:' $LC/${P}_${ARM}.log) generated" | tee -a $Q
  cp $RT $LC/${P}_${ARM}_runtime.py; cp -r experiments/results_${ARM}_${P} $LC/ 2>/dev/null || true; cp $RT.paper.bak $RT
done
cd $REPO; for ip in $WORKERS; do rsync -az -e "ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" $REPO/runtime/ ubuntu@$ip:$REPO/runtime/; done
python3 - <<'PY'
import json, os
man=[{"problem":"ring_kv","cand":f"livecal_{a}","kind":"file","path":f"/home/ubuntu/livecal/ring_kv_{a}_runtime.py"} for a in ("snapshot","v4live") if os.path.exists(f"/home/ubuntu/livecal/ring_kv_{a}_runtime.py")]
json.dump(man, open("/home/ubuntu/ap7/manifest_livecal_rkv.json","w"), indent=1); print(len(man))
PY
stage "QUEUE51 measure ring_kv deployed runtimes, 3 repeats"
for r in 1 2 3; do MANIFEST=$AP/manifest_livecal_rkv.json SWEEP_OUT=/home/ubuntu/sweep_livecal_rkv_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE51 DONE"
