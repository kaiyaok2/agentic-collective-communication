#!/bin/bash
# queue31 (new cluster, after queue30): live-calibration end-to-end reproduction.
#  a) fresh single-node Phase-1 probe on THIS cluster -> /home/ubuntu/hwprobe/hw_probe_measured.json
#  b) for dxe and alltoallv: run the full pipeline (Phase 1 -> 3 -> 4 -> 5, LLM sonnet, strategy-enumerate, HW gates on
#     7 nodes) twice: SNAPSHOT table (as shipped) vs LIVE probe table (OVERLAY_HW_TABLE); keep logs + deployed runtimes
#  c) measure the four deployed runtimes with the per-problem harness, 3 repeats
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log; LC=/home/ubuntu/livecal; mkdir -p $LC
source /home/ubuntu/venv.env; source /home/ubuntu/llm.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE30 DONE" $Q; do sleep 20; done
stage "QUEUE31 a) fresh Phase-1 probe (single node, SIZES<=4MB, no collective_permute)"
cp $AP/train_hwprobe_7node.py $REPO/training/
SECTIONS=trivial,xla_ops,index_select,memcpy,graph_launch,back_to_back,collectives SIZES=4096,32768,65536,524288,4194304 WORKERS_OVERRIDE=' ' PERCALL_PROBE=0 RESULTS_DIR=/home/ubuntu/hwprobe \
  timeout 2400 bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe_new >> /home/ubuntu/hwprobe.log 2>&1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_hwprob[e]'
if [ ! -f /home/ubuntu/hwprobe/hw_probe_measured.json ]; then
  [ -f /home/ubuntu/hwprobe/hw_probe_partial.json ] && python3 $AP/derive_table.py /home/ubuntu/hwprobe/hw_probe_partial.json /home/ubuntu/hwprobe/hw_probe_measured.json || cp /home/ubuntu/hwprobe/hw_probe_measured_prevcluster.json /home/ubuntu/hwprobe/hw_probe_measured.json
fi
stage "QUEUE31 b) pipeline runs: snapshot vs live table, dxe + alltoallv"
python3 $AP/live_probe_table.py $REPO >> $Q 2>&1 || true
WCSV=$(echo $WORKERS | tr ' ' ',')
cd $REPO
for P in dxe alltoallv; do
  RT=runtime/trainium_${P}_7node.py; [ -f $RT.paper.bak ] || cp $RT $RT.paper.bak
  for ARM in snapshot live; do
    stage "QUEUE31 run_search --problem $P [$ARM]"
    cp $RT.paper.bak $RT
    ENVX=""; [ $ARM = live ] && ENVX="OVERLAY_HW_TABLE=/home/ubuntu/hwprobe/hw_probe_measured.json"
    env $ENVX NEURON_VENV=$VENV PYTHONPATH=$REPO PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH timeout 5400 $VENV/bin/python experiments/run_search.py \
      --problem $P --llm-model sonnet --phase3-style strategy-enumerate --max-rounds 4 --num-nodes 7 --worker-addrs "$WCSV" --hw-eval \
      --output-dir experiments/results_${ARM}_${P} > $LC/${P}_${ARM}.log 2>&1
    echo "  rc=$? $(grep -cE 'Phase 5|Generated:' $LC/${P}_${ARM}.log) phase5-lines" | tee -a $Q
    cp $RT $LC/${P}_${ARM}_runtime.py; cp -r experiments/results_${ARM}_${P} $LC/ 2>/dev/null || true
    cp $RT.paper.bak $RT
    for ip in $WORKERS; do rsync -az -e "ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" $REPO/runtime/ ubuntu@$ip:$REPO/runtime/; done
  done
done
python3 - <<'PY'
import json, os
man=[{"problem":p,"cand":f"livecal_{a}","kind":"file","path":f"/home/ubuntu/livecal/{p}_{a}_runtime.py"} for p in ("dxe","alltoallv") for a in ("snapshot","live") if os.path.exists(f"/home/ubuntu/livecal/{p}_{a}_runtime.py")]
json.dump(man, open("/home/ubuntu/ap7/manifest_livecal.json","w"), indent=1); print(len(man), "deployed runtimes to measure")
PY
stage "QUEUE31 c) measure deployed runtimes, 3 repeats"
for r in 1 2 3; do MANIFEST=$AP/manifest_livecal.json SWEEP_OUT=/home/ubuntu/sweep_livecal_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE31 DONE"
