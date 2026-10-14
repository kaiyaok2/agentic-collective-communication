#!/bin/bash
# queue12: after the HW probe run finishes -> (a) re-run the xla_ops probe section (SIGHUP'd compile in run 1),
# (b) measure the leak-free generic search's "found" programs on HW (sim-exploit check), (c) resume queue11 (EXDIM A-Q1).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
while pgrep -f 'train_hwprob[e]' >/dev/null; do sleep 15; done
stage "QUEUE12 a) probe re-run: SECTIONS=xla_ops"
cp $AP/train_hwprobe_7node.py /home/ubuntu/OverlayCCL/training/
SECTIONS=xla_ops WORKERS_OVERRIDE=' ' PERCALL_PROBE=0 RESULTS_DIR=/home/ubuntu/hwprobe bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe_xlaops >> /home/ubuntu/hwprobe.log 2>&1
stage "QUEUE12 b) HW sweep of generic-search found programs ($(python3 -c "import json;print(len(json.load(open('$AP/manifest_found.json'))))") cands)"
MANIFEST=$AP/manifest_found.json SWEEP_OUT=/home/ubuntu/sweep_found bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "QUEUE12 c) resume queue11 (EXDIM sweep)"
bash $AP/queue11_ap7.sh
stage "QUEUE12 DONE"
