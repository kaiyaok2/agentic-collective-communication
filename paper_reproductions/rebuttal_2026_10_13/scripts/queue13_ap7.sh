#!/bin/bash
# queue13 (post-incident chain, replaces queue12): the HW probe's 8 MB all_gather compile ran >15 min and the
# master became unreachable; this chain redoes the missing probe sections with the collective sizes capped, then
# measures the generic-search "found" programs on HW, then resumes queue11 (EXDIM, A-Q1).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
cp $AP/train_hwprobe_7node.py /home/ubuntu/OverlayCCL/training/
stage "QUEUE13 a) probe full re-run (master /tmp wiped by reboot -> cold cache), SIZES<=4MB (8MB AG compile thrashed the host)"
COLD=0 SIZES=4096,32768,65536,524288,4194304 WORKERS_OVERRIDE=' ' PERCALL_PROBE=0 RESULTS_DIR=/home/ubuntu/hwprobe \
  bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe_rerun >> /home/ubuntu/hwprobe.log 2>&1
stage "QUEUE13 b) HW sweep of generic-search found programs ($(python3 -c "import json;print(len(json.load(open('$AP/manifest_found.json'))))") cands)"
MANIFEST=$AP/manifest_found.json SWEEP_OUT=/home/ubuntu/sweep_found bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "QUEUE13 c) resume queue11 (EXDIM sweep)"
bash $AP/queue11_ap7.sh
stage "QUEUE13 DONE"
