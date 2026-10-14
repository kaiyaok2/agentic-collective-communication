#!/bin/bash
# queue16 (after queue15/queue11): (a) uniform_a2a seed-library candidates on the per-primitive harness (120 steps),
# (b) retry of the 224-rank collective-latency probe (collectives section only).
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE15 DONE" $Q; do sleep 20; done
stage "QUEUE16 a) uniform_a2a seed-library candidates (3) on per-primitive harness, 120 steps"
STEPS=120 MANIFEST=$AP/manifest_ua2a.json SWEEP_OUT=/home/ubuntu/sweep_ua2a bash $AP/sweep_candidates.sh >> $Q 2>&1
stage "QUEUE16 b) 224-rank collective latency probe retry (collectives only, SIZES<=512KB, AR-only 4/8/16MB)"
cp $AP/train_hwprobe_7node.py /home/ubuntu/OverlayCCL/training/
SECTIONS=collectives SIZES=4096,65536,524288 AR_ONLY_SIZES=4194304,8388608,16777216 PERCALL_PROBE=0 PROBE_ITERS=11 RESULTS_DIR=/home/ubuntu/hwprobe224 \
  timeout 1500 bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe224b >> /home/ubuntu/hwprobe.log 2>&1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_hwprob[e]'
source /home/ubuntu/venv.env; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_hwprob[e]'" 2>/dev/null; done
stage "QUEUE16 DONE"
