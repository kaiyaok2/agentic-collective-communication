#!/bin/bash
# queue15: when the a2av sweep ends (queue14 reaches stage b), take over: 7-node collective-latency probe (fits the
# 224-rank curves for the simulator), then run queue11 (EXDIM OLMoE) ourselves.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE14 b)" $Q; do sleep 10; done
pkill -f 'queue14_ap[7]'; pkill -f 'queue11_ap[7]'; pkill -f 'run_ap7.s[h]'; sleep 1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'
source /home/ubuntu/venv.env
for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'" 2>/dev/null; done
stage "QUEUE15 a) 7-node (224-rank) collective latency probe: SECTIONS=collectives,back_to_back SIZES<=512KB (+AR-only 4/8/16MB)"
cp $AP/train_hwprobe_7node.py /home/ubuntu/OverlayCCL/training/
SECTIONS=trivial,graph_launch,back_to_back,collectives SIZES=4096,32768,65536,524288 AR_ONLY_SIZES=4194304,8388608,16777216 PERCALL_PROBE=0 PROBE_ITERS=11 RESULTS_DIR=/home/ubuntu/hwprobe224 \
  bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe224 >> /home/ubuntu/hwprobe.log 2>&1
stage "QUEUE15 b) queue11 (EXDIM OLMoE)"
bash $AP/queue11_ap7.sh
stage "QUEUE15 DONE"
