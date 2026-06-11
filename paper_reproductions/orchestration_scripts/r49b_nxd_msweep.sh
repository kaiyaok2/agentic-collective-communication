#!/bin/bash
# M-sweep on NXD-clean harness: baseline+agent at M={4,8,16}
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
OUTDIR=/home/ubuntu/nxd_msweep
mkdir -p $OUTDIR
HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $OUTDIR/HEARTBEAT; }

run_one() {
  local NAME=$1; local BACKEND=$2; local M=$3; local PORT=$4
  HB "$NAME start (backend=$BACKEND M=$M steps=300)"
  cd $REPO
  source $VENV/bin/activate
  timeout 1800 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=32 \
    --master_addr=127.0.0.1 --master_port=$PORT \
    experiments/model_extension/train_llama_nxd_clean.py \
    --backend $BACKEND --microbatches $M --steps 300 --warmup 3 --realtok --n-batches 350 \
    > $OUTDIR/$NAME.log 2>&1
  HB "$NAME exit=$?"
  sleep 5
}

HB "nxd-msweep start"
run_one M4_baseline  baseline 4  58620
run_one M4_agent     agent    4  58621
run_one M8_baseline  baseline 8  58622
run_one M8_agent     agent    8  58623
run_one M16_baseline baseline 16 58624
run_one M16_agent    agent    16 58625
HB "nxd-msweep done"
