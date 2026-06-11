#!/bin/bash
# Llama-block NXD-clean descent: baseline (per_mb) vs agent (bundled), M=2, 300 steps each.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
OUTDIR=/home/ubuntu/nxd_clean_descent
mkdir -p $OUTDIR
HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $OUTDIR/HEARTBEAT; }

sudo pkill -9 -f torchrun 2>/dev/null || true
sleep 3

run_one() {
  local NAME=$1; local BACKEND=$2; local PORT=$3
  HB "$NAME start (backend=$BACKEND M=2 steps=300)"
  cd $REPO
  source $VENV/bin/activate
  timeout 1800 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=32 \
    --master_addr=127.0.0.1 --master_port=$PORT \
    experiments/model_extension/train_llama_nxd_clean.py \
    --backend $BACKEND --microbatches 2 --steps 300 --warmup 3 --realtok --n-batches 150 \
    > $OUTDIR/$NAME.log 2>&1
  HB "$NAME exit=$?"
  sleep 5
}

HB "nxd-clean full descent start"
run_one baseline_M2 baseline 58610
run_one agent_M2    agent    58611
HB "nxd-clean done"
