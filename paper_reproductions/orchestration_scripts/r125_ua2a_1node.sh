#!/bin/bash
# r125: 1-node ua2a per-call probe at chunk=16384.
# Waits for r124 (train_llama_nxd_mb) to finish to avoid sharing a Neuron device.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
R125=/home/ubuntu/r125_ua2a_1node
mkdir -p $R125
HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R125/HEARTBEAT; }

# Wait for r124 to be quiet on this box.
while pgrep -f train_llama_nxd_mb >/dev/null 2>&1; do
  HB "waiting for r124 (train_llama_nxd_mb) to finish before probe"
  sleep 60
done

HB "r125 start"
cd $REPO
source $VENV/bin/activate

run_probe() {
  local NAME=$1; local BACKEND=$2; local CHUNK=$3; local PORT=$4
  HB "$NAME start (backend=$BACKEND chunk=$CHUNK)"
  local OUT=$R125/$NAME
  mkdir -p $OUT
  timeout 900 $TORCHRUN --nnodes=1 --node_rank=0 --nproc_per_node=32 \
    --master_addr=127.0.0.1 --master_port=$PORT \
    training/train_ua2a_sweep_7node.py \
    --backend $BACKEND --chunk $CHUNK --warmup 10 --iters 30 \
    --tag "${NAME}" \
    > $OUT/node_0.log 2>&1
  HB "$NAME exit=$?"
  sleep 5
}

# chunk=16384 is the size called for table 3/12 corrected 1-node ua2a row.
run_probe baseline_c16384 baseline 16384 58200
run_probe agent_c16384    agent    16384 58201

HB "r125 done"
