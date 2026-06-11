#!/bin/bash
# r124: single-node Llama-7B per-microbatch sweep
# Sweep M in {1,2,4,8} for both baseline (per_mb) and agent (bundled).
# Single trn1.32xlarge, TP=32 DP=1, no PP cross-node.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
R124=/home/ubuntu/r124_single_node
mkdir -p $R124
HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R124/HEARTBEAT; }

# Kill any lingering runs from previous attempts on this box.
sudo pkill -9 -f train_llama 2>/dev/null || true
sudo pkill -9 -f torchrun 2>/dev/null || true
sudo pkill -9 -f neuronx-cc 2>/dev/null || true
find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null || true
sleep 5

run_one() {
  local NAME=$1; local BACKEND=$2; local M=$3; local STEPS=$4; local PORT=$5
  HB "$NAME start (backend=$BACKEND M=$M steps=$STEPS)"
  local OUT=$R124/$NAME
  mkdir -p $OUT
  cd $REPO
  source $VENV/bin/activate
  timeout 10800 $TORCHRUN --nnodes=1 --node_rank=0 --nproc_per_node=32 \
    --master_addr=127.0.0.1 --master_port=$PORT \
    experiments/model_extension/train_llama_nxd_mb.py \
    --backend $BACKEND --microbatches $M --steps $STEPS --warmup 3 \
    > $OUT/node_0.log 2>&1
  local EX=$?
  HB "$NAME exit=$EX"
  sleep 5
}

HB "r124 start (single-node Llama-7B M-sweep)"
run_one M1_baseline baseline 1 200 58100
run_one M2_baseline baseline 2 200 58101
run_one M4_baseline baseline 4 200 58102
run_one M8_baseline baseline 8 100 58103
run_one M1_agent    agent    1 200 58110
run_one M2_agent    agent    2 200 58111
run_one M4_agent    agent    4 200 58112
run_one M8_agent    agent    8 100 58113
HB "r124 done"
