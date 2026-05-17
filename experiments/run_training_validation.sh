#!/bin/bash
set -euo pipefail

WORKER=172.31.55.245
MASTER=$(hostname -I | awk '{print $1}')
VENV=/opt/aws_neuronx_venv_pytorch_2_9
PROJECT=/home/ubuntu/trainium-llm-search

for task in "$@"; do
  PORT=$((29600 + RANDOM % 100))
  echo "=== Running $task on port $PORT ==="

  cd "$PROJECT"
  source "$VENV/bin/activate"
  export NEURON_RT_NUM_CORES=32 FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
  export MASTER_ADDR=$MASTER MASTER_PORT=$PORT
  export NEURON_CC_FLAGS='--retry_failed_compilation'

  torchrun --nnodes=2 --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT} \
    experiments/validate_all_training.py --n-steps 15 --warmup 5 --tasks $task 2>&1 &
  MASTER_PID=$!

  ssh -o StrictHostKeyChecking=no ubuntu@${WORKER} \
  "cd $PROJECT && \
   source $VENV/bin/activate && \
   export NEURON_RT_NUM_CORES=32 FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 && \
   export MASTER_ADDR=$MASTER MASTER_PORT=$PORT && \
   export NEURON_CC_FLAGS='--retry_failed_compilation' && \
   torchrun --nnodes=2 --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT} \
     experiments/validate_all_training.py --n-steps 15 --warmup 5 --tasks $task" 2>&1 &
  WORKER_PID=$!

  wait $MASTER_PID
  EXIT_CODE=$?
  wait $WORKER_PID 2>/dev/null || true
  echo "$task exit: $EXIT_CODE"
  echo ""

  sleep 15
done
