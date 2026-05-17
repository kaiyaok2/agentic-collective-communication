#!/bin/bash
set -euo pipefail

NEURON_VENV=/opt/aws_neuronx_venv_pytorch_2_9
MASTER_ADDR=$(hostname -I | awk '{print $1}')
WORKER=172.31.55.245
PROJECT_DIR=/home/ubuntu/trainium-llm-search
PORT=29510

export PATH=$NEURON_VENV/bin:$PATH
export NEURON_RT_NUM_CORES=32
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$PORT
export NEURON_NUM_RECENT_MODELS_TO_KEEP=1
export NEURON_CC_FLAGS='--retry_failed_compilation'
export XLA_TRANSFER_SEED_ASYNC=1

SCRIPT=${1:-training/train_ring_kv.py}
BACKEND=${2:-evolved}
STEPS=${3:-2}

NEURON_ENV="export PATH=$NEURON_VENV/bin:\$PATH; \
export NEURON_RT_NUM_CORES=32; \
export FI_PROVIDER=efa; \
export FI_EFA_USE_DEVICE_RDMA=1; \
export NCCL_PROTO=simple; \
export XLA_TRANSFER_SEED_ASYNC=1; \
export NEURON_NUM_RECENT_MODELS_TO_KEEP=1; \
export NEURON_CC_FLAGS='--retry_failed_compilation'"

TORCHRUN_CMD="$NEURON_VENV/bin/torchrun \
    --nnodes=2 \
    --nproc_per_node=32 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$PORT \
    $SCRIPT --backend $BACKEND --steps $STEPS --warmup 1"

echo "=== Smoke test: $SCRIPT $BACKEND $STEPS steps ==="
echo "Master: $MASTER_ADDR  Worker: $WORKER  Port: $PORT"

# Launch worker
ssh -o StrictHostKeyChecking=no "ubuntu@${WORKER}" \
    "cd $PROJECT_DIR && $NEURON_ENV && \
     export MASTER_ADDR=$MASTER_ADDR && \
     export MASTER_PORT=$PORT && \
     $TORCHRUN_CMD" \
    > /tmp/smoke_worker.log 2>&1 &
WPID=$!

# Launch master
cd "$PROJECT_DIR"
$TORCHRUN_CMD
MEXIT=$?

wait $WPID 2>/dev/null || true

echo "Master exit: $MEXIT"
if [ $MEXIT -ne 0 ]; then
    echo "Worker log tail:"
    tail -20 /tmp/smoke_worker.log 2>/dev/null
fi
