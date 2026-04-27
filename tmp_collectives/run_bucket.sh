#!/bin/bash
set -euo pipefail

VENV=/opt/aws_neuronx_venv_pytorch_2_9
MASTER_ADDR=172.31.48.122
WORKER=172.31.55.245
TRUN_ARGS="--nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d"
DIR=/home/ubuntu/trainium-llm-search/tmp_collectives
PORT=29530

export PATH=$VENV/bin:$PATH

scp -o StrictHostKeyChecking=no "$DIR/bench_grad_bucket.py" ubuntu@$WORKER:/tmp/bench_bucket3.py 2>/dev/null

ssh -o StrictHostKeyChecking=no ubuntu@$WORKER \
    "export PATH=$VENV/bin:\$PATH && \
     export NEURON_RT_NUM_CORES=32 && \
     export FI_PROVIDER=efa && \
     export FI_EFA_USE_DEVICE_RDMA=1 && \
     $VENV/bin/torchrun $TRUN_ARGS --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
         /tmp/bench_bucket3.py" &
WPID=$!

export NEURON_RT_NUM_CORES=32
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

$VENV/bin/torchrun $TRUN_ARGS --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
    "$DIR/bench_grad_bucket.py" 2>&1 | grep -v DeprecationWarning

wait $WPID 2>/dev/null || true
echo "Gradient bucketing benchmark complete."
