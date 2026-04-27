#!/bin/bash
set -euo pipefail

VENV=/opt/aws_neuronx_venv_pytorch_2_9
MASTER_ADDR=172.31.48.122
WORKER=172.31.55.245
TRUN_ARGS="--nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d"
DIR=/home/ubuntu/trainium-llm-search/tmp_collectives

export PATH=$VENV/bin:$PATH

run_bench() {
    local script=$1
    local name=$2
    local port=$3
    local cache_name=$4

    echo ""
    echo "======================================================"
    echo "  Running: $name"
    echo "======================================================"

    # Clear caches
    rm -rf /tmp/neuron_cache_${cache_name} /tmp/ubuntu/neuroncc_compile_workdir 2>/dev/null
    ssh -o StrictHostKeyChecking=no ubuntu@$WORKER \
        "rm -rf /tmp/neuron_cache_${cache_name} /tmp/ubuntu/neuroncc_compile_workdir" 2>/dev/null

    # Copy script to worker
    scp -o StrictHostKeyChecking=no "$DIR/$script" ubuntu@$WORKER:/tmp/bench_${cache_name}.py 2>/dev/null

    # Launch worker
    ssh -o StrictHostKeyChecking=no ubuntu@$WORKER \
        "export PATH=$VENV/bin:\$PATH && \
         export NEURON_RT_NUM_CORES=32 && \
         export FI_PROVIDER=efa && \
         export FI_EFA_USE_DEVICE_RDMA=1 && \
         $VENV/bin/torchrun $TRUN_ARGS --rdzv_endpoint=${MASTER_ADDR}:${port} \
             /tmp/bench_${cache_name}.py" &
    local WPID=$!

    # Launch master
    export NEURON_RT_NUM_CORES=32
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1

    $VENV/bin/torchrun $TRUN_ARGS --rdzv_endpoint=${MASTER_ADDR}:${port} \
        "$DIR/$script" 2>&1 | grep -v DeprecationWarning

    wait $WPID 2>/dev/null || true
    echo "  $name complete."
}

echo "Starting all 3 collective benchmarks on 2x trn1.32xlarge (64 ranks)"

run_bench bench_overlap.py           "#1 Overlapped Comm/Compute"   29522 overlap
run_bench bench_ring_attn.py         "#2 Ring Attention"            29523 ring
run_bench bench_grad_bucket.py       "#3 Gradient Bucketing"        29524 bucket

echo ""
echo "======================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "======================================================"
