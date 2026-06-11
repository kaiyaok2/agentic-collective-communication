#!/bin/bash
# Run DeepSeek-MoE-Lite training benchmark on 2x trn1.32xlarge.
# Usage: bash training/run_benchmark.sh [steps]
set -euo pipefail

STEPS=${1:-100}
WARMUP=5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$SCRIPT_DIR/train.py"
MASTER_ADDR="172.31.48.122"
WORKER="172.31.55.245"
MASTER_PORT=29500

VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN="$VENV/bin/torchrun"
RESULTS_DIR="$SCRIPT_DIR/results"

ENV="export PATH=$VENV/bin:\$PATH && \
export NEURON_RT_NUM_CORES=32 && \
export FI_PROVIDER=efa && \
export FI_EFA_USE_DEVICE_RDMA=1 && \
export MASTER_ADDR=$MASTER_ADDR && \
export MASTER_PORT=$MASTER_PORT && \
export RESULTS_DIR=$RESULTS_DIR"

TRUN_ARGS="--nproc_per_node=32 --nnodes=2 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

# Ensure worker has the script
scp -o StrictHostKeyChecking=no "$TRAIN" ubuntu@$WORKER:/tmp/train_moe.py

run_backend() {
    local backend=$1
    echo "======================================================"
    echo "  Training with backend: $backend  (steps=$STEPS)"
    echo "======================================================"

    # Launch worker
    ssh -o StrictHostKeyChecking=no ubuntu@$WORKER \
        "$ENV && cd /tmp && $TORCHRUN $TRUN_ARGS /tmp/train_moe.py --backend $backend --steps $STEPS --warmup $WARMUP" &
    WORKER_PID=$!

    # Launch master
    export PATH=$VENV/bin:$PATH
    export NEURON_RT_NUM_CORES=32
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export RESULTS_DIR=$RESULTS_DIR

    $TORCHRUN $TRUN_ARGS "$TRAIN" --backend "$backend" --steps "$STEPS" --warmup "$WARMUP"

    wait $WORKER_PID 2>/dev/null || true
    echo ""
    echo "Backend $backend complete."
    echo ""
}

mkdir -p "$RESULTS_DIR"

echo "Starting AllToAllV training benchmark"
echo "  Master: $MASTER_ADDR"
echo "  Worker: $WORKER"
echo "  Steps:  $STEPS"
echo ""

# Run agent-generated (from runtime/), then AG+RS baseline
run_backend agent
run_backend agrs

echo "======================================================"
echo "  ALL DONE — results in $RESULTS_DIR/"
echo "======================================================"

# Print comparison
if [ -f "$RESULTS_DIR/evolved.json" ] && [ -f "$RESULTS_DIR/agrs.json" ]; then
    python3 -c "
import json
e = json.load(open('$RESULTS_DIR/evolved.json'))
a = json.load(open('$RESULTS_DIR/agrs.json'))
print()
print(f'  Agent-Evolved: {e[\"avg_ms\"]:.1f} ms/step  (total {e[\"total_s\"]:.1f}s)')
print(f'  AG+RS Baseline: {a[\"avg_ms\"]:.1f} ms/step  (total {a[\"total_s\"]:.1f}s)')
sp = (a['avg_ms'] - e['avg_ms']) / a['avg_ms'] * 100
print(f'  Evolved is {sp:+.1f}% vs AG+RS')
"
fi
