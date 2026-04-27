#!/bin/bash
# Run convergence training for all 3 non-AllToAllV collectives (baseline vs evolved) on 2 nodes.
#
# Each run trains DeepSeek-MoE-Lite comparing evolved vs baseline implementations.
#
# Usage:
#   ./training/run_all_collectives.sh                         # 2 nodes, 5000 steps
#   STEPS=1000 ./training/run_all_collectives.sh              # quick test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WORKER="${WORKER:-172.31.55.245}"
STEPS="${STEPS:-5000}"
WARMUP="${WARMUP:-5}"
NPROC="${NPROC_PER_NODE:-32}"
NEURON_VENV="${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
MASTER_ADDR="$(hostname -I | awk '{print $1}')"
NNODES=2

echo "=========================================="
echo "All-Collective Training Benchmark"
echo "=========================================="
echo "  Master:     $MASTER_ADDR"
echo "  Worker:     $WORKER"
echo "  Steps:      $STEPS"
echo "  Procs/node: $NPROC"
echo "  Venv:       $NEURON_VENV"
echo "=========================================="

# Sync code
echo "[0/6] Syncing code to worker..."
bash "$PROJECT_DIR/experiments/sync_nodes.sh" "$WORKER"

NEURON_ENV="export PATH=$NEURON_VENV/bin:\$PATH; \
export NEURON_RT_NUM_CORES=32; \
export NEURON_NUM_RECENT_MODELS_TO_KEEP=1; \
export FI_PROVIDER=efa; \
export FI_EFA_USE_DEVICE_RDMA=1; \
export NCCL_PROTO=simple; \
export XLA_TRANSFER_SEED_ASYNC=1; \
export NEURON_CC_FLAGS='--retry_failed_compilation'"

run_job() {
    local job_num="$1"
    local label="$2"
    local script="$3"
    local backend="$4"
    local port="$5"

    echo ""
    echo "=========================================="
    echo "[$job_num/6] $label ($backend)"
    echo "=========================================="

    local TORCHRUN_CMD="$NEURON_VENV/bin/torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$port \
        $script --backend $backend --steps $STEPS --warmup $WARMUP"

    # Launch worker
    ssh -o StrictHostKeyChecking=no "ubuntu@${WORKER}" \
        "cd $PROJECT_DIR && $NEURON_ENV && \
         export MASTER_ADDR=$MASTER_ADDR && \
         export MASTER_PORT=$port && \
         $TORCHRUN_CMD" \
        > "/tmp/worker_${label}_${backend}.log" 2>&1 &
    local worker_pid=$!

    # Launch master
    cd "$PROJECT_DIR"
    export PATH=$NEURON_VENV/bin:$PATH
    export NEURON_RT_NUM_CORES=32
    export NEURON_NUM_RECENT_MODELS_TO_KEEP=1
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$port
    export NEURON_CC_FLAGS='--retry_failed_compilation'
    export XLA_TRANSFER_SEED_ASYNC=1

    $TORCHRUN_CMD
    local master_exit=$?

    wait "$worker_pid" 2>/dev/null || true

    if [ $master_exit -ne 0 ]; then
        echo "  FAILED (exit $master_exit)"
        echo "  Worker log tail:"
        tail -20 "/tmp/worker_${label}_${backend}.log" 2>/dev/null | sed 's/^/    /'
    else
        echo "  DONE"
    fi

    return $master_exit
}

# ---- Uniform AllToAll ----
run_job 1 "ua2a" "training/train_uniform_a2a.py" "agent" 29501 || true
run_job 2 "ua2a" "training/train_uniform_a2a.py" "baseline" 29502 || true

# ---- Fused ReduceScatter ----
run_job 3 "frs" "training/train_fused_reducescatter.py" "agent" 29503 || true
run_job 4 "frs" "training/train_fused_reducescatter.py" "baseline" 29504 || true

# ---- Ring KV ----
run_job 5 "rkv" "training/train_ring_kv.py" "agent" 29505 || true
run_job 6 "rkv" "training/train_ring_kv.py" "baseline" 29506 || true

echo ""
echo "=========================================="
echo "All runs complete. Results in training/results/"
echo "=========================================="
ls -la "$PROJECT_DIR/training/results/"ua2a_*.json "$PROJECT_DIR/training/results/"frs_*.json "$PROJECT_DIR/training/results/"rkv_*.json 2>/dev/null
