#!/bin/bash
# Launch a multi-node torchrun job across N trn1.32xlarge nodes.
#
# This script:
#   1. Syncs the project to all worker nodes
#   2. Launches torchrun on each worker node via SSH (background)
#   3. Launches torchrun on this (master) node (foreground)
#   4. Collects output from all nodes
#
# Usage:
#   ./experiments/multinode_launch.sh --workers 172.31.55.245 -- \
#       python experiments/real_alltoallv_bench.py --algo agent --sizes 1024 --mode worker
#
#   ./experiments/multinode_launch.sh --workers 172.31.55.245 -- \
#       python experiments/run_search.py --pattern moe --num-nodes 2
#
# Environment:
#   NPROC_PER_NODE: processes per node (default: 32)
#   NEURON_VENV:    path to neuron venv (default: /opt/aws_neuronx_venv_pytorch_2_9)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NPROC_PER_NODE="${NPROC_PER_NODE:-32}"
NEURON_VENV="${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Parse --workers
WORKERS=()
SCRIPT_ARGS=()
parsing_workers=false
past_separator=false
for arg in "$@"; do
    if [ "$past_separator" = true ]; then
        SCRIPT_ARGS+=("$arg")
    elif [ "$arg" = "--" ]; then
        past_separator=true
    elif [ "$arg" = "--workers" ]; then
        parsing_workers=true
    elif [ "$parsing_workers" = true ]; then
        if [[ "$arg" == --* ]]; then
            parsing_workers=false
        else
            WORKERS+=("$arg")
            continue
        fi
    fi
done

if [ ${#WORKERS[@]} -eq 0 ]; then
    echo "Usage: $0 --workers <ip1> [ip2 ...] -- <command>"
    echo "Example: $0 --workers 172.31.55.245 -- python experiments/run_search.py --pattern moe --num-nodes 2"
    exit 1
fi

NUM_NODES=$(( ${#WORKERS[@]} + 1 ))
MASTER_ADDR="$(hostname -I | awk '{print $1}')"

echo "=========================================="
echo "Multi-Node Launch"
echo "=========================================="
echo "  Master:     $MASTER_ADDR (this node)"
echo "  Workers:    ${WORKERS[*]}"
echo "  Num nodes:  $NUM_NODES"
echo "  Procs/node: $NPROC_PER_NODE"
echo "  Venv:       $NEURON_VENV"
echo "  Command:    ${SCRIPT_ARGS[*]}"
echo "=========================================="

# Sync code to workers
echo "[1/3] Syncing code to workers..."
bash "$SCRIPT_DIR/sync_nodes.sh" "${WORKERS[@]}"

# FI_EFA_USE_DEVICE_RDMA and other EFA env vars for optimal performance
NEURON_ENV="export PATH=$NEURON_VENV/bin:\$PATH; \
export NEURON_RT_NUM_CORES=32; \
export FI_PROVIDER=efa; \
export FI_EFA_USE_DEVICE_RDMA=1; \
export NCCL_PROTO=simple; \
export XLA_TRANSFER_SEED_ASYNC=1; \
export NEURON_CC_FLAGS='--retry_failed_compilation'"

# Build torchrun command
TORCHRUN_CMD="$NEURON_VENV/bin/torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ${SCRIPT_ARGS[*]}"

# Launch on worker nodes via SSH
echo "[2/3] Launching workers..."
WORKER_PIDS=()
WORKER_LOGS=()
for i in "${!WORKERS[@]}"; do
    worker="${WORKERS[$i]}"
    node_rank=$(( i + 1 ))
    log_file="/tmp/multinode_worker_${node_rank}.log"
    WORKER_LOGS+=("$log_file")

    echo "  Starting worker $node_rank on $worker..."
    ssh -o StrictHostKeyChecking=no "ubuntu@${worker}" \
        "cd $PROJECT_DIR && $NEURON_ENV && \
         export MASTER_ADDR=$MASTER_ADDR && \
         export MASTER_PORT=$MASTER_PORT && \
         $TORCHRUN_CMD" \
        > "$log_file" 2>&1 &
    WORKER_PIDS+=($!)
done

# Launch on master node (foreground)
echo "[3/3] Launching master (this node)..."
cd "$PROJECT_DIR"
export PATH=$NEURON_VENV/bin:$PATH
export NEURON_RT_NUM_CORES=32
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

$TORCHRUN_CMD
MASTER_EXIT=$?

# Wait for workers and collect status
echo ""
echo "Master exited with code $MASTER_EXIT"
for i in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$i]}"
    log="${WORKER_LOGS[$i]}"
    node_rank=$(( i + 1 ))
    wait "$pid" 2>/dev/null || true
    worker_exit=$?
    echo "Worker $node_rank exited with code $worker_exit"
    if [ "$worker_exit" -ne 0 ] && [ -f "$log" ]; then
        echo "  Last 20 lines of worker $node_rank log:"
        tail -20 "$log" | sed 's/^/    /'
    fi
done

exit $MASTER_EXIT
