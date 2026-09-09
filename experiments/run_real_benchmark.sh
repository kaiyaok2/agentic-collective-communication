#!/bin/bash
# Run real AllToAllV benchmark on trn1.32xlarge (single or multi-node).
#
# Compares the search-optimized hierarchical AllToAllV against topology-unaware
# baselines (default ring, allgather). Each algorithm runs in its own torchrun
# process to stay within Neuron's communication group limit.
#
# Single-node usage:
#   ./experiments/run_real_benchmark.sh                              # defaults
#   ./experiments/run_real_benchmark.sh --sizes 1024,4096 --iters 50 # custom
#
# Multi-node usage (set env vars before running):
#   NUM_NODES=2 MASTER_ADDR=172.31.48.122 WORKER_ADDRS=172.31.55.245 \
#       ./experiments/run_real_benchmark.sh
#
# Benchmark scope (set SCOPE env var):
#   SCOPE=intra ./experiments/run_real_benchmark.sh    # intra-node only (32 ranks)
#   SCOPE=full  ./experiments/run_real_benchmark.sh    # full cross-node (N*32 ranks)
#   SCOPE=both  ./experiments/run_real_benchmark.sh    # both + cross-node penalty table
#
# Or run a single algorithm directly:
#   torchrun --nproc_per_node=32 experiments/real_alltoallv_bench.py --algo hierarchical
#
# Requirements: trn1.32xlarge with torch-xla and torch-neuronx installed.
# For multi-node: all nodes must have the same code and network connectivity.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate Neuron venv
NEURON_VENV="${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_9}"
if [ -f "$NEURON_VENV/bin/activate" ]; then
    source "$NEURON_VENV/bin/activate"
fi

# Configuration (override via environment variables)
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
WORKER_ADDRS="${WORKER_ADDRS:-}"
SCOPE="${SCOPE:-both}"

EXTRA_ARGS="--scope $SCOPE"
if [ "$NUM_NODES" -gt 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --num-nodes $NUM_NODES --master-addr $MASTER_ADDR"
    if [ -n "$WORKER_ADDRS" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --worker-addrs $WORKER_ADDRS"
        echo "Syncing code to workers..."
        IFS=',' read -ra ADDRS <<< "$WORKER_ADDRS"
        bash "$SCRIPT_DIR/sync_nodes.sh" "${ADDRS[@]}"
    fi
    echo "Multi-node mode: $NUM_NODES nodes, master=$MASTER_ADDR, workers=$WORKER_ADDRS, scope=$SCOPE"
fi

# Set EFA environment
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NEURON_RT_NUM_CORES=32

python experiments/real_alltoallv_bench.py --algo all \
    --output experiments/results/real_benchmark_results.json \
    $EXTRA_ARGS \
    "$@"
