#!/bin/bash
# Full pipeline: search (normal + Sorcar) + training (agent vs baseline) for all 4 problems.
# Each search generates runtime/trainium_*.py, then training uses those dynamically.
#
# Usage:
#   bash experiments/run_full_pipeline.sh                    # full run
#   TRAIN_STEPS=200 bash experiments/run_full_pipeline.sh    # quick test
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"

TRAIN_STEPS="${TRAIN_STEPS:-5000}"
MASTER="172.31.48.122"
WORKER="172.31.55.245"
VENV="/opt/aws_neuronx_venv_pytorch_2_9"
NPROC=32

PROBLEMS=("alltoallv" "uniform_a2a" "fused_reducescatter" "ring_kv")
BASELINES=("agrs" "baseline" "baseline" "baseline")

echo "=========================================="
echo "Full Pipeline: Search + Training"
echo "  Problems: ${PROBLEMS[*]}"
echo "  Train steps: $TRAIN_STEPS"
echo "  Time: $(date)"
echo "=========================================="

# Sync code to worker
echo "[sync] Syncing to worker..."
bash experiments/sync_nodes.sh "$WORKER"

ENV_SETUP="export PATH=$VENV/bin:\$PATH; \
export NEURON_RT_NUM_CORES=32; \
export NEURON_NUM_RECENT_MODELS_TO_KEEP=1; \
export FI_PROVIDER=efa; \
export FI_EFA_USE_DEVICE_RDMA=1; \
export XLA_TRANSFER_SEED_ASYNC=1; \
export NEURON_CC_FLAGS='--retry_failed_compilation'"

# ---------- Phase A: Search (normal + Sorcar) ----------
# Normal search first
for prob in "${PROBLEMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  SEARCH: $prob (normal)"
    echo "=========================================="
    CMD="python3 -u experiments/run_search.py \
        --problem $prob --pattern moe --llm-model opus \
        --output-dir experiments/results \
        --num-nodes 2 --master-addr $MASTER --worker-addrs $WORKER --hw-eval"

    LOG="/tmp/bench_${prob}_normal.log"
    if $CMD > "$LOG" 2>&1; then
        echo "  OK ($(tail -1 "$LOG"))"
    else
        echo "  FAIL (exit $?). Log: $LOG"
    fi
done

# Back up normal search runtime modules before sorcar overwrites them
cp -r runtime/ runtime_normal_backup/

# Sorcar search
for prob in "${PROBLEMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  SEARCH: $prob (sorcar)"
    echo "=========================================="
    CMD="python3 -u experiments/run_search.py \
        --problem $prob --pattern moe --llm-model opus \
        --output-dir experiments/results \
        --num-nodes 2 --master-addr $MASTER --worker-addrs $WORKER --hw-eval --kiss-sorcar"

    LOG="/tmp/bench_${prob}_sorcar.log"
    if $CMD > "$LOG" 2>&1; then
        echo "  OK ($(tail -1 "$LOG"))"
    else
        echo "  FAIL (exit $?). Log: $LOG"
    fi
done

# Restore normal search runtime modules for training
cp runtime_normal_backup/trainium_*.py runtime/

# Re-sync after search generated new runtime/ modules
bash experiments/sync_nodes.sh "$WORKER"

# ---------- Phase B: Training (agent vs baseline) ----------
run_train() {
    local prob="$1"
    local backend="$2"
    local port="$3"
    local script="$4"

    echo ""
    echo "=========================================="
    echo "  TRAIN: $prob ($backend) steps=$TRAIN_STEPS"
    echo "=========================================="

    TORCHRUN="$VENV/bin/torchrun \
        --nnodes=2 --nproc_per_node=$NPROC \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$port \
        $script --backend $backend --steps $TRAIN_STEPS --warmup 5"

    # Worker
    ssh -o StrictHostKeyChecking=no "ubuntu@$WORKER" \
        "cd $PROJECT && $ENV_SETUP && \
         export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && \
         $TORCHRUN" \
        > "/tmp/train_${prob}_${backend}_worker.log" 2>&1 &
    local wpid=$!

    # Master
    export PATH="$VENV/bin:$PATH"
    export NEURON_RT_NUM_CORES=32
    export NEURON_NUM_RECENT_MODELS_TO_KEEP=1
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export XLA_TRANSFER_SEED_ASYNC=1
    export NEURON_CC_FLAGS='--retry_failed_compilation'
    export MASTER_ADDR="$MASTER"
    export MASTER_PORT="$port"

    local LOG="/tmp/train_${prob}_${backend}.log"
    if $TORCHRUN > "$LOG" 2>&1; then
        echo "  OK"
        tail -5 "$LOG" | grep -i -E "avg|throughput|wall|final" | sed 's/^/  /'
    else
        echo "  FAIL (exit $?)"
        tail -5 "$LOG" | sed 's/^/  /'
    fi
    wait "$wpid" 2>/dev/null || true
}

SCRIPTS=("training/train.py" "training/train_uniform_a2a.py" \
         "training/train_fused_reducescatter.py" "training/train_ring_kv.py")

port=29600
for i in "${!PROBLEMS[@]}"; do
    prob="${PROBLEMS[$i]}"
    base="${BASELINES[$i]}"
    script="${SCRIPTS[$i]}"

    run_train "$prob" "agent" "$port" "$script"
    port=$((port + 1))
    run_train "$prob" "$base" "$port" "$script"
    port=$((port + 1))
done

echo ""
echo "=========================================="
echo "Full pipeline complete. $(date)"
echo "=========================================="
