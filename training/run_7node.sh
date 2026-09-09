#!/bin/bash
# 7-node DeepSeek-MoE-Lite benchmark on 7x trn1.32xlarge (224 ranks).
# Usage: bash run_7node.sh <train_script> <backend> <steps> <out_tag>
#   train_script: training/train.py, training/train_uniform_a2a.py, etc.
#   backend     : evolved|agrs|baseline|agent (depends on script)
#   steps       : integer (e.g. 5000)
#   out_tag     : results filename prefix written to results/7node/
set -euo pipefail

TRAIN_REL=${1:?train script path relative to repo}
BACKEND=${2:?backend}
STEPS=${3:-5000}
OUT_TAG=${4:-${BACKEND}}
WARMUP=${WARMUP:-5}

REPO=/home/ubuntu/agentic-collective-communication
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
NNODES=7
NPROC=32
PORT=29500

VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
RESULTS_DIR=$REPO/training/results/7node
LOG_DIR=$REPO/training/results/7node/logs
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

KEY=/home/ubuntu/.ssh/Kaiyao.pem
chmod 600 "$KEY" 2>/dev/null || true

COMMON_ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
export NEURON_RT_NUM_CORES=32 && \
export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && \
export FI_PROVIDER=efa && \
export FI_EFA_USE_DEVICE_RDMA=1 && \
export FI_EFA_FORK_SAFE=1 && \
export OFI_NCCL_USE_IPV6_TCP=0 && \
export MASTER_ADDR=$MASTER && \
export MASTER_PORT=$PORT && \
export RESULTS_DIR=$RESULTS_DIR"

TRUN_ARGS="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "============================================================"
echo "  7-node training: $TRAIN_REL  backend=$BACKEND  steps=$STEPS"
echo "  Master: $MASTER   Workers: ${WORKERS[*]}"
echo "  Results: $RESULTS_DIR  Logs: $LOG_DIR"
echo "============================================================"

# Rsync repo to workers (cheap if already in sync).
for ip in "${WORKERS[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='training/results' --exclude='/tmp/neuron_cache' \
        "$REPO/" ubuntu@$ip:$REPO/ &
done
wait

WORKER_PIDS=()
NODE_RANK=1
for ip in "${WORKERS[@]}"; do
    LOG="$LOG_DIR/${OUT_TAG}_node${NODE_RANK}_${ip}.log"
    echo "  -> launching node_rank=$NODE_RANK on $ip (log: $LOG)"
    ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip \
        "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN_ARGS --node_rank=$NODE_RANK $REPO/$TRAIN_REL --backend $BACKEND --steps $STEPS --warmup $WARMUP" \
        > "$LOG" 2>&1 &
    WORKER_PIDS+=($!)
    NODE_RANK=$((NODE_RANK+1))
done

# Master node
MASTER_LOG="$LOG_DIR/${OUT_TAG}_node0_master.log"
echo "  -> launching node_rank=0 on master (log: $MASTER_LOG)"
eval "$COMMON_ENV"
cd "$REPO"
$TORCHRUN $TRUN_ARGS --node_rank=0 "$REPO/$TRAIN_REL" \
    --backend "$BACKEND" --steps "$STEPS" --warmup "$WARMUP" 2>&1 | tee "$MASTER_LOG"

MASTER_RC=${PIPESTATUS[0]}
echo "  master exit code: $MASTER_RC"

# Wait for workers
fail=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then fail=$((fail+1)); fi
done

echo "============================================================"
echo "  Run complete. master_rc=$MASTER_RC  worker_failures=$fail"
echo "  Worker logs in $LOG_DIR"
echo "============================================================"

exit $MASTER_RC
