#!/bin/bash
# N-node DeepSeek-MoE-Lite benchmark (set NNODES + WORKER_IPS to use a subset).
# Usage:
#   NNODES=2 WORKER_IPS="172.31.17.80" bash run_Nnode.sh training/train.py evolved 50 smoke2
# The master is always this host (rank 0).
set -euo pipefail

TRAIN_REL=${1:?train script path relative to repo}
BACKEND=${2:?backend}
STEPS=${3:-5000}
OUT_TAG=${4:-${BACKEND}}
WARMUP=${WARMUP:-5}

REPO=/home/ubuntu/agentic-collective-communication
MASTER=172.31.19.201
NNODES=${NNODES:?must set NNODES}
WORKER_IPS=${WORKER_IPS:?must set WORKER_IPS space-separated}
NPROC=${NPROC:-32}
PORT=${PORT:-29500}

VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
RESULTS_DIR=$REPO/training/results/${NNODES}node
LOG_DIR=$RESULTS_DIR/logs
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

KEY=/home/ubuntu/.ssh/Kaiyao.pem
chmod 600 "$KEY" 2>/dev/null || true

EXTRA_ENV=${EXTRA_ENV:-""}

COMMON_ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
export NEURON_RT_NUM_CORES=$NPROC && \
export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && \
export FI_PROVIDER=efa && \
export FI_EFA_USE_DEVICE_RDMA=1 && \
export FI_EFA_FORK_SAFE=1 && \
export MASTER_ADDR=$MASTER && \
export MASTER_PORT=$PORT && \
export RESULTS_DIR=$RESULTS_DIR${EXTRA_ENV:+ && $EXTRA_ENV}"

# Let torch_xla auto-set NEURON_RT_ROOT_COMM_ID to ${MASTER_ADDR}:62182.
MASTER_EXTRA_ENV=""

TRUN_ARGS="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "============================================================"
echo "  Training: $TRAIN_REL  backend=$BACKEND  steps=$STEPS  nnodes=$NNODES"
echo "  Master: $MASTER   Workers: $WORKER_IPS"
echo "  Results: $RESULTS_DIR  Logs: $LOG_DIR"
echo "============================================================"

WORKER_PIDS=()
NODE_RANK=1
for ip in $WORKER_IPS; do
    LOG="$LOG_DIR/${OUT_TAG}_node${NODE_RANK}_${ip}.log"
    echo "  -> launching node_rank=$NODE_RANK on $ip"
    ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip \
        "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN_ARGS --node_rank=$NODE_RANK $REPO/$TRAIN_REL --backend $BACKEND --steps $STEPS --warmup $WARMUP" \
        > "$LOG" 2>&1 &
    WORKER_PIDS+=($!)
    NODE_RANK=$((NODE_RANK+1))
done

MASTER_LOG="$LOG_DIR/${OUT_TAG}_node0_master.log"
echo "  -> launching node_rank=0 on master"
eval "$COMMON_ENV"
eval "$MASTER_EXTRA_ENV"
cd "$REPO"
$TORCHRUN $TRUN_ARGS --node_rank=0 "$REPO/$TRAIN_REL" \
    --backend "$BACKEND" --steps "$STEPS" --warmup "$WARMUP" 2>&1 | tee "$MASTER_LOG"

MASTER_RC=${PIPESTATUS[0]}
echo "  master exit code: $MASTER_RC"

fail=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then fail=$((fail+1)); fi
done

echo "============================================================"
echo "  Run complete. master_rc=$MASTER_RC  worker_failures=$fail"
echo "============================================================"
exit $MASTER_RC
