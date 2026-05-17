#!/bin/bash
# 7-node OLMoE-10B comparison launcher.
# Usage: bash training/run_olmoe_compare.sh <backend> [STEPS]
#   backend: baseline | agent  (AllToAllV; defaults sweep individual collectives)
#   STEPS  : default 1000
# Env knobs:
#   GRAD_SYNC=baseline|agent   (default baseline)
#   CE=baseline|agent          (default baseline)
#   WARMUP=N                   (default 5)
set -uo pipefail

BACKEND=${1:?backend (baseline|agent)}
STEPS=${2:-1000}
GRAD_SYNC=${GRAD_SYNC:-baseline}
CE=${CE:-baseline}
WARMUP=${WARMUP:-5}
NPROC=32
NNODES=7
PORT=${PORT:-29520}

REPO=/home/ubuntu/agentic-collective-communication
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
chmod 600 "$KEY" 2>/dev/null || true

RESULTS_DIR=$REPO/training/results/olmoe_7node
TAG="a2av-${BACKEND}_gs-${GRAD_SYNC}_ce-${CE}"
LOG_DIR=$RESULTS_DIR/logs/${TAG}
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Sync repo to all workers (rsync is incremental, fast if already in sync)
for ip in "${WORKERS[@]}"; do
  rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    --exclude=.git --exclude=__pycache__ --exclude="*.pyc" \
    --exclude=training/results --exclude=/tmp/neuron_cache \
    "$REPO/" "ubuntu@$ip:$REPO/" &
done
wait

COMMON_ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
export NEURON_RT_NUM_CORES=32 && \
export NEURON_NUM_RECENT_MODELS_TO_KEEP=1 && \
export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && \
export FI_PROVIDER=efa && \
export FI_EFA_USE_DEVICE_RDMA=1 && \
export FI_EFA_FORK_SAFE=1 && \
export MASTER_ADDR=$MASTER && \
export MASTER_PORT=$PORT && \
export RESULTS_DIR=$RESULTS_DIR"

TRUN_ARGS="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "============================================================"
echo "  OLMoE 7-node training  $TAG  steps=$STEPS  warmup=$WARMUP"
echo "  Started: $(date -u)"
echo "============================================================"

WORKER_PIDS=()
NODE_RANK=1
for ip in "${WORKERS[@]}"; do
  WLOG="$LOG_DIR/node${NODE_RANK}_${ip}.log"
  ssh -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$ip" \
    "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN_ARGS --node_rank=$NODE_RANK $REPO/training/train_olmoe10b.py --backend $BACKEND --grad-sync $GRAD_SYNC --ce $CE --steps $STEPS --warmup $WARMUP" \
    > "$WLOG" 2>&1 &
  WORKER_PIDS+=($!)
  NODE_RANK=$((NODE_RANK+1))
done

MLOG="$LOG_DIR/node0_master.log"
eval "$COMMON_ENV"
cd "$REPO"
$TORCHRUN $TRUN_ARGS --node_rank=0 "$REPO/training/train_olmoe10b.py" \
  --backend "$BACKEND" --grad-sync "$GRAD_SYNC" --ce "$CE" --steps "$STEPS" --warmup "$WARMUP" > "$MLOG" 2>&1
MRC=$?

# Pull the JSON from whichever node produced it (rank-0 may live on a worker)
JSON_NAME="olmoe10b_${TAG}.json"
JSON="$RESULTS_DIR/${JSON_NAME}"
PHASE_NAME="olmoe10b_${TAG}_phase.json"
PHASE="$RESULTS_DIR/${PHASE_NAME}"
# Always pull the freshest copies from whichever worker has them (rank-0 may live on a worker).
# We compare mtime: the worker with the newest file wins.
PULL_FROM=""
NEWEST=0
for ip in "${WORKERS[@]}"; do
  T=$(ssh -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$ip" "stat -c %Y $RESULTS_DIR/${JSON_NAME} 2>/dev/null" 2>/dev/null)
  if [ -n "$T" ] && [ "$T" -gt "$NEWEST" ]; then NEWEST=$T; PULL_FROM=$ip; fi
done
if [ -n "$PULL_FROM" ]; then
  scp -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$PULL_FROM:$RESULTS_DIR/${JSON_NAME}" "$JSON" 2>/dev/null
  scp -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$PULL_FROM:$RESULTS_DIR/${PHASE_NAME}" "$PHASE" 2>/dev/null
fi

fail=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then fail=$((fail+1)); fi
done

echo "============================================================"
echo "  $TAG complete  master_rc=$MRC  worker_failures=$fail  $(date -u)"
echo "  json=${JSON} ($([ -f "$JSON" ] && echo "OK" || echo "MISSING"))"
echo "============================================================"

[ -f "$JSON" ] && exit 0 || exit 1
