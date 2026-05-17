#!/bin/bash
# 7-node distributed-CE training launcher.
# Usage: bash training/run_dxe_compare.sh <backend> [STEPS]
#   backend: baseline | agent
set -uo pipefail

BACKEND=${1:?backend (baseline|agent)}
STEPS=${2:-100}
WARMUP=${WARMUP:-5}
NPROC=32
NNODES=7
PORT=${PORT:-29620}

REPO=/home/ubuntu/agentic-collective-communication
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
chmod 600 "$KEY" 2>/dev/null || true

RESULTS_DIR=$REPO/training/results/dxe_7node
LOG_DIR=$RESULTS_DIR/logs/${BACKEND}
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

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

echo "================  DXE 7-node $BACKEND  STEPS=$STEPS  $(date -u)  ================"
WORKER_PIDS=()
NODE_RANK=1
for ip in "${WORKERS[@]}"; do
  WLOG="$LOG_DIR/node${NODE_RANK}_${ip}.log"
  ssh -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$ip" \
    "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN_ARGS --node_rank=$NODE_RANK $REPO/training/train_dxe.py --backend $BACKEND --steps $STEPS --warmup $WARMUP" \
    > "$WLOG" 2>&1 &
  WORKER_PIDS+=($!)
  NODE_RANK=$((NODE_RANK+1))
done

MLOG="$LOG_DIR/node0_master.log"
eval "$COMMON_ENV"
cd "$REPO"
$TORCHRUN $TRUN_ARGS --node_rank=0 "$REPO/training/train_dxe.py" --backend "$BACKEND" --steps "$STEPS" --warmup "$WARMUP" > "$MLOG" 2>&1
MRC=$?

JSON="$RESULTS_DIR/dxe_${BACKEND}.json"
if [ ! -f "$JSON" ]; then
  for ip in "${WORKERS[@]}"; do
    REMOTE=$(ssh -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$ip" "ls $JSON 2>/dev/null | head -1" 2>/dev/null)
    if [ -n "$REMOTE" ]; then
      scp -q -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$ip:$REMOTE" "$JSON" 2>/dev/null
      break
    fi
  done
fi
fail=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then fail=$((fail+1)); fi
done
echo "================  DXE $BACKEND master_rc=$MRC failures=$fail json=$([ -f "$JSON" ] && echo OK || echo MISSING)  ================"
[ -f "$JSON" ] && exit 0 || exit 1
