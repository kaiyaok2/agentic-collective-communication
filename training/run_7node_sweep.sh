#!/bin/bash
# 7-node sweep: run all 5 collective training scripts with both backends.
# Output: training/results/7node/<task>/<backend>_converge.json
#
# Usage:  bash training/run_7node_sweep.sh [STEPS]
# Default STEPS = 5000 (matches single-node README table).

set -uo pipefail

STEPS="${1:-5000}"
WARMUP="${WARMUP:-5}"
NPROC=32
PORT_BASE=29500

REPO=/home/ubuntu/agentic-collective-communication
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
NNODES=7

VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
chmod 600 "$KEY" 2>/dev/null || true

RESULTS_ROOT=$REPO/training/results/7node
LOG_ROOT=$RESULTS_ROOT/logs
SUMMARY=$RESULTS_ROOT/SWEEP_SUMMARY.txt
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

# (label, train_script_rel, backend)
JOBS=(
  "a2av:training/train.py:evolved"
  "a2av:training/train.py:agrs"
  "ua2a:training/train_uniform_a2a.py:evolved"
  "ua2a:training/train_uniform_a2a.py:baseline"
  "rkv:training/train_ring_kv.py:evolved"
  "rkv:training/train_ring_kv.py:baseline"
)

echo "============================================================" | tee -a "$SUMMARY"
echo "  7-NODE SWEEP  steps=$STEPS  warmup=$WARMUP  started $(date -u)" | tee -a "$SUMMARY"
echo "  Master: $MASTER   Workers: ${WORKERS[*]}" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# Sync repo to all workers once at start.
echo "[sync] rsync to all workers..." | tee -a "$SUMMARY"
for ip in "${WORKERS[@]}"; do
  rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    --exclude=.git --exclude=__pycache__ --exclude="*.pyc" \
    --exclude=training/results --exclude=/tmp/neuron_cache \
    "$REPO/" ubuntu@$ip:$REPO/ &
done
wait
echo "[sync] done" | tee -a "$SUMMARY"

JOB_IDX=0
for spec in "${JOBS[@]}"; do
  JOB_IDX=$((JOB_IDX+1))
  IFS=":" read -r LABEL TRAIN_REL BACKEND <<< "$spec"
  PORT=$((PORT_BASE + JOB_IDX))
  TAG="${LABEL}_${BACKEND}"
  JOB_RESULTS_DIR="$RESULTS_ROOT/$TAG"
  mkdir -p "$JOB_RESULTS_DIR"
  LOG_DIR="$LOG_ROOT/$TAG"
  mkdir -p "$LOG_DIR"

  echo "" | tee -a "$SUMMARY"
  echo "============================================================" | tee -a "$SUMMARY"
  echo "  [$JOB_IDX/10] $TAG  ($TRAIN_REL --backend $BACKEND)" | tee -a "$SUMMARY"
  echo "  Port: $PORT  Started: $(date -u)" | tee -a "$SUMMARY"
  echo "============================================================" | tee -a "$SUMMARY"

  COMMON_ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
export NEURON_RT_NUM_CORES=32 && \
export NEURON_NUM_RECENT_MODELS_TO_KEEP=1 && \
export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && \
export FI_PROVIDER=efa && \
export FI_EFA_USE_DEVICE_RDMA=1 && \
export FI_EFA_FORK_SAFE=1 && \
export MASTER_ADDR=$MASTER && \
export MASTER_PORT=$PORT && \
export RESULTS_DIR=$JOB_RESULTS_DIR"

  TRUN_ARGS="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

  WORKER_PIDS=()
  NODE_RANK=1
  for ip in "${WORKERS[@]}"; do
    WLOG="$LOG_DIR/node${NODE_RANK}_${ip}.log"
    ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip \
      "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN_ARGS --node_rank=$NODE_RANK $REPO/$TRAIN_REL --backend $BACKEND --steps $STEPS --warmup $WARMUP" \
      > "$WLOG" 2>&1 &
    WORKER_PIDS+=($!)
    NODE_RANK=$((NODE_RANK+1))
  done

  MLOG="$LOG_DIR/node0_master.log"
  eval "$COMMON_ENV"
  cd "$REPO"
  T0=$(date +%s)
  $TORCHRUN $TRUN_ARGS --node_rank=0 "$REPO/$TRAIN_REL" \
    --backend "$BACKEND" --steps "$STEPS" --warmup "$WARMUP" > "$MLOG" 2>&1
  MRC=$?
  T1=$(date +%s)
  ELAPSED=$((T1-T0))

  fail=0
  for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then fail=$((fail+1)); fi
  done

  # Find result JSON across master + workers (rank 0 may live anywhere).
  JSON_FOUND=""
  for d in "$JOB_RESULTS_DIR" ; do
    for f in "$d"/*_converge.json; do
      [ -f "$f" ] && JSON_FOUND="$f"
    done
  done
  if [ -z "$JSON_FOUND" ]; then
    for ip in "${WORKERS[@]}"; do
      REMOTE_JSON=$(ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip "ls $JOB_RESULTS_DIR/*_converge.json 2>/dev/null | head -1" 2>/dev/null)
      if [ -n "$REMOTE_JSON" ]; then
        scp -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$ip:"$REMOTE_JSON" "$JOB_RESULTS_DIR/" 2>/dev/null
        JSON_FOUND="$JOB_RESULTS_DIR/$(basename $REMOTE_JSON)"
        break
      fi
    done
  fi

  STATUS="OK"
  [ $MRC -ne 0 ] && STATUS="MASTER_FAIL_$MRC"
  [ $fail -gt 0 ] && STATUS="$STATUS WORKER_FAIL_$fail"
  [ -z "$JSON_FOUND" ] && STATUS="$STATUS NO_JSON"

  echo "  [$JOB_IDX/10] $TAG  status=$STATUS  elapsed=${ELAPSED}s  json=${JSON_FOUND:-NONE}" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "  SWEEP DONE  $(date -u)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
