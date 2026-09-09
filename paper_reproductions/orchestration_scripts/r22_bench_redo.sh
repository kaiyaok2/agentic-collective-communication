#!/bin/bash
# r22_bench_redo.sh — re-run h7_bench + n1_bench with v5 scripts that have the
# corrected cross-scope-inversion shapes (baseline = M small dispatches in a loop,
# agent = 1 big bundled dispatch on M-stacked buffer; same total bytes per iter).
#
# Mirrors r21_pipeline.sh::phase4_h7bench + phase5_n1bench but writes to
# /home/ubuntu/r22/<stack>/{h7_bench_v5,n1_bench_v5}/.
set -u

R22=/home/ubuntu/r22
REPO=/home/ubuntu/agentic-collective-communication
KEY=/home/ubuntu/.ssh/Kaiyao.pem
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN="$VENV/bin/torchrun"
MASTER=172.31.19.201
CCFLAGS="--model-type=transformer --enable-saturate-infinity"
RT_PAGE=64
HEARTBEAT=$R22/HEARTBEAT_BENCH_V5

WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
STACKS=(${STACKS:-baseline strategy-enumerate cc-react multi-island})

probs_v5=(a2av tp_mlp fsdp_prefetch llama_block_ar pp_send_recv)

deploy_baseline() {
  cd $REPO && git checkout main -- runtime/ >/dev/null 2>&1 || true
  # restore the v6 grad_ar that fits in 32MB-bucketed HBM
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > runtime/trainium_grad_ar_7node.py 2>/dev/null || true
}

deploy_agent_picks() {
  local STACK_DIR=$1
  cd $REPO
  for f in $STACK_DIR/runtime_per_problem/*/trainium_*_7node.py; do
    [ -f "$f" ] && cp "$f" runtime/
  done
}

rsync_repo() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --delete \
      --exclude='neuron_cache' --exclude='.git' --exclude='*.pyc' \
      $REPO/ ubuntu@$ip:$REPO/ >/dev/null 2>&1 &
  done
  wait
}

kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun 2>/dev/null || true" &
  done
  wait
  sleep 3
}

ensure_dirs() {
  mkdir -p /tmp/h7_bench /tmp/tp_search
  rm -f /tmp/h7_bench/*.json
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "mkdir -p /tmp/h7_bench; rm -f /tmp/h7_bench/*.json" &
  done
  wait
}

phase_h7bench_v5() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R22/$STACK/h7_bench_v5
    mkdir -p $OUT
    [ -f $OUT/result ] && { echo "[h7bench_v5 $STACK] skip (result exists)"; continue; }
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
    rsync_repo
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    for prob in "${probs_v5[@]}"; do
      local port=$((46000 + RANDOM % 1000))
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
        export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
      local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
      local SCRIPT=experiments/h7_bench/bench_${prob}_v5.py
      local NR=1
      for ip in "${WORKER_LIST[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT" \
          > $OUT/${prob}_w${NR}.log 2>&1 &
        NR=$((NR+1))
      done
      eval "$ENV_VARS"; cd $REPO
      timeout 1200 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT > $OUT/${prob}_m0.log 2>&1
      wait
      for ip in "${WORKER_LIST[@]}"; do
        rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include="${prob}_*.json" --exclude='*' \
          "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/" 2>/dev/null &
      done
      rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
      wait
    done
    local T1=$(date -u +%s)
    echo "dur=$((T1-T0))s" > $OUT/result
    echo "[h7bench_v5 $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

phase_n1bench_v5() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R22/$STACK/n1_bench_v5
    mkdir -p $OUT
    [ -f $OUT/result ] && { echo "[n1bench_v5 $STACK] skip"; continue; }
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    for prob in "${probs_v5[@]}"; do
      local port=$((51000 + RANDOM % 5000))
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
      local SCRIPT=experiments/h7_bench/bench_${prob}_v5.py
      eval "$ENV_VARS"; cd $REPO
      timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 --master_addr=$MASTER --master_port=$port \
        $REPO/$SCRIPT > $OUT/${prob}_m0.log 2>&1
      rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
    done
    local T1=$(date -u +%s)
    echo "dur=$((T1-T0))s" > $OUT/result
    echo "[n1bench_v5 $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

phase_h7bench_v5
phase_n1bench_v5
