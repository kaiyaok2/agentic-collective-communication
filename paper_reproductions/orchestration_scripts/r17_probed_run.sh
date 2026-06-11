#!/bin/bash
# R17 — re-run training with PERCALL_PROBE=1 to populate Table 2's
# "7-node training" per-call column, plus the bench-script fixes
# (a2av baseline, ua2a baseline, new grad_ar bench).
#
# Reuses R16's agent picks; only changes:
#   - PERCALL_PROBE=1 in training env
#   - Llama steps 1000 -> 5000 (user request: "more steps on Llama as it is faster")
#   - Adds train_uniform_a2a_7node.py and train_ring_kv.py to the training set
#     so ua2a/rkv have a real 7-node-training column
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
PY=$VENV/bin/python3
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

R16=/home/ubuntu/r16          # reuse agent picks from R16
R17=/home/ubuntu/r17
HEARTBEAT=$R17/HEARTBEAT
mkdir -p $R17
echo "R17 START $(date -u)" >> $HEARTBEAT

CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64

STACKS=(baseline strategy-enumerate cc-react multi-island)

rsync_repo_to_workers() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" &
  done
  wait
}

kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  pkill -9 -f train_  2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_" 2>/dev/null &
  done
  wait
  sleep 2
}

collect_artifacts() {
  local OUT=$1
  mkdir -p "$OUT"
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:${OUT}/" "${OUT}/" 2>/dev/null &
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/tp_search/" "${OUT}/tp_search/" 2>/dev/null &
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/h7_bench/" 2>/dev/null &
  done
  wait
}

ensure_tp_search_dir() {
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search /tmp/h7_bench" &
  done
  wait
  mkdir -p /tmp/tp_search /tmp/h7_bench
}

deploy_dev_baseline() {
  cd $REPO
  git checkout -- runtime/ 2>/dev/null || true
}

deploy_agent_picks() {
  local STYLE_DIR="$1"
  cd $REPO
  if [ -d "$STYLE_DIR/runtime_per_problem" ]; then
    for f in $STYLE_DIR/runtime_per_problem/trainium_*.py; do
      [ -f "$f" ] && cp "$f" runtime/$(basename $f)
    done
  fi
}

# ============================ TRAINING ==================================

run_olmoe() {
  local STACK="$1"
  local OUT=$R17/$STACK/training/olmoe_default
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[olmoe $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then
    deploy_dev_baseline; local FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R16/$STACK; local FLAGS="--backend agent --grad-sync agent --ce agent"
  fi
  rsync_repo_to_workers
  echo "[olmoe $STACK] start $(date -u)" >> $HEARTBEAT
  kill_all
  local T0=$(date -u +%s)
  local port=$((42000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local ARGS="$FLAGS --steps 1000 --warmup 50"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_olmoe10b.py $ARGS" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 5400 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_olmoe10b.py $ARGS > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  collect_artifacts "$OUT"
  echo "[olmoe $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

run_single_collective_training() {
  # Runs a per-problem 7-node training script (ua2a or ring_kv).
  # ua2a uses --backend baseline|agent; rkv uses --backend baseline|evolved.
  local STACK="$1"
  local SCRIPT="$2"        # e.g. training/train_uniform_a2a_7node.py
  local TAG="$3"           # e.g. ua2a_7node
  local AGENT_NAME="${4:-agent}"  # 'agent' for ua2a, 'evolved' for rkv
  local OUT=$R17/$STACK/training/$TAG
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[$TAG $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then
    deploy_dev_baseline; local BACKEND_ARG=baseline
  else
    deploy_agent_picks $R16/$STACK; local BACKEND_ARG="$AGENT_NAME"
  fi
  rsync_repo_to_workers
  echo "[$TAG $STACK] start $(date -u)" >> $HEARTBEAT
  kill_all
  local T0=$(date -u +%s)
  local port=$((43000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local ARGS="--backend $BACKEND_ARG --steps 200 --warmup 30"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $ARGS" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 1800 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $ARGS > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  collect_artifacts "$OUT"
  echo "[$TAG $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

run_llama() {
  local STACK="$1"
  local SCRIPT="$2"         # experiments/model_extension/train_llama_e2e_amp{2,3}.py
  local LBASE="$3"          # llama_amp3 / llama_amp2
  local BACKEND="$4"        # per_mb / bundled
  local TAG="${LBASE}_${BACKEND}"
  local OUT=$R17/$STACK/training/$TAG
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[$TAG $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then deploy_dev_baseline
  else deploy_agent_picks $R16/$STACK; fi
  ensure_tp_search_dir
  rsync_repo_to_workers
  echo "[$TAG $STACK] start $(date -u)" >> $HEARTBEAT
  kill_all
  local T0=$(date -u +%s)
  local port=$((44000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  # User request: more steps on Llama because it's much faster.
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $BACKEND 5000" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $BACKEND 5000 > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  collect_artifacts "$OUT"
  echo "[$TAG $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

# ============================ BENCH FIXES ===============================

run_bench_fix() {
  # Re-run the 3 bench scripts that previously failed / didn't exist.
  # Uses the multi-island stack's agent runtimes (closest match to deployed).
  local OUT=$R17/bench_fixes
  mkdir -p $OUT
  deploy_agent_picks $R16/multi-island
  rsync_repo_to_workers
  ensure_tp_search_dir
  for prob in a2av ua2a grad_ar; do
    [ -f $OUT/${prob}_result ] && { echo "[bench $prob] cached" >> $HEARTBEAT; continue; }
    echo "[bench $prob] start $(date -u)" >> $HEARTBEAT
    kill_all
    local T0=$(date -u +%s)
    local port=$((46000 + RANDOM % 1000))
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
    local SCRIPT=experiments/h7_bench/bench_${prob}.py
    local NR=1
    for ip in "${WORKER_LIST[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT" \
        > $OUT/${prob}_w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV_VARS"; cd $REPO
    timeout 1200 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT > $OUT/${prob}_m0.log 2>&1
    local RC=$?; wait
    local T1=$(date -u +%s)
    echo "RC=$RC dur=$((T1-T0))s" > $OUT/${prob}_result
    collect_artifacts "$OUT"
    echo "[bench $prob] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

# ============================ ORCHESTRATION =============================

# Bench fixes first — fastest, fills 1-node-bench/7-node-bench blanks.
run_bench_fix

for STACK in "${STACKS[@]}"; do
  # OLMoE: a2av + dxe + grad_ar in one script
  run_olmoe "$STACK"
  # Single-collective 7-node training
  run_single_collective_training "$STACK" training/train_uniform_a2a_7node.py ua2a_7node agent
  run_single_collective_training "$STACK" training/train_ring_kv.py rkv_7node evolved
  # Llama: pp_send_recv + tp_fsdp + vocab_dxe + grad_ar_llama
  for amp in amp3 amp2; do
    for be in per_mb bundled; do
      run_llama "$STACK" experiments/model_extension/train_llama_e2e_$amp.py llama_$amp $be
    done
  done
done

echo "R17 DONE $(date -u)" >> $HEARTBEAT
