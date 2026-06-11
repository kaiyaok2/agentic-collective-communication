#!/bin/bash
# R18 follow-up — runs ONLY after r18_full.sh completes. Fills the remaining
# cells in Table 2 with no footnotes:
#   - per-stack 1-node bench for all 9 problems
#   - per-stack Llama amp2/amp3 with probes
#   - per-stack ua2a/rkv 7-node training with add_step_closure probes
#   - per-stack 7-node bench rerun for a2av/ua2a using the v2 autograd-Function-wrapped harness
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

R16=/home/ubuntu/r16
R18=/home/ubuntu/r18
HEARTBEAT=$R18/HEARTBEAT
mkdir -p $R18
echo "R18 FOLLOWUP START $(date -u)" >> $HEARTBEAT

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
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_" 2>/dev/null &
  done
  wait
  sleep 2
}
ensure_dirs() {
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search /tmp/h7_bench; rm -f /tmp/h7_bench/*.json /tmp/tp_search/*.json" &
  done
  rm -f /tmp/h7_bench/*.json /tmp/tp_search/*.json 2>/dev/null
  wait
}
deploy_baseline() {
  cd $REPO
  git checkout main -- runtime/ 2>/dev/null
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
}
deploy_agent_picks() {
  local STYLE_DIR="$1"
  cd $REPO
  for prob_dir in $STYLE_DIR/runtime_per_problem/*/; do
    for f in $prob_dir/trainium_*_7node.py $prob_dir/trainium_*[!7node].py; do
      [ -f "$f" ] && cp "$f" runtime/$(basename $f) 2>/dev/null
    done
  done
}

# ---- 1-node bench ----
run_n1_bench_per_stack() {
  local STACK="$1"
  local OUT=$R18/$STACK/n1_bench
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[n1bench $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R16/$STACK; fi
  rsync_repo_to_workers
  kill_all; ensure_dirs
  echo "[n1bench $STACK] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  for prob in a2av ua2a rkv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar grad_ar; do
    local port=$((47000 + RANDOM % 1000))
    local script_name=bench_${prob}.py
    # Use v2 harness for a2av/ua2a (autograd-wrap dodges the all_gather regression)
    [ "$prob" = "a2av" ] && script_name=bench_a2av_v2.py
    [ "$prob" = "ua2a" ] && script_name=bench_ua2a_v2.py
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
    timeout 1200 $TORCHRUN --nproc_per_node=32 --nnodes=1 --master_addr=$MASTER --master_port=$port \
      $REPO/experiments/h7_bench/$script_name > $OUT/${prob}_m0.log 2>&1
    rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
    echo "  [n1bench $STACK/$prob] done $(date -u)" >> $HEARTBEAT
  done
  local T1=$(date -u +%s)
  echo "dur=$((T1-T0))s" > $OUT/result
  echo "[n1bench $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

# ---- 7-node bench rerun (a2av/ua2a baseline using v2 harness) ----
run_h7_bench_v2_per_stack() {
  local STACK="$1"
  local OUT=$R18/$STACK/h7_bench_v2
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[h7bench_v2 $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R16/$STACK; fi
  rsync_repo_to_workers
  kill_all; ensure_dirs
  echo "[h7bench_v2 $STACK] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  for prob in a2av ua2a; do
    local port=$((48000 + RANDOM % 1000))
    local script_name=bench_${prob}_v2.py
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
    local NR=1
    for ip in "${WORKER_LIST[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/h7_bench/$script_name" \
        > $OUT/${prob}_w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV_VARS"; cd $REPO
    timeout 1500 $TORCHRUN $TRUN --node_rank=0 $REPO/experiments/h7_bench/$script_name > $OUT/${prob}_m0.log 2>&1
    wait
    for ip in "${WORKER_LIST[@]}"; do
      rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include="${prob}_*.json" --exclude='*' \
        "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/" 2>/dev/null &
    done
    rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
    wait
    echo "  [h7bench_v2 $STACK/$prob] done $(date -u)" >> $HEARTBEAT
  done
  local T1=$(date -u +%s)
  echo "dur=$((T1-T0))s" > $OUT/result
}

# ---- 7-node ua2a / rkv training ----
run_single_collective() {
  local STACK="$1"
  local SCRIPT="$2"
  local TAG="$3"
  local AGENT_NAME="${4:-agent}"
  local OUT=$R18/$STACK/training/$TAG
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[$TAG $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then deploy_baseline; local BACKEND=baseline
  else deploy_agent_picks $R16/$STACK; local BACKEND="$AGENT_NAME"; fi
  rsync_repo_to_workers; kill_all; ensure_dirs
  echo "[$TAG $STACK] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  local port=$((43000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local ARGS="--backend $BACKEND --steps 200 --warmup 30"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $ARGS" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $ARGS > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:${OUT}/" "${OUT}/" 2>/dev/null &
  done
  wait
  echo "[$TAG $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

# ---- Llama amp2/amp3 with probes per stack ----
run_llama() {
  local STACK="$1"; local SCRIPT="$2"; local LBASE="$3"; local BACKEND="$4"
  local TAG="${LBASE}_${BACKEND}"
  local OUT=$R18/$STACK/training/$TAG
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[$TAG $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R16/$STACK; fi
  rsync_repo_to_workers; kill_all; ensure_dirs
  echo "[$TAG $STACK] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  local port=$((44000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $BACKEND 2000" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $BACKEND 2000 > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:${OUT}/" "${OUT}/" 2>/dev/null &
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/tp_search/" "${OUT}/tp_search/" 2>/dev/null &
  done
  wait
  echo "[$TAG $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

# ============= ORCHESTRATION =============

# 1) Per-stack rerun of h7 bench for a2av/ua2a with v2 harness (replaces the broken 7n cells)
for STACK in "${STACKS[@]}"; do
  run_h7_bench_v2_per_stack "$STACK"
done

# 2) Per-stack 1-node bench for all 9 problems
for STACK in "${STACKS[@]}"; do
  run_n1_bench_per_stack "$STACK"
done

# 3) Per-stack ua2a + rkv 7-node training (add_step_closure probes)
for STACK in "${STACKS[@]}"; do
  run_single_collective "$STACK" training/train_uniform_a2a_7node.py ua2a_7node agent
  run_single_collective "$STACK" training/train_ring_kv.py rkv_7node evolved
done

# 4) Per-stack Llama amp2/amp3 per_mb + bundled with probes
for STACK in "${STACKS[@]}"; do
  for amp in amp3 amp2; do
    for be in per_mb bundled; do
      run_llama "$STACK" experiments/model_extension/train_llama_e2e_$amp.py llama_$amp $be
    done
  done
done

echo "R18 FOLLOWUP DONE $(date -u)" >> $HEARTBEAT
