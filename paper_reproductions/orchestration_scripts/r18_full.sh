#!/bin/bash
# R18 — full per-stack pipeline using add_step_closure probes (no .item() sync).
# Runs OLMoE + Llama amp2/amp3 + ua2a 7n + rkv 7n training for all 4 stacks
# WITH probes, AND runs h7_bench independently per stack so bench numbers
# reflect each stack's actually-deployed runtime (no rsync collision).
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

R16=/home/ubuntu/r16   # agent-pick source
R18=/home/ubuntu/r18
HEARTBEAT=$R18/HEARTBEAT
mkdir -p $R18
echo "R18 START $(date -u)" >> $HEARTBEAT
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
collect() {
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

run_training_olmoe() {
  local STACK="$1"
  local OUT=$R18/$STACK/training/olmoe_default
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[olmoe $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then
    deploy_baseline; local FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R16/$STACK; local FLAGS="--backend agent --grad-sync agent --ce agent"
  fi
  rsync_repo_to_workers
  echo "[olmoe $STACK] start $(date -u)" >> $HEARTBEAT
  kill_all
  ensure_dirs
  local T0=$(date -u +%s)
  local port=$((42000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local ARGS="$FLAGS --steps 200 --warmup 30"
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
  collect "$OUT"
  echo "[olmoe $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

run_h7bench_per_stack() {
  local STACK="$1"
  local OUT=$R18/$STACK/h7_bench
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[h7bench $STACK] cached" >> $HEARTBEAT; return; }
  if [ "$STACK" = "baseline" ]; then
    deploy_baseline
  else
    deploy_agent_picks $R16/$STACK
  fi
  rsync_repo_to_workers
  ensure_dirs
  echo "[h7bench $STACK] start $(date -u)" >> $HEARTBEAT
  kill_all
  local T0=$(date -u +%s)
  for prob in a2av ua2a rkv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar grad_ar; do
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
    timeout 1500 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT > $OUT/${prob}_m0.log 2>&1
    wait
    # Collect JSON only for THIS problem from /tmp/h7_bench/ on all nodes
    for ip in "${WORKER_LIST[@]}"; do
      rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include="${prob}_*.json" --exclude='*' \
        "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/" 2>/dev/null &
    done
    rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
    wait
    echo "  [h7bench $STACK/$prob] done $(date -u)" >> $HEARTBEAT
  done
  local T1=$(date -u +%s)
  echo "dur=$((T1-T0))s" > $OUT/result
  echo "[h7bench $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

# ============= ORCHESTRATION =============

for STACK in "${STACKS[@]}"; do
  run_h7bench_per_stack "$STACK"
  run_training_olmoe "$STACK"
done

echo "R18 DONE $(date -u)" >> $HEARTBEAT
