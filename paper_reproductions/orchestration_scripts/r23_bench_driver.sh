#!/bin/bash
# R23 v6 bench driver: runs each of {tp_mlp, fsdp_prefetch, pp_send_recv,
# llama_block_ar} v6 bench on 1-node (master only, 32 ranks) and 7-node
# (224 ranks). Each variant runs WARMUP=0 and reports cold first-iter.
# Loops over the 4 stacks (baseline + 3 agent search styles).
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
R22=/home/ubuntu/r22
R23=/home/ubuntu/r23
mkdir -p $R23
PROBS=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar)
STACKS=(baseline strategy-enumerate cc-react multi-island)

rsync_repo() {
  for ip in "${WORKERS[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" &
  done
  wait
}
kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun" 2>/dev/null &
  done
  wait
  sleep 2
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

run_1node() {
  local STACK=$1; local PROB=$2; local OUT=$R23/$STACK/n1_bench_v6/$PROB
  mkdir -p $OUT
  local PORT=$((43000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
  eval "$ENV"; cd $REPO
  timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 \
    $REPO/experiments/h7_bench/bench_${PROB}_v6.py > $OUT/m0.log 2>&1
  echo "[1n $STACK $PROB] exit=$? $(date -u)" >> $R23/HEARTBEAT_BENCH
}
run_7node() {
  local STACK=$1; local PROB=$2; local OUT=$R23/$STACK/h7_bench_v6/$PROB
  mkdir -p $OUT
  local PORT=$((44000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"
  local NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/h7_bench/bench_${PROB}_v6.py" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout 600 $TORCHRUN $TRUN --node_rank=0 \
    $REPO/experiments/h7_bench/bench_${PROB}_v6.py > $OUT/m0.log 2>&1
  RC=$?; wait
  echo "[7n $STACK $PROB] exit=$RC $(date -u)" >> $R23/HEARTBEAT_BENCH
}

echo "R23 BENCH START $(date -u)" > $R23/HEARTBEAT_BENCH
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for PROB in "${PROBS[@]}"; do
    kill_all
    run_1node $STACK $PROB
    kill_all
    run_7node $STACK $PROB
  done
done
echo "R23 BENCH DONE $(date -u)" >> $R23/HEARTBEAT_BENCH
