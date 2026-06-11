#!/bin/bash
# R18 v3 bench retry — fills the a2av/ua2a baseline cells that the v1/v2 bench
# harnesses cannot produce due to a torch_xla 2.9 regression in
# xm.all_gather(unsqueeze(0), dim=0). v3 uses torch.distributed primitives
# directly (all_gather_into_tensor + reduce_scatter_tensor with explicit
# output allocation) so the shape inference can't go wrong.
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
echo "R18 V3 BENCH START $(date -u)" >> $HEARTBEAT
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
ensure_dirs() {
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/h7_bench; rm -f /tmp/h7_bench/a2av_*.json /tmp/h7_bench/ua2a_*.json" &
  done
  rm -f /tmp/h7_bench/a2av_*.json /tmp/h7_bench/ua2a_*.json 2>/dev/null
  wait
}

run_v3_per_stack() {
  local STACK="$1"
  local OUT7=$R18/$STACK/h7_bench_v3
  local OUT1=$R18/$STACK/n1_bench_v3
  mkdir -p $OUT7 $OUT1
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R16/$STACK; fi
  rsync_repo_to_workers
  ensure_dirs

  # ---- 7-node v3 ----
  for prob in a2av ua2a; do
    local port=$((49000 + RANDOM % 1000))
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
    local SCRIPT=experiments/h7_bench/bench_${prob}_v3.py
    kill_all
    local NR=1
    for ip in "${WORKER_LIST[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT" \
        > $OUT7/${prob}_w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV_VARS"; cd $REPO
    timeout 1200 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT > $OUT7/${prob}_m0.log 2>&1
    wait
    for ip in "${WORKER_LIST[@]}"; do
      rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include="${prob}_*.json" --exclude='*' \
        "ubuntu@${ip}:/tmp/h7_bench/" "${OUT7}/" 2>/dev/null &
    done
    rsync -a /tmp/h7_bench/${prob}_*.json $OUT7/ 2>/dev/null || true
    wait
    echo "  [v3-7n $STACK/$prob] done $(date -u)" >> $HEARTBEAT
  done

  ensure_dirs
  # ---- 1-node v3 ----
  for prob in a2av ua2a; do
    local port=$((50000 + RANDOM % 1000))
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
    local SCRIPT=experiments/h7_bench/bench_${prob}_v3.py
    timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 --master_addr=$MASTER --master_port=$port \
      $REPO/$SCRIPT > $OUT1/${prob}_m0.log 2>&1
    rsync -a /tmp/h7_bench/${prob}_*.json $OUT1/ 2>/dev/null || true
    echo "  [v3-1n $STACK/$prob] done $(date -u)" >> $HEARTBEAT
  done
}

for STACK in "${STACKS[@]}"; do
  echo "[v3 $STACK] start $(date -u)" >> $HEARTBEAT
  run_v3_per_stack "$STACK"
  echo "[v3 $STACK] done $(date -u)" >> $HEARTBEAT
done

echo "R18 V3 BENCH DONE $(date -u)" >> $HEARTBEAT
