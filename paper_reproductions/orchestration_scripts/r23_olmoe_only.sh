#!/bin/bash
# R23 — OLMoE only, fixed wait_device_ops probe pattern. 4 stacks, late-phase
# probe window steps 200-230 (30 steps × ~16 calls × 2 syncs per step).
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64
R22=/home/ubuntu/r22
R23=/home/ubuntu/r23
HEARTBEAT=$R23/HEARTBEAT
mkdir -p $R23

STACKS=(baseline strategy-enumerate cc-react multi-island)

rsync_repo() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" &
  done
  wait
}
kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  pkill -9 -f train_ 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_" 2>/dev/null &
  done
  wait
  sleep 3
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
collect() {
  local OUT=$1
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:${OUT}/" "${OUT}/" 2>/dev/null &
  done
  wait
}

echo "R23 START $(date -u)" >> $HEARTBEAT
for STACK in "${STACKS[@]}"; do
  OUT=$R23/$STACK/training/olmoe_default
  mkdir -p $OUT
  if [ "$STACK" = "baseline" ]; then
    deploy_baseline; FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R22/$STACK; FLAGS="--backend agent --grad-sync agent --ce agent"
  fi
  rsync_repo
  kill_all
  T0=$(date -u +%s)
  port=$((42000 + RANDOM % 1000))
  ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1 && export PROBE_START_STEP=200 && export PROBE_END_STEP=230"
  TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  ARGS="$FLAGS --steps 250 --warmup 30"
  NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_olmoe10b.py $ARGS" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_olmoe10b.py $ARGS > $OUT/m0.log 2>&1
  RC=$?
  wait
  T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  collect $OUT
  echo "[olmoe $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
done
echo "R23 OLMoE DONE $(date -u)" >> $HEARTBEAT
