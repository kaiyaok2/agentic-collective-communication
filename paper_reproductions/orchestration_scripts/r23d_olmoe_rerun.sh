#!/bin/bash
# R23d: OLMoE rerun with grad_ar probe stripped and --grad-sync baseline forced
# for all stacks (so strat-enum's bad 32 MB grad_ar doesn't bloat step times).
# Only AllToAllV and dxe per-call probes recorded.
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
R22=/home/ubuntu/r22
R23=/home/ubuntu/r23
HEARTBEAT=$R23/HEARTBEAT_OLMOE2
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
  pkill -9 -f train_olmoe 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_olmoe" 2>/dev/null &
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
  cd $REPO
  for prob_dir in $1/runtime_per_problem/*/; do
    for f in $prob_dir/trainium_*_7node.py $prob_dir/trainium_*[!7node].py; do
      [ -f "$f" ] && cp "$f" runtime/$(basename $f) 2>/dev/null
    done
  done
  # Override grad_ar with baseline (per-tensor) regardless of stack
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
}
collect() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:$1/" "$1/" 2>/dev/null &
  done
  wait
}

echo "R23 OLMOE2 START $(date -u)" >> $HEARTBEAT
for STACK in "${STACKS[@]}"; do
  OUT=$R23/$STACK/training/olmoe_baseline_gar
  mkdir -p $OUT
  if [ "$STACK" = "baseline" ]; then
    deploy_baseline; FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R22/$STACK
    # Force grad-sync baseline (use per-tensor instead of evolved grad_ar runtime)
    FLAGS="--backend agent --grad-sync baseline --ce agent"
  fi
  rsync_repo
  kill_all
  T0=$(date -u +%s)
  port=$((51000 + RANDOM % 1000))
  ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1 && export PROBE_START_STEP=200 && export PROBE_END_STEP=230"
  TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port} --rdzv_conf=timeout=600"
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
  echo "[olmoe2 $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
done
echo "R23 OLMOE2 DONE $(date -u)" >> $HEARTBEAT
