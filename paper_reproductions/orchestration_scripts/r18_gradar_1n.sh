#!/bin/bash
# Re-run grad_ar 1-node bench for all 4 stacks. The followup hit EADDRINUSE
# on port 47100; using randomized ports here.
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
echo "R18 GRADAR_1N START $(date -u)" >> $HEARTBEAT
CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64

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

for STACK in baseline strategy-enumerate cc-react multi-island; do
  OUT=$R18/$STACK/n1_bench
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R16/$STACK; fi
  rsync_repo_to_workers
  kill_all
  rm -f /tmp/h7_bench/grad_ar_*.json
  port=$((51000 + RANDOM % 5000))
  ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  eval "$ENV_VARS"; cd $REPO
  timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 --master_addr=$MASTER --master_port=$port \
    $REPO/experiments/h7_bench/bench_grad_ar.py > $OUT/grad_ar_m0.log 2>&1
  rsync -a /tmp/h7_bench/grad_ar_*.json $OUT/ 2>/dev/null || true
  echo "[gradar_1n $STACK] done $(date -u)" >> $HEARTBEAT
done

echo "R18 GRADAR_1N DONE $(date -u)" >> $HEARTBEAT
