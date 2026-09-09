#!/bin/bash
# R23e: extra bench reruns (a2av v6 + ring_kv v6) not in main post driver.
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
HB=$R23/HEARTBEAT_EXTRA
mkdir -p $R23
PROBS=(a2av ring_kv)
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
  pkill -9 -f 'bench_(a2av|ring_kv)_v6\.py' 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f 'bench_(a2av|ring_kv)_v6.py'" 2>/dev/null &
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
}

echo "R23 EXTRA BENCH START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  [ "$STACK" = "baseline" ] && continue  # skip baseline stack: redundant with agent stacks' baseline_fn measurements
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for PROB in "${PROBS[@]}"; do
    # 1-node bench (master only)
    OUT=$R23/$STACK/n1_bench_v6/$PROB
    mkdir -p $OUT
    kill_all
    PORT=$((52000 + RANDOM % 1000))
    ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
      export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
    eval "$ENV"; cd $REPO
    timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 \
      $REPO/experiments/h7_bench/bench_${PROB}_v6.py > $OUT/m0.log 2>&1
    echo "[1n $STACK $PROB v6] exit=$? $(date -u)" >> $HB

    # 7-node bench
    OUT=$R23/$STACK/h7_bench_v6/$PROB
    mkdir -p $OUT
    kill_all
    PORT=$((53000 + RANDOM % 1000))
    ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
    TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT} --rdzv_conf=timeout=600"
    NR=1
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/h7_bench/bench_${PROB}_v6.py" \
        > $OUT/w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV"; cd $REPO
    timeout 900 $TORCHRUN $TRUN --node_rank=0 $REPO/experiments/h7_bench/bench_${PROB}_v6.py > $OUT/m0.log 2>&1
    RC=$?; wait
    echo "[7n $STACK $PROB v6] exit=$RC $(date -u)" >> $HB
  done
done
echo "R23 EXTRA BENCH DONE $(date -u)" >> $HB
