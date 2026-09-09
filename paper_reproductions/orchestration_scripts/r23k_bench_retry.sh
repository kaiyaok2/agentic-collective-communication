#!/bin/bash
# R23k: rerun ua2a + ring_kv v6 benches for 3 agent stacks × 1n+7n.
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
HB=$R23/HEARTBEAT_K
mkdir -p $R23
STACKS=(strategy-enumerate cc-react multi-island)
PROBS=(ua2a ring_kv)

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
  pkill -9 -f 'bench_.*_v6\.py' 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "sudo pkill -9 -u ubuntu -f python" 2>/dev/null &
  done
  wait
  sleep 3
}
deploy_agent_picks() {
  cd $REPO
  for prob_dir in $1/runtime_per_problem/*/; do
    for f in $prob_dir/trainium_*_7node.py $prob_dir/trainium_*[!7node].py; do
      [ -f "$f" ] && cp "$f" runtime/$(basename $f) 2>/dev/null
    done
  done
}
run_1n() {
  local SCRIPT="$1"; local OUT="$2"
  kill_all
  mkdir -p "$OUT"
  local port=$((58000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  eval "$ENV"; cd $REPO
  timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 $SCRIPT > $OUT/m0.log 2>&1
}
run_7n() {
  local SCRIPT="$1"; local OUT="$2"
  kill_all
  mkdir -p "$OUT"
  local port=$((59000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=static --rdzv_endpoint=${MASTER}:${port} --master_addr=${MASTER} --master_port=${port}"
  local NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $SCRIPT" > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout 600 $TORCHRUN $TRUN --node_rank=0 $SCRIPT > $OUT/m0.log 2>&1
  local RC=$?
  if [ $RC -eq 124 ]; then
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "sudo pkill -9 -u ubuntu -f python" 2>/dev/null &
    done; wait
  fi
  wait
}

echo "R23K START $(date -u)" > $HB
for STACK in "${STACKS[@]}"; do
  deploy_agent_picks $R22/$STACK
  rsync_repo
  for PROB in "${PROBS[@]}"; do
    OUT=$R23/$STACK/n1_bench_v6/$PROB
    run_1n $REPO/experiments/h7_bench/bench_${PROB}_v6.py $OUT
    echo "[1n $STACK $PROB v6] exit=$? $(date -u)" >> $HB
    OUT=$R23/$STACK/h7_bench_v6/$PROB
    T0=$(date -u +%s); run_7n $REPO/experiments/h7_bench/bench_${PROB}_v6.py $OUT
    T1=$(date -u +%s)
    echo "[7n $STACK $PROB v6] dur=$((T1-T0))s $(date -u)" >> $HB
  done
done
echo "R23K ALL DONE $(date -u)" >> $HB
