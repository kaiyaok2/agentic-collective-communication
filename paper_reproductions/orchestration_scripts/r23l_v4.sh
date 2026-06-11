#!/bin/bash
# R23l v4: ring_kv cold-cache-per-variant — proper version that doesn't overwrite script.
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
HB=$R23/HEARTBEAT_L4
mkdir -p $R23

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
  pkill -9 -f 'ring_kv_variant_only' 2>/dev/null || true
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

run_cold_variant() {
  local STACK=$1; local VARIANT=$2; local SCOPE=$3
  local OUT=$R23/$STACK/${SCOPE}_bench_v6_cold/ring_kv
  mkdir -p $OUT
  kill_all
  local CACHE=/tmp/cold_$$_$RANDOM
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p $CACHE" 2>/dev/null &
  done
  mkdir -p $CACHE
  wait
  local port=$((60000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=$CACHE && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  if [ "$SCOPE" = "n1" ]; then
    eval "$ENV"; cd $REPO
    timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 /tmp/ring_kv_variant_only.py $VARIANT > $OUT/${VARIANT}_m0.log 2>&1
  else
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=static --rdzv_endpoint=${MASTER}:${port} --master_addr=${MASTER} --master_port=${port}"
    local NR=1
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR /tmp/ring_kv_variant_only.py $VARIANT" \
        > $OUT/${VARIANT}_w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV"; cd $REPO
    timeout 600 $TORCHRUN $TRUN --node_rank=0 /tmp/ring_kv_variant_only.py $VARIANT > $OUT/${VARIANT}_m0.log 2>&1
    wait
  fi
  rm -rf $CACHE 2>/dev/null
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "rm -rf $CACHE" 2>/dev/null &
  done
  wait
}

echo "R23L4 START $(date -u)" > $HB
for STACK in strategy-enumerate cc-react multi-island; do
  deploy_agent_picks $R22/$STACK
  rsync_repo
  # Push variant script to workers via rsync (since master's /tmp/ring_kv_variant_only.py exists)
  for ip in "${WORKERS[@]}"; do
    rsync -az /tmp/ring_kv_variant_only.py "ubuntu@${ip}:/tmp/ring_kv_variant_only.py" -e "ssh -i $KEY -o StrictHostKeyChecking=no" &
  done
  wait
  for SCOPE in n1 h7; do
    for VARIANT in baseline agent; do
      T0=$(date -u +%s)
      run_cold_variant $STACK $VARIANT $SCOPE
      T1=$(date -u +%s)
      echo "[$SCOPE $STACK ring_kv $VARIANT cold] dur=$((T1-T0))s $(date -u)" >> $HB
    done
  done
done
echo "R23L4 ALL DONE $(date -u)" >> $HB
