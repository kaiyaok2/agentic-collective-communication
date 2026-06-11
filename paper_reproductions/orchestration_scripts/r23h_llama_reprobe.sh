#!/bin/bash
# R23h: Llama amp1-4 rerun with split fsdp_prefetch + tp_mlp probes in per_mb mode
# (and bundled mode unchanged with fsdp_prefetch_bundled + tp_mlp_bundled).
# Skip baseline stack (per_mb is shared).
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
HB=$R23/HEARTBEAT_H
mkdir -p $R23

STACKS_AGENT=(strategy-enumerate cc-react multi-island)
AMPS=(amp1 amp2 amp3 amp4)

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
  pkill -9 -f train_llama 2>/dev/null || true
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
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
}

run_static_7n() {
  local SCRIPT="$1"; local ARGS="$2"; local OUT="$3"; local TO="${4:-1200}"
  kill_all
  mkdir -p "$OUT"
  local port=$((54000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=static --rdzv_endpoint=${MASTER}:${port} --master_addr=${MASTER} --master_port=${port}"
  local NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $SCRIPT $ARGS" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout $TO $TORCHRUN $TRUN --node_rank=0 $SCRIPT $ARGS > $OUT/m0.log 2>&1
  local RC=$?
  if [ $RC -eq 124 ]; then
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "sudo pkill -9 -u ubuntu -f python" 2>/dev/null &
    done
    wait
  fi
  wait
  return $RC
}

echo "R23H LLAMA-REPROBE START $(date -u)" > $HB
for STACK in "${STACKS_AGENT[@]}"; do
  deploy_agent_picks $R22/$STACK
  rsync_repo
  for amp in "${AMPS[@]}"; do
    for be in per_mb bundled; do
      TAG="llama_${amp}_${be}_reprobe"
      OUT=$R23/$STACK/training/$TAG
      for ip in "${WORKERS[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search; rm -f /tmp/tp_search/llama_e2e_${amp}_*" 2>/dev/null &
      done
      rm -f /tmp/tp_search/llama_e2e_${amp}_* 2>/dev/null
      wait
      T0=$(date -u +%s)
      run_static_7n $REPO/experiments/model_extension/train_llama_e2e_${amp}.py "$be 1000" $OUT 1200
      RC=$?
      T1=$(date -u +%s)
      mkdir -p $OUT/tp_search
      cp /tmp/tp_search/llama_e2e_${amp}_${be}* $OUT/tp_search/ 2>/dev/null
      for ip in "${WORKERS[@]}"; do
        rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
          --include='*.json' --exclude='*' \
          "ubuntu@${ip}:/tmp/tp_search/" "$OUT/tp_search/" 2>/dev/null &
      done
      wait
      echo "[$TAG $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
    done
  done
done
echo "R23H LLAMA-REPROBE DONE $(date -u)" >> $HB
