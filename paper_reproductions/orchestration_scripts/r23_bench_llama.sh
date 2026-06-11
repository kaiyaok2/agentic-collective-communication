#!/bin/bash
# R23 bench-then-llama driver: runs v6 benches (4 problems × 4 stacks × 1n+7n)
# then Llama amp redo (4 amps × 4 stacks × 2 backends). Launched after R23 OLMoE.
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
HB=$R23/HEARTBEAT_BL
mkdir -p $R23

PROBS=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar)
STACKS=(baseline strategy-enumerate cc-react multi-island)
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
  pkill -9 -f train_ 2>/dev/null || true
  pkill -9 -f 'bench_.*_v6.py' 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_; pkill -9 -f 'bench_.*_v6.py'" 2>/dev/null &
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
collect() {
  for ip in "${WORKERS[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:$1/" "$1/" 2>/dev/null &
  done
  wait
}

# ----- BENCH PHASE -----
echo "R23 BENCH START $(date -u)" > $HB
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for PROB in "${PROBS[@]}"; do
    # 1-node bench (master only, 32 ranks)
    OUT=$R23/$STACK/n1_bench_v6/$PROB
    mkdir -p $OUT
    kill_all
    PORT=$((43000 + RANDOM % 1000))
    ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
      export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
    eval "$ENV"; cd $REPO
    timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 \
      $REPO/experiments/h7_bench/bench_${PROB}_v6.py > $OUT/m0.log 2>&1
    RC=$?
    echo "[1n $STACK $PROB] exit=$RC $(date -u)" >> $HB

    # 7-node bench (224 ranks)
    OUT=$R23/$STACK/h7_bench_v6/$PROB
    mkdir -p $OUT
    kill_all
    PORT=$((44000 + RANDOM % 1000))
    ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
    TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"
    NR=1
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
    echo "[7n $STACK $PROB] exit=$RC $(date -u)" >> $HB
  done
done

# ----- LLAMA PHASE -----
echo "R23 LLAMA START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for amp in "${AMPS[@]}"; do
    for be in per_mb bundled; do
      TAG="llama_${amp}_${be}"
      OUT=$R23/$STACK/training/$TAG
      mkdir -p $OUT
      kill_all
      for ip in "${WORKERS[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search; rm -f /tmp/tp_search/llama_e2e_${amp}_*" 2>/dev/null &
      done
      rm -f /tmp/tp_search/llama_e2e_${amp}_* 2>/dev/null
      wait
      T0=$(date -u +%s)
      port=$((45000 + RANDOM % 1000))
      ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
        export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1 && \
        export PROBE_START_STEP=800 && export PROBE_END_STEP=1000"
      TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
      NR=1
      for ip in "${WORKERS[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/model_extension/train_llama_e2e_${amp}.py $be 1000" \
          > $OUT/w${NR}.log 2>&1 &
        NR=$((NR+1))
      done
      eval "$ENV"; cd $REPO
      timeout 1800 $TORCHRUN $TRUN --node_rank=0 $REPO/experiments/model_extension/train_llama_e2e_${amp}.py $be 1000 > $OUT/m0.log 2>&1
      RC=$?; wait
      T1=$(date -u +%s)
      echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
      # Collect from master /tmp and all workers
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
echo "R23 BENCH+LLAMA DONE $(date -u)" >> $HB
