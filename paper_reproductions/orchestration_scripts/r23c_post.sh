#!/bin/bash
# R23 post-OLMoE: rerun 7n benches (v6 + v7) with explicit init sync, plus
# cc-react amp3 per_mb (failed in main run). Each 7n run holds the master
# torchrun open until ALL ranks complete via a sync read after the bench.
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
HB=$R23/HEARTBEAT_POST
mkdir -p $R23

PROBS_V6=(tp_mlp llama_block_ar ring_kv)
PROBS_V7=(fsdp_prefetch pp_send_recv)
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
  pkill -9 -f train_olmoe 2>/dev/null || true
  pkill -9 -f train_llama 2>/dev/null || true
  pkill -9 -f 'bench_.*_v[67].py' 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_olmoe; pkill -9 -f train_llama; pkill -9 -f 'bench_.*_v[67].py'" 2>/dev/null &
  done
  wait
  sleep 5
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

run_7n_bench() {
  local STACK=$1; local PROB=$2; local VER=$3
  local OUT=$R23/$STACK/h7_bench_${VER}/$PROB
  mkdir -p $OUT
  kill_all
  local PORT=$((46000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT && \
    export TORCH_DIST_INIT_BARRIER=1"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT} --rdzv_conf=timeout=600"
  local NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/h7_bench/bench_${PROB}_${VER}.py" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout 900 $TORCHRUN $TRUN --node_rank=0 \
    $REPO/experiments/h7_bench/bench_${PROB}_${VER}.py > $OUT/m0.log 2>&1
  RC=$?
  # Wait for all worker SSHs to finish before next iteration (avoids killing live workers)
  wait
  echo "[7n $STACK $PROB $VER] exit=$RC $(date -u)" >> $HB
}

run_1n_bench() {
  local STACK=$1; local PROB=$2; local VER=$3
  local OUT=$R23/$STACK/n1_bench_${VER}/$PROB
  mkdir -p $OUT
  kill_all
  local PORT=$((45000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
  eval "$ENV"; cd $REPO
  timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 \
    $REPO/experiments/h7_bench/bench_${PROB}_${VER}.py > $OUT/m0.log 2>&1
  echo "[1n $STACK $PROB $VER] exit=$? $(date -u)" >> $HB
}

run_cc_react_amp3_per_mb() {
  local OUT=$R23/cc-react/training/llama_amp3_per_mb
  mkdir -p $OUT
  kill_all
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search; rm -f /tmp/tp_search/llama_e2e_amp3_per_mb*" 2>/dev/null &
  done
  rm -f /tmp/tp_search/llama_e2e_amp3_per_mb*
  wait
  local T0=$(date -u +%s)
  local port=$((47000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1 && \
    export PROBE_START_STEP=800 && export PROBE_END_STEP=1000"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/model_extension/train_llama_e2e_amp3.py per_mb 1000" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout 1800 $TORCHRUN $TRUN --node_rank=0 $REPO/experiments/model_extension/train_llama_e2e_amp3.py per_mb 1000 > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  mkdir -p $OUT/tp_search
  cp /tmp/tp_search/llama_e2e_amp3_per_mb* $OUT/tp_search/ 2>/dev/null
  for ip in "${WORKERS[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/tp_search/" "$OUT/tp_search/" 2>/dev/null &
  done
  wait
  echo "[amp3 per_mb cc-react rerun] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
}

echo "R23 POST START $(date -u)" > $HB

# Phase 1: rerun 7n benches (1n is fine from main run; just 7n needs rerun)
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for PROB in "${PROBS_V6[@]}"; do
    run_7n_bench $STACK $PROB v6
  done
  for PROB in "${PROBS_V7[@]}"; do
    run_1n_bench $STACK $PROB v7
    run_7n_bench $STACK $PROB v7
  done
done

# Phase 2: cc-react amp3 per_mb rerun
deploy_agent_picks $R22/cc-react
rsync_repo
run_cc_react_amp3_per_mb


# Phase 3: per-problem training under new amp1 shapes (S=1024, M=4)
echo "R23 PERPROB START $(date -u)" >> $HB
PERPROB=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar)
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for PROB in "${PERPROB[@]}"; do
    OUT=$R23/$STACK/training/${PROB}_7node_amp1
    mkdir -p $OUT
    kill_all
    T0=$(date -u +%s)
    port=$((48000 + RANDOM % 1000))
    if [ "$STACK" = "baseline" ]; then BFLAG="--backend baseline"; else BFLAG="--backend agent"; fi
    ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
      export PERCALL_PROBE=1 && export PROBE_START_STEP=200 && export PROBE_END_STEP=250"
    TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port} --rdzv_conf=timeout=600"
    NR=1
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_${PROB}_7node.py $BFLAG --steps 250 --warmup 30" \
        > $OUT/w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV"; cd $REPO
    timeout 2400 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_${PROB}_7node.py $BFLAG --steps 250 --warmup 30 > $OUT/m0.log 2>&1
    RC=$?; wait
    T1=$(date -u +%s)
    echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
    echo "[perprob $PROB $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
  done
done
echo "R23 PERPROB DONE $(date -u)" >> $HB

# Phase 4: Llama amp1-4 full-bundled rerun (step_bundled now bundles tp_fsdp + vocab_dxe too)
echo "R23 LLAMA-FULL START $(date -u)" >> $HB
AMPS=(amp1 amp2 amp3 amp4)
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  [ "$STACK" = "baseline" ] && continue  # skip baseline stack: per_mb is shared, bundled would use main canonical (not meaningful)
  rsync_repo
  for amp in "${AMPS[@]}"; do
    for be in per_mb bundled; do
      TAG="llama_${amp}_${be}_full"
      OUT=$R23/$STACK/training/$TAG
      mkdir -p $OUT
      kill_all
      for ip in "${WORKERS[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search; rm -f /tmp/tp_search/llama_e2e_${amp}_*" 2>/dev/null &
      done
      rm -f /tmp/tp_search/llama_e2e_${amp}_* 2>/dev/null
      wait
      T0=$(date -u +%s)
      port=$((49000 + RANDOM % 1000))
      ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
        export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1 && \
        export PROBE_START_STEP=800 && export PROBE_END_STEP=1000"
      TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port} --rdzv_conf=timeout=600"
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
echo "R23 LLAMA-FULL DONE $(date -u)" >> $HB

# Phase 5: Ring KV 7-node training with reduced model dims to fit 224-rank HBM
echo "R23 RING-KV START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  OUT=$R23/$STACK/training/ring_kv_7node
  mkdir -p $OUT
  kill_all
  T0=$(date -u +%s)
  port=$((50000 + RANDOM % 1000))
  if [ "$STACK" = "baseline" ]; then BFLAG="--backend baseline"; else BFLAG="--backend evolved"; fi
  ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1 && export PROBE_START_STEP=100 && export PROBE_END_STEP=130 && \
    export RKV_DM=512 && export RKV_HEADS=8 && export RKV_LAYERS=2 && export RKV_NEXP=16 && \
    export RKV_EXDIM=512 && export RKV_VOCAB=4096 && export RKV_SEQLEN=128"
  TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port} --rdzv_conf=timeout=600"
  NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_ring_kv.py $BFLAG --steps 150" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout 1800 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_ring_kv.py $BFLAG --steps 150 > $OUT/m0.log 2>&1
  RC=$?; wait
  T1=$(date -u +%s)
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
  echo "[ring_kv $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
done
echo "R23 RING-KV DONE $(date -u)" >> $HB
echo "R23 POST DONE $(date -u)" >> $HB
