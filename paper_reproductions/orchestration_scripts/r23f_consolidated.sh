#!/bin/bash
# R23f consolidated driver: Llama amp1-4 full-bundled + per-problem amp1 training
# + Ring KV training + OLMoE rerun + extra benches. Uses STATIC rdzv backend to
# avoid the rank-0-on-worker TCPStore reachability issue.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
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
HB=$R23/HEARTBEAT_F
mkdir -p $R23

STACKS=(baseline strategy-enumerate cc-react multi-island)
AMPS=(amp1 amp2 amp3 amp4)
PERPROB=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar)

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
  pkill -9 -f train_tp_mlp 2>/dev/null || true
  pkill -9 -f train_fsdp 2>/dev/null || true
  pkill -9 -f train_pp_send 2>/dev/null || true
  pkill -9 -f train_llama_block 2>/dev/null || true
  pkill -9 -f train_ring_kv 2>/dev/null || true
  pkill -9 -f 'bench_.*v6\.py' 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "pkill -9 -f torchrun; pkill -9 -f 'train_(olmoe|llama|tp|fsdp|pp|ring)'; pkill -9 -f 'bench_.*v6.py'" 2>/dev/null &
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
  # Override grad_ar with baseline (per-tensor) to avoid bad runtime regressions
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
}

# Run a 7-node training script with STATIC rdzv backend
# Args: $1=script path, $2=script args, $3=output dir, $4=timeout_sec
run_static_7n() {
  local SCRIPT="$1"; local ARGS="$2"; local OUT="$3"; local TO="${4:-1800}"
  kill_all
  mkdir -p "$OUT"
  local port=$((54000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
    export PERCALL_PROBE=1"
  # static backend: --rdzv_backend=static --rdzv_endpoint=<master>:<port> --node_rank=<NR>
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
  wait
  echo "RC=$RC" > $OUT/result
  return $RC
}

# Run a 7-node bench with STATIC rdzv (master is rank 0 -> output goes to m0.log)
run_static_bench_7n() {
  local SCRIPT="$1"; local OUT="$2"; local TO="${3:-600}"
  kill_all
  mkdir -p "$OUT"
  local port=$((55000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=static --rdzv_endpoint=${MASTER}:${port} --master_addr=${MASTER} --master_port=${port}"
  local NR=1
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $SCRIPT" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV"; cd $REPO
  timeout $TO $TORCHRUN $TRUN --node_rank=0 $SCRIPT > $OUT/m0.log 2>&1
  local RC=$?
  wait
  return $RC
}

run_1n_bench() {
  local SCRIPT="$1"; local OUT="$2"
  kill_all
  mkdir -p "$OUT"
  local port=$((56000 + RANDOM % 1000))
  local ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
    export BENCH_OUT=$OUT && export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  eval "$ENV"; cd $REPO
  timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 $SCRIPT > $OUT/m0.log 2>&1
}

# === PHASE F1: Llama amp1-4 full-bundled rerun (skip baseline stack) ===
echo "R23F LLAMA START $(date -u)" > $HB
for STACK in "${STACKS[@]}"; do
  [ "$STACK" = "baseline" ] && continue
  deploy_agent_picks $R22/$STACK
  rsync_repo
  for amp in "${AMPS[@]}"; do
    for be in per_mb bundled; do
      TAG="llama_${amp}_${be}_fullF"
      OUT=$R23/$STACK/training/$TAG
      for ip in "${WORKERS[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "rm -f /tmp/tp_search/llama_e2e_${amp}_*" 2>/dev/null &
      done
      rm -f /tmp/tp_search/llama_e2e_${amp}_* 2>/dev/null
      wait
      T0=$(date -u +%s)
      run_static_7n $REPO/experiments/model_extension/train_llama_e2e_${amp}.py "$be 1000" $OUT 1800
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
echo "R23F LLAMA DONE $(date -u)" >> $HB

# === PHASE F2: per-problem amp1 7n training (S=1024 M=4 shapes) ===
echo "R23F PERPROB START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  for PROB in "${PERPROB[@]}"; do
    OUT=$R23/$STACK/training/${PROB}_7node_amp1F
    if [ "$STACK" = "baseline" ]; then BFLAG="--backend baseline"; else BFLAG="--backend agent"; fi
    T0=$(date -u +%s)
    run_static_7n $REPO/training/train_${PROB}_7node.py "$BFLAG --steps 250 --warmup 30" $OUT 1800
    RC=$?
    T1=$(date -u +%s)
    echo "[perprob $PROB $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
  done
done
echo "R23F PERPROB DONE $(date -u)" >> $HB

# === PHASE F3: Ring KV 7n training (reduced dims) ===
echo "R23F RING-KV START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  OUT=$R23/$STACK/training/ring_kv_7node_F
  if [ "$STACK" = "baseline" ]; then BFLAG="--backend baseline"; else BFLAG="--backend evolved"; fi
  # Per-call probe window 100-130
  RKV_ENV="export RKV_DM=512 && export RKV_HEADS=8 && export RKV_LAYERS=2 && export RKV_NEXP=16 && \
    export RKV_EXDIM=512 && export RKV_VOCAB=4096 && export RKV_SEQLEN=128 && \
    export PROBE_START_STEP=100 && export PROBE_END_STEP=130"
  # Wrap RKV_ENV around run_static_7n by exporting in shell first
  eval "$RKV_ENV"
  T0=$(date -u +%s)
  run_static_7n $REPO/training/train_ring_kv.py "$BFLAG --steps 150" $OUT 1800
  RC=$?
  T1=$(date -u +%s)
  echo "[ring_kv $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
done
echo "R23F RING-KV DONE $(date -u)" >> $HB

# === PHASE F4: OLMoE rerun without grad_ar probe ===
echo "R23F OLMOE2 START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R22/$STACK
    FLAGS="--backend agent --grad-sync baseline --ce agent"
  fi
  rsync_repo
  OUT=$R23/$STACK/training/olmoe_no_gar
  export PROBE_START_STEP=200 PROBE_END_STEP=230
  T0=$(date -u +%s)
  run_static_7n $REPO/training/train_olmoe10b.py "$FLAGS --steps 250 --warmup 30" $OUT 3600
  RC=$?
  T1=$(date -u +%s)
  echo "[olmoe2 $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
done
echo "R23F OLMOE2 DONE $(date -u)" >> $HB

# === PHASE F5: extra benches a2av + ring_kv v6 (skip baseline stack) ===
echo "R23F EXTRA-BENCH START $(date -u)" >> $HB
for STACK in "${STACKS[@]}"; do
  [ "$STACK" = "baseline" ] && continue
  deploy_agent_picks $R22/$STACK
  rsync_repo
  for PROB in a2av ring_kv; do
    OUT=$R23/$STACK/n1_bench_v6/$PROB
    run_1n_bench $REPO/experiments/h7_bench/bench_${PROB}_v6.py $OUT
    echo "[1n $STACK $PROB v6] exit=$? $(date -u)" >> $HB
    OUT=$R23/$STACK/h7_bench_v6/$PROB
    T0=$(date -u +%s)
    run_static_bench_7n $REPO/experiments/h7_bench/bench_${PROB}_v6.py $OUT 900
    RC=$?
    T1=$(date -u +%s)
    echo "[7n $STACK $PROB v6] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
  done
done
echo "R23F EXTRA-BENCH DONE $(date -u)" >> $HB
echo "R23F ALL DONE $(date -u)" >> $HB
