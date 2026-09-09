#!/bin/bash
# R23g continuation: pick up after F2 pp_send_recv hang.
# Phase G1: continue F2 PERPROB for cc-react and multi-island (skip pp_send_recv to avoid 100-min hangs).
# Phase G2: F3 Ring KV (4 stacks).
# Phase G3: F4 OLMoE rerun without grad_ar (4 stacks).
# Phase G4: F5 extra benches a2av + ring_kv (3 stacks, skip baseline).
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
HB=$R23/HEARTBEAT_G
mkdir -p $R23

STACKS_AGENT=(cc-react multi-island)  # skip baseline + strat-enum already done
PERPROB=(tp_mlp fsdp_prefetch llama_block_ar)  # skip pp_send_recv
ALL_STACKS=(baseline strategy-enumerate cc-react multi-island)

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
  pkill -9 -f 'train_(olmoe|llama|tp|fsdp|pp|ring)' 2>/dev/null || true
  pkill -9 -f 'bench_.*_v6\.py' 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "sudo pkill -9 -u ubuntu -f python; sudo pkill -9 -u ubuntu -f torchrun" 2>/dev/null &
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
  # Force-kill workers after master timeout to prevent hangs
  if [ $RC -eq 124 ]; then
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "sudo pkill -9 -u ubuntu -f python; sudo pkill -9 -u ubuntu -f torchrun" 2>/dev/null &
    done
    wait
  fi
  wait
  echo "RC=$RC" > $OUT/result
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
  if [ $RC -eq 124 ]; then
    for ip in "${WORKERS[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "sudo pkill -9 -u ubuntu -f python" 2>/dev/null &
    done
    wait
  fi
  wait
  return $RC
}

echo "R23G START $(date -u)" > $HB

# === G1: continue F2 PERPROB for cc-react + multi-island (skip pp_send_recv) ===
for STACK in "${STACKS_AGENT[@]}"; do
  deploy_agent_picks $R22/$STACK
  rsync_repo
  for PROB in "${PERPROB[@]}"; do
    OUT=$R23/$STACK/training/${PROB}_7node_amp1F
    BFLAG="--backend agent"
    T0=$(date -u +%s)
    run_static_7n $REPO/training/train_${PROB}_7node.py "$BFLAG --steps 250 --warmup 30" $OUT 1200
    RC=$?
    T1=$(date -u +%s)
    echo "[perprob $PROB $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
  done
done
echo "R23G PERPROB DONE $(date -u)" >> $HB

# === G2: F3 Ring KV (all 4 stacks) ===
for STACK in "${ALL_STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
  rsync_repo
  OUT=$R23/$STACK/training/ring_kv_7node_F
  if [ "$STACK" = "baseline" ]; then BFLAG="--backend baseline"; else BFLAG="--backend evolved"; fi
  export RKV_DM=512 RKV_HEADS=8 RKV_LAYERS=2 RKV_NEXP=16 RKV_EXDIM=512 RKV_VOCAB=4096 RKV_SEQLEN=128
  export PROBE_START_STEP=100 PROBE_END_STEP=130
  T0=$(date -u +%s)
  run_static_7n $REPO/training/train_ring_kv.py "$BFLAG --steps 150" $OUT 1200
  RC=$?
  T1=$(date -u +%s)
  echo "[ring_kv $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
done
echo "R23G RING-KV DONE $(date -u)" >> $HB

# === G3: F4 OLMoE rerun without grad_ar (4 stacks; force baseline grad-sync) ===
for STACK in "${ALL_STACKS[@]}"; do
  if [ "$STACK" = "baseline" ]; then
    deploy_baseline; FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R22/$STACK
    FLAGS="--backend agent --grad-sync baseline --ce agent"
  fi
  rsync_repo
  OUT=$R23/$STACK/training/olmoe_no_gar_F
  export PROBE_START_STEP=200 PROBE_END_STEP=230
  T0=$(date -u +%s)
  run_static_7n $REPO/training/train_olmoe10b.py "$FLAGS --steps 250 --warmup 30" $OUT 2400
  RC=$?
  T1=$(date -u +%s)
  echo "[olmoe2 $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
done
echo "R23G OLMOE2 DONE $(date -u)" >> $HB

# === G4: extra benches a2av + ring_kv v6 (skip baseline stack) ===
for STACK in "${STACKS_AGENT[@]}" strategy-enumerate; do
  deploy_agent_picks $R22/$STACK
  rsync_repo
  for PROB in a2av ring_kv; do
    OUT=$R23/$STACK/n1_bench_v6/$PROB
    run_1n_bench $REPO/experiments/h7_bench/bench_${PROB}_v6.py $OUT
    echo "[1n $STACK $PROB v6] exit=$? $(date -u)" >> $HB
    OUT=$R23/$STACK/h7_bench_v6/$PROB
    T0=$(date -u +%s)
    run_static_bench_7n $REPO/experiments/h7_bench/bench_${PROB}_v6.py $OUT 600
    RC=$?
    T1=$(date -u +%s)
    echo "[7n $STACK $PROB v6] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HB
  done
done
echo "R23G ALL DONE $(date -u)" >> $HB
