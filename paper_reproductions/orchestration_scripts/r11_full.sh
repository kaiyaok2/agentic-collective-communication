#!/bin/bash
# R11 — FULL paper-numbers pipeline for tables 2-7 and figure 3
# 3 styles (strategy-enumerate, cc-react, multi-island) × 9 problems
# Survives reboots via /home/ubuntu/r11/ + per-phase state files
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
set -u

# ─── Paths / cluster ─────────────────────────────────────────────────────
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
PY=$VENV/bin/python3
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
WORKERS=$(IFS=,; echo "${WORKER_LIST[*]}")

R11=/home/ubuntu/r11
HEARTBEAT=$R11/HEARTBEAT
STATE=$R11/STATE
mkdir -p $R11 $STATE $R11/_kg_v6
echo "R11 FULL START $(date -u)" >> $HEARTBEAT

# Locked compile flags for all training (training only — search uses its own)
CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64

PROBLEMS=(alltoallv uniform_a2a dxe grad_ar ring_kv pp_send_recv tp_mlp fsdp_prefetch llama_block_ar)
STYLES=(strategy-enumerate cc-react multi-island)
STYLE_FLAGS=(strategy-enumerate cc-react multi-island)
HEAVY_FIRST=(grad_ar dxe fsdp_prefetch tp_mlp llama_block_ar pp_send_recv ring_kv uniform_a2a alltoallv)

# Training run plan (script + tag)
declare -a OLMOE_RUNS=(
  "training/train_olmoe10b.py            olmoe_default"
  "training/olmoe_sweep_s128.py          olmoe_s128"
  "training/olmoe_sweep_l4.py            olmoe_l4"
  "training/olmoe_sweep_d1024_s512.py    olmoe_d1024_s512"
)
declare -a LLAMA_RUNS=(
  "experiments/model_extension/train_llama_e2e_amp3.py per_mb  llama_amp3_per_mb"
  "experiments/model_extension/train_llama_e2e_amp3.py bundled llama_amp3_bundled"
  "experiments/model_extension/train_llama_e2e_amp2.py per_mb  llama_amp2_per_mb"
  "experiments/model_extension/train_llama_e2e_amp2.py bundled llama_amp2_bundled"
)

# ─── helpers ─────────────────────────────────────────────────────────────
restore_v6_kg() {
  cd $REPO
  git checkout main -- runtime/ 2>/dev/null
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > runtime/trainium_grad_ar_7node.py
}

snapshot_v6_kg() {
  restore_v6_kg
  cp $REPO/runtime/trainium_*.py $R11/_kg_v6/ 2>/dev/null
}

rsync_repo() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      "$REPO/" ubuntu@$ip:$REPO/ &
  done; wait
}

kill_all() {
  killall -9 torchrun 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "killall -9 torchrun python3 2>/dev/null; true" &
  done; wait; sleep 3
}

deploy_picks() {
  local STYLE_DIR="$1"
  restore_v6_kg
  for P in "${PROBLEMS[@]}"; do
    cp "$STYLE_DIR/runtime_per_problem/$P/trainium_${P}"*.py $REPO/runtime/ 2>/dev/null
  done
}

deploy_baseline() {
  # Paper baseline = v6 KG runtime
  restore_v6_kg
}

# olmoe_viability TAG  →  0 = OK, non-zero = OOM/fail
olmoe_viability() {
  local TAG="$1"
  local OUT=$R11/_gate/$TAG
  mkdir -p $OUT
  kill_all
  rsync_repo
  local port=$((40000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_olmoe10b.py --backend agent --grad-sync agent --ce agent --steps 5 --warmup 1" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 900 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_olmoe10b.py --backend agent --grad-sync agent --ce agent --steps 5 --warmup 1 > $OUT/m0.log 2>&1
  local RC=$?; wait
  if grep -qi 'NRT_RESOURCE\|Failed to allocate' $OUT/m0.log $OUT/w*.log 2>/dev/null; then RC=2; fi
  echo "[gate $TAG] RC=$RC $(date -u)" >> $HEARTBEAT
  return $RC
}

# ─── Phase 1: search ─────────────────────────────────────────────────────
phase1_search() {
  [ -f $STATE/phase1_done ] && { echo "PHASE 1 skip (state)" >> $HEARTBEAT; return; }
  echo "PHASE 1: SEARCH (3 styles × 9 problems) $(date -u)" >> $HEARTBEAT
  snapshot_v6_kg
  rsync_repo
  for i in 0 1 2; do
    local STYLE=${STYLES[$i]}
    local STYLE_FLAG=${STYLE_FLAGS[$i]}
    mkdir -p $R11/$STYLE/searches $R11/$STYLE/runtime_per_problem
    for P in "${PROBLEMS[@]}"; do
      local OUT=$R11/$STYLE/searches/$P
      mkdir -p $OUT
      if [ -f $OUT/results_${P}.json ]; then
        echo "[search $STYLE/$P] cached" >> $HEARTBEAT
        continue
      fi
      echo "[search $STYLE/$P] start $(date -u)" >> $HEARTBEAT
      local T0=$(date -u +%s)
      restore_v6_kg
      cd $REPO
      timeout 2400 $PY $REPO/experiments/run_search.py \
        --problem $P --phase3-style $STYLE_FLAG \
        --max-rounds 4 --num-nodes 7 \
        --master-addr $MASTER --worker-addrs $WORKERS \
        --output-dir $OUT > $OUT/run.log 2>&1
      local RC=$?
      local T1=$(date -u +%s)
      echo "[search $STYLE/$P] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
      mkdir -p $R11/$STYLE/runtime_per_problem/$P
      cp $REPO/runtime/trainium_${P}*.py $R11/$STYLE/runtime_per_problem/$P/ 2>/dev/null
    done
  done
  touch $STATE/phase1_done
  echo "PHASE 1 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 1.5: gate per style ───────────────────────────────────────────
phase1_gate() {
  [ -f $STATE/gate_done ] && { echo "GATE skip" >> $HEARTBEAT; return; }
  echo "PHASE 1.5: OLMoE viability gate $(date -u)" >> $HEARTBEAT
  for STYLE in "${STYLES[@]}"; do
    local STYLE_DIR=$R11/$STYLE
    deploy_picks $STYLE_DIR
    rsync_repo
    if olmoe_viability "${STYLE}_init"; then
      echo "[gate $STYLE] PASS first-try" >> $HEARTBEAT
      continue
    fi
    for P in "${HEAVY_FIRST[@]}"; do
      local KG_F=$R11/_kg_v6/trainium_${P}_7node.py
      if [ -f $KG_F ]; then
        echo "[gate $STYLE] retry: fallback $P to v6 KG" >> $HEARTBEAT
        cp $KG_F $STYLE_DIR/runtime_per_problem/$P/trainium_${P}_7node.py
        deploy_picks $STYLE_DIR
        rsync_repo
        if olmoe_viability "${STYLE}_retry_$P"; then
          echo "[gate $STYLE] PASS after $P fallback" >> $HEARTBEAT
          break
        fi
      fi
    done
  done
  touch $STATE/gate_done
  echo "GATE DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 2: 7-node training (high-priority paper data) ─────────────────
run_olmoe_training() {
  local STACK_DIR="$1"   # baseline runtime dir or $R11/$STYLE
  local STACK_TAG="$2"   # e.g. baseline, strategy-enumerate
  local SCRIPT="$3"      # e.g. training/train_olmoe10b.py
  local TAG="$4"         # e.g. olmoe_default
  local OUT=$R11/$STACK_TAG/training/${TAG}
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[train ${STACK_TAG}/${TAG}] cached" >> $HEARTBEAT; return; }
  if [ "$STACK_TAG" = "baseline" ]; then deploy_baseline; else deploy_picks $STACK_DIR; fi
  echo "[train ${STACK_TAG}/${TAG}] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  kill_all
  rsync_repo
  local port=$((42000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local ARGS="--backend agent --grad-sync agent --ce agent --steps 1000 --warmup 50"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $ARGS" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 5400 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $ARGS > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "ResultRC=$RC dur=$((T1-T0))s $(date -u)" > $OUT/result
  for j in $OUT/olmoe10b_*.json; do [ -f "$j" ] && cp "$j" "$OUT/" 2>/dev/null; done
  echo "[train ${STACK_TAG}/${TAG}] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

run_llama_training() {
  local STACK_DIR="$1" STACK_TAG="$2" SCRIPT="$3" BACKEND="$4" TAG="$5"
  local OUT=$R11/$STACK_TAG/training/${TAG}
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[train ${STACK_TAG}/${TAG}] cached" >> $HEARTBEAT; return; }
  if [ "$STACK_TAG" = "baseline" ]; then deploy_baseline; else deploy_picks $STACK_DIR; fi
  echo "[train ${STACK_TAG}/${TAG}] start backend=$BACKEND $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  kill_all
  rsync_repo
  local port=$((44000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $BACKEND 1000" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 2400 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $BACKEND 1000 > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  echo "ResultRC=$RC dur=$((T1-T0))s $(date -u)" > $OUT/result
  echo "[train ${STACK_TAG}/${TAG}] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

phase2_training() {
  [ -f $STATE/phase2_done ] && { echo "PHASE 2 skip" >> $HEARTBEAT; return; }
  echo "PHASE 2: 7-NODE TRAINING $(date -u)" >> $HEARTBEAT
  for STACK in baseline "${STYLES[@]}"; do
    if [ "$STACK" = "baseline" ]; then DIR=""; else DIR=$R11/$STACK; fi
    mkdir -p $R11/$STACK/training
    for line in "${OLMOE_RUNS[@]}"; do
      set -- $line
      run_olmoe_training "$DIR" "$STACK" "$1" "$2"
    done
    for line in "${LLAMA_RUNS[@]}"; do
      set -- $line
      run_llama_training "$DIR" "$STACK" "$1" "$2" "$3"
    done
  done
  touch $STATE/phase2_done
  echo "PHASE 2 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 3: 7-node h7_bench (Table 2 cols 2 + Fig 3) ──────────────────
run_h7_bench() {
  local STACK_DIR="$1" STACK_TAG="$2" PROB="$3"
  local OUT=$R11/$STACK_TAG/h7_bench
  mkdir -p $OUT
  [ -f $OUT/${PROB}.json ] && { echo "[h7bench ${STACK_TAG}/$PROB] cached" >> $HEARTBEAT; return; }
  if [ "$STACK_TAG" = "baseline" ]; then deploy_baseline; else deploy_picks $STACK_DIR; fi
  rsync_repo
  echo "[h7bench ${STACK_TAG}/$PROB] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  cd $REPO
  PORT=$((45000 + RANDOM % 1000)) bash $REPO/experiments/h7_bench/run_bench.sh $PROB > $OUT/${PROB}_run.log 2>&1
  cp /tmp/h7_bench/results/${PROB}.json $OUT/${PROB}.json 2>/dev/null
  cp -r /tmp/h7_bench/logs/${PROB} $OUT/${PROB}_logs 2>/dev/null
  local T1=$(date -u +%s)
  echo "[h7bench ${STACK_TAG}/$PROB] dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

phase3_h7bench() {
  [ -f $STATE/phase3_done ] && { echo "PHASE 3 skip" >> $HEARTBEAT; return; }
  echo "PHASE 3: 7-NODE H7_BENCH $(date -u)" >> $HEARTBEAT
  for STACK in baseline "${STYLES[@]}"; do
    if [ "$STACK" = "baseline" ]; then DIR=""; else DIR=$R11/$STACK; fi
    for PROB in a2av ua2a rkv dxe; do
      run_h7_bench "$DIR" "$STACK" "$PROB"
    done
  done
  touch $STATE/phase3_done
  echo "PHASE 3 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 4: 1-node h7_bench (Table 2 col 1 + Fig 3) ──────────────────
# Uses the same bench scripts but constrained to 1 node (the master only).
phase4_n1bench() {
  [ -f $STATE/phase4_done ] && { echo "PHASE 4 skip" >> $HEARTBEAT; return; }
  echo "PHASE 4: 1-NODE BENCH $(date -u)" >> $HEARTBEAT
  for STACK in baseline "${STYLES[@]}"; do
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_picks $R11/$STACK; fi
    rsync_repo
    local OUT=$R11/$STACK/n1_bench
    mkdir -p $OUT
    for PROB in a2av ua2a rkv dxe; do
      [ -f $OUT/${PROB}.json ] && continue
      echo "[n1bench ${STACK}/$PROB] start $(date -u)" >> $HEARTBEAT
      local T0=$(date -u +%s)
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE"
      eval "$ENV_VARS"
      timeout 600 $TORCHRUN --nproc_per_node=32 --standalone $REPO/experiments/h7_bench/bench_${PROB}.py > $OUT/${PROB}_run.log 2>&1
      cp /tmp/h7_bench/results/${PROB}.json $OUT/${PROB}.json 2>/dev/null
      local T1=$(date -u +%s)
      echo "[n1bench ${STACK}/$PROB] dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
    done
  done
  touch $STATE/phase4_done
  echo "PHASE 4 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Main ────────────────────────────────────────────────────────────────
snapshot_v6_kg
phase1_search
phase1_gate
phase2_training
phase3_h7bench
phase4_n1bench
echo "R11 ALL DONE $(date -u)" >> $HEARTBEAT
