#!/bin/bash
# R14 — full 4-stack pipeline (baseline + 3 agent styles) for paper Tables 2-7 + Figure 3
# Driver bugs fixed: (c) baseline stack uses developer baseline flags, (d) JSONs rsynced from all workers after each run
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
PY=$VENV/bin/python3
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
WORKERS=$(IFS=,; echo "${WORKER_LIST[*]}")

R14=/home/ubuntu/r14
HEARTBEAT=$R14/HEARTBEAT
STATE=$R14/STATE
mkdir -p $R14 $STATE $R14/_kg_v6 $R14/_gate
echo "R14 FULL START $(date -u)" >> $HEARTBEAT

# Locked compile flags for all training
CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64

PROBLEMS=(alltoallv uniform_a2a dxe grad_ar ring_kv pp_send_recv tp_mlp fsdp_prefetch llama_block_ar)
AGENT_STYLES=(strategy-enumerate cc-react multi-island)
HEAVY_FIRST=(grad_ar dxe fsdp_prefetch tp_mlp llama_block_ar pp_send_recv ring_kv uniform_a2a alltoallv)

# OLMoE training runs (script + tag)
declare -a OLMOE_RUNS=(
  "training/train_olmoe10b.py            olmoe_default"
  "training/olmoe_sweep_s128.py          olmoe_s128"
  "training/olmoe_sweep_l4.py            olmoe_l4"
  "training/olmoe_sweep_d1024_s512.py    olmoe_d1024_s512"
)
# Llama: each script has per_mb (baseline) AND bundled (agent) backends
declare -a LLAMA_SCRIPTS=(
  "experiments/model_extension/train_llama_e2e_amp3.py llama_amp3"
  "experiments/model_extension/train_llama_e2e_amp2.py llama_amp2"
)

# ─── runtime / repo helpers ──────────────────────────────────────────────
restore_main_runtime() {
  cd $REPO
  git checkout main -- runtime/ 2>/dev/null
}

snapshot_v6_kg() {
  restore_main_runtime
  # v6 grad_ar (the only file that matters for fallback; main is heavier-mem)
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py
  cp $REPO/runtime/trainium_*.py $R14/_kg_v6/ 2>/dev/null
}

rsync_repo_to_workers() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      "$REPO/" ubuntu@$ip:$REPO/ &
  done; wait
}

# Collect JSONs and logs from ALL workers into master $OUT after a training run.
# Looks for olmoe10b_*.json in $OUT on each worker, and /tmp/tp_search/*.json
collect_artifacts_from_workers() {
  local OUT="$1"
  mkdir -p "$OUT"
  for ip in "${WORKER_LIST[@]}"; do
    # Pull anything the worker dropped in $OUT (rank-0 worker writes here)
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:${OUT}/" "${OUT}/" 2>/dev/null &
    # Pull Llama's hardcoded /tmp/tp_search/ if any
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/tp_search/" "${OUT}/" 2>/dev/null &
  done; wait
}

ensure_tp_search_dir_on_workers() {
  # Llama scripts hardcode /tmp/tp_search/ as JSON output dir; make sure it exists
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip 'mkdir -p /tmp/tp_search' &
  done; wait
}

kill_all() {
  killall -9 torchrun python3 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "killall -9 torchrun python3 2>/dev/null; true" &
  done; wait; sleep 3
}

# Deploy each agent style's Phase-5 picks into runtime/
deploy_agent_picks() {
  local STYLE_DIR="$1"
  restore_main_runtime
  for P in "${PROBLEMS[@]}"; do
    cp "$STYLE_DIR/runtime_per_problem/$P/trainium_${P}"*.py $REPO/runtime/ 2>/dev/null
  done
}

# Baseline stack: don't deploy any agent picks — restore main runtime as a clean baseline,
# AND pass --backend baseline / --grad-sync baseline / --ce baseline so the OLMoE script
# uses its hardcoded developer-baseline functions instead of calling into runtime/.
deploy_dev_baseline() {
  restore_main_runtime
  # main's grad_ar/dxe/etc are agent-evolved but won't be called when flags=baseline
  # Still restore v6 grad_ar so any incidental import path doesn't OOM the cache
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
}

# ─── OLMoE viability gate (for agent stacks) ─────────────────────────────
olmoe_viability() {
  local TAG="$1"
  local OUT=$R14/_gate/$TAG
  mkdir -p $OUT
  kill_all
  rsync_repo_to_workers
  local port=$((40000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT"
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
  rsync_repo_to_workers
  for STYLE in "${AGENT_STYLES[@]}"; do
    mkdir -p $R14/$STYLE/searches $R14/$STYLE/runtime_per_problem
    for P in "${PROBLEMS[@]}"; do
      local OUT=$R14/$STYLE/searches/$P
      mkdir -p $OUT
      if [ -f $OUT/results_${P}.json ]; then
        echo "[search $STYLE/$P] cached" >> $HEARTBEAT
        continue
      fi
      echo "[search $STYLE/$P] start $(date -u)" >> $HEARTBEAT
      local T0=$(date -u +%s)
      snapshot_v6_kg   # always reset KG runtime before each search
      cd $REPO
      timeout 2400 $PY $REPO/experiments/run_search.py \
        --problem $P --phase3-style $STYLE \
        --max-rounds 4 --num-nodes 7 \
        --master-addr $MASTER --worker-addrs $WORKERS \
        --output-dir $OUT > $OUT/run.log 2>&1
      local RC=$?
      local T1=$(date -u +%s)
      echo "[search $STYLE/$P] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
      mkdir -p $R14/$STYLE/runtime_per_problem/$P
      cp $REPO/runtime/trainium_${P}*.py $R14/$STYLE/runtime_per_problem/$P/ 2>/dev/null
    done
  done
  touch $STATE/phase1_done
  echo "PHASE 1 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 1.5: gate (per agent style) ──────────────────────────────────
phase1_gate() {
  [ -f $STATE/gate_done ] && { echo "GATE skip" >> $HEARTBEAT; return; }
  echo "PHASE 1.5: OLMoE viability gate $(date -u)" >> $HEARTBEAT
  for STYLE in "${AGENT_STYLES[@]}"; do
    local STYLE_DIR=$R14/$STYLE
    deploy_agent_picks $STYLE_DIR
    rsync_repo_to_workers
    if olmoe_viability "${STYLE}_init"; then
      echo "[gate $STYLE] PASS first-try" >> $HEARTBEAT
      continue
    fi
    for P in "${HEAVY_FIRST[@]}"; do
      local KG_F=$R14/_kg_v6/trainium_${P}_7node.py
      if [ -f $KG_F ]; then
        echo "[gate $STYLE] retry: fallback $P to v6 KG" >> $HEARTBEAT
        cp $KG_F $STYLE_DIR/runtime_per_problem/$P/trainium_${P}_7node.py
        deploy_agent_picks $STYLE_DIR
        rsync_repo_to_workers
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

# ─── Phase 2: training (per stack × per config) ─────────────────────────
run_olmoe_training() {
  local STACK_TAG="$1"   # "baseline" or agent style
  local SCRIPT="$2"
  local CFG_TAG="$3"
  local OUT=$R14/$STACK_TAG/training/$CFG_TAG
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[train ${STACK_TAG}/${CFG_TAG}] cached" >> $HEARTBEAT; return; }

  # Pick flags + runtime deploy based on stack
  local OLMOE_FLAGS
  if [ "$STACK_TAG" = "baseline" ]; then
    deploy_dev_baseline
    OLMOE_FLAGS="--backend baseline --grad-sync baseline --ce baseline"
  else
    deploy_agent_picks $R14/$STACK_TAG
    OLMOE_FLAGS="--backend agent --grad-sync agent --ce agent"
  fi
  rsync_repo_to_workers

  echo "[train ${STACK_TAG}/${CFG_TAG}] start flags='$OLMOE_FLAGS' $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  kill_all
  rsync_repo_to_workers
  local port=$((42000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  local ARGS="$OLMOE_FLAGS --steps 1000 --warmup 50"
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
  echo "RC=$RC dur=$((T1-T0))s $(date -u)" > $OUT/result
  collect_artifacts_from_workers "$OUT"
  echo "[train ${STACK_TAG}/${CFG_TAG}] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

run_llama_training() {
  local STACK_TAG="$1"
  local SCRIPT="$2"
  local LBASE_TAG="$3"  # llama_amp3 or llama_amp2
  local BACKEND="$4"    # per_mb (baseline) or bundled (agent)
  local CFG_TAG="${LBASE_TAG}_${BACKEND}"
  local OUT=$R14/$STACK_TAG/training/$CFG_TAG
  mkdir -p $OUT
  [ -f $OUT/result ] && { echo "[train ${STACK_TAG}/${CFG_TAG}] cached" >> $HEARTBEAT; return; }

  if [ "$STACK_TAG" = "baseline" ]; then
    deploy_dev_baseline
  else
    deploy_agent_picks $R14/$STACK_TAG
  fi
  ensure_tp_search_dir_on_workers
  rsync_repo_to_workers

  echo "[train ${STACK_TAG}/${CFG_TAG}] start backend=$BACKEND $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  kill_all
  rsync_repo_to_workers
  local port=$((44000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
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
  echo "RC=$RC dur=$((T1-T0))s $(date -u)" > $OUT/result
  collect_artifacts_from_workers "$OUT"
  echo "[train ${STACK_TAG}/${CFG_TAG}] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

phase2_training() {
  [ -f $STATE/phase2_done ] && { echo "PHASE 2 skip" >> $HEARTBEAT; return; }
  echo "PHASE 2: 7-NODE TRAINING $(date -u)" >> $HEARTBEAT
  # 4 stacks: baseline + 3 agent
  for STACK in baseline "${AGENT_STYLES[@]}"; do
    mkdir -p $R14/$STACK/training
    for line in "${OLMOE_RUNS[@]}"; do
      set -- $line
      run_olmoe_training "$STACK" "$1" "$2"
    done
    for line in "${LLAMA_SCRIPTS[@]}"; do
      set -- $line
      local SCRIPT="$1" LBASE_TAG="$2"
      run_llama_training "$STACK" "$SCRIPT" "$LBASE_TAG" "per_mb"
      run_llama_training "$STACK" "$SCRIPT" "$LBASE_TAG" "bundled"
    done
  done
  touch $STATE/phase2_done
  echo "PHASE 2 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 3: 7-node h7_bench ──────────────────────────────────────────
run_h7_bench() {
  local STACK_TAG="$1" PROB="$2"
  local OUT=$R14/$STACK_TAG/h7_bench/$PROB
  mkdir -p $OUT
  [ -f $OUT/${PROB}.json ] && { echo "[h7bench ${STACK_TAG}/$PROB] cached" >> $HEARTBEAT; return; }
  if [ "$STACK_TAG" = "baseline" ]; then deploy_dev_baseline; else deploy_agent_picks $R14/$STACK_TAG; fi
  rsync_repo_to_workers
  echo "[h7bench ${STACK_TAG}/$PROB] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  cd $REPO
  PORT=$((45000 + RANDOM % 1000)) bash $REPO/experiments/h7_bench/run_bench.sh $PROB > $OUT/${PROB}_run.log 2>&1
  cp /tmp/h7_bench/results/${PROB}.json $OUT/${PROB}.json 2>/dev/null
  cp -r /tmp/h7_bench/logs/${PROB} $OUT/${PROB}_logs 2>/dev/null
  collect_artifacts_from_workers "$OUT"
  local T1=$(date -u +%s)
  echo "[h7bench ${STACK_TAG}/$PROB] dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

phase3_h7bench() {
  [ -f $STATE/phase3_done ] && { echo "PHASE 3 skip" >> $HEARTBEAT; return; }
  echo "PHASE 3: 7-NODE H7_BENCH $(date -u)" >> $HEARTBEAT
  for STACK in baseline "${AGENT_STYLES[@]}"; do
    for PROB in a2av ua2a rkv dxe; do
      run_h7_bench "$STACK" "$PROB"
    done
  done
  touch $STATE/phase3_done
  echo "PHASE 3 DONE $(date -u)" >> $HEARTBEAT
}

# ─── Phase 4: 1-node h7_bench ──────────────────────────────────────────
phase4_n1bench() {
  [ -f $STATE/phase4_done ] && { echo "PHASE 4 skip" >> $HEARTBEAT; return; }
  echo "PHASE 4: 1-NODE BENCH $(date -u)" >> $HEARTBEAT
  for STACK in baseline "${AGENT_STYLES[@]}"; do
    if [ "$STACK" = "baseline" ]; then deploy_dev_baseline; else deploy_agent_picks $R14/$STACK; fi
    rsync_repo_to_workers
    local OUT=$R14/$STACK/n1_bench
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
echo "R14 ALL DONE $(date -u)" >> $HEARTBEAT
