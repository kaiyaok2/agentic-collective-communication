#!/bin/bash
# R19 — full re-run after simulator fix. The previous search converged to
# `per_mb_loop` for tp_mlp/fsdp_prefetch because the simulator double-charged
# cat/stack bytes that are actually fused with the downstream collective.
# After the fix (correctness_test.py fusion-credit walk extension), bundled
# should win for ALL multi-microbatch problems.
#
# Pipeline:
#   Phase 1: search 9 problems × 3 styles
#   Phase 2: deploy per-stack agent picks
#   Phase 3: per-stack OLMoE training with add_step_closure probes
#   Phase 4: per-stack h7_bench
#   Phase 5: per-stack n1_bench
#   Phase 6: ua2a/rkv 7n + Llama amp3/amp2 training (per-stack, probed)
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
CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64

R16=/home/ubuntu/r16
R19=/home/ubuntu/r19
HEARTBEAT=$R19/HEARTBEAT
STATE=$R19/STATE
mkdir -p $R19 $STATE
echo "R19 FULL START $(date -u)" >> $HEARTBEAT
PROBLEMS=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar grad_ar dxe ring_kv uniform_a2a alltoallv)
AGENT_STYLES=(strategy-enumerate cc-react multi-island)
STACKS=(baseline strategy-enumerate cc-react multi-island)

# =================== HELPERS ===================
rsync_repo_to_workers() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" &
  done
  wait
}
kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  pkill -9 -f train_ 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun; pkill -9 -f train_" 2>/dev/null &
  done
  wait
  sleep 2
}
ensure_dirs() {
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search /tmp/h7_bench; rm -f /tmp/h7_bench/*.json /tmp/tp_search/*.json" &
  done
  rm -f /tmp/h7_bench/*.json /tmp/tp_search/*.json 2>/dev/null
  wait
}
deploy_baseline() {
  cd $REPO
  git checkout main -- runtime/ 2>/dev/null
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
}
deploy_agent_picks() {
  local STYLE_DIR="$1"
  cd $REPO
  for prob_dir in $STYLE_DIR/runtime_per_problem/*/; do
    for f in $prob_dir/trainium_*_7node.py $prob_dir/trainium_*[!7node].py; do
      [ -f "$f" ] && cp "$f" runtime/$(basename $f) 2>/dev/null
    done
  done
}
collect() {
  local OUT=$1
  mkdir -p "$OUT"
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" "ubuntu@${ip}:${OUT}/" "${OUT}/" 2>/dev/null &
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/tp_search/" "${OUT}/tp_search/" 2>/dev/null &
  done
  wait
}

# =================== PHASE 1: SEARCH ===================
phase1_search() {
  [ -f $STATE/phase1_done ] && { echo "PHASE 1 cached" >> $HEARTBEAT; return; }
  echo "PHASE 1: search 9 problems × 3 styles $(date -u)" >> $HEARTBEAT
  for P in "${PROBLEMS[@]}"; do
    for STYLE in "${AGENT_STYLES[@]}"; do
      local OUT=$R19/$STYLE/searches/$P
      [ -f $OUT/done ] && { echo "[search $STYLE/$P] cached" >> $HEARTBEAT; continue; }
      mkdir -p $OUT
      deploy_baseline
      rsync_repo_to_workers
      echo "[search $STYLE/$P] start $(date -u)" >> $HEARTBEAT
      local T0=$(date -u +%s)
      timeout 2400 $PY $REPO/experiments/run_search.py \
        --problem $P --phase3-style $STYLE \
        --max-rounds 4 --num-nodes 7 \
        --master-addr $MASTER --worker-addrs $WORKERS \
        --output-dir $OUT > $OUT/run.log 2>&1
      local RC=$?
      local T1=$(date -u +%s)
      mkdir -p $R19/$STYLE/runtime_per_problem/$P
      cp $REPO/runtime/trainium_${P}*.py $R19/$STYLE/runtime_per_problem/$P/ 2>/dev/null
      echo "RC=$RC dur=$((T1-T0))s" > $OUT/done
      echo "[search $STYLE/$P] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
    done
  done
  touch $STATE/phase1_done
  echo "PHASE 1 DONE $(date -u)" >> $HEARTBEAT
}

# =================== PHASE 3: OLMoE training (per stack) ===================
phase3_olmoe() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R19/$STACK/training/olmoe_default
    mkdir -p $OUT
    [ -f $OUT/result ] && { echo "[olmoe $STACK] cached" >> $HEARTBEAT; continue; }
    if [ "$STACK" = "baseline" ]; then
      deploy_baseline; local FLAGS="--backend baseline --grad-sync baseline --ce baseline"
    else
      deploy_agent_picks $R19/$STACK; local FLAGS="--backend agent --grad-sync agent --ce agent"
    fi
    rsync_repo_to_workers
    echo "[olmoe $STACK] start $(date -u)" >> $HEARTBEAT
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    local port=$((42000 + RANDOM % 1000))
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
      export PERCALL_PROBE=1"
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
    local ARGS="$FLAGS --steps 200 --warmup 30"
    local NR=1
    for ip in "${WORKER_LIST[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_olmoe10b.py $ARGS" \
        > $OUT/w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV_VARS"; cd $REPO
    timeout 5400 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_olmoe10b.py $ARGS > $OUT/m0.log 2>&1
    local RC=$?; wait
    local T1=$(date -u +%s)
    echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
    collect "$OUT"
    echo "[olmoe $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

# =================== PHASE 4: Llama amp3/amp2 per stack ===================
phase4_llama() {
  for STACK in "${STACKS[@]}"; do
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R19/$STACK; fi
    rsync_repo_to_workers
    for amp in amp3 amp2; do
      for be in per_mb bundled; do
        local TAG="llama_${amp}_${be}"
        local OUT=$R19/$STACK/training/$TAG
        mkdir -p $OUT
        [ -f $OUT/result ] && { echo "[$TAG $STACK] cached" >> $HEARTBEAT; continue; }
        echo "[$TAG $STACK] start $(date -u)" >> $HEARTBEAT
        kill_all; ensure_dirs
        local T0=$(date -u +%s)
        local port=$((44000 + RANDOM % 1000))
        local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
          export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
          export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
          export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
          export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1"
        local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
        local NR=1
        for ip in "${WORKER_LIST[@]}"; do
          ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
            "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/model_extension/train_llama_e2e_$amp.py $be 2000" \
            > $OUT/w${NR}.log 2>&1 &
          NR=$((NR+1))
        done
        eval "$ENV_VARS"; cd $REPO
        timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/experiments/model_extension/train_llama_e2e_$amp.py $be 2000 > $OUT/m0.log 2>&1
        local RC=$?; wait
        local T1=$(date -u +%s)
        echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
        collect "$OUT"
        echo "[$TAG $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
      done
    done
  done
}

# =================== PHASE 5: h7_bench per stack ===================
phase5_h7bench() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R19/$STACK/h7_bench
    mkdir -p $OUT
    [ -f $OUT/result ] && { echo "[h7bench $STACK] cached" >> $HEARTBEAT; continue; }
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R19/$STACK; fi
    rsync_repo_to_workers
    echo "[h7bench $STACK] start $(date -u)" >> $HEARTBEAT
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    for prob in a2av ua2a rkv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar grad_ar; do
      local port=$((46000 + RANDOM % 1000))
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
        export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
      local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
      local SCRIPT=experiments/h7_bench/bench_${prob}.py
      # v3 baseline path for a2av/ua2a
      if [ "$prob" = "a2av" ] || [ "$prob" = "ua2a" ]; then SCRIPT=experiments/h7_bench/bench_${prob}_v3.py; fi
      local NR=1
      for ip in "${WORKER_LIST[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT" \
          > $OUT/${prob}_w${NR}.log 2>&1 &
        NR=$((NR+1))
      done
      eval "$ENV_VARS"; cd $REPO
      timeout 1500 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT > $OUT/${prob}_m0.log 2>&1
      wait
      for ip in "${WORKER_LIST[@]}"; do
        rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include="${prob}_*.json" --exclude='*' \
          "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/" 2>/dev/null &
      done
      rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
      wait
      echo "  [h7bench $STACK/$prob] $(date -u)" >> $HEARTBEAT
    done
    local T1=$(date -u +%s)
    echo "dur=$((T1-T0))s" > $OUT/result
    echo "[h7bench $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

# =================== ORCHESTRATION ===================

phase1_search
phase3_olmoe
phase4_llama
phase5_h7bench

echo "R19 DONE $(date -u)" >> $HEARTBEAT
