#!/bin/bash
# R22 — late-phase .item() probes + per-problem training scripts + Llama amp1/4 + UA2A search rerun
# Imports per-stack runtime_per_problem from R21 for all problems EXCEPT ua2a (which is re-searched).
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

R21=/home/ubuntu/r21
R22=/home/ubuntu/r22
HEARTBEAT=$R22/HEARTBEAT
STATE=$R22/STATE
mkdir -p $R22 $STATE
echo "R22 START $(date -u)" >> $HEARTBEAT

STACKS=(baseline strategy-enumerate cc-react multi-island)
AGENT_STYLES=(strategy-enumerate cc-react multi-island)
PER_PROB_TRAINS=(tp_mlp fsdp_prefetch llama_block_ar pp_send_recv)
LLAMA_AMPS=(amp1 amp2 amp3 amp4)

rsync_repo() {
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
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search /tmp/h7_bench; rm -f /tmp/h7_bench/ua2a*.json /tmp/tp_search/*.json" &
  done
  rm -f /tmp/h7_bench/ua2a*.json /tmp/tp_search/*.json 2>/dev/null
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

# Bootstrap: import R21 runtime_per_problem (all 9 problems) for all 3 agent stacks
bootstrap_from_r21() {
  for STYLE in "${AGENT_STYLES[@]}"; do
    [ -d $R22/$STYLE/runtime_per_problem ] && continue
    mkdir -p $R22/$STYLE
    cp -r $R21/$STYLE/runtime_per_problem $R22/$STYLE/
  done
}

# Phase 1: UA2A search × 3 styles (user request: only ua2a needs re-search; rest reused from R21)
phase1_ua2a_search() {
  [ -f $STATE/phase1_done ] && { echo "PHASE 1 cached" >> $HEARTBEAT; return; }
  echo "PHASE 1: UA2A search x 3 styles $(date -u)" >> $HEARTBEAT
  for STYLE in "${AGENT_STYLES[@]}"; do
    local OUT=$R22/$STYLE/searches/uniform_a2a
    [ -f $OUT/done ] && continue
    mkdir -p $OUT
    cd $REPO && git checkout main -- runtime/ 2>/dev/null
    git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
    rsync_repo
    local T0=$(date -u +%s)
    SEARCH_TAG="$STYLE/uniform_a2a" TOKEN_LOG=/tmp/r22_token_usage.jsonl \
      timeout 2400 $PY $REPO/experiments/run_search.py \
      --problem uniform_a2a --phase3-style $STYLE \
      --max-rounds 4 --num-nodes 7 \
      --master-addr $MASTER --worker-addrs $WORKERS \
      --output-dir $OUT > $OUT/run.log 2>&1
    local RC=$?
    local T1=$(date -u +%s)
    mkdir -p $R22/$STYLE/runtime_per_problem/uniform_a2a
    cp $REPO/runtime/trainium_uniform_a2a*.py $R22/$STYLE/runtime_per_problem/uniform_a2a/ 2>/dev/null
    echo "RC=$RC dur=$((T1-T0))s" > $OUT/done
    echo "[search $STYLE/ua2a] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
  touch $STATE/phase1_done
}

# Phase 2: OLMoE training x 4 stacks with LATE-phase .item() probes
phase2_olmoe() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R22/$STACK/training/olmoe_default
    mkdir -p $OUT
    [ -f $OUT/result ] && continue
    if [ "$STACK" = "baseline" ]; then
      deploy_baseline; local FLAGS="--backend baseline --grad-sync baseline --ce baseline"
    else
      deploy_agent_picks $R22/$STACK; local FLAGS="--backend agent --grad-sync agent --ce agent"
    fi
    rsync_repo
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    local port=$((42000 + RANDOM % 1000))
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
      export PERCALL_PROBE=1 && export PROBE_START_STEP=170 && export PROBE_END_STEP=250"
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
    # 250 steps total, probe activates from step 170-250 (80 samples). Warmup 30.
    local ARGS="$FLAGS --steps 250 --warmup 30"
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

# Phase 3: per-problem training (tp_mlp, fsdp_prefetch, llama_block_ar, pp_send_recv) x 4 stacks
phase3_per_problem() {
  for STACK in "${STACKS[@]}"; do
    for prob in "${PER_PROB_TRAINS[@]}"; do
      local OUT=$R22/$STACK/training/${prob}_7node
      mkdir -p $OUT
      [ -f $OUT/result ] && continue
      if [ "$STACK" = "baseline" ]; then
        deploy_baseline; local BACKEND=baseline
      else
        deploy_agent_picks $R22/$STACK; local BACKEND=agent
      fi
      rsync_repo
      kill_all; ensure_dirs
      local T0=$(date -u +%s)
      local port=$((43000 + RANDOM % 1000))
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
        export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
        export PERCALL_PROBE=1 && export PROBE_START_STEP=170 && export PROBE_END_STEP=250"
      local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
      local ARGS="--backend $BACKEND --steps 250 --warmup 30"
      local NR=1
      for ip in "${WORKER_LIST[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/training/train_${prob}_7node.py $ARGS" \
          > $OUT/w${NR}.log 2>&1 &
        NR=$((NR+1))
      done
      eval "$ENV_VARS"; cd $REPO
      timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/training/train_${prob}_7node.py $ARGS > $OUT/m0.log 2>&1
      local RC=$?; wait
      local T1=$(date -u +%s)
      echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
      collect "$OUT"
      echo "[$prob $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
    done
  done
}

# Phase 4: ua2a + rkv 7n training x 4 stacks
phase4_ua2a_rkv() {
  for STACK in "${STACKS[@]}"; do
    for prob in uniform_a2a ring_kv; do
      local TAG=${prob}_7node
      local OUT=$R22/$STACK/training/$TAG
      mkdir -p $OUT
      [ -f $OUT/result ] && continue
      if [ "$STACK" = "baseline" ]; then
        deploy_baseline; local BACKEND_ARG=baseline
      else
        deploy_agent_picks $R22/$STACK
        if [ "$prob" = "ring_kv" ]; then local BACKEND_ARG=evolved; else local BACKEND_ARG=agent; fi
      fi
      rsync_repo
      kill_all; ensure_dirs
      local T0=$(date -u +%s)
      local port=$((44000 + RANDOM % 1000))
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
        export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
        export PERCALL_PROBE=1 && export PROBE_START_STEP=170 && export PROBE_END_STEP=250"
      local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
      local SCRIPT=training/train_${prob}_7node.py
      [ "$prob" = "ring_kv" ] && SCRIPT=training/train_ring_kv.py
      local ARGS="--backend $BACKEND_ARG --steps 250 --warmup 30"
      local NR=1
      for ip in "${WORKER_LIST[@]}"; do
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/$SCRIPT $ARGS" \
          > $OUT/w${NR}.log 2>&1 &
        NR=$((NR+1))
      done
      eval "$ENV_VARS"; cd $REPO
      timeout 3600 $TORCHRUN $TRUN --node_rank=0 $REPO/$SCRIPT $ARGS > $OUT/m0.log 2>&1
      local RC=$?; wait
      local T1=$(date -u +%s)
      echo "RC=$RC dur=$((T1-T0))s" > $OUT/result
      collect "$OUT"
      echo "[$prob $STACK] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
    done
  done
}

# Phase 5: Llama amp1/2/3/4 x 4 stacks x 2 backends (per_mb/bundled), 2000 steps
phase5_llama() {
  for STACK in "${STACKS[@]}"; do
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
    rsync_repo
    for amp in "${LLAMA_AMPS[@]}"; do
      for be in per_mb bundled; do
        local TAG="llama_${amp}_${be}"
        local OUT=$R22/$STACK/training/$TAG
        mkdir -p $OUT
        [ -f $OUT/result ] && continue
        kill_all; ensure_dirs
        local T0=$(date -u +%s)
        local port=$((45000 + RANDOM % 1000))
        local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
          export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
          export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
          export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
          export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export PERCALL_PROBE=1 && \
          export PROBE_START_STEP=1700 && export PROBE_END_STEP=2000"
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

# Phase 6: ua2a v4 bench × 4 stacks (other problems' bench reused from R21)
phase6_ua2a_bench() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R22/$STACK/h7_bench_v4
    mkdir -p $OUT
    [ -f $OUT/result ] && continue
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R22/$STACK; fi
    rsync_repo
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    local port=$((47000 + RANDOM % 1000))
    local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
      export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
      export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
      export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
      export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
    local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
    local NR=1
    for ip in "${WORKER_LIST[@]}"; do
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR $REPO/experiments/h7_bench/bench_ua2a_v4.py" \
        > $OUT/ua2a_w${NR}.log 2>&1 &
      NR=$((NR+1))
    done
    eval "$ENV_VARS"; cd $REPO
    timeout 1200 $TORCHRUN $TRUN --node_rank=0 $REPO/experiments/h7_bench/bench_ua2a_v4.py > $OUT/ua2a_m0.log 2>&1
    wait
    for ip in "${WORKER_LIST[@]}"; do
      rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include="ua2a_v4*.json" --exclude='*' \
        "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/" 2>/dev/null &
    done
    rsync -a /tmp/h7_bench/ua2a_v4*.json $OUT/ 2>/dev/null || true
    wait
    local T1=$(date -u +%s)
    echo "dur=$((T1-T0))s" > $OUT/result
    echo "[ua2a_v4 $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

bootstrap_from_r21
phase1_ua2a_search
phase2_olmoe
phase3_per_problem
phase4_ua2a_rkv
phase5_llama
phase6_ua2a_bench

echo "R22 DONE $(date -u)" >> $HEARTBEAT
