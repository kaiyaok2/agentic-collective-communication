#!/bin/bash
# R20 — full pipeline after strategy-enum validation passed all 9 problems.
# Phase 1b: cc-react + multi-island search (strategy-enum already done)
# Phase 2: OLMoE training per stack (4 stacks)
# Phase 3: Llama amp3/amp2 per stack
# Phase 4: h7_bench per stack
# Phase 5: n1_bench per stack (with v3 a2av/ua2a)
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

R20=/home/ubuntu/r20
HEARTBEAT=$R20/HEARTBEAT
STATE=$R20/STATE
mkdir -p $R20 $STATE
echo "R20 FULL PIPELINE START $(date -u)" >> $HEARTBEAT

PROBLEMS=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar grad_ar dxe ring_kv uniform_a2a alltoallv)
REMAINING_STYLES=(cc-react multi-island)
STACKS=(baseline strategy-enumerate cc-react multi-island)

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

# PHASE 1b: cc-react + multi-island search
phase1b_search() {
  [ -f $STATE/phase1b_done ] && { echo "PHASE 1b cached" >> $HEARTBEAT; return; }
  echo "PHASE 1b: cc-react + multi-island search $(date -u)" >> $HEARTBEAT
  for P in "${PROBLEMS[@]}"; do
    for STYLE in "${REMAINING_STYLES[@]}"; do
      local OUT=$R20/$STYLE/searches/$P
      [ -f $OUT/done ] && continue
      mkdir -p $OUT
      cd $REPO && git checkout main -- runtime/ 2>/dev/null
      git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
      rsync_repo
      local T0=$(date -u +%s)
      timeout 2400 $PY $REPO/experiments/run_search.py \
        --problem $P --phase3-style $STYLE \
        --max-rounds 4 --num-nodes 7 \
        --master-addr $MASTER --worker-addrs $WORKERS \
        --output-dir $OUT > $OUT/run.log 2>&1
      local RC=$?
      local T1=$(date -u +%s)
      mkdir -p $R20/$STYLE/runtime_per_problem/$P
      cp $REPO/runtime/trainium_${P}*.py $R20/$STYLE/runtime_per_problem/$P/ 2>/dev/null
      echo "RC=$RC dur=$((T1-T0))s" > $OUT/done
      echo "[search $STYLE/$P] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
    done
  done
  touch $STATE/phase1b_done
}

# PHASE 2: OLMoE training per stack
phase2_olmoe() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R20/$STACK/training/olmoe_default
    mkdir -p $OUT
    [ -f $OUT/result ] && continue
    if [ "$STACK" = "baseline" ]; then
      deploy_baseline; local FLAGS="--backend baseline --grad-sync baseline --ce baseline"
    else
      deploy_agent_picks $R20/$STACK; local FLAGS="--backend agent --grad-sync agent --ce agent"
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

# PHASE 3: Llama amp3/amp2
phase3_llama() {
  for STACK in "${STACKS[@]}"; do
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R20/$STACK; fi
    rsync_repo
    for amp in amp3 amp2; do
      for be in per_mb bundled; do
        local TAG="llama_${amp}_${be}"
        local OUT=$R20/$STACK/training/$TAG
        mkdir -p $OUT
        [ -f $OUT/result ] && continue
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

# PHASE 4: h7_bench per stack
phase4_h7bench() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R20/$STACK/h7_bench
    mkdir -p $OUT
    [ -f $OUT/result ] && continue
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R20/$STACK; fi
    rsync_repo
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
    done
    local T1=$(date -u +%s)
    echo "dur=$((T1-T0))s" > $OUT/result
    echo "[h7bench $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

# PHASE 5: n1_bench per stack
phase5_n1bench() {
  for STACK in "${STACKS[@]}"; do
    local OUT=$R20/$STACK/n1_bench
    mkdir -p $OUT
    [ -f $OUT/result ] && continue
    if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_agent_picks $R20/$STACK; fi
    rsync_repo
    kill_all; ensure_dirs
    local T0=$(date -u +%s)
    for prob in a2av ua2a rkv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar grad_ar; do
      local port=$((51000 + RANDOM % 5000))
      local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
        export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
        export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
        export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
      local SCRIPT=experiments/h7_bench/bench_${prob}.py
      if [ "$prob" = "a2av" ] || [ "$prob" = "ua2a" ]; then SCRIPT=experiments/h7_bench/bench_${prob}_v3.py; fi
      timeout 600 $TORCHRUN --nproc_per_node=32 --nnodes=1 --master_addr=$MASTER --master_port=$port \
        $REPO/$SCRIPT > $OUT/${prob}_m0.log 2>&1
      rsync -a /tmp/h7_bench/${prob}_*.json $OUT/ 2>/dev/null || true
    done
    local T1=$(date -u +%s)
    echo "dur=$((T1-T0))s" > $OUT/result
    echo "[n1bench $STACK] done dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
  done
}

phase1b_search
phase2_olmoe
phase3_llama
phase4_h7bench
phase5_n1bench

echo "R20 FULL PIPELINE DONE $(date -u)" >> $HEARTBEAT
