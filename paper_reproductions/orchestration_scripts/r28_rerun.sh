#!/bin/bash
# R28: rerun blanks in Table 2 + ua2a 7n bench under strat-enum
# Items:
#  1) ua2a 7n bench strategy-enumerate (fishy 1350.20 ms)
#  2) Uniform A2A per-problem training (3 stacks; try smaller payload)
#  3) Ring KV per-problem training (strategy-enumerate, cc-react retries)
#  4) Layer-block AR strategy-enumerate per-problem training
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R28=/home/ubuntu/r28
mkdir -p $R28

H() { echo "[$(date -u +%H:%M:%S)] $*"; }
HB() { echo "$(date -u +%H:%M:%S) $*" >> $R28/HEARTBEAT; }

rsync_repo() {
  for ip in "${WORKERS[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" >/dev/null 2>&1 &
  done
  wait
}

kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null; find /home/ubuntu/neuron_cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  sleep 5
}

purge_locks() {
  find /home/ubuntu/neuron_cache -name '*.lock' -delete 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "find /home/ubuntu/neuron_cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
}

deploy_stack() {
  local STACK=$1
  local SRC=/home/ubuntu/r22/$STACK/runtime_per_problem
  for prob_dir in $SRC/*/; do
    for f in "$prob_dir"trainium_*_7node.py "$prob_dir"trainium_*[!7node].py; do
      [ -f "$f" ] && cp "$f" $REPO/runtime/$(basename $f) 2>/dev/null
    done
  done
  rsync_repo
}

launch_workers() {
  local OUTDIR=$1; shift
  local CMD=$@
  for i in "${!WORKERS[@]}"; do
    local NODE_RANK=$((i+1))
    local ip=${WORKERS[$i]}
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$ip \
      "cd $REPO && nohup bash -c 'source $VENV/bin/activate && NODE_RANK=$NODE_RANK NEURON_RT_ROOT_COMM_ID=$MASTER:$LB_PORT NEURON_PJRT_PROCESSES_NUM_DEVICES=32,32,32,32,32,32,32 $CMD' > $OUTDIR/node_${NODE_RANK}.log 2>&1 &" \
      &
  done
  wait
}

# =========================================================
# JOB 1: ua2a 7n bench rerun under strategy-enumerate
# =========================================================
job_ua2a_bench() {
  HB "JOB1 ua2a 7n bench strategy-enumerate"
  local STACK=strategy-enumerate
  local OUT=$R28/$STACK/h7_bench_v6
  mkdir -p $OUT
  kill_all
  purge_locks
  deploy_stack $STACK
  LB_PORT=29500
  H "  launching ua2a v6 7n bench"
  # workers
  for i in "${!WORKERS[@]}"; do
    local NR=$((i+1))
    local ip=${WORKERS[$i]}
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "cd $REPO && nohup $TORCHRUN --nnodes=7 --node_rank=$NR --nproc_per_node=32 --rdzv_backend=static --master_addr=$MASTER --master_port=29500 experiments/h7_bench/bench_ua2a_v6.py > $OUT/node_${NR}.log 2>&1 &" \
      &
  done
  wait
  # master
  cd $REPO
  timeout 600 $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=29500 \
    experiments/h7_bench/bench_ua2a_v6.py > $OUT/node_0.log 2>&1
  local EX=$?
  HB "JOB1 ua2a 7n bench exit=$EX"
  grep -E "warm_med|ua2a|baseline|agent" $OUT/node_0.log | tail -20 > $OUT/summary.txt
}

# =========================================================
# JOB 2: ua2a per-problem training (3 stacks; small payload)
# =========================================================
job_ua2a_training() {
  HB "JOB2 ua2a per-problem training"
  for STACK in strategy-enumerate cc-react multi-island; do
    HB "  ua2a training stack=$STACK"
    kill_all
    purge_locks
    deploy_stack $STACK
    for BE in baseline agent; do
      local OUT=$R28/$STACK/training/ua2a/$BE
      mkdir -p $OUT
      # Train with smaller payload: lower CAP/chunk to fit
      for i in "${!WORKERS[@]}"; do
        local NR=$((i+1))
        local ip=${WORKERS[$i]}
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "cd $REPO && nohup $TORCHRUN --nnodes=7 --node_rank=$NR --nproc_per_node=32 --rdzv_backend=static --master_addr=$MASTER --master_port=29501 training/train_uniform_a2a_7node.py --backend $BE --steps 220 --warmup 20 > $OUT/node_${NR}.log 2>&1 &" \
          &
      done
      wait
      cd $REPO
      timeout 900 $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
        --rdzv_backend=static --master_addr=$MASTER --master_port=29501 \
        training/train_uniform_a2a_7node.py --backend $BE --steps 220 --warmup 20 \
        > $OUT/node_0.log 2>&1
      local EX=$?
      HB "  ua2a $STACK $BE exit=$EX"
      grep -E "steady|percall" $OUT/node_0.log | tail -10 > $OUT/summary.txt
    done
  done
}

# =========================================================
# JOB 3: ring_kv per-problem training retries
# =========================================================
job_ring_kv_training() {
  HB "JOB3 ring_kv per-problem training retries"
  for STACK in strategy-enumerate cc-react; do
    HB "  ring_kv training stack=$STACK"
    kill_all
    purge_locks
    deploy_stack $STACK
    for BE in baseline agent; do
      local OUT=$R28/$STACK/training/ring_kv/$BE
      mkdir -p $OUT
      for i in "${!WORKERS[@]}"; do
        local NR=$((i+1))
        local ip=${WORKERS[$i]}
        ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
          "cd $REPO && nohup $TORCHRUN --nnodes=7 --node_rank=$NR --nproc_per_node=32 --rdzv_backend=static --master_addr=$MASTER --master_port=29502 training/train_ring_kv_7node.py --backend $BE --steps 220 --warmup 20 > $OUT/node_${NR}.log 2>&1 &" \
          &
      done
      wait
      cd $REPO
      timeout 600 $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
        --rdzv_backend=static --master_addr=$MASTER --master_port=29502 \
        training/train_ring_kv_7node.py --backend $BE --steps 220 --warmup 20 \
        > $OUT/node_0.log 2>&1
      local EX=$?
      HB "  ring_kv $STACK $BE exit=$EX"
      grep -E "steady|ring_kv" $OUT/node_0.log | tail -10 > $OUT/summary.txt
    done
  done
}

# =========================================================
# JOB 4: llama_block_ar strategy-enumerate training
# =========================================================
job_lbar_strat_training() {
  HB "JOB4 llama_block_ar strategy-enumerate training"
  STACK=strategy-enumerate
  kill_all
  purge_locks
  deploy_stack $STACK
  for BE in baseline agent; do
    local OUT=$R28/$STACK/training/llama_block_ar/$BE
    mkdir -p $OUT
    for i in "${!WORKERS[@]}"; do
      local NR=$((i+1))
      local ip=${WORKERS[$i]}
      ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
        "cd $REPO && nohup $TORCHRUN --nnodes=7 --node_rank=$NR --nproc_per_node=32 --rdzv_backend=static --master_addr=$MASTER --master_port=29503 training/train_llama_block_ar_7node.py --backend $BE --steps 220 --warmup 20 > $OUT/node_${NR}.log 2>&1 &" \
        &
    done
    wait
    cd $REPO
    timeout 900 $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
      --rdzv_backend=static --master_addr=$MASTER --master_port=29503 \
      training/train_llama_block_ar_7node.py --backend $BE --steps 220 --warmup 20 \
      > $OUT/node_0.log 2>&1
    local EX=$?
    HB "  lbar $STACK $BE exit=$EX"
    grep -E "steady|block_ar" $OUT/node_0.log | tail -10 > $OUT/summary.txt
  done
}

HB "R28 start"
job_ua2a_bench
job_ua2a_training
job_ring_kv_training
job_lbar_strat_training
HB "R28 done"
