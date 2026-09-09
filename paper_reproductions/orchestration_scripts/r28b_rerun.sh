#!/bin/bash
# R28b: rerun blanks in Table 2 (fixed: create OUT dirs on workers too)
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

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R28/HEARTBEAT; }

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
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null; find /home/ubuntu/neuron_cache -name '*.lock' -delete 2>/dev/null; find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null; find /var/tmp/neuron-compile-cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  # wait for Neuron driver to release cores
  sleep 30
}

mkdir_all() {
  local D=$1
  mkdir -p $D
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "mkdir -p $D" >/dev/null 2>&1 &
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

run_torchrun_7n() {
  # Args: PORT OUT_DIR TIMEOUT_S MASTER_CMD
  local PORT=$1; local OUT=$2; local TO=$3; shift 3
  local CMD="$@"
  # Stage launcher script on each worker so ssh -f exits immediately.
  local LSH=/tmp/r28_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT
cd $REPO
exec $TORCHRUN --nnodes=7 --node_rank=__NR__ --nproc_per_node=32 \\
  --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \\
  $CMD > $OUT/node___NR__.log 2>&1
EOF
  for i in "${!WORKERS[@]}"; do
    local NR=$((i+1))
    local ip=${WORKERS[$i]}
    sed "s/__NR__/$NR/g" $LSH > $LSH.$NR
    scp -i $KEY -o StrictHostKeyChecking=no -q $LSH.$NR ubuntu@$ip:$LSH &
  done
  wait
  for i in "${!WORKERS[@]}"; do
    local NR=$((i+1))
    local ip=${WORKERS[$i]}
    ssh -fn -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$ip \
      "chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
  done
  sleep 3
  cd $REPO
  source $VENV/bin/activate
  export NEURON_RT_NUM_SOFTWARE_NQS=8
  export NEURON_RT_EXEC_TIMEOUT=300
  timeout $TO $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $CMD > $OUT/node_0.log 2>&1
  return $?
}

# =====================================
# JOB 1: ua2a 7n bench strategy-enumerate
# =====================================
job_ua2a_bench() {
  HB "JOB1 ua2a 7n bench strategy-enumerate"
  STACK=strategy-enumerate
  OUT=$R28/$STACK/h7_bench_v6
  kill_all
  deploy_stack $STACK
  mkdir_all $OUT
  run_torchrun_7n 29500 $OUT 600 experiments/h7_bench/bench_ua2a_v6.py
  EX=$?
  HB "  JOB1 exit=$EX"
  grep -E "warm_med|ua2a|baseline|agent|Error|Traceback" $OUT/node_0.log | tail -25 > $OUT/summary.txt
  cat $OUT/summary.txt
}

# =====================================
# JOB 2: ua2a per-problem training (3 stacks)
# =====================================
job_ua2a_training() {
  HB "JOB2 ua2a per-problem training"
  for STACK in strategy-enumerate cc-react multi-island; do
    HB "  ua2a training stack=$STACK"
    kill_all
    deploy_stack $STACK
    for BE in baseline agent; do
      OUT=$R28/$STACK/training/ua2a/$BE
      mkdir_all $OUT
      run_torchrun_7n 29501 $OUT 1800 training/train_uniform_a2a_7node.py --backend $BE --steps 150 --warmup 10
      EX=$?
      HB "    $STACK $BE exit=$EX"
      grep -E "steady|Steady|Backend|Error|Traceback|SIGABRT" $OUT/node_0.log | tail -15 > $OUT/summary.txt
    done
  done
}

# =====================================
# JOB 3: ring_kv per-problem training (strat-enum + cc-react)
# =====================================
job_ring_kv_training() {
  HB "JOB3 ring_kv per-problem training retries"
  for STACK in strategy-enumerate cc-react; do
    HB "  ring_kv stack=$STACK"
    kill_all
    deploy_stack $STACK
    for BE in baseline agent; do
      OUT=$R28/$STACK/training/ring_kv/$BE
      mkdir_all $OUT
      run_torchrun_7n 29502 $OUT 900 training/train_ring_kv_7node.py --backend $BE --steps 200 --warmup 10
      EX=$?
      HB "    $STACK $BE exit=$EX"
      grep -E "steady|Backend|Error|Traceback|NRT" $OUT/node_0.log | tail -15 > $OUT/summary.txt
    done
  done
}

# =====================================
# JOB 4: llama_block_ar strategy-enumerate training
# =====================================
job_lbar_strat_training() {
  HB "JOB4 llama_block_ar strategy-enumerate training"
  STACK=strategy-enumerate
  kill_all
  deploy_stack $STACK
  for BE in baseline agent; do
    OUT=$R28/$STACK/training/llama_block_ar/$BE
    mkdir_all $OUT
    run_torchrun_7n 29503 $OUT 1200 training/train_llama_block_ar_7node.py --backend $BE --steps 200 --warmup 10
    EX=$?
    HB "    $BE exit=$EX"
    grep -E "steady|Backend|Error|Traceback|CCOM" $OUT/node_0.log | tail -15 > $OUT/summary.txt
  done
}

HB "R28b start"
SKIP_BENCH=${SKIP_BENCH:-0}
if [ "$SKIP_BENCH" != "1" ]; then
  job_ua2a_bench
fi
job_ua2a_training
job_ring_kv_training
job_lbar_strat_training
HB "R28b done"
