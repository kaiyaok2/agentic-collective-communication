#!/bin/bash
# r34b: re-run strategy-enumerate search for ua2a (print bug fixed) + measure agent
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R34B=/home/ubuntu/r34b_ua2a_strat
MAIN_BACKUP=/home/ubuntu/r34_ua2a_strat/runtime_main_backup
mkdir -p $R34B

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R34B/HEARTBEAT; }
restore_main() { cp $MAIN_BACKUP/trainium_uniform_a2a*.py $REPO/runtime/; }

rsync_repo() {
  for ip in "${WORKERS[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" >/dev/null 2>&1 &
  done
  wait
}

clean_disk() {
  cd /tmp && rm -rf tmp* ubuntu/neuroncc_compile_workdir/* neuron-core-dump/* 2>/dev/null; cd /
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "cd /tmp && sudo rm -rf tmp* ubuntu/neuroncc_compile_workdir/* neuron-core-dump/* 2>/dev/null" &
  done
  wait
}

kill_all() {
  pkill -9 -f torchrun 2>/dev/null || true
  pkill -9 -f neuronx-cc 2>/dev/null || true
  pkill -9 -f run_search 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null; sudo pkill -9 -u ubuntu -f neuronx-cc 2>/dev/null; find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  sleep 30
}

run_training() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R34B/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r34b_launch_${PORT}.sh
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
      "mkdir -p $OUT && chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
  done
  sleep 3
  cd $REPO
  source $VENV/bin/activate
  timeout $TIMEOUT $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $CMD > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R34b start"

# Stage 1: search (with simulator + print fix)
HB "STAGE 1: strategy-enumerate search (with simulator) for uniform_a2a"
WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
cd $REPO
source $VENV/bin/activate
mkdir -p $R34B/search_logs
timeout 2400 python3 -u experiments/run_search.py \
    --problem uniform_a2a --phase3-style strategy-enumerate --hw-eval \
    --llm-model sonnet --max-rounds 4 \
    --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
    --output-dir $R34B/search_logs > $R34B/search_logs/search.log 2>&1
EX=$?
HB "  search exit=$EX"

mkdir -p $R34B/deployed_runtimes
cp $REPO/runtime/trainium_uniform_a2a*.py $R34B/deployed_runtimes/ 2>/dev/null
HB "deployed runtime summary"
for f in $R34B/deployed_runtimes/trainium_*.py; do
  desc=$(grep -m1 '"""' $f | head -1 | tr -d '"')
  HB "  $(basename $f): $desc"
done

# Stage 2: measure agent only (baseline already done in r34: 2841 ms)
clean_disk
kill_all
sleep 30

run_training "ua2a_agent" 29881 1500 training/train_uniform_a2a_7node.py --backend agent --steps 150 --warmup 10

# Stage 3: restore main
HB "restoring main..."
restore_main
rsync_repo
clean_disk
HB "R34b done"
