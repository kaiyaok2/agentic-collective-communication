#!/bin/bash
# R28e: lbar cc-react + multi-island (strat-enum done in r28b).
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R28E=/home/ubuntu/r28e
mkdir -p $R28E

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R28E/HEARTBEAT; }

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
  pkill -9 -f neuronx-cc 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null; sudo pkill -9 -u ubuntu -f neuronx-cc 2>/dev/null; find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  sleep 30
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

run_bench() {
  local STACK=$1; local BE=$2; local PORT=$3
  local OUT=$R28E/$STACK/training/llama_block_ar/$BE
  HB "lbar $STACK $BE start"
  kill_all
  deploy_stack $STACK
  local LSH=/tmp/r28e_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT
cd $REPO
exec $TORCHRUN --nnodes=7 --node_rank=__NR__ --nproc_per_node=32 \\
  --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \\
  training/train_llama_block_ar_7node.py --backend $BE --steps 200 --warmup 10 > $OUT/node___NR__.log 2>&1
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
  mkdir -p $OUT
  cd $REPO
  source $VENV/bin/activate
  timeout 1200 $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    training/train_llama_block_ar_7node.py --backend $BE --steps 200 --warmup 10 \
    > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $STACK $BE exit=$EX"
  cp /home/ubuntu/agentic-collective-communication/training/results/percall_r22/llama_block_ar_${BE}_step.json $OUT/ 2>/dev/null
}

HB "R28e start"
HB "initial Neuron driver wait..."
kill_all
sleep 60
HB "wait done"
for STACK in cc-react multi-island; do
  for BE in baseline agent; do
    run_bench $STACK $BE 29603
  done
done
HB "R28e done"
