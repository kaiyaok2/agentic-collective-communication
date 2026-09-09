#!/bin/bash
# R28c: extra benches — cc-react + multi-island ua2a 7n bench for Table 2 consistency
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R28C=/home/ubuntu/r28c
mkdir -p $R28C

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R28C/HEARTBEAT; }

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
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  sleep 3
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
  local STACK=$1; local PORT=$2
  local OUT=$R28C/$STACK/h7_bench_v6
  HB "ua2a 7n bench $STACK"
  kill_all
  deploy_stack $STACK
  mkdir -p $OUT
  local LSH=/tmp/r28c_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
source $VENV/bin/activate
mkdir -p $OUT
cd $REPO
exec $TORCHRUN --nnodes=7 --node_rank=__NR__ --nproc_per_node=32 \\
  --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \\
  experiments/h7_bench/bench_ua2a_v6.py > $OUT/node___NR__.log 2>&1
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
  timeout 600 $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    experiments/h7_bench/bench_ua2a_v6.py > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $STACK exit=$EX"
  grep -E "bench.*ua2a" $OUT/node_0.log | grep -v Deprec | tail -10 > $OUT/summary.txt
  cp /tmp/h7_bench/ua2a_*.json $OUT/ 2>/dev/null
  cat $OUT/summary.txt
}

HB "R28c start"
run_bench cc-react 29610
run_bench multi-island 29611
HB "R28c done"
