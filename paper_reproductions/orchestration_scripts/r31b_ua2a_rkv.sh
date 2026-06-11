#!/bin/bash
# r31b: faithfully re-probe ua2a and ring_kv with force-deployed no-sim
# Phase-3 first-enumeration agent code (instead of Phase-5's baseline-seed
# fallback that was deployed in r28_nosim).
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R31B=/home/ubuntu/r31b_ua2a_rkv
mkdir -p $R31B

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R31B/HEARTBEAT; }

backup_main_runtimes() { mkdir -p $R31B/runtime_main_backup; cp $REPO/runtime/trainium_*.py $R31B/runtime_main_backup/ 2>/dev/null; }
restore_main_runtimes() { cp $R31B/runtime_main_backup/trainium_*.py $REPO/runtime/ 2>/dev/null; }

deploy_forced_nosim() {
  cp /home/ubuntu/trainium_uniform_a2a_7node_nosim_forced.py $REPO/runtime/trainium_uniform_a2a_7node.py
  cp /home/ubuntu/trainium_uniform_a2a_7node_nosim_forced.py $REPO/runtime/trainium_uniform_a2a.py
  cp /home/ubuntu/trainium_ring_kv_7node_nosim_forced.py $REPO/runtime/trainium_ring_kv_7node.py
  cp /home/ubuntu/trainium_ring_kv_7node_nosim_forced.py $REPO/runtime/trainium_ring_kv.py
}

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
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null; sudo pkill -9 -u ubuntu -f neuronx-cc 2>/dev/null; find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  sleep 30
}

run() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R31B/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r31b_launch_${PORT}.sh
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

HB "R31b-ua2a-rkv start (force-deploy no-sim Phase-3 first-enumeration code)"
backup_main_runtimes
deploy_forced_nosim
clean_disk
kill_all
sleep 60
HB "wait done"

run "ua2a_baseline" 29711 1500 training/train_uniform_a2a_7node.py --backend baseline --steps 150 --warmup 5
run "ua2a_agent"    29712 1500 training/train_uniform_a2a_7node.py --backend agent    --steps 150 --warmup 5
run "ring_kv_baseline" 29713 1500 training/train_ring_kv_7node.py   --backend baseline --steps 150 --warmup 5
run "ring_kv_agent"    29714 1500 training/train_ring_kv_7node.py   --backend agent    --steps 150 --warmup 5

HB "restoring main runtimes..."
restore_main_runtimes
rsync_repo
clean_disk
HB "R31b-ua2a-rkv done"
