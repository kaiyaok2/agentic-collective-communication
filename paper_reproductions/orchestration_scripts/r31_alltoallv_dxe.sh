#!/bin/bash
# r31: measure no-sim alltoallv and dxe at training scope (4 runs).
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
NOSIM_REPO=/tmp/r28_smoke/acc-r28
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R31=/home/ubuntu/r31_a2av_dxe
mkdir -p $R31

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R31/HEARTBEAT; }

backup_main_runtimes() { mkdir -p $R31/runtime_main_backup; cp $REPO/runtime/trainium_*.py $R31/runtime_main_backup/ 2>/dev/null; }
deploy_nosim_runtimes() { for f in $NOSIM_REPO/runtime/trainium_*.py; do [ -f "$f" ] && cp "$f" $REPO/runtime/$(basename "$f"); done; }
restore_main_runtimes() { cp $R31/runtime_main_backup/trainium_*.py $REPO/runtime/ 2>/dev/null; }

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
  local OUT=$R31/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r31_launch_${PORT}.sh
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

HB "R31-a2av-dxe start"
backup_main_runtimes
deploy_nosim_runtimes
clean_disk
kill_all
sleep 60
HB "wait done"

run "alltoallv_baseline" 29701 1500 training/train_alltoallv_7node.py --backend baseline --steps 150 --warmup 5
run "alltoallv_agent"    29702 1500 training/train_alltoallv_7node.py --backend agent    --steps 150 --warmup 5
run "dxe_baseline"       29703 1500 training/train_dxe_7node.py        --backend baseline --steps 150 --warmup 5
run "dxe_agent"          29704 1500 training/train_dxe_7node.py        --backend agent    --steps 150 --warmup 5

HB "restoring..."
restore_main_runtimes
rsync_repo
clean_disk
HB "R31-a2av-dxe done"
