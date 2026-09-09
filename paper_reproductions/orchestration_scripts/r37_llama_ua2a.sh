#!/bin/bash
# r37: (a) Rerun Llama-7B per_mb + bundled at 7-node with dxe removed from the script.
#      (b) Toy ua2a training at 7-node (small shapes) to fill in tab:perproblem ‡.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R37=/home/ubuntu/r37_llama_ua2a
mkdir -p $R37

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R37/HEARTBEAT; }

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
  sleep 20
}

run_training() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R37/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r37_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT /tmp/tp_search
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
      "mkdir -p $OUT /tmp/tp_search && chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
  done
  sleep 3
  cd $REPO
  source $VENV/bin/activate
  mkdir -p /tmp/tp_search
  timeout $TIMEOUT $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $CMD > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R37 start"

# --- Part A: Llama-7B no-dxe at 7-node ---
HB "PART A: Llama-7B per_mb + bundled (dxe removed from script)"
run_training "llama7b_per_mb"  29950 1500 experiments/model_extension/train_llama_e2e_7b.py per_mb  200
run_training "llama7b_bundled" 29951 1500 experiments/model_extension/train_llama_e2e_7b.py bundled 200

# --- Part B: toy ua2a 7-node training ---
HB "PART B: toy ua2a 7-node training"
run_training "toy_ua2a_baseline" 29960 1200 training/train_uniform_a2a_toy_7node.py --backend baseline --steps 150 --warmup 10
run_training "toy_ua2a_agent"    29961 1200 training/train_uniform_a2a_toy_7node.py --backend agent    --steps 150 --warmup 10

HB "R37 done"
