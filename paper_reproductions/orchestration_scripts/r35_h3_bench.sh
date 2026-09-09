#!/bin/bash
# r35: 3-node generalization microbenchmarks for the 8 primitives.
# Uses 3 nodes (MASTER + 2 workers = 96 ranks) and runs h7-style benches.
# The runtime files' _world_size is set at init() from xr.world_size() so
# they auto-adapt; _NUM_DEVICES/_NUM_NODES are passed for diagnostics only.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
# Use only 2 workers (master + 2 = 3 nodes = 96 ranks)
WORKERS=(172.31.17.80 172.31.24.136)
R35=/home/ubuntu/r35_h3_bench
mkdir -p $R35

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R35/HEARTBEAT; }

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

run_bench() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R35/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r35_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT /tmp/h7_bench
cd $REPO
exec $TORCHRUN --nnodes=3 --node_rank=__NR__ --nproc_per_node=32 \\
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
      "mkdir -p $OUT /tmp/h7_bench && chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
  done
  sleep 3
  cd $REPO
  source $VENV/bin/activate
  mkdir -p /tmp/h7_bench
  timeout $TIMEOUT $TORCHRUN --nnodes=3 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $CMD > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R35-h3-bench start (3 nodes = 96 ranks)"

# 8 primitives via h7_bench scripts. v6/v7 are the most recent SOTA versions.
run_bench "alltoallv"     29900 600 experiments/h7_bench/bench_a2av_v6.py
run_bench "dxe"           29901 600 experiments/h7_bench/bench_dxe.py
run_bench "uniform_a2a"   29902 600 experiments/h7_bench/bench_ua2a_v6.py
run_bench "ring_kv"       29903 600 experiments/h7_bench/bench_ring_kv_v6.py
run_bench "fsdp_prefetch" 29904 600 experiments/h7_bench/bench_fsdp_prefetch_v7.py
run_bench "tp_mlp"        29905 600 experiments/h7_bench/bench_tp_mlp_v6.py
run_bench "pp_send_recv"  29906 600 experiments/h7_bench/bench_pp_send_recv_v7.py
run_bench "llama_block_ar" 29907 600 experiments/h7_bench/bench_llama_block_ar_v6.py

HB "R35-h3-bench done"
