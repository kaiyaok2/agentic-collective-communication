#!/bin/bash
# Full no-sim measurement: per-problem (6 scripts × 2 backends) + Llama-7B e2e
# Uses NOSIM runtimes (deployed from /tmp/r28_smoke/acc-r28/runtime/) overlaid
# onto main repo. Restores main runtimes at end.
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
R30=/home/ubuntu/r30_nosim_full
mkdir -p $R30

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R30/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R30/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R30/runtime_main_backup/ 2>/dev/null
}

deploy_nosim_runtimes() {
  for f in $NOSIM_REPO/runtime/trainium_*.py; do
    [ -f "$f" ] && cp "$f" $REPO/runtime/$(basename "$f")
  done
}

restore_main_runtimes() {
  cp $R30/runtime_main_backup/trainium_*.py $REPO/runtime/ 2>/dev/null
}

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

run() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R30/$NAME
  HB "$NAME start"
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r30_launch_${PORT}.sh
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

HB "R30-nosim-full start"
backup_main_runtimes
deploy_nosim_runtimes
HB "initial wait..."
kill_all
sleep 60
HB "wait done"

# === Per-problem: 6 scripts (alltoallv and dxe live in OLMoE e2e) ===
run "tp_mlp_baseline"           29501 1500 training/train_tp_mlp_7node.py        --backend baseline --steps 150 --warmup 10
run "tp_mlp_agent"              29502 1500 training/train_tp_mlp_7node.py        --backend agent    --steps 150 --warmup 10
run "fsdp_baseline"             29503 1500 training/train_fsdp_prefetch_7node.py --backend baseline --steps 150 --warmup 10
run "fsdp_agent"                29504 1500 training/train_fsdp_prefetch_7node.py --backend agent    --steps 150 --warmup 10
run "pp_baseline"               29505 1500 training/train_pp_send_recv_7node.py  --backend baseline --steps 150 --warmup 10
run "pp_agent"                  29506 1500 training/train_pp_send_recv_7node.py  --backend agent    --steps 150 --warmup 10
run "lbar_baseline"             29507 1500 training/train_llama_block_ar_7node.py --backend baseline --steps 150 --warmup 10
run "lbar_agent"                29508 1500 training/train_llama_block_ar_7node.py --backend agent    --steps 150 --warmup 10
run "ring_kv_baseline"          29509 1500 training/train_ring_kv_7node.py       --backend baseline --steps 150 --warmup 10
run "ring_kv_agent"             29510 1500 training/train_ring_kv_7node.py       --backend agent    --steps 150 --warmup 10
run "ua2a_baseline"             29511 1500 training/train_uniform_a2a_7node.py   --backend baseline --steps 150 --warmup 10
run "ua2a_agent"                29512 1500 training/train_uniform_a2a_7node.py   --backend agent    --steps 150 --warmup 10

# === Llama-7B e2e with no-sim runtimes ===
run "llama7b_per_mb"  29520 2400 experiments/model_extension/train_llama_e2e_7b.py per_mb 200
run "llama7b_bundled" 29521 2400 experiments/model_extension/train_llama_e2e_7b.py bundled 200

HB "restoring main runtimes..."
restore_main_runtimes
rsync_repo
HB "R30-nosim-full done"
