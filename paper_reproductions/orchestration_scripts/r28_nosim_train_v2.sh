#!/bin/bash
# No-sim training, using the MAIN /home/ubuntu repo (known-working) but with
# the no-sim winners swapped into runtime/. Backs up + restores main runtimes.
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

R28NT=/home/ubuntu/r28_nosim_train
mkdir -p $R28NT

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R28NT/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R28NT/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R28NT/runtime_main_backup/ 2>/dev/null
}

deploy_nosim_runtimes() {
  for f in $NOSIM_REPO/runtime/trainium_*.py; do
    [ -f "$f" ] && cp "$f" $REPO/runtime/$(basename "$f")
  done
}

restore_main_runtimes() {
  cp $R28NT/runtime_main_backup/trainium_*.py $REPO/runtime/ 2>/dev/null
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

run_training() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local SCRIPT_AND_ARGS="$@"
  local OUT=$R28NT/$NAME
  HB "$NAME start"
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r28nt_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT
cd $REPO
exec $TORCHRUN --nnodes=7 --node_rank=__NR__ --nproc_per_node=32 \\
  --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \\
  $SCRIPT_AND_ARGS > $OUT/node___NR__.log 2>&1
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
    $SCRIPT_AND_ARGS > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R28-nosim-train start (using main repo with no-sim runtimes swapped)"
backup_main_runtimes
deploy_nosim_runtimes

HB "initial Neuron driver wait..."
kill_all
sleep 60
HB "wait done"

# Llama-side per-problem (smaller models, faster)
run_training "tp_mlp_baseline"      29401 1500 training/train_tp_mlp_7node.py        --backend baseline --steps 150 --warmup 10
cp $REPO/training/results/percall_r22/tp_mlp_baseline_step.json $R28NT/tp_mlp_baseline/ 2>/dev/null || true
run_training "tp_mlp_agent"         29402 1500 training/train_tp_mlp_7node.py        --backend agent    --steps 150 --warmup 10
cp $REPO/training/results/percall_r22/tp_mlp_agent_step.json    $R28NT/tp_mlp_agent/    2>/dev/null || true
run_training "lbar_baseline"        29403 1500 training/train_llama_block_ar_7node.py --backend baseline --steps 150 --warmup 10
cp $REPO/training/results/percall_r22/llama_block_ar_baseline_step.json $R28NT/lbar_baseline/ 2>/dev/null || true
run_training "lbar_agent"           29404 1500 training/train_llama_block_ar_7node.py --backend agent    --steps 150 --warmup 10
cp $REPO/training/results/percall_r22/llama_block_ar_agent_step.json    $R28NT/lbar_agent/    2>/dev/null || true
run_training "ring_kv_baseline"     29405 1500 training/train_ring_kv_7node.py        --backend baseline --steps 150 --warmup 10
cp /tmp/h7_bench/ring_kv_7node_baseline.json $R28NT/ring_kv_baseline/ 2>/dev/null || true
run_training "ring_kv_agent"        29406 1500 training/train_ring_kv_7node.py        --backend agent    --steps 150 --warmup 10
cp /tmp/h7_bench/ring_kv_7node_agent.json    $R28NT/ring_kv_agent/    2>/dev/null || true

# OLMoE e2e: 250 steps with no-sim a2av + no-sim dxe
run_training "olmoe_baseline" 29410 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
cp $REPO/training/results/olmoe_*7node*/olmoe*baseline*.json $R28NT/olmoe_baseline/ 2>/dev/null || true
run_training "olmoe_agent"    29411 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30
cp $REPO/training/results/olmoe_*7node*/olmoe*agent*.json $R28NT/olmoe_agent/ 2>/dev/null || true

# Llama amp1 e2e
run_training "llama_amp1_per_mb"  29420 2400 experiments/model_extension/train_llama_e2e_amp1.py per_mb 200
run_training "llama_amp1_bundled" 29421 2400 experiments/model_extension/train_llama_e2e_amp1.py bundled 200

HB "restoring main runtimes..."
restore_main_runtimes
rsync_repo
HB "R28-nosim-train done"
