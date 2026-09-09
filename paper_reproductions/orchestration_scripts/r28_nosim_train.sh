#!/bin/bash
# Run training measurements for no-sim ablation results.
# Uses /tmp/r28_smoke/acc-r28 as the repo so the no-sim runtimes are picked up.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u

REPO=/tmp/r28_smoke/acc-r28
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

R28NT=/home/ubuntu/r28_nosim_train
mkdir -p $R28NT

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R28NT/HEARTBEAT; }

rsync_repo_to_workers() {
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "mkdir -p /tmp/r28_smoke/acc-r28" >/dev/null 2>&1 &
  done
  wait
  for ip in "${WORKERS[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
      "$REPO/" "ubuntu@$ip:/tmp/r28_smoke/acc-r28/" >/dev/null 2>&1 &
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
  rsync_repo_to_workers
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
  mkdir -p $OUT
  cd $REPO
  source $VENV/bin/activate
  timeout $TIMEOUT $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $SCRIPT_AND_ARGS > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R28-nosim-train start"
HB "initial repo sync to workers..."
rsync_repo_to_workers
HB "initial Neuron driver wait..."
kill_all
sleep 60
HB "wait done"

# === Per-problem training: just measure the ones with potential agent wins ===
# For no-sim: only pp_send_recv, tp_mlp, dxe have non-baseline-clone agent code
# But run all for completeness; report from json/log

# Llama-side per-problem
run_training "tp_mlp_baseline"      29401 1500 training/train_tp_mlp_7node.py        --backend baseline --steps 150 --warmup 10
cp /home/ubuntu/agentic-collective-communication/training/results/percall_r22/tp_mlp_baseline_step.json $R28NT/tp_mlp_baseline/ 2>/dev/null || true
run_training "tp_mlp_agent"         29402 1500 training/train_tp_mlp_7node.py        --backend agent    --steps 150 --warmup 10
cp /home/ubuntu/agentic-collective-communication/training/results/percall_r22/tp_mlp_agent_step.json $R28NT/tp_mlp_agent/ 2>/dev/null || true
run_training "lbar_baseline"        29403 1500 training/train_llama_block_ar_7node.py --backend baseline --steps 150 --warmup 10
cp /home/ubuntu/agentic-collective-communication/training/results/percall_r22/llama_block_ar_baseline_step.json $R28NT/lbar_baseline/ 2>/dev/null || true
run_training "lbar_agent"           29404 1500 training/train_llama_block_ar_7node.py --backend agent    --steps 150 --warmup 10
cp /home/ubuntu/agentic-collective-communication/training/results/percall_r22/llama_block_ar_agent_step.json $R28NT/lbar_agent/ 2>/dev/null || true
run_training "ring_kv_baseline"     29405 1500 training/train_ring_kv_7node.py        --backend baseline --steps 150 --warmup 10
run_training "ring_kv_agent"        29406 1500 training/train_ring_kv_7node.py        --backend agent    --steps 150 --warmup 10

# === OLMoE e2e: 250 steps ===
run_training "olmoe_baseline" 29410 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
run_training "olmoe_agent"    29411 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30

# === Llama amp1 e2e ===
run_training "llama_amp1_per_mb"  29420 2400 experiments/model_extension/train_llama_e2e_amp1.py per_mb 200
run_training "llama_amp1_bundled" 29421 2400 experiments/model_extension/train_llama_e2e_amp1.py bundled 200

HB "R28-nosim-train done"
