#!/bin/bash
# r38: 3-node end-to-end training for OLMoE-10B (3node variant) and Llama-7B (3node variant)
# using the existing 7-node strategy-enum agent code, ported to ws=96 by changing
# _WORLD/_NUM_DEVICES/_NUM_NODES constants only.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
# 3-node setup: master + 2 workers
WORKERS=(172.31.17.80 172.31.24.136)
R38=/home/ubuntu/r38_3node_e2e
mkdir -p $R38

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R38/HEARTBEAT; }

deploy_3node_runtimes() {
  for f in /tmp/3node_runtimes/trainium_*_3node.py; do
    cp "$f" $REPO/runtime/$(basename "$f")
  done
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
  sleep 20
}

run_training() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R38/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r38_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT /tmp/tp_search
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
      "mkdir -p $OUT /tmp/tp_search && chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
  done
  sleep 3
  cd $REPO
  source $VENV/bin/activate
  mkdir -p /tmp/tp_search
  timeout $TIMEOUT $TORCHRUN --nnodes=3 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $CMD > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R38 3-node end-to-end start"
deploy_3node_runtimes
HB "deployed _3node runtime variants"

# OLMoE 3-node (250 steps, scaled-down VOCAB)
run_training "olmoe_baseline_3n" 29970 1800 training/train_olmoe10b_3node.py --backend baseline --grad-sync baseline --ce baseline --steps 200 --warmup 20
run_training "olmoe_agent_3n"    29971 1800 training/train_olmoe10b_3node.py --backend agent    --grad-sync baseline --ce agent    --steps 200 --warmup 20

# Llama 3-node
run_training "llama_per_mb_3n"  29980 1500 experiments/model_extension/train_llama_e2e_7b_3node.py per_mb  200
run_training "llama_bundled_3n" 29981 1500 experiments/model_extension/train_llama_e2e_7b_3node.py bundled 200

# Remove the 3node deploys (cleanup)
for f in /tmp/3node_runtimes/trainium_*_3node.py; do rm -f $REPO/runtime/$(basename "$f"); done
rsync_repo
HB "R38 done"
