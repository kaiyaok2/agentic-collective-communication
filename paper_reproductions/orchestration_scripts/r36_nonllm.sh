#!/bin/bash
# r36: Two non-LLM Phase-3 baselines (--phase3-style none, --phase3-style random)
# For each style: run searches for 8 problems (fast, no LLM), then measure OLMoE e2e
# baseline+agent. Restore main runtimes between styles.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R36=/home/ubuntu/r36_nonllm_phase3
mkdir -p $R36

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R36/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R36/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R36/runtime_main_backup/
}

restore_main_runtimes() {
  cp $R36/runtime_main_backup/trainium_*.py $REPO/runtime/
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
  pkill -9 -f run_search 2>/dev/null || true
  for ip in "${WORKERS[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
      "sudo pkill -9 -u ubuntu -f python 2>/dev/null; sudo pkill -9 -u ubuntu -f torchrun 2>/dev/null; sudo pkill -9 -u ubuntu -f neuronx-cc 2>/dev/null; find /tmp/neuron_cache -name '*.lock' -delete 2>/dev/null" >/dev/null 2>&1 &
  done
  wait
  sleep 30
}

run_training() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R36/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r36_launch_${PORT}.sh
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

run_searches_for_style() {
  local STYLE=$1
  HB "=== STYLE: $STYLE: running 8 problem searches ==="
  PROBLEMS=(alltoallv uniform_a2a ring_kv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar)
  WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
  cd $REPO
  source $VENV/bin/activate
  mkdir -p $R36/${STYLE}_search_logs $R36/${STYLE}_deployed_runtimes
  for P in "${PROBLEMS[@]}"; do
    HB "  search $P ($STYLE) start"
    mkdir -p $R36/${STYLE}_search_logs/$P
    timeout 900 python3 -u experiments/run_search.py \
        --problem $P --phase3-style $STYLE --hw-eval \
        --llm-model sonnet --max-rounds 4 \
        --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
        --output-dir $R36/${STYLE}_search_logs/$P > $R36/${STYLE}_search_logs/$P/search.log 2>&1
    HB "    search $P exit=$?"
    [ -f $REPO/runtime/trainium_${P}_7node.py ] && cp $REPO/runtime/trainium_${P}_7node.py $R36/${STYLE}_deployed_runtimes/
    [ -f $REPO/runtime/trainium_${P}.py ] && cp $REPO/runtime/trainium_${P}.py $R36/${STYLE}_deployed_runtimes/
  done
}

deploy_style_runtimes() {
  local STYLE=$1
  for f in $R36/${STYLE}_deployed_runtimes/trainium_*.py; do
    [ -f "$f" ] && cp "$f" $REPO/runtime/$(basename "$f")
  done
}

HB "R36 start"
backup_main_runtimes

# --- STYLE: none ---
run_searches_for_style "none"
deploy_style_runtimes "none"
clean_disk
kill_all
sleep 30
run_training "olmoe_baseline_none" 29920 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
run_training "olmoe_agent_none"    29921 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30
restore_main_runtimes

# --- STYLE: random ---
run_searches_for_style "random"
deploy_style_runtimes "random"
clean_disk
kill_all
sleep 30
run_training "olmoe_baseline_random" 29922 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
run_training "olmoe_agent_random"    29923 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30
restore_main_runtimes
rsync_repo
clean_disk
HB "R36 done"
