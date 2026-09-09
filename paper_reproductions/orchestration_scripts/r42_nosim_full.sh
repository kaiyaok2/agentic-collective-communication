#!/bin/bash
# r42: Full no-sim search + OLMoE e2e using the new iterative no-sim Phase 3
# + h7-style small-shape Phase 5 LLM-judge. Picks up r41's verified ua2a/a2av
# searches; runs the remaining 6 problems + OLMoE-10B end-to-end.
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
R42=/home/ubuntu/r42_nosim_full
R41=/home/ubuntu/r41_nosim_test
mkdir -p $R42

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R42/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R42/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R42/runtime_main_backup/
}
restore_main_runtimes() {
  cp $R42/runtime_main_backup/trainium_*.py $REPO/runtime/
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
  sleep 20
}

run_training() {
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; shift 3
  local CMD="$@"
  local OUT=$R42/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r42_launch_${PORT}.sh
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

HB "R42 start: 6 remaining no-sim searches + OLMoE e2e"
backup_main_runtimes
WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
cd $REPO
source $VENV/bin/activate
mkdir -p $R42/search_logs

# Phase A: run no-sim search for 6 remaining problems
# (ua2a + alltoallv already done in r41)
for P in dxe ring_kv tp_mlp pp_send_recv fsdp_prefetch llama_block_ar; do
  HB "  search $P --no-simulator iterative start"
  mkdir -p $R42/search_logs/$P
  timeout 1500 python3 -u experiments/run_search.py \
      --problem $P --phase3-style strategy-enumerate --no-simulator --hw-eval \
      --llm-model sonnet --max-rounds 4 \
      --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
      --output-dir $R42/search_logs/$P > $R42/search_logs/$P/search.log 2>&1
  HB "    search $P exit=$?"
  mkdir -p $R42/deployed_runtimes
  [ -f $REPO/runtime/trainium_${P}_7node.py ] && cp $REPO/runtime/trainium_${P}_7node.py $R42/deployed_runtimes/
  [ -f $REPO/runtime/trainium_${P}.py ] && cp $REPO/runtime/trainium_${P}.py $R42/deployed_runtimes/
done

# Copy r41's ua2a + alltoallv deployed runtimes
for P in uniform_a2a alltoallv; do
  [ -f $R41/deployed_runtimes/trainium_${P}_7node.py ] && cp $R41/deployed_runtimes/trainium_${P}_7node.py $R42/deployed_runtimes/
  [ -f $R41/deployed_runtimes/trainium_${P}.py ] && cp $R41/deployed_runtimes/trainium_${P}.py $R42/deployed_runtimes/
done

HB "deployed runtime summary:"
for f in $R42/deployed_runtimes/trainium_*.py; do
  desc=$(grep -m1 '"""' $f | head -1 | tr -d '"')
  HB "  $(basename $f): $desc"
done

# Phase B: deploy + OLMoE e2e
restore_main_runtimes
for f in $R42/deployed_runtimes/trainium_*.py; do
  cp "$f" $REPO/runtime/$(basename "$f")
done
clean_disk
kill_all
sleep 30

run_training "olmoe_baseline" 30100 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
run_training "olmoe_agent"    30101 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30

# Per-problem 7-node training probes (these match the §7.2 table rows)
run_training "tp_mlp_baseline"   30110 1500 training/train_tp_mlp_7node.py        --backend baseline --steps 150 --warmup 10
run_training "tp_mlp_agent"      30111 1500 training/train_tp_mlp_7node.py        --backend agent    --steps 150 --warmup 10
run_training "fsdp_baseline"     30112 1500 training/train_fsdp_prefetch_7node.py --backend baseline --steps 150 --warmup 10
run_training "fsdp_agent"        30113 1500 training/train_fsdp_prefetch_7node.py --backend agent    --steps 150 --warmup 10
run_training "lbar_baseline"     30114 1500 training/train_llama_block_ar_7node.py --backend baseline --steps 150 --warmup 10
run_training "lbar_agent"        30115 1500 training/train_llama_block_ar_7node.py --backend agent    --steps 150 --warmup 10
run_training "ring_kv_baseline"  30116 1500 training/train_ring_kv_7node.py       --backend baseline --steps 150 --warmup 10
run_training "ring_kv_agent"     30117 1500 training/train_ring_kv_7node.py       --backend agent    --steps 150 --warmup 10
run_training "ua2a_baseline"     30118 1500 training/train_uniform_a2a_7node.py   --backend baseline --steps 150 --warmup 5
run_training "ua2a_agent"        30119 1500 training/train_uniform_a2a_7node.py   --backend agent    --steps 150 --warmup 5
run_training "alltoallv_baseline" 30120 1500 training/train_alltoallv_7node.py    --backend baseline --steps 150 --warmup 5
run_training "alltoallv_agent"   30121 1500 training/train_alltoallv_7node.py    --backend agent    --steps 150 --warmup 5
run_training "dxe_baseline"      30122 1500 training/train_dxe_7node.py           --backend baseline --steps 150 --warmup 5
run_training "dxe_agent"         30123 1500 training/train_dxe_7node.py           --backend agent    --steps 150 --warmup 5
run_training "pp_baseline"       30124 1500 training/train_pp_send_recv_7node.py  --backend baseline --steps 150 --warmup 10
run_training "pp_agent"          30125 1500 training/train_pp_send_recv_7node.py  --backend agent    --steps 150 --warmup 10

HB "restoring main..."
restore_main_runtimes
rsync_repo
clean_disk
HB "R42 done"
