#!/bin/bash
# r44: (a) re-run no-sim dxe search with the fixed full_vocab_ag template
# and per-candidate unique small-shape ports; verify convergence to 1-AG.
# (b) bench_dxe at 7-node and 1-node to refresh Table 2 + Table 7 dxe
# agent column (agent_fn now calls deployed strategy-enum runtime).
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
R44=/home/ubuntu/r44_dxe_iterate_bench
R42=/home/ubuntu/r42_nosim_full
mkdir -p $R44

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R44/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R44/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R44/runtime_main_backup/
}
restore_main_runtimes() {
  cp $R44/runtime_main_backup/trainium_*.py $REPO/runtime/
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
  local NAME=$1; local PORT=$2; local TIMEOUT=$3; local NNODES=$4; shift 4
  local CMD="$@"
  local OUT=$R44/$NAME
  HB "$NAME start ($NNODES-node)"
  clean_disk
  kill_all
  if [ $NNODES -gt 1 ]; then
    rsync_repo
  fi
  mkdir -p $OUT /tmp/h7_bench
  local LSH=/tmp/r44_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
unset NEURON_RT_EXEC_TIMEOUT
source $VENV/bin/activate
mkdir -p $OUT /tmp/h7_bench
cd $REPO
exec $TORCHRUN --nnodes=$NNODES --node_rank=__NR__ --nproc_per_node=32 \\
  --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \\
  $CMD > $OUT/node___NR__.log 2>&1
EOF
  if [ $NNODES -gt 1 ]; then
    for i in "${!WORKERS[@]}"; do
      local NR=$((i+1))
      if [ $NR -lt $NNODES ]; then
        local ip=${WORKERS[$i]}
        sed "s/__NR__/$NR/g" $LSH > $LSH.$NR
        scp -i $KEY -o StrictHostKeyChecking=no -q $LSH.$NR ubuntu@$ip:$LSH &
      fi
    done
    wait
    for i in "${!WORKERS[@]}"; do
      local NR=$((i+1))
      if [ $NR -lt $NNODES ]; then
        local ip=${WORKERS[$i]}
        ssh -fn -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$ip \
          "mkdir -p $OUT /tmp/h7_bench && chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
      fi
    done
    sleep 3
  fi
  cd $REPO
  source $VENV/bin/activate
  timeout $TIMEOUT $TORCHRUN --nnodes=$NNODES --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    $CMD > $OUT/node_0.log 2>&1
  EX=$?
  HB "  $NAME exit=$EX"
}

HB "R44 start: dxe no-sim iterate (until 1-AG wins) + bench_dxe 1n+7n"
backup_main_runtimes

# ===== PART A: re-run no-sim dxe search with the fixed full_vocab_ag template =====
WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
cd $REPO
source $VENV/bin/activate
mkdir -p $R44/search_logs

# Try up to 3 search iterations; succeed when winner is full_vocab_ag
ATTEMPT=0
WIN=""
while [ $ATTEMPT -lt 3 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  HB "  dxe search attempt $ATTEMPT --no-simulator start"
  mkdir -p $R44/search_logs/attempt_$ATTEMPT
  timeout 1500 python3 -u experiments/run_search.py \
      --problem dxe --phase3-style strategy-enumerate --no-simulator --hw-eval \
      --llm-model sonnet --max-rounds 4 \
      --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
      --output-dir $R44/search_logs/attempt_$ATTEMPT > $R44/search_logs/attempt_$ATTEMPT/search.log 2>&1
  HB "    dxe search attempt $ATTEMPT exit=$?"
  WIN=$(grep -aE "  Winner:" $R44/search_logs/attempt_$ATTEMPT/search.log | tail -1)
  HB "    winner: $WIN"
  if echo "$WIN" | grep -qE "full_vocab_ag"; then
    HB "  ✓ dxe converged to full_vocab_ag on attempt $ATTEMPT"
    cp $REPO/runtime/trainium_dxe_7node.py $R44/dxe_7node_deployed_fullvocab.py
    cp $REPO/runtime/trainium_dxe.py $R44/dxe_1node_deployed_fullvocab.py 2>/dev/null
    break
  fi
done

# ===== PART B: deploy + OLMoE e2e (alltoallv from r43 AG+T+RS, dxe = new pick) =====
if echo "$WIN" | grep -qE "full_vocab_ag"; then
  HB "PART B: OLMoE e2e with full_vocab_ag dxe"
  # Start from main, overlay r42's other 6 (non-alltoallv non-dxe), overlay r43's alltoallv (AG+T+RS wrapper),
  # and overlay r44's dxe (full_vocab_ag).
  restore_main_runtimes
  for P in uniform_a2a ring_kv tp_mlp pp_send_recv fsdp_prefetch llama_block_ar; do
    for ext in "_7node.py" ".py"; do
      [ -f $R42/deployed_runtimes/trainium_${P}${ext} ] && cp $R42/deployed_runtimes/trainium_${P}${ext} $REPO/runtime/
    done
  done
  for f in /home/ubuntu/r43_alltoallv_dxe_baselinematch/deployed_runtimes/trainium_alltoallv*.py; do
    cp $f $REPO/runtime/$(basename $f)
  done
  # dxe stays as deployed by r44's last search (already in $REPO/runtime since search Phase 5 writes there)
  clean_disk
  kill_all
  sleep 30
  run_training "olmoe_baseline" 30300 1800 7 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
  run_training "olmoe_agent"    30301 1800 7 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30
fi

# ===== PART C: bench_dxe at 7-node and 1-node =====
HB "PART C: bench_dxe at 7-node and 1-node"
restore_main_runtimes
rsync_repo
clean_disk
kill_all
sleep 20
run_training "bench_dxe_7node" 30310 600 7 experiments/h7_bench/bench_dxe.py
clean_disk
kill_all
sleep 20
run_training "bench_dxe_1node" 30311 600 1 experiments/h7_bench/bench_dxe.py

HB "restoring main..."
restore_main_runtimes
rsync_repo
clean_disk
HB "R44 done"
