#!/bin/bash
# r40: Honest no-sim measurements following user's directive workflow.
#   - Rerun --no-simulator search for alltoallv and uniform_a2a with the
#     _StubXM shape-gate fix in place; AG+RS should now pass Phase 4
#     shape gate and the LLM-judge should pick it by HW microbench.
#   - Reuse r33's no-sim dxe runtime (no change).
#   - Rerun OLMoE-10B baseline + agent at 7 nodes; the deployed alltoallv
#     and uniform_a2a being structurally identical to the inline baseline
#     should give ~1.0x (or slightly worse because no-sim dxe is slower).
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
R40=/home/ubuntu/r40_honest_nosim
NOSIM_DIR=/home/ubuntu/r33_nosim_full_judge/nosim_runtimes
mkdir -p $R40

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R40/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R40/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R40/runtime_main_backup/
}

restore_main_runtimes() {
  cp $R40/runtime_main_backup/trainium_*.py $REPO/runtime/
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
  local OUT=$R40/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r40_launch_${PORT}.sh
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

HB "R40 honest-nosim start"
backup_main_runtimes

# Re-search alltoallv + uniform_a2a with --no-simulator + fixed stub
WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
cd $REPO
source $VENV/bin/activate
mkdir -p $R40/search_logs
for P in alltoallv uniform_a2a; do
  HB "  search $P --no-simulator --hw-eval start"
  mkdir -p $R40/search_logs/$P
  timeout 1500 python3 -u experiments/run_search.py \
      --problem $P --phase3-style strategy-enumerate --no-simulator --hw-eval \
      --llm-model sonnet --max-rounds 4 \
      --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
      --output-dir $R40/search_logs/$P > $R40/search_logs/$P/search.log 2>&1
  HB "    search $P exit=$?"
  # Save deployed
  mkdir -p $R40/deployed_runtimes
  [ -f $REPO/runtime/trainium_${P}_7node.py ] && cp $REPO/runtime/trainium_${P}_7node.py $R40/deployed_runtimes/
  [ -f $REPO/runtime/trainium_${P}.py ] && cp $REPO/runtime/trainium_${P}.py $R40/deployed_runtimes/
done

HB "deployed runtime summary:"
for f in $R40/deployed_runtimes/trainium_*.py; do
  desc=$(grep -m1 '"""' $f | head -1 | tr -d '"')
  HB "  $(basename $f): $desc"
done

# Now: restore main, then overlay deployed alltoallv + ua2a + r33-no-sim dxe
restore_main_runtimes
for f in $R40/deployed_runtimes/trainium_*.py; do
  cp "$f" $REPO/runtime/$(basename "$f")
done
cp $NOSIM_DIR/trainium_dxe_7node.py $REPO/runtime/trainium_dxe_7node.py
cp $NOSIM_DIR/trainium_dxe.py       $REPO/runtime/trainium_dxe.py

clean_disk
kill_all
sleep 30

# OLMoE e2e
run_training "olmoe_baseline" 29990 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
run_training "olmoe_agent"    29991 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30

HB "restoring main..."
restore_main_runtimes
rsync_repo
clean_disk
HB "R40 done"
