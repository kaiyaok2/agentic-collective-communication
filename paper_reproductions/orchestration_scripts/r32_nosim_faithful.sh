#!/bin/bash
# r32: faithful no-simulator ablation.
# 1. Backup main runtimes + r28_nosim outputs (preserve Tables 2-6 artifacts)
# 2. Re-run no-simulator search for all 8 problems → fresh runtime/ files
#    (Phase 5 cost_score sentinel ensures Phase 5 does NOT fall back to baseline seed)
# 3. Deploy fresh nosim runtimes + measure per-problem + OLMoE e2e + Llama-7B e2e
# 4. Restore main runtimes (Tables 2-6 reproducible)
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
R32=/home/ubuntu/r32_nosim_faithful
mkdir -p $R32

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R32/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R32/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R32/runtime_main_backup/
  HB "  backed up $(ls $R32/runtime_main_backup/ | wc -l) main runtime files to $R32/runtime_main_backup/"
}

restore_main_runtimes() {
  cp $R32/runtime_main_backup/trainium_*.py $REPO/runtime/
}

deploy_fresh_nosim_runtimes() {
  # Copy the newly-search-written runtimes from $R32/nosim_runtimes/ to $REPO/runtime/
  for f in $R32/nosim_runtimes/trainium_*.py; do
    [ -f "$f" ] && cp "$f" $REPO/runtime/$(basename "$f")
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
  local OUT=$R32/$NAME
  HB "$NAME start"
  clean_disk
  kill_all
  rsync_repo
  mkdir -p $OUT
  local LSH=/tmp/r32_launch_${PORT}.sh
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

# ====================================================================
# STAGE 0: Backup main runtimes (preserve Tables 2-6 reproducibility)
# ====================================================================
HB "R32-nosim-faithful start"
backup_main_runtimes

# ====================================================================
# STAGE 1: Re-run no-simulator search for 8 problems
# Output written to runtime/, which we'll archive to $R32/nosim_runtimes/
# ====================================================================
HB "STAGE 1: running no-simulator search for 8 problems"
PROBLEMS=(alltoallv uniform_a2a ring_kv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar)
WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
cd $REPO
source $VENV/bin/activate
mkdir -p $R32/nosim_search_logs $R32/nosim_runtimes
for P in "${PROBLEMS[@]}"; do
  HB "  search $P --no-simulator start"
  mkdir -p $R32/nosim_search_logs/$P
  timeout 1500 python3 -u experiments/run_search.py \
      --problem $P --phase3-style strategy-enumerate --no-simulator \
      --llm-model sonnet --max-rounds 4 \
      --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
      --output-dir $R32/nosim_search_logs/$P > $R32/nosim_search_logs/$P/search.log 2>&1
  HB "  search $P exit=$?"
  # Archive whatever runtime file got written
  [ -f $REPO/runtime/trainium_${P}_7node.py ] && cp $REPO/runtime/trainium_${P}_7node.py $R32/nosim_runtimes/
  [ -f $REPO/runtime/trainium_${P}.py ] && cp $REPO/runtime/trainium_${P}.py $R32/nosim_runtimes/
done

# Summary: list deployed code for each problem
HB "=== nosim search complete; deployed runtime summary ==="
for f in $R32/nosim_runtimes/trainium_*.py; do
  desc=$(grep -m1 '"""' $f | head -1 | tr -d '"')
  HB "  $(basename $f): $desc"
done

# ====================================================================
# STAGE 2: Run measurements with the freshly-searched nosim runtimes
# ====================================================================
HB "STAGE 2: deploy + measure"
deploy_fresh_nosim_runtimes
clean_disk
kill_all
sleep 60
HB "wait done"

# Per-problem (8): 6 standalone + 2 in-OLMoE (alltoallv, dxe via OLMoE e2e)
run_training "tp_mlp_baseline"   29801 1500 training/train_tp_mlp_7node.py        --backend baseline --steps 150 --warmup 10
run_training "tp_mlp_agent"      29802 1500 training/train_tp_mlp_7node.py        --backend agent    --steps 150 --warmup 10
run_training "fsdp_baseline"     29803 1500 training/train_fsdp_prefetch_7node.py --backend baseline --steps 150 --warmup 10
run_training "fsdp_agent"        29804 1500 training/train_fsdp_prefetch_7node.py --backend agent    --steps 150 --warmup 10
run_training "pp_baseline"       29805 1500 training/train_pp_send_recv_7node.py  --backend baseline --steps 150 --warmup 10
run_training "pp_agent"          29806 1500 training/train_pp_send_recv_7node.py  --backend agent    --steps 150 --warmup 10
run_training "lbar_baseline"     29807 1500 training/train_llama_block_ar_7node.py --backend baseline --steps 150 --warmup 10
run_training "lbar_agent"        29808 1500 training/train_llama_block_ar_7node.py --backend agent    --steps 150 --warmup 10
run_training "ring_kv_baseline"  29809 1500 training/train_ring_kv_7node.py       --backend baseline --steps 150 --warmup 10
run_training "ring_kv_agent"     29810 1500 training/train_ring_kv_7node.py       --backend agent    --steps 150 --warmup 10
run_training "ua2a_baseline"     29811 1500 training/train_uniform_a2a_7node.py   --backend baseline --steps 150 --warmup 10
run_training "ua2a_agent"        29812 1500 training/train_uniform_a2a_7node.py   --backend agent    --steps 150 --warmup 10
run_training "alltoallv_baseline" 29813 1500 training/train_alltoallv_7node.py    --backend baseline --steps 150 --warmup 5
run_training "alltoallv_agent"   29814 1500 training/train_alltoallv_7node.py    --backend agent    --steps 150 --warmup 5
run_training "dxe_baseline"      29815 1500 training/train_dxe_7node.py           --backend baseline --steps 150 --warmup 5
run_training "dxe_agent"         29816 1500 training/train_dxe_7node.py           --backend agent    --steps 150 --warmup 5

# OLMoE e2e (baseline already on r28 main; rerun under same runtimes for clean comparison)
run_training "olmoe_baseline"    29820 1800 training/train_olmoe10b.py --backend baseline --grad-sync baseline --ce baseline --steps 250 --warmup 30
run_training "olmoe_agent"       29821 1800 training/train_olmoe10b.py --backend agent    --grad-sync baseline --ce agent    --steps 250 --warmup 30

# Llama-7B e2e
run_training "llama7b_per_mb"    29830 2400 experiments/model_extension/train_llama_e2e_7b.py per_mb 200
run_training "llama7b_bundled"   29831 2400 experiments/model_extension/train_llama_e2e_7b.py bundled 200

# ====================================================================
# STAGE 3: Restore main runtimes (Tables 2-6 reproducibility preserved)
# ====================================================================
HB "STAGE 3: restoring main runtimes..."
restore_main_runtimes
rsync_repo
clean_disk
HB "R32-nosim-faithful done"
