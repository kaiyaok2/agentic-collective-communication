#!/bin/bash
# R20 Phase 1: search ALL 9 problems with strategy-enumerate ONLY.
# Validates that the simulator+HW-gate fix drives convergence:
#   - tp_mlp/fsdp_prefetch/pp_send_recv → bundled
#   - alltoallv → pack+1AG+slice
#   - uniform_a2a → 1AG+slice
#   - ring_kv → per-slot AG
#   - dxe → 2×AR
#   - grad_ar → chunked bucketed AR
#   - llama_block_ar → bundled stacked AR
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
PY=$VENV/bin/python3
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
WORKERS=$(IFS=,; echo "${WORKER_LIST[*]}")

R20=/home/ubuntu/r20
HEARTBEAT=$R20/HEARTBEAT
mkdir -p $R20
echo "R20 STRATEGY-ENUM PHASE 1 START $(date -u)" >> $HEARTBEAT

PROBLEMS=(tp_mlp fsdp_prefetch pp_send_recv llama_block_ar grad_ar dxe ring_kv uniform_a2a alltoallv)

rsync_repo() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
      "$REPO/" "ubuntu@$ip:$REPO/" &
  done
  wait
}

# Reset runtime/ to dev baseline before each search
cd $REPO && git checkout main -- runtime/ 2>/dev/null
git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
rsync_repo

for P in "${PROBLEMS[@]}"; do
  OUT=$R20/strategy-enumerate/searches/$P
  [ -f $OUT/done ] && { echo "[search strategy-enum/$P] cached" >> $HEARTBEAT; continue; }
  mkdir -p $OUT
  cd $REPO && git checkout main -- runtime/ 2>/dev/null
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
  rsync_repo
  echo "[search strategy-enum/$P] start $(date -u)" >> $HEARTBEAT
  T0=$(date -u +%s)
  timeout 2400 $PY $REPO/experiments/run_search.py \
    --problem $P --phase3-style strategy-enumerate \
    --max-rounds 4 --num-nodes 7 \
    --master-addr $MASTER --worker-addrs $WORKERS \
    --output-dir $OUT > $OUT/run.log 2>&1
  RC=$?
  T1=$(date -u +%s)
  mkdir -p $R20/strategy-enumerate/runtime_per_problem/$P
  cp $REPO/runtime/trainium_${P}*.py $R20/strategy-enumerate/runtime_per_problem/$P/ 2>/dev/null
  echo "RC=$RC dur=$((T1-T0))s" > $OUT/done
  echo "[search strategy-enum/$P] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
done

echo "R20 STRATEGY-ENUM PHASE 1 DONE $(date -u)" >> $HEARTBEAT
