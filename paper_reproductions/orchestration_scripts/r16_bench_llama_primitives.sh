#!/bin/bash
# Bench redo for the 4 Llama primitives: pp_send_recv, tp_mlp, fsdp_prefetch, llama_block_ar
# at both 1-node and 7-node scopes, across 4 stacks.
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
set -u

REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

R16=/home/ubuntu/r16
HEARTBEAT=$R16/HEARTBEAT_BENCH

CCFLAGS='--hbm-scratchpad-page-size=64 --optlevel=2'
RT_PAGE=64

# Stage new bench scripts on master + workers
mkdir -p /tmp/h7_bench
cp $REPO/experiments/h7_bench/bench_*.py /tmp/h7_bench/
for ip in "${WORKER_LIST[@]}"; do
  scp -q -i $KEY -o StrictHostKeyChecking=no $REPO/experiments/h7_bench/bench_*.py ubuntu@$ip:/tmp/h7_bench/ &
done; wait

restore_main_runtime() { cd $REPO; git checkout main -- runtime/ 2>/dev/null; }
snapshot_v6_kg() {
  restore_main_runtime
  git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py
}
rsync_repo() {
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --exclude='.git' --exclude='__pycache__' "$REPO/" ubuntu@$ip:$REPO/ &
  done; wait
}
deploy_picks() {
  local STYLE_DIR="$1"
  snapshot_v6_kg
  for P in alltoallv uniform_a2a dxe grad_ar ring_kv pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
    cp "$STYLE_DIR/runtime_per_problem/$P/trainium_${P}"*.py $REPO/runtime/ 2>/dev/null
  done
}
deploy_baseline() { snapshot_v6_kg; }
kill_all() {
  killall -9 torchrun python3 2>/dev/null || true
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "killall -9 torchrun python3 2>/dev/null; true" &
  done; wait; sleep 3
}
collect_worker_jsons() {
  local OUT="$1"
  for ip in "${WORKER_LIST[@]}"; do
    rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" --include='*.json' --exclude='*' \
      "ubuntu@${ip}:/tmp/h7_bench/" "${OUT}/" 2>/dev/null &
  done; wait
}

run_h7_bench() {
  local STACK="$1" PROB="$2"
  local OUT=$R16/$STACK/h7_bench_llama/$PROB
  mkdir -p $OUT
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_picks $R16/$STACK; fi
  rsync_repo
  kill_all
  local port=$((47000 + RANDOM % 1000))
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
    export PYTHONPATH=$REPO && export MASTER_ADDR=$MASTER && export MASTER_PORT=$port"
  local TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
  echo "[h7bench-llama ${STACK}/$PROB] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  local NR=1
  for ip in "${WORKER_LIST[@]}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
      "$ENV_VARS && cd $REPO && $TORCHRUN $TRUN --node_rank=$NR /tmp/h7_bench/bench_${PROB}.py" \
      > $OUT/w${NR}.log 2>&1 &
    NR=$((NR+1))
  done
  eval "$ENV_VARS"; cd $REPO
  timeout 600 $TORCHRUN $TRUN --node_rank=0 /tmp/h7_bench/bench_${PROB}.py > $OUT/m0.log 2>&1
  local RC=$?; wait
  local T1=$(date -u +%s)
  collect_worker_jsons "$OUT"
  cp /tmp/h7_bench/${PROB}_*.json $OUT/ 2>/dev/null
  echo "[h7bench-llama ${STACK}/$PROB] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

run_n1_bench() {
  local STACK="$1" PROB="$2"
  local OUT=$R16/$STACK/n1_bench_llama/$PROB
  mkdir -p $OUT
  if [ "$STACK" = "baseline" ]; then deploy_baseline; else deploy_picks $R16/$STACK; fi
  rsync_repo
  kill_all
  echo "[n1bench-llama ${STACK}/$PROB] start $(date -u)" >> $HEARTBEAT
  local T0=$(date -u +%s)
  local ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
    export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
    export NEURON_CC_FLAGS=\"$CCFLAGS\" && export NEURON_SCRATCHPAD_PAGE_SIZE=$RT_PAGE && \
    export PYTHONPATH=$REPO"
  eval "$ENV_VARS"
  timeout 600 $TORCHRUN --nproc_per_node=32 --standalone /tmp/h7_bench/bench_${PROB}.py > $OUT/m0.log 2>&1
  local RC=$?
  local T1=$(date -u +%s)
  cp /tmp/h7_bench/${PROB}_*.json $OUT/ 2>/dev/null
  echo "[n1bench-llama ${STACK}/$PROB] exit=$RC dur=$((T1-T0))s $(date -u)" >> $HEARTBEAT
}

echo "BENCH-LLAMA START $(date -u)" >> $HEARTBEAT
for STACK in baseline strategy-enumerate cc-react multi-island; do
  for PROB in pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
    run_h7_bench "$STACK" "$PROB"
  done
done
for STACK in baseline strategy-enumerate cc-react multi-island; do
  for PROB in pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
    run_n1_bench "$STACK" "$PROB"
  done
done
echo "BENCH-LLAMA DONE $(date -u)" >> $HEARTBEAT
