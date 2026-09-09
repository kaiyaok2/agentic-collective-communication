#!/bin/bash
# 7-node E2E launcher for the 10B-TP family training scripts (venv 2_8).
# Usage: e2e_launch.sh <script.py> <backend> <tag> [extra args...]
#   backend in {baseline, sorcar}. Extra args forwarded (e.g. --nmb 8 --fuse).
set -u
SCRIPT=$1; BACKEND=$2; TAG=$3; shift 3
EXTRA="$*"
NNODES=7
MASTER=172.31.19.204
WORKERS=(172.31.26.146 172.31.26.149 172.31.22.121 172.31.28.85 172.31.19.153 172.31.31.94)
KEY=/home/ubuntu/.ssh/Kaiyao.pem
VENV=/opt/aws_neuronx_venv_pytorch_2_8
PORT=${PORT:-29500}
STEPS=${STEPS:-30}
SEED=${SEED:-42}
CACHE=${CACHE:-/home/ubuntu/neuron_cache_e2e}
LOGD=/home/ubuntu/e2e_sweep/logs/${TAG}
mkdir -p "$LOGD" "$CACHE" /home/ubuntu/e2e_sweep

# sync script to all workers (flat /home/ubuntu path)
BN=$(basename "$SCRIPT")
for ip in "${WORKERS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8 ubuntu@$ip "mkdir -p $CACHE" 2>/dev/null
  scp -q -i $KEY -o StrictHostKeyChecking=no "$SCRIPT" ubuntu@$ip:/home/ubuntu/"$BN"
done

ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH \
&& export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 \
&& export PJRT_DEVICE=NEURON && export NEURON_RT_LOG_LEVEL=ERROR \
&& export NEURON_COMPILE_CACHE_URL=$CACHE \
&& export NEURON_RT_NUM_CORES=32 && export OMP_NUM_THREADS=1"
ARGS="--backend $BACKEND --steps $STEPS --seed $SEED $EXTRA"
TRUN="--nnodes=$NNODES --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "[e2e] $(date -u) launch $TAG backend=$BACKEND port=$PORT args=[$ARGS]"
NR=1
for ip in "${WORKERS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
    "$ENV && cd /home/ubuntu && timeout 5400 torchrun $TRUN --node_rank=$NR /home/ubuntu/$BN $ARGS" \
    > "$LOGD/n${NR}.log" 2>&1 &
  NR=$((NR+1))
done
eval "$ENV"
cd /home/ubuntu
timeout 5400 torchrun $TRUN --node_rank=0 /home/ubuntu/$BN $ARGS > "$LOGD/n0_master.log" 2>&1
RC=$?
echo "[e2e] $TAG master_rc=$RC $(date -u)"
wait
grep -h "RESULT_JSON" "$LOGD"/*.log 2>/dev/null | head -1
exit $RC
