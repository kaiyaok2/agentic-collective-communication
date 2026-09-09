#!/bin/bash
set -uo pipefail
# Usage: bash run_bench.sh <problem>
#   problem: a2av | ua2a | rkv | dxe
PROBLEM=${1:?problem}
NPROC=32; NNODES=7; MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
VENV=/opt/aws_neuronx_venv_pytorch_2_9
KEY=/home/ubuntu/.ssh/Kaiyao.pem
SCRIPT=/tmp/h7_bench/bench_${PROBLEM}.py
PORT=${PORT:-32500}
LOGD=/tmp/h7_bench/logs/${PROBLEM}
mkdir -p $LOGD

# Sync script + ensure runtime/ available on workers (rsync repo)
REPO=/home/ubuntu/agentic-collective-communication
for ip in "${WORKERS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip 'mkdir -p /tmp/h7_bench' 2>/dev/null
  scp -q -i $KEY -o StrictHostKeyChecking=no $SCRIPT ubuntu@$ip:$SCRIPT
done

ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && export NEURON_RT_NUM_CORES=32 && export NEURON_NUM_RECENT_MODELS_TO_KEEP=1 && export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT && export PYTHONPATH=$REPO"
TRUN="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "[h7-bench] === $(date -u) launching $PROBLEM (port=$PORT) ==="
NR=1
for ip in "${WORKERS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "$ENV && $VENV/bin/torchrun $TRUN --node_rank=$NR $SCRIPT" > $LOGD/n${NR}_${ip}.log 2>&1 &
  NR=$((NR+1))
done
eval "$ENV"
timeout 900 $VENV/bin/torchrun $TRUN --node_rank=0 $SCRIPT > $LOGD/n0_master.log 2>&1
echo "[h7-bench] $PROBLEM master_rc=$? at $(date -u)"
wait
for ip in "${WORKERS[@]}"; do
  scp -q -i $KEY -o StrictHostKeyChecking=no "ubuntu@$ip:/tmp/h7_bench/${PROBLEM}_*.json" /tmp/h7_bench/ 2>/dev/null
done
echo "[h7-bench] $PROBLEM DONE $(date -u)"
