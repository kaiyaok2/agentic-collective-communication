#!/bin/bash
set -uo pipefail
PROB=${1:?problem}
BACKEND=${2:?backend}
NNODES=${3:-7}
NPROC=32; MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
VENV=/opt/aws_neuronx_venv_pytorch_2_9
KEY=/home/ubuntu/.ssh/Kaiyao.pem
SCRIPT=/tmp/tp_search/percall_modext.py
PORT=${PORT:-33860}
LOGD=/tmp/tp_search/logs_percall/${PROB}_${BACKEND}_${NNODES}n; mkdir -p $LOGD
for ip in "${WORKERS[@]:0:$((NNODES-1))}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "mkdir -p /tmp/tp_search"
  scp -q -i $KEY -o StrictHostKeyChecking=no $SCRIPT ubuntu@$ip:$SCRIPT
done
ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && export NEURON_RT_NUM_CORES=32 && export NEURON_NUM_RECENT_MODELS_TO_KEEP=1 && export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT"
TRUN="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"
NR=1
if [ "$NNODES" -gt 1 ]; then
  for ip in "${WORKERS[@]:0:$((NNODES-1))}"; do
    ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "$ENV && $VENV/bin/torchrun $TRUN --node_rank=$NR $SCRIPT $PROB $BACKEND" > $LOGD/n${NR}_${ip}.log 2>&1 &
    NR=$((NR+1))
  done
fi
eval "$ENV"
timeout 600 $VENV/bin/torchrun $TRUN --node_rank=0 $SCRIPT $PROB $BACKEND > $LOGD/n0_master.log 2>&1
rc=$?
wait
for ip in "${WORKERS[@]:0:$((NNODES-1))}"; do
  scp -q -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip:/tmp/tp_search/percall_${PROB}_${BACKEND}.json /tmp/tp_search/ 2>/dev/null
done
echo "[percall] $PROB/$BACKEND ${NNODES}n rc=$rc done"
