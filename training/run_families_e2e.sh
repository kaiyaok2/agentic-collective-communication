#!/bin/bash
# 7-node launcher for train_families_e2e_7node.py
# Usage: run_families_e2e.sh <arch> <backend> <master_ip> <workers_file> [steps]
set -u
ARCH=$1
BACKEND=$2
MASTER_IP=$3
WORKERS=$4
STEPS=${5:-40}
SEED=${6:-42}
NNODES=7

export_env='export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 PJRT_DEVICE=NEURON NEURON_RT_LOG_LEVEL=ERROR'
SCRIPT=/home/ubuntu/train_families_e2e_7node.py
CMD_ARGS="--arch $ARCH --backend $BACKEND --steps $STEPS --seed $SEED"

master_cmd="$export_env && cd /home/ubuntu && timeout 5400 torchrun --nnodes=$NNODES --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29500 --node_rank=0 $SCRIPT $CMD_ARGS"

RANK=1
for W in $(cat $WORKERS); do
  worker_cmd="$export_env && cd /home/ubuntu && timeout 5400 torchrun --nnodes=$NNODES --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29500 --node_rank=$RANK $SCRIPT $CMD_ARGS"
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$W "$worker_cmd" > /tmp/families_worker_$RANK.log 2>&1 &
  RANK=$((RANK+1))
done

bash -c "$master_cmd" 2>&1
wait
