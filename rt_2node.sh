#!/bin/bash
set -u
RUNTIME_FILE=$1; PROBLEM=$2; MASTER_IP=$3; WORKER_IP=$4; N_ITERS=${5:-100}
LOG_PREFIX=/tmp/rt_$$
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$WORKER_IP "mkdir -p $(dirname $RUNTIME_FILE)" 2>/dev/null || true
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q $RUNTIME_FILE ubuntu@$WORKER_IP:$RUNTIME_FILE 2>/dev/null || true
master_cmd="cd /home/ubuntu && export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:\$PATH FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 PJRT_DEVICE=NEURON NEURON_RT_LOG_LEVEL=ERROR PROBLEM=$PROBLEM RUNTIME_FILE=$RUNTIME_FILE N_ITERS=$N_ITERS && timeout 600 torchrun --nnodes=2 --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29500 --node_rank=0 /home/ubuntu/rt_run_v12.py"
worker_cmd="cd /home/ubuntu && export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:\$PATH FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 PJRT_DEVICE=NEURON NEURON_RT_LOG_LEVEL=ERROR PROBLEM=$PROBLEM RUNTIME_FILE=$RUNTIME_FILE N_ITERS=$N_ITERS && timeout 600 torchrun --nnodes=2 --nproc_per_node=32 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29500 --node_rank=1 /home/ubuntu/rt_run_v12.py"
bash -c "$master_cmd" > ${LOG_PREFIX}_master.log 2>&1 &
MPID=$!
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$WORKER_IP "$worker_cmd" > ${LOG_PREFIX}_worker.log 2>&1 &
WPID=$!
wait $MPID; wait $WPID
MS_PER_ITER=$(grep 'RT_TIME_MS_PER_ITER' ${LOG_PREFIX}_master.log | awk '{print $2}')
echo "MS_PER_ITER=$MS_PER_ITER"
