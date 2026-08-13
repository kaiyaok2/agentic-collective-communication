#!/bin/bash
set -u
CODE_FILE=$1; PROBLEM=$2; MASTER_IP=$3; WORKER_IP=$4
NNODES=2
NPROC=32
LOG_PREFIX=/tmp/hwgate_$$

# ensure worker dir + copy file
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$WORKER_IP "mkdir -p $(dirname $CODE_FILE)" 2>/dev/null || true
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q $CODE_FILE ubuntu@$WORKER_IP:$CODE_FILE 2>/dev/null || true

master_cmd="cd /home/ubuntu && export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:\$PATH FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 PJRT_DEVICE=NEURON NEURON_RT_LOG_LEVEL=ERROR && timeout 240 torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29500 --node_rank=0 /home/ubuntu/hw_gate_run.py --problem $PROBLEM --code-file $CODE_FILE"
worker_cmd="cd /home/ubuntu && export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:\$PATH FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 PJRT_DEVICE=NEURON NEURON_RT_LOG_LEVEL=ERROR && timeout 240 torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:29500 --node_rank=1 /home/ubuntu/hw_gate_run.py --problem $PROBLEM --code-file $CODE_FILE"

bash -c "$master_cmd" > ${LOG_PREFIX}_master.log 2>&1 &
MPID=$!
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$WORKER_IP "$worker_cmd" > ${LOG_PREFIX}_worker.log 2>&1 &
WPID=$!
wait $MPID; MRC=$?
wait $WPID; WRC=$?

if grep -q 'HW_GATE_PASS rank=0' ${LOG_PREFIX}_master.log; then
  exit 0
fi
exit $([ $MRC -ne 0 ] && echo $MRC || echo $WRC)
