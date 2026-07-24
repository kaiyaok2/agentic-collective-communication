#!/bin/bash
exec > /home/ubuntu/rt_xor.log 2>&1
export PATH=/home/ubuntu/venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
MASTER_IP=172.31.17.9
WORKER_IP=172.31.26.232
mkdir -p /home/ubuntu/rt_xor
clean_zombies() {
  local nids
  nids=$(sudo fuser /dev/neuron0 /dev/neuron1 /dev/neuron2 /dev/neuron3 /dev/neuron4 /dev/neuron5 /dev/neuron6 /dev/neuron7 /dev/neuron8 /dev/neuron9 /dev/neuron10 /dev/neuron11 /dev/neuron12 /dev/neuron13 /dev/neuron14 /dev/neuron15 2>&1 | tr ':m' ' ' | tr -s ' ' | tr ' ' '\n' | grep -E '^[0-9]{4,}$' | sort -u)
  if [ -n "$nids" ]; then sudo kill -9 $nids 2>/dev/null; fi
  ssh -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$WORKER_IP "wids=\$(sudo fuser /dev/neuron0 /dev/neuron1 /dev/neuron2 /dev/neuron3 /dev/neuron4 /dev/neuron5 /dev/neuron6 /dev/neuron7 /dev/neuron8 /dev/neuron9 /dev/neuron10 /dev/neuron11 /dev/neuron12 /dev/neuron13 /dev/neuron14 /dev/neuron15 2>&1 | tr ':m' ' ' | tr -s ' ' | tr ' ' '\n' | grep -E '^[0-9]{4,}\$' | sort -u); if [ -n \"\$wids\" ]; then sudo kill -9 \$wids 2>/dev/null; fi"
  sleep 15
}
PROBLEM=xor_grid_bcast
PORT=65200
for VARIANT in baseline kiss_pick; do
  PORT=$((PORT + 4))
  echo "=== [$(date -u -Iseconds)] $PROBLEM $VARIANT port=$PORT ==="
  clean_zombies
  export PROBLEM=$PROBLEM VARIANT=$VARIANT
  ssh -f -n -o StrictHostKeyChecking=no -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$WORKER_IP \
    "export PATH=/home/ubuntu/venv/bin:\$PATH; export LD_LIBRARY_PATH=/opt/aws/neuron/lib; export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1; export PROBLEM=$PROBLEM VARIANT=$VARIANT; nohup /home/ubuntu/venv/bin/torchrun --nnodes=2 --node_rank=1 --nproc_per_node=32 --rdzv_backend=static --master_addr=$MASTER_IP --master_port=$PORT /tmp/real_training_multi.py > /tmp/w_rtxor_${VARIANT}.log 2>&1 &"
  sleep 5
  timeout 300 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=32 --rdzv_backend=static --master_addr=$MASTER_IP --master_port=$PORT /tmp/real_training_multi.py > /home/ubuntu/rt_xor/${VARIANT}_m.log 2>&1
  MS=$(grep '^REAL_MS_PER_ITER' /home/ubuntu/rt_xor/${VARIANT}_m.log 2>/dev/null | awk '{print $2}')
  LOSS_F=$(grep '^REAL_LOSS_FINAL' /home/ubuntu/rt_xor/${VARIANT}_m.log 2>/dev/null | awk '{print $2}')
  echo "$PROBLEM,$VARIANT,${MS:-NA},${LOSS_F:-NA}" >> /home/ubuntu/rt_xor/summary.csv
done
echo "RT_XOR DONE $(date -u -Iseconds)"
touch /tmp/rt_xor_done
aws s3 sync /home/ubuntu/rt_xor s3://overlayccl-kaiyao-artifacts/2026-07-24-cb-v4/rt_xor/ 2>&1 | tail -3
