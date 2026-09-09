#!/bin/bash
set -uo pipefail
# Usage: bash rt_launch.sh <problem> <tag> <runtime_file>
# Launches rt_taxonomy.py across the 7-node CB (224 ranks) via torchrun.
PROBLEM=${1:?problem}
TAG=${2:?tag}
RUNTIME_FILE=${3:?runtime_file}
NPROC=32; NNODES=7
MASTER=172.31.19.204
WORKERS=(172.31.26.146 172.31.26.149 172.31.22.121 172.31.28.85 172.31.19.153 172.31.31.94)
VENV=/opt/aws_neuronx_venv_pytorch_2_8
KEY=/home/ubuntu/.ssh/Kaiyao.pem
SCRIPT=/home/ubuntu/rt_taxonomy.py
PORT=${PORT:-32611}
RT_WORLD=${RT_WORLD:-224}
RT_ITERS=${RT_ITERS:-20}
RT_OUTDIR=${RT_OUTDIR:-/home/ubuntu/rt_sweep}
CACHE=${CACHE:-/home/ubuntu/neuron_cache_rt}
LOGD=/home/ubuntu/rt_sweep/logs/${PROBLEM}.${TAG}
mkdir -p "$LOGD" "$RT_OUTDIR"

REPO=/home/ubuntu/acc
# Sync script + runtime file + repo bits workers need.
RT_DIR=$(dirname "$RUNTIME_FILE")
for ip in "${WORKERS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8 ubuntu@$ip "mkdir -p /home/ubuntu/rt_sweep $CACHE $RT_DIR" 2>/dev/null
  scp -q -i $KEY -o StrictHostKeyChecking=no "$SCRIPT" ubuntu@$ip:"$SCRIPT"
  scp -q -i $KEY -o StrictHostKeyChecking=no "$RUNTIME_FILE" ubuntu@$ip:"$RUNTIME_FILE"
  # sync the search package (workers lack /home/ubuntu/acc)
  rsync -a -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    --exclude='__pycache__' --exclude='session_logs*' --exclude='.git' \
    /home/ubuntu/acc/search /home/ubuntu/acc/experiments ubuntu@$ip:/home/ubuntu/acc/ 2>/dev/null
done

ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH \
&& export NEURON_RT_NUM_CORES=32 && export NEURON_NUM_RECENT_MODELS_TO_KEEP=1 \
&& export NEURON_COMPILE_CACHE_URL=$CACHE \
&& export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 \
&& export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT && export PYTHONPATH=$REPO \
&& export PROBLEM=$PROBLEM && export RT_TAG=$TAG && export RUNTIME_FILE=$RUNTIME_FILE \
&& export RT_WORLD=$RT_WORLD && export RT_ITERS=$RT_ITERS && export RT_OUTDIR=$RT_OUTDIR \
&& export OMP_NUM_THREADS=1"
TRUN="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "[rt] $(date -u) launch $PROBLEM/$TAG (port=$PORT world=$RT_WORLD)"
NR=1
for ip in "${WORKERS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
    "$ENV && torchrun $TRUN --node_rank=$NR $SCRIPT" > "$LOGD/n${NR}.log" 2>&1 &
  NR=$((NR+1))
done
eval "$ENV"
timeout 900 torchrun $TRUN --node_rank=0 $SCRIPT > "$LOGD/n0_master.log" 2>&1
RC=$?
echo "[rt] $PROBLEM/$TAG master_rc=$RC $(date -u)"
wait
# rank 0 can land on any node under c10d rendezvous; harvest RT_RESULT from ALL
# node logs and (re)write the json on master so the orchestrator finds it.
RES=$(grep -h "RT_RESULT" "$LOGD"/*.log 2>/dev/null | head -1)
echo "$RES"
if [ -n "$RES" ]; then
  echo "${RES#RT_RESULT }" > "$RT_OUTDIR/${PROBLEM}.${TAG}.json"
fi
exit $RC
