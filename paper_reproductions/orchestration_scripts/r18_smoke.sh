#!/bin/bash
# R18 smoke: 50-step OLMoE baseline with new add_step_closure probes.
# Confirms no NRT_RESOURCE OOM before launching the full multi-stack rerun.
set +u
[ -f "$HOME/.profile" ] && . "$HOME/.profile"
set -u
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKER_LIST=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
OUT=/home/ubuntu/r18_smoke_olmoe
mkdir -p $OUT
# Restore developer baseline runtime (R16 helper).
cd $REPO && git checkout main -- runtime/ 2>/dev/null
git show acc-orphan-v6:runtime/trainium_grad_ar_7node.py > $REPO/runtime/trainium_grad_ar_7node.py 2>/dev/null
for ip in "${WORKER_LIST[@]}"; do
  rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=training/results \
    "$REPO/" "ubuntu@$ip:$REPO/" &
done
wait
pkill -9 -f torchrun 2>/dev/null || true
for ip in "${WORKER_LIST[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip "pkill -9 -f torchrun" 2>/dev/null &
done
wait
sleep 2
port=42777
ENV_VARS="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
  export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache && \
  export NEURON_CC_FLAGS='--hbm-scratchpad-page-size=64 --optlevel=2' && export NEURON_SCRATCHPAD_PAGE_SIZE=64 && \
  export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
  export MASTER_ADDR=$MASTER && export MASTER_PORT=$port && export RESULTS_DIR=$OUT && \
  export PERCALL_PROBE=1"
TRUN="--nproc_per_node=32 --nnodes=7 --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${port}"
ARGS="--backend baseline --grad-sync baseline --ce baseline --steps 50 --warmup 5"
NR=1
for ip in "${WORKER_LIST[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$ip \
    "$ENV_VARS && cd $REPO && $VENV/bin/torchrun $TRUN --node_rank=$NR $REPO/training/train_olmoe10b.py $ARGS" \
    > $OUT/w${NR}.log 2>&1 &
  NR=$((NR+1))
done
eval "$ENV_VARS"; cd $REPO
timeout 1500 $VENV/bin/torchrun $TRUN --node_rank=0 $REPO/training/train_olmoe10b.py $ARGS > $OUT/m0.log 2>&1
echo "exit=$?"
wait
ls -la $OUT/*.json 2>/dev/null
