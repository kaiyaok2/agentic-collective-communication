#!/bin/bash
# Llama-7B end-to-end (experiments/model_extension/train_llama_e2e_7b.py) on the rebuttal cluster.
# Usage: bash run_llama7b.sh <per_mb|bundled> <steps> <tag>
# Env: COLD=1 wipes /tmp/neuron_cache on all nodes first (wall time then includes compile).
# Output: /home/ubuntu/llama7b_results/<tag>.json (harvested from whichever node held the bench rank),
#         wall seconds appended to /home/ubuntu/llama7b_results/runs.log
set -uo pipefail
BACKEND=${1:?per_mb|bundled}; STEPS=${2:-6000}; TAG=${3:-${BACKEND}_${STEPS}}
REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
WORKERS=${WORKERS_OVERRIDE:-$WORKERS}
MASTER=${MASTER_PRIV:-$(hostname -I | awk '{print $1}')}
PORT=${PORT:-29610}
NNODES=$(( $(echo $WORKERS | wc -w) + 1 ))
OUT=/home/ubuntu/llama7b_results; LOG_DIR=$OUT/logs; mkdir -p $LOG_DIR
KEY=/home/ubuntu/.ssh/Kaiyao.pem
SSHK="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
COMMON_ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && \
export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
export OFI_NCCL_USE_IPV6_TCP=0 && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT && export PYTHONPATH=$REPO"
TRUN="--nproc_per_node=32 --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"
SCRIPT=${SCRIPT:-$REPO/experiments/model_extension/train_llama_e2e_7b.py}

echo "=== [$(date -u +%T)] llama7b backend=$BACKEND steps=$STEPS tag=$TAG cold=${COLD:-0} ==="
for ip in $WORKERS; do
  rsync -az --delete -e "$SSHK" --exclude='.git' --exclude='__pycache__' --exclude='training/results' "$REPO/" ubuntu@$ip:$REPO/ &
done; wait
mkdir -p /tmp/tp_search; for ip in $WORKERS; do $SSHK ubuntu@$ip "mkdir -p /tmp/tp_search; rm -f /tmp/tp_search/llama_e2e_amp2_${BACKEND}*.json; pkill -9 -f torchrun 2>/dev/null; true" </dev/null; done
rm -f /tmp/tp_search/llama_e2e_amp2_${BACKEND}*.json; pkill -9 -f torchrun 2>/dev/null
if [ "${COLD:-0}" = "1" ]; then rm -rf /tmp/neuron_cache; for ip in $WORKERS; do $SSHK ubuntu@$ip "rm -rf /tmp/neuron_cache" </dev/null; done; echo "  cold cache"; fi
for i in $(seq 1 30); do busy=0; [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] || busy=1
  for ip in $WORKERS; do [ "$($SSHK ubuntu@$ip 'sudo fuser /dev/neuron* 2>/dev/null | wc -w' </dev/null 2>/dev/null)" = "0" ] || busy=1; done
  [ $busy = 0 ] && break; sleep 3; done
[ $busy = 0 ] && echo "  neuron cores free" || echo "  WARNING: neuron cores still held after 90s"
T0=$(date +%s); NODE_RANK=1; PIDS=""
for ip in $WORKERS; do
  $SSHK ubuntu@$ip "$COMMON_ENV && cd $REPO && $VENV/bin/torchrun $TRUN --node_rank=$NODE_RANK $SCRIPT $BACKEND $STEPS" > $LOG_DIR/${TAG}_node${NODE_RANK}.log 2>&1 </dev/null &
  PIDS="$PIDS $!"; NODE_RANK=$((NODE_RANK+1))
done
bash -c "$COMMON_ENV && cd $REPO && $VENV/bin/torchrun $TRUN --node_rank=0 $SCRIPT $BACKEND $STEPS" > $LOG_DIR/${TAG}_node0.log 2>&1
RC=$?; wait $PIDS; T1=$(date +%s)
# harvest the bench-rank json from whichever node has it
for ip in $WORKERS; do scp -q -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:/tmp/tp_search/llama_e2e_amp2_${BACKEND}.json $OUT/${TAG}.json 2>/dev/null && break; done
[ -f $OUT/${TAG}.json ] || cp /tmp/tp_search/llama_e2e_amp2_${BACKEND}.json $OUT/${TAG}.json 2>/dev/null
echo "  rc=$RC wall=$((T1-T0))s"; grep -h "\[bench\]" $LOG_DIR/${TAG}_node*.log | head -2
echo "TAG=$TAG BACKEND=$BACKEND STEPS=$STEPS COLD=${COLD:-0} WALL_S=$((T1-T0)) RC=$RC $(date -u +%FT%TZ)" >> $OUT/runs.log
exit $RC
