#!/bin/bash
# Generic N-node torchrun launcher for the rebuttal cluster (run ON master).
# Usage: bash run_ap7.sh <train_script_rel_to_repo> <backend> <steps> <out_tag> [extra args...]
# Env: WORKERS (space-separated worker private IPs; default from /home/ubuntu/venv.env),
#      MASTER_PRIV (default: first private IP), PORT (default 29600), RESULTS_DIR, WARMUP,
#      COLD=1 -> wipe /tmp/neuron_cache on all nodes before the run (compile-time measurement).
set -uo pipefail
TRAIN_REL=${1:?train script}; BACKEND=${2:?backend}; STEPS=${3:-300}; OUT_TAG=${4:-$BACKEND}; shift 4 || true
EXTRA="$*"
REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env            # VENV=..., WORKERS="..."
WORKERS=${WORKERS_OVERRIDE:-${WORKERS:?}}
MASTER=${MASTER_PRIV:-$(hostname -I | awk '{print $1}')}
PORT=${PORT:-29600}
WARMUP=${WARMUP:-5}
NPROC=32
NNODES=$(( $(echo $WORKERS | wc -w) + 1 ))
RESULTS_DIR=${RESULTS_DIR:-$REPO/training/results/7node}
LOG_DIR=$RESULTS_DIR/logs
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
KEY=/home/ubuntu/.ssh/Kaiyao.pem
SSHK="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
TORCHRUN=$VENV/bin/torchrun
COMMON_ENV="export PATH=$VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH && \
export NEURON_RT_NUM_CORES=32 && export NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache && \
export FI_PROVIDER=efa && export FI_EFA_USE_DEVICE_RDMA=1 && export FI_EFA_FORK_SAFE=1 && \
export OFI_NCCL_USE_IPV6_TCP=0 && export MASTER_ADDR=$MASTER && export MASTER_PORT=$PORT && \
export RESULTS_DIR=$RESULTS_DIR && export PYTHONPATH=$REPO && export PERCALL_PROBE=${PERCALL_PROBE:-1} PROBE_START_STEP=${PROBE_START_STEP:-280} PROBE_END_STEP=${PROBE_END_STEP:-300}"
TRUN="--nproc_per_node=$NPROC --nnodes=$NNODES --rdzv_backend=c10d --rdzv_endpoint=${MASTER}:${PORT}"

echo "=== [$(date -u +%T)] $TRAIN_REL backend=$BACKEND steps=$STEPS tag=$OUT_TAG nnodes=$NNODES ==="
# sync code to workers (runtime/ edits must reach every node: torchrun uses each node's LOCAL copy)
for ip in $WORKERS; do
  rsync -az --delete -e "$SSHK" --exclude='.git' --exclude='__pycache__' --exclude='training/results' "$REPO/" ubuntu@$ip:$REPO/ &
done; wait
if [ "${COLD:-0}" = "1" ]; then
  rm -rf /tmp/neuron_cache; for ip in $WORKERS; do $SSHK ubuntu@$ip "rm -rf /tmp/neuron_cache" </dev/null; done
  echo "  cold cache on all nodes"
fi
# kill stale neuron procs
for ip in $WORKERS; do $SSHK ubuntu@$ip "pkill -9 -f torchru[n] 2>/dev/null; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]' 2>/dev/null; true" </dev/null; done
pkill -9 -f 'torchru[n]' 2>/dev/null; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]' 2>/dev/null
# wait until no process holds a Neuron device on any node (cores are released a few seconds after a kill)
free_cores () { [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ]; }
for i in $(seq 1 30); do
  busy=0; free_cores || busy=1
  for ip in $WORKERS; do [ "$($SSHK ubuntu@$ip 'sudo fuser /dev/neuron* 2>/dev/null | wc -w' </dev/null 2>/dev/null)" = "0" ] || busy=1; done
  [ $busy = 0 ] && break; sleep 3
done
[ $busy = 0 ] && echo "  neuron cores free" || echo "  WARNING: neuron cores still held after 90s"
for i in $(seq 1 20); do pb=0
  (ss -ltn 2>/dev/null | grep -qE ":(62182|$PORT) ") && pb=1
  for ip in $WORKERS; do [ "$($SSHK ubuntu@$ip "ss -ltn 2>/dev/null | grep -cE ':(62182|$PORT) '" </dev/null 2>/dev/null)" = "0" ] || pb=1; done
  [ $pb = 0 ] && break; sleep 3; done
[ $pb = 0 ] && echo "  ports free" || echo "  WARNING: port 62182/$PORT still bound"

T0=$(date +%s)
NODE_RANK=1; PIDS=""
for ip in $WORKERS; do
  LOG="$LOG_DIR/${OUT_TAG}_node${NODE_RANK}.log"
  $SSHK ubuntu@$ip "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=$NODE_RANK $REPO/$TRAIN_REL --backend $BACKEND --steps $STEPS --warmup $WARMUP $EXTRA" > "$LOG" 2>&1 </dev/null &
  PIDS="$PIDS $!"; NODE_RANK=$((NODE_RANK+1))
done
LOG0="$LOG_DIR/${OUT_TAG}_node0.log"
bash -c "$COMMON_ENV && cd $REPO && $TORCHRUN $TRUN --node_rank=0 $REPO/$TRAIN_REL --backend $BACKEND --steps $STEPS --warmup $WARMUP $EXTRA" > "$LOG0" 2>&1
RC=$?
wait $PIDS
T1=$(date +%s)
echo "  rc=$RC wall=$((T1-T0))s"
grep -hE "Steady step|steady_mean|steady_ms|ms/step|final_loss|Final loss" $LOG_DIR/${OUT_TAG}_node*.log | head -4
echo "WALL_S=$((T1-T0)) RC=$RC TAG=$OUT_TAG BACKEND=$BACKEND STEPS=$STEPS COLD=${COLD:-0} $(date -u +%FT%TZ)" >> $RESULTS_DIR/runs.log
exit $RC
