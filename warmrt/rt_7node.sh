#!/bin/bash
# Warm-cache RT of ONE candidate file across all 7 nodes (224 ranks) via rt_diverge.py.
# Usage: rt_7node.sh <RUNTIME_FILE> <PROBLEM> <MASTER_IP> "<WORKER_IP_1 ... WORKER_IP_6>" [N_ITERS]
# Runs ON the master. Launches torchrun node_rank=0 locally + node_rank=i on each worker.
set -u
RUNTIME_FILE=$1; PROBLEM=$2; MASTER_IP=$3; WORKERS=$4; N_ITERS=${5:-50}
LOG_PREFIX=/tmp/rt7_$$
NPROC=32
MASTER_PORT=29500
read -r -a WARR <<< "$WORKERS"
NNODES=$(( ${#WARR[@]} + 1 ))
NEURON_VENV=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_8}

# NEURON_RT_NUM_CORES + MASTER_ADDR/MASTER_PORT are load-bearing at multi-node:
# torch_xla auto-derives NEURON_RT_ROOT_COMM_ID=${MASTER_ADDR}:62182 from them,
# which is how cross-node CCOM forms. Without them collectives silently hang.
ENV="export PATH=$NEURON_VENV/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:\$PATH \
NEURON_RT_NUM_CORES=$NPROC NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache \
MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT \
FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 PJRT_DEVICE=NEURON \
NEURON_RT_LOG_LEVEL=ERROR ACC_REPO=/home/ubuntu/agentic-collective-communication \
PROBLEM=$PROBLEM RUNTIME_FILE=$RUNTIME_FILE N_ITERS=$N_ITERS NUM_NODES=$NNODES"
TR="timeout 3000 torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
--rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:$MASTER_PORT"
SCRIPT_PY=/home/ubuntu/agentic-collective-communication/warmrt/rt_diverge.py

# distribute the candidate file to workers (repo/search already present on each node)
for i in "${!WARR[@]}"; do
  W=${WARR[$i]}
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@"$W" \
    "mkdir -p $(dirname "$RUNTIME_FILE")" 2>/dev/null || true
  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q \
    "$RUNTIME_FILE" ubuntu@"$W":"$RUNTIME_FILE" 2>/dev/null || true
done

kill_all_torchruns() {
  pkill -9 -f "torchrun .*rdzv_endpoint=$MASTER_IP:$MASTER_PORT" 2>/dev/null
  pkill -9 -f "rt_diverge.py" 2>/dev/null
  for W in "${WARR[@]}"; do
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@"$W" \
      "pkill -9 -f torchrun 2>/dev/null; pkill -9 -f rt_diverge.py 2>/dev/null" 2>/dev/null || true
  done
}

# master = node_rank 0
bash -c "cd /home/ubuntu && $ENV && $TR --node_rank=0 $SCRIPT_PY" > ${LOG_PREFIX}_r0.log 2>&1 &
PIDS=($!)
# workers = node_rank 1..N-1
for i in "${!WARR[@]}"; do
  W=${WARR[$i]}; NR=$(( i + 1 ))
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@"$W" \
    "cd /home/ubuntu && $ENV && $TR --node_rank=$NR $SCRIPT_PY" > ${LOG_PREFIX}_r${NR}.log 2>&1 &
  PIDS+=($!)
done

# --- straggler watchdog ---------------------------------------------------
# A dead peer makes the survivors spin in a barrier-retry loop emitting
# "Connection refused - retrying" for the full 3000s timeout. That signature
# NEVER appears during a healthy compile (only when a peer proc is gone), so
# keying on it is safe: kill early -> a one-node death fails in ~1 min, not 50.
( 
  for _ in $(seq 1 120); do
    sleep 5
    grep -qE "RT_TIME_MS_PER_ITER" ${LOG_PREFIX}_r*.log 2>/dev/null && break
    if grep -qE "Connection refused - retrying|CCOM WARN Connect to|Timeout waiting for incoming connection|barrier recv:|Root Cause \(first observed failure\)|torch.distributed.elastic.multiprocessing.api: .*failed" ${LOG_PREFIX}_r*.log 2>/dev/null; then
      echo "[watchdog] barrier-hang signature -> killing torchruns (dead peer)" >> ${LOG_PREFIX}_r0.log
      kill_all_torchruns
      break
    fi
  done
) &
WATCHDOG_PID=$!

for p in "${PIDS[@]}"; do wait "$p"; done
kill "$WATCHDOG_PID" 2>/dev/null; wait "$WATCHDOG_PID" 2>/dev/null

# core-settle: guarantee this rep's ranks release the Neuron cores before we
# return -- the sweep launches the next rep immediately, and a lingering rank
# (esp. after a watchdog kill) holding a core makes the next rep die with
# NRT_FAILURE. Best-effort, ~30s cap, master + workers.
for _ in $(seq 1 15); do
  nm=$(pgrep -f "rt_diverge\\.py" 2>/dev/null | wc -l)
  [ "${nm:-0}" -le 0 ] && break
  pgrep -f "rt_diverge\\.py" | xargs -r kill -9 2>/dev/null
  sleep 2
done
for W in "${WARR[@]}"; do
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@"$W" \
    "pgrep -f rt_diverge.py | xargs -r kill -9 2>/dev/null" 2>/dev/null || true
done

# PJRT/Neuron assigns xla ordinal 0 to an arbitrary physical proc, so rank 0's
# RT_TIME print can land in ANY node's log — grep them all.
MS=$(grep -h 'RT_TIME_MS_PER_ITER' ${LOG_PREFIX}_r*.log 2>/dev/null | awk '{print $2}' | head -1)
echo "MS_PER_ITER=${MS:-FAIL}"
if [ -z "${MS:-}" ]; then echo "---- r0 tail ----"; tail -25 ${LOG_PREFIX}_r0.log; fi
