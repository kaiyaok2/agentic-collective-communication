#!/bin/bash
# R28-nosim ablation: run strategy-enumerate WITHOUT simulator-guided
# refinement; deploy results, compare to baseline / r28 main.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u

# Anthropic key - read from env file (kept out of git)
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env

REPO=/tmp/r28_smoke/acc-r28
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)

R28N=/home/ubuntu/r28_nosim
mkdir -p $R28N

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R28N/HEARTBEAT; }

# Problems to search (8-problem set per r28 paper)
PROBLEMS=(alltoallv uniform_a2a ring_kv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar)

HB "R28-nosim start"
cd $REPO
source $VENV/bin/activate

# Pattern map (alltoallv-family uses 'moe', others default 'moe')
PATTERN="moe"

for P in "${PROBLEMS[@]}"; do
  OUT=$R28N/search/$P
  mkdir -p $OUT
  HB "  search $P (--no-simulator) start"
  WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
  # Run search in background; capture output
  timeout 1200 python3 -u experiments/run_search.py \
      --problem $P --phase3-style strategy-enumerate --no-simulator \
      --llm-model sonnet --max-rounds 4 \
      --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
      --output-dir $OUT > $OUT/search.log 2>&1
  EX=$?
  HB "  search $P exit=$EX (see $OUT/search.log)"
done

HB "R28-nosim search complete"

# Summarize candidates
HB "=== Summary ==="
for P in "${PROBLEMS[@]}"; do
  OUT=$R28N/search/$P
  if [ -f $OUT/search.log ]; then
    LINE=$(grep -E "Winner|No correct candidates|Best|FALLBACK|fallback" $OUT/search.log | tail -3 | tr '\n' '|')
    HB "  $P: $LINE"
  fi
done

HB "R28-nosim done"
