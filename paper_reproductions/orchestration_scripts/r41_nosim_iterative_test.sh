#!/bin/bash
# r41: Test iterative no-sim search on ua2a and alltoallv with new
# Phase-3 iterative refinement + small-shape Phase-5 LLM-judge.
# Verify both converge to AG+T+RS / AG+RS before scaling up.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
[ -f /home/ubuntu/.anthropic_env ] && . /home/ubuntu/.anthropic_env
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
KEY=/home/ubuntu/.ssh/Kaiyao.pem
MASTER=172.31.19.201
WORKERS=(172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240)
R41=/home/ubuntu/r41_nosim_test
mkdir -p $R41

HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R41/HEARTBEAT; }

backup_main_runtimes() {
  mkdir -p $R41/runtime_main_backup
  cp $REPO/runtime/trainium_*.py $R41/runtime_main_backup/
}
restore_main_runtimes() {
  cp $R41/runtime_main_backup/trainium_*.py $REPO/runtime/
}

HB "R41 start: testing iterative no-sim on ua2a + alltoallv"
backup_main_runtimes
WORKER_CSV=$(IFS=','; echo "${WORKERS[*]}")
cd $REPO
source $VENV/bin/activate
mkdir -p $R41/search_logs

for P in uniform_a2a alltoallv; do
  HB "  search $P --no-simulator iterative start"
  mkdir -p $R41/search_logs/$P
  timeout 1500 python3 -u experiments/run_search.py \
      --problem $P --phase3-style strategy-enumerate --no-simulator --hw-eval \
      --llm-model sonnet --max-rounds 4 \
      --num-nodes 7 --master-addr $MASTER --worker-addrs $WORKER_CSV \
      --output-dir $R41/search_logs/$P > $R41/search_logs/$P/search.log 2>&1
  HB "    search $P exit=$?"
  mkdir -p $R41/deployed_runtimes
  [ -f $REPO/runtime/trainium_${P}_7node.py ] && cp $REPO/runtime/trainium_${P}_7node.py $R41/deployed_runtimes/
  [ -f $REPO/runtime/trainium_${P}.py ] && cp $REPO/runtime/trainium_${P}.py $R41/deployed_runtimes/
done

HB "deployed code summary:"
for f in $R41/deployed_runtimes/trainium_*.py; do
  desc=$(grep -m1 '"""' $f | head -1 | tr -d '"')
  HB "  $(basename $f): $desc"
done

# Restore main so cluster is clean
restore_main_runtimes
HB "R41 done"
