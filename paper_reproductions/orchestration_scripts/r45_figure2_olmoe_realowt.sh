#!/bin/bash
# r45: OLMoE-10B 2500-step real-OpenWebText loss curve (Figure 2, Appendix A).
# Runs baseline and agent backends back-to-back on the 7-node cluster, then
# regenerates figures/loss_curve.pdf from the per-step loss arrays.
#
# Cluster: 7 x trn1.32xlarge (ws=224). Set MASTER + WORKERS env or accept
# defaults.
#
# NOTE: 2500 steps is hours of cluster time and was not re-run during the
# session-recovery; this script is the documented path that produced the
# published trajectory.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH
REPO=/home/ubuntu/agentic-collective-communication
VENV=/opt/aws_neuronx_venv_pytorch_2_9
TORCHRUN=$VENV/bin/torchrun
KEY=${SSH_KEY:-/home/ubuntu/.ssh/Kaiyao.pem}
MASTER=${MASTER_IP:-172.31.19.201}
WORKERS=(${WORKER_IPS:-172.31.17.80 172.31.24.136 172.31.27.22 172.31.18.238 172.31.20.12 172.31.27.240})
R45=/home/ubuntu/r45_figure2
mkdir -p $R45
HB() { echo "$(date -u +%H:%M:%S) $*" | tee -a $R45/HEARTBEAT; }

rsync_repo() {
  for ip in "${WORKERS[@]}"; do
    rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
      --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
      "$REPO/" "ubuntu@$ip:$REPO/" >/dev/null 2>&1 &
  done
  wait
}

run() {
  local NAME=$1; local BACKEND=$2; local STEPS=$3; local PORT=$4
  HB "$NAME start (backend=$BACKEND steps=$STEPS realtok=ON)"
  local OUT=$R45/$NAME
  mkdir -p $OUT
  rsync_repo
  local LSH=/tmp/r45_launch_${PORT}.sh
  cat > $LSH <<EOF
#!/bin/bash
unset NEURON_RT_NUM_SOFTWARE_NQS
source $VENV/bin/activate
mkdir -p $OUT
cd $REPO
RESULTS_DIR=$OUT exec $TORCHRUN --nnodes=7 --node_rank=__NR__ --nproc_per_node=32 \
  --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
  training/train_olmoe10b.py --backend $BACKEND --ce $BACKEND --grad-sync baseline \
  --steps $STEPS --realtok > $OUT/node___NR__.log 2>&1
EOF
  for i in "${!WORKERS[@]}"; do
    local NR=$((i+1))
    local ip=${WORKERS[$i]}
    sed "s/__NR__/$NR/g" $LSH > $LSH.$NR
    scp -i $KEY -o StrictHostKeyChecking=no -q $LSH.$NR ubuntu@$ip:$LSH &
  done
  wait
  for i in "${!WORKERS[@]}"; do
    local NR=$((i+1))
    local ip=${WORKERS[$i]}
    ssh -fn -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$ip \
      "chmod +x $LSH && nohup bash $LSH < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
  done
  sleep 3
  source $VENV/bin/activate
  RESULTS_DIR=$OUT $TORCHRUN --nnodes=7 --node_rank=0 --nproc_per_node=32 \
    --rdzv_backend=static --master_addr=$MASTER --master_port=$PORT \
    training/train_olmoe10b.py --backend $BACKEND --ce $BACKEND --grad-sync baseline \
    --steps $STEPS --realtok > $OUT/node_0.log 2>&1
  HB "$NAME exit=$?"
}

HB 'r45 Figure-2 OLMoE real-OWT 2500-step start'
run baseline_2500 baseline 2500 29801
run agent_2500    agent    2500 29802
HB 'r45 done -- regenerating figures/loss_curve.pdf'
source $VENV/bin/activate
python figures/plot_loss_curve.py \
  --baseline $R45/baseline_2500/olmoe10b_a2av-baseline_gs-baseline_ce-baseline.json \
  --agent    $R45/agent_2500/olmoe10b_a2av-agent_gs-baseline_ce-agent.json \
  --out figures/loss_curve.pdf
HB 'r45 figure regenerated'
