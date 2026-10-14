#!/bin/bash
# queue24: bisect the B>1 NRT_EXEC_OOB. R1: B=2 DM1024 with the baseline CE (isolates the dxe path);
# R2: B=1 S=512 DM1024 (T=512 without a batch dim); each with the agent AllToAllV. Then baseline backend of R2.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3; FL=$4
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 1800 bash $AP/run_ap7.sh $V $B 120 $T $FL >> $Q 2>&1; harvest $D; }
sed -E 's/^BSZ *=.*/BSZ       = 1/; s/^SEQLEN *=.*/SEQLEN    = 512/' $REPO/training/train_olmoe10b_B2fix_DM1024.py > $REPO/training/train_olmoe10b_B1S512fix_DM1024.py
stage "QUEUE24 bisect OOB: R1 B2 DM1024 agent a2av + BASELINE ce"
run olmoe_B2fix_DM1024_agent_cebase training/train_olmoe10b_B2fix_DM1024.py agent "--ce baseline"
stage "QUEUE24 R2: B1 S512 DM1024 agent (ce agent), then baseline backend"
run olmoe_B1S512fix_DM1024_agent training/train_olmoe10b_B1S512fix_DM1024.py agent "--ce agent"
run olmoe_B1S512fix_DM1024_baseline training/train_olmoe10b_B1S512fix_DM1024.py baseline ""
stage "QUEUE24 DONE"
