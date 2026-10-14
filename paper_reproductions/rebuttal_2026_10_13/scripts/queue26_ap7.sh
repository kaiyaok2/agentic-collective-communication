#!/bin/bash
# queue26 (last runs on this cluster): TOPK=4 + EXDIM=8192 x agent/stub/baseline; then 2 more reps of ring_kv found_ga.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
run () { T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 1000 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py; }
sed -E 's/^TOPK *=.*/TOPK      = 4/; s/^EXDIM *=.*/EXDIM     = 8192/; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/' $REPO/training/train_olmoe10b.py > $REPO/training/train_olmoe10b_topk4ex8192.py
stage "QUEUE26 OLMoE TOPK=4 + EXDIM=8192 x agent/stub/baseline"
run olmoe_topk4ex8192_agent    training/train_olmoe10b_topk4ex8192.py agent    -
run olmoe_topk4ex8192_nocomm   training/train_olmoe10b_topk4ex8192.py agent    $AP/cands/alltoallv__nocomm__olmoe.py
run olmoe_topk4ex8192_baseline training/train_olmoe10b_topk4ex8192.py baseline -
stage "QUEUE26 ring_kv found_ga repeats"
python3 -c "import json; json.dump([c for c in json.load(open('$AP/manifest_found.json')) if c['problem']=='ring_kv'], open('$AP/manifest_rkvfound.json','w'))"
for r in 2 3; do MANIFEST=$AP/manifest_rkvfound.json SWEEP_OUT=/home/ubuntu/sweep_found_r$r bash $AP/sweep_candidates.sh >> $Q 2>&1; done
stage "QUEUE26 DONE"
