#!/bin/bash
# queue8 (after queue7): A-Q1 — collective fraction + larger-token configs that fit on 7 nodes.
# nocomm = agent path with an identity AllToAllV stub (timing only). f = 1 - T_nocomm/T_baseline.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
mk () { # <name> <bsz> <seq> <layers> <dm>
  V=training/train_olmoe10b_$1.py
  sed -E "s/^BSZ *=.*/BSZ       = $2/; s/^SEQLEN *=.*/SEQLEN    = $3/; s/^LAYERS *=.*/LAYERS    = $4/; s/^DM *=.*/DM        = $5/; s/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/$V
  echo $V
}
run () { # <tag> <script> <backend> <runtime-file|->
  T=$1; V=$2; B=$3; RT=$4
  [ "$RT" != "-" ] && cp $RT $REPO/runtime/trainium_alltoallv_7node.py || cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D timeout 3000 bash $AP/run_ap7.sh $V $B 160 $T $FL >> $Q 2>&1; harvest $D
  cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
}
stage "E4d collective fraction at the headline config (nocomm stub) + larger-token configs that fit"
V0=$(mk hl_w100 1 256 8 2048)
run olmoe_hl_nocomm $V0 agent $AP/cands/alltoallv__nocomm__olmoe.py
for cfg in "B2_S256_L8_DM1024:2:256:8:1024" "B4_S256_L8_DM1024:4:256:8:1024" "B1_S512_L16_DM1024:1:512:16:1024"; do
  IFS=: read -r name bsz seq lay dm <<< "$cfg"
  V=$(mk $name $bsz $seq $lay $dm)
  run olmoe_${name}_baseline $V baseline -
  run olmoe_${name}_agent    $V agent    -
  run olmoe_${name}_nocomm   $V agent    $AP/cands/alltoallv__nocomm__olmoe.py
done
stage "QUEUE8 DONE"
