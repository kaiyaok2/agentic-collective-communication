#!/bin/bash
# queue6 (after queue5): second pass of the per-problem sweep for variance (n=2 per candidate),
# plus PP S=512 baseline/paper repeat.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "E2-pass2: full per-problem sweep repeat (variance)"
SWEEP_OUT=/home/ubuntu/sweep_results_pass2 bash $AP/sweep_candidates.sh 'alltoallv|dxe|fsdp|llama_block|ring_kv|tp_mlp' >> $Q 2>&1
stage "E2c-pass2: PP S=512 baseline + paper repeat"
VP=training/train_pp_send_recv_7node_s512.py
export RESULTS_DIR=/home/ubuntu/sweep_results_pass2/raw_pp512; mkdir -p $RESULTS_DIR
for spec in "harness_baseline:baseline:" "paper_7f79b897:agent:$AP/cands/pp_send_recv__paper__7f79b897.py"; do
  IFS=: read -r T B F <<< "$spec"
  [ -n "$F" ] && python3 $AP/deploy_candidate.py pp_send_recv --file $F >> $Q
  rm -f $RESULTS_DIR/pp_send_recv_${B}_step.json; for ip in $WORKERS; do ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip "mkdir -p $RESULTS_DIR; rm -f $RESULTS_DIR/pp_send_recv_${B}_step.json" </dev/null; done
  timeout 1800 bash $AP/run_ap7.sh $VP $B 300 pp512p2_$T >> $Q 2>&1
  J=$RESULTS_DIR/pp_send_recv_${B}_step.json
  [ -f $J ] || for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$J $J 2>/dev/null && break; done
  [ -f $J ] && python3 -c "import json; d=json.load(open('$J')); d.update(problem='pp_send_recv_S512', cand='$T'); d.pop('step_times_ms',None); d.pop('losses',None); d['measured_ms']=d.get('steady_ms') or d.get('avg_ms'); json.dump(d,open('/home/ubuntu/sweep_results_pass2/pp_send_recv_S512__$T.json','w'),indent=1); print('   pp512p2 $T', d['measured_ms'])" | tee -a $Q
  python3 $AP/deploy_candidate.py pp_send_recv --restore >/dev/null
done
stage "QUEUE6 DONE"
