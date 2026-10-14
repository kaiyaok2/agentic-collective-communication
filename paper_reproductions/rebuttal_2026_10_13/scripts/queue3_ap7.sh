#!/bin/bash
# E3b: OLMoE-scope (whole 224-rank training step) measurement of the AllToAllV alternatives, deployed
# through the agent path, so sim-vs-measured can be stated at the scope the headline uses (C-Q1/A-Q2).
# ag_slice_cat (paper) and the developer baseline are covered by E5; here: AG+RS via agent path, and
# the ag_t_rs_inline_match seed template. 160 steps, steady window from step 100.
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL
source /home/ubuntu/venv.env
Q=/home/ubuntu/queue.log
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "E3b OLMoE-scope AllToAllV alternatives via agent path (160 steps)"
V=training/train_olmoe10b_w100.py
sed -E "s/times\[200:\]/times[100:]/; s/len\(times\) - 200/len(times) - 100/; s/from step 200/from step 100/" $REPO/training/train_olmoe10b.py > $REPO/$V
run_one () {  # <tag> <deploy args...>
  T=$1; shift
  python3 $AP/deploy_candidate.py alltoallv "$@" | tee -a $Q
  D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  RESULTS_DIR=$D timeout 3600 bash $AP/run_ap7.sh $V agent 160 $T --ce agent >> $Q 2>&1
  for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
  python3 $AP/deploy_candidate.py alltoallv --restore >/dev/null
}
run_one olmoe_a2av_AGRS_agentpath   --file $AP/cands/alltoallv__r42__1b07b1ae.py
run_one olmoe_a2av_ag_t_rs_inline   --builtin ag_t_rs_inline_match
run_one olmoe_a2av_ag_slice_cat_w100 --file $AP/cands/alltoallv__paper__7ee07e87.py
stage "E2b re-sweep: alltoallv + ring_kv seed templates (fixed template/shim)"
cd /home/ubuntu/sweep_results && for f in alltoallv__builtin_*.json ring_kv__builtin_*.json; do python3 -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if d.get('measured_ms') is None else 1)" && rm -f $f; done; cd /home/ubuntu
bash $AP/sweep_candidates.sh 'alltoallv|ring_kv' >> $Q 2>&1

stage "E2c PP per-problem at S=512 (S=1024 stacked buffer OOMs at 224 ranks): 4 candidates + baseline"
VP=training/train_pp_send_recv_7node_s512.py
sed -E "s/^S *= *1024/S = 512/" $REPO/training/train_pp_send_recv_7node.py > $REPO/$VP
export RESULTS_DIR=/home/ubuntu/sweep_results/raw_pp512; mkdir -p $RESULTS_DIR
pp_one () {  # <tag> <backend> [deploy args]
  T=$1; B=$2; shift 2
  [ "$B" = agent ] && python3 $AP/deploy_candidate.py pp_send_recv "$@" | tee -a $Q
  rm -f $RESULTS_DIR/pp_send_recv_${B}_step.json; for ip in $WORKERS; do ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip "mkdir -p $RESULTS_DIR; rm -f $RESULTS_DIR/pp_send_recv_${B}_step.json" </dev/null; done
  timeout 1800 bash $AP/run_ap7.sh $VP $B 300 pp512_$T >> $Q 2>&1
  F=$RESULTS_DIR/pp_send_recv_${B}_step.json
  [ -f $F ] || for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$F $F 2>/dev/null && break; done
  [ -f $F ] && python3 -c "import json; d=json.load(open('$F')); d.update(problem='pp_send_recv_S512', cand='$T'); d.pop('step_times_ms',None); d.pop('losses',None); d['measured_ms']=d.get('steady_ms') or d.get('avg_ms'); json.dump(d,open('/home/ubuntu/sweep_results/pp_send_recv_S512__$T.json','w'),indent=1); print('   pp512', '$T', d['measured_ms'])" | tee -a $Q
  python3 $AP/deploy_candidate.py pp_send_recv --restore >/dev/null
}
pp_one harness_baseline baseline
pp_one paper_7f79b897 agent --file $AP/cands/pp_send_recv__paper__7f79b897.py
pp_one r42_144ba34c agent --file $AP/cands/pp_send_recv__r42__144ba34c.py
pp_one r36none_755c6cc9 agent --file $AP/cands/pp_send_recv__r36none__755c6cc9.py
pp_one r28nosim_train_cd9857e6 agent --file $AP/cands/pp_send_recv__r28nosim_train__cd9857e6.py
unset RESULTS_DIR
stage "E5b OLMoE agent re-runs (seed1, seed2) after alltoallv runtime fix"
for seed in 1 2; do
  V=training/train_olmoe10b_seed$seed.py
  T=olmoe_seed${seed}_agent; D=/home/ubuntu/olmoe_results/$T; mkdir -p $D
  RESULTS_DIR=$D timeout 3600 bash $AP/run_ap7.sh $V agent 250 $T --ce agent >> $Q 2>&1
  for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$D/*.json $D/ 2>/dev/null; done
done
stage "QUEUE3 DONE"
