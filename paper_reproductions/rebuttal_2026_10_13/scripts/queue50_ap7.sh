#!/bin/bash
# queue50 (cluster 2, post-reboot): 1) headline reproductions (OLMoE baseline/agent; Llama-7B per_mb/bundled);
# 2) two more single-node Phase-1 probes; 3) 224-rank collective probe attempt (bf16, no device RNG).
set -uo pipefail
AP=/home/ubuntu/ap7; REPO=/home/ubuntu/OverlayCCL; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
harvest () { for ip in $WORKERS; do scp -q -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$ip:$1/*.json $1/ 2>/dev/null; done; }
cp $AP/train_hwprobe_7node.py $REPO/training/
for ip in $WORKERS; do rsync -az -e "ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" $REPO/training/ ubuntu@$ip:$REPO/training/; done
stage "QUEUE50 1a) OLMoE headline on cluster 2: baseline, agent"
cp $REPO/runtime/trainium_alltoallv_7node.py.paper.bak $REPO/runtime/trainium_alltoallv_7node.py
sed -E 's/times\[200:\]/times[100:]/g; s/len\(times\) - 200/len(times) - 100/; s/len\(times\) > 200/len(times) > 100/; s/from step 200/from step 100/' $REPO/training/train_olmoe10b.py > $REPO/training/train_olmoe10b_hl_w100.py
for B in baseline agent; do D=/home/ubuntu/olmoe_results/olmoe_c2_headline_$B; mkdir -p $D; FL=""; [ $B = agent ] && FL="--ce agent"
  RESULTS_DIR=$D PERCALL_PROBE=0 timeout 2400 bash $AP/run_ap7.sh training/train_olmoe10b_hl_w100.py $B 160 olmoe_c2_headline_$B $FL >> $Q 2>&1; harvest $D; done
stage "QUEUE50 1b) Llama-7B on cluster 2: per_mb, bundled (300 steps)"
for B in per_mb bundled; do RESULTS_DIR=/home/ubuntu/llama7b_results timeout 2400 bash $AP/run_llama7b.sh $B 300 >> $Q 2>&1; done
stage "QUEUE50 2) two more single-node Phase-1 probes"
for r in 2 3; do SIZES=4096,32768,65536,524288,4194304 SECTIONS=trivial,xla_ops,index_select,memcpy,graph_launch,back_to_back,collectives WORKERS_OVERRIDE=' ' PERCALL_PROBE=0 RESULTS_DIR=/home/ubuntu/hwprobe_r$r timeout 2400 bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe_r$r >> /home/ubuntu/hwprobe.log 2>&1; pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_hwprob[e]'; done
stage "QUEUE50 3) 224-rank collective probe attempt (bf16, no device RNG)"
PROBE_BF16=1 SECTIONS=collectives,back_to_back SIZES=4096,65536,524288 AR_ONLY_SIZES=4194304,8388608 PERCALL_PROBE=0 PROBE_ITERS=11 RESULTS_DIR=/home/ubuntu/hwprobe224c timeout 1500 bash $AP/run_ap7.sh training/train_hwprobe_7node.py probe 1 hwprobe224c >> /home/ubuntu/hwprobe.log 2>&1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_hwprob[e]'; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_hwprob[e]'" 2>/dev/null; done
stage "QUEUE50 DONE"
