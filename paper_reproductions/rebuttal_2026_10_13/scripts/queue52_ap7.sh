#!/bin/bash
# queue52 (after queue51): leak-free generic search against the FINAL simulator (v16b config) on 4 workers' CPUs, 2 seeds; then HW-gate any find.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
until grep -q "QUEUE51 DONE" $Q; do sleep 20; done
stage "QUEUE52 leak-free search vs final simulator (v16b) on worker CPUs"
W=($WORKERS); K="-i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
declare -a PR=("alltoallv,uniform_a2a" "ring_kv,dxe" "pp_send_recv,llama_block_ar" "tp_mlp,fsdp_prefetch")
for i in 0 1 2 3; do ip=${W[$i]}
  rsync -az -e "ssh $K" /home/ubuntu/ap7/ ubuntu@$ip:/home/ubuntu/ap7/; rsync -az -e "ssh $K" /home/ubuntu/hwprobe/ ubuntu@$ip:/home/ubuntu/hwprobe/
  rsync -az --delete -e "ssh $K" --exclude=.git --exclude=__pycache__ --exclude=training/results /home/ubuntu/OverlayCCL/ ubuntu@$ip:/home/ubuntu/OverlayCCL/
  ssh -n $K ubuntu@$ip "mkdir -p /home/ubuntu/sweep_results; cd /home/ubuntu/OverlayCCL && nohup setsid bash -c 'for s in 0 1; do PROBLEMS=${PR[$i]} SCORER=v4 FWDBWD=2 TABLE=/home/ubuntu/hwprobe/hw_probe_measured_cluster1.json BUDGET=60 SEED=\$s PYTHONPATH=. nice -n 19 /opt/aws_neuronx_venv_pytorch_2_9/bin/python /home/ubuntu/ap7/generic_search.py; done' > /home/ubuntu/generic_search_v16b.log 2>&1 < /dev/null & disown"
done
# wait for all four
for i in $(seq 1 240); do n=0; for j in 0 1 2 3; do ssh -n $K ubuntu@${W[$j]} 'pgrep -c -f generic_searc[h]' 2>/dev/null | grep -qv '^0$' && n=$((n+1)); done; [ $n = 0 ] && break; sleep 60; done
mkdir -p /home/ubuntu/gs_v16b; for j in 0 1 2 3; do for f in $(ssh -n $K ubuntu@${W[$j]} 'ls /home/ubuntu/sweep_results/generic_search_v4_s*.json 2>/dev/null'); do scp -q $K ubuntu@${W[$j]}:$f /home/ubuntu/gs_v16b/${W[$j]}_$(basename $f); done; done
stage "QUEUE52 DONE ($(ls /home/ubuntu/gs_v16b | wc -l) result files)"
