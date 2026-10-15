#!/bin/bash
# queue57: larger-budget leak-free search vs the final simulator (v16b config), budget 400, methods random/GA/beam (+rules), 2 seeds, 6 workers' CPUs.
set -uo pipefail
AP=/home/ubuntu/ap7; Q=/home/ubuntu/queue.log
source /home/ubuntu/venv.env
stage () { echo "### [$(date -u +%T)] $1" | tee -a $Q; }
stage "QUEUE57 leak-free search vs v16b, BUDGET=400, 6 workers"
W=($WORKERS); K="-i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
declare -a PR=("alltoallv" "uniform_a2a,ring_kv" "dxe" "pp_send_recv,llama_block_ar" "tp_mlp" "fsdp_prefetch")
for i in 0 1 2 3 4 5; do ip=${W[$i]}
  rsync -az -e "ssh $K" /home/ubuntu/ap7/ ubuntu@$ip:/home/ubuntu/ap7/; rsync -az -e "ssh $K" /home/ubuntu/hwprobe/ ubuntu@$ip:/home/ubuntu/hwprobe/
  rsync -az --delete -e "ssh $K" --exclude=.git --exclude=__pycache__ --exclude=training/results /home/ubuntu/OverlayCCL/ ubuntu@$ip:/home/ubuntu/OverlayCCL/
  ssh -n $K ubuntu@$ip "mkdir -p /home/ubuntu/sweep_results_b400; cd /home/ubuntu/OverlayCCL && nohup setsid bash -c 'for s in 0 1; do PROBLEMS=${PR[$i]} SCORER=v4 FWDBWD=2 TABLE=/home/ubuntu/hwprobe/hw_probe_measured_cluster1.json BUDGET=400 SEED=\$s OUTDIR=/home/ubuntu/sweep_results_b400 PYTHONPATH=. nice -n 19 /opt/aws_neuronx_venv_pytorch_2_9/bin/python /home/ubuntu/ap7/generic_search.py; done' > /home/ubuntu/generic_search_b400.log 2>&1 < /dev/null & disown"
done
for i in $(seq 1 400); do n=0; for j in 0 1 2 3 4 5; do ssh -n $K ubuntu@${W[$j]} 'pgrep -c -f generic_searc[h]' 2>/dev/null | grep -qv '^0$' && n=$((n+1)); done; [ $n = 0 ] && break; sleep 60; done
mkdir -p /home/ubuntu/gs_b400; for j in 0 1 2 3 4 5; do for f in $(ssh -n $K ubuntu@${W[$j]} 'ls /home/ubuntu/sweep_results_b400/generic_search_v4_s*.json /home/ubuntu/sweep_results/generic_search_v4_s*.json 2>/dev/null'); do scp -q $K ubuntu@${W[$j]}:$f /home/ubuntu/gs_b400/${W[$j]}_$(basename $f); done; done
stage "QUEUE57 DONE ($(ls /home/ubuntu/gs_b400 | wc -l) files)"
