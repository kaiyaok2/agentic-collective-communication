#!/bin/bash
source /home/ubuntu/venv.env; K="-i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"; W=($WORKERS)
declare -A PR=([4]="tp_mlp" [5]="fsdp_prefetch")
for i in 4 5; do ip=${W[$i]}
  rsync -az -e "ssh $K" /home/ubuntu/ap7/ ubuntu@$ip:/home/ubuntu/ap7/; rsync -az -e "ssh $K" /home/ubuntu/hwprobe/ ubuntu@$ip:/home/ubuntu/hwprobe/
  rsync -az --delete -e "ssh $K" --exclude=.git --exclude=__pycache__ --exclude=training/results /home/ubuntu/OverlayCCL/ ubuntu@$ip:/home/ubuntu/OverlayCCL/
  ssh -n $K ubuntu@$ip "mkdir -p /home/ubuntu/sweep_results_b400; cd /home/ubuntu/OverlayCCL && nohup setsid bash -c 'for s in 0 1; do PROBLEMS=${PR[$i]} SCORER=v4 FWDBWD=2 TABLE=/home/ubuntu/hwprobe/hw_probe_measured_cluster1.json BUDGET=400 SEED=\$s OUTDIR=/home/ubuntu/sweep_results_b400 PYTHONPATH=. nice -n 19 /opt/aws_neuronx_venv_pytorch_2_9/bin/python /home/ubuntu/ap7/generic_search.py; done' > /home/ubuntu/generic_search_b400.log 2>&1 < /dev/null & disown; sleep 1; pgrep -c -f generic_searc[h]"
done
