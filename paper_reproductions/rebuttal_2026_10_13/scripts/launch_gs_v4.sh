#!/bin/bash
# launch_gs_v4.sh <worker-ip> <problems-csv> : run the leak-free generic search (SCORER=v4, 2 seeds) on a worker, detached
IP=$1; PR=$2; K="-i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
rsync -az -e "ssh $K" /home/ubuntu/ap7/ ubuntu@$IP:/home/ubuntu/ap7/ ; rsync -az -e "ssh $K" /home/ubuntu/hwprobe/ ubuntu@$IP:/home/ubuntu/hwprobe/
rsync -az --delete -e "ssh $K" --exclude=.git --exclude=__pycache__ --exclude=training/results /home/ubuntu/OverlayCCL/ ubuntu@$IP:/home/ubuntu/OverlayCCL/
ssh -n $K ubuntu@$IP "mkdir -p /home/ubuntu/sweep_results /home/ubuntu/sweep_v2; cd /home/ubuntu/OverlayCCL && nohup setsid bash -c 'for s in 0 1; do PROBLEMS=$PR SCORER=v4 BUDGET=60 SEED=\$s PYTHONPATH=. nice -n 19 /opt/aws_neuronx_venv_pytorch_2_9/bin/python /home/ubuntu/ap7/generic_search.py; done' > /home/ubuntu/generic_search_v4.log 2>&1 < /dev/null & disown; sleep 1; pgrep -c -f generic_searc[h]"
