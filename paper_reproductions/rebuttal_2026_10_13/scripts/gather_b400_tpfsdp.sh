#!/bin/bash
source /home/ubuntu/venv.env; K="-i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"; W=($WORKERS)
for j in 4 5; do echo -n "W$j procs=$(ssh -n $K ubuntu@${W[$j]} 'pgrep -c -f generic_searc[h]') "; for f in $(ssh -n $K ubuntu@${W[$j]} 'ls /home/ubuntu/sweep_results_b400/generic_search_v4_s*.json 2>/dev/null'); do scp -q $K ubuntu@${W[$j]}:$f /home/ubuntu/gs_b400/${W[$j]}_$(basename $f); done; done; echo; ls /home/ubuntu/gs_b400 | wc -l
