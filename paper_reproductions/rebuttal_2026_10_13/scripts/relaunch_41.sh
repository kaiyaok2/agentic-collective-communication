#!/bin/bash
pkill -f 'queue40_ap[7]' 2>/dev/null; pkill -f 'queue39_ap[7]' 2>/dev/null; pkill -f 'run_ap7.s[h]' 2>/dev/null; sleep 1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'
source /home/ubuntu/venv.env; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'" 2>/dev/null; done
echo "### queue40 stopped: DM4096 L4 -> NRT_EXEC_HW_ERR_COLLECTIVES (stuck collective); DM4096 infeasible on this stack -> queue41" >> /home/ubuntu/queue.log
L=$(ls -t /home/ubuntu/olmoe_results/olmoe_dm4096_L4_topk2_ex4096_nocomm/logs/*node0.log 2>/dev/null | head -1); grep -m2 -E 'Error|OOB|allocate|INTERNAL' $L 2>/dev/null | grep -v _sso | cut -c1-160
cd /home/ubuntu; nohup setsid bash /home/ubuntu/ap7/queue41_ap7.sh > /home/ubuntu/queue41_nohup.log 2>&1 < /dev/null & disown
sleep 1; pgrep -fa 'queue4[01]_ap[7]|queue39' | grep -v 'bash -c' | cut -c1-50
