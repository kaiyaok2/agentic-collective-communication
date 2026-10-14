#!/bin/bash
pkill -f 'queue38_ap[7]' 2>/dev/null; pkill -f 'queue39_ap[7]' 2>/dev/null; pkill -f 'run_ap7.s[h]' 2>/dev/null; sleep 1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'
source /home/ubuntu/venv.env; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'" 2>/dev/null; done
echo "### queue38 (DM4096 L8) stopped: device OOM -> queue40 (L4/L2 configs)" >> /home/ubuntu/queue.log
cd /home/ubuntu
nohup setsid bash /home/ubuntu/ap7/queue40_ap7.sh > /home/ubuntu/queue40_nohup.log 2>&1 < /dev/null & disown
nohup setsid bash /home/ubuntu/ap7/queue39_ap7.sh > /home/ubuntu/queue39_nohup.log 2>&1 < /dev/null & disown
sleep 1; pgrep -fa 'queue[34][0-9]_ap[7]' | grep -v 'bash -c' | cut -c1-50
