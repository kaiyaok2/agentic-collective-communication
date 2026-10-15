#!/bin/bash
pkill -f 'queue55_ap[7]' 2>/dev/null; pkill -f 'sweep_candidates.s[h]' 2>/dev/null; pkill -f 'run_ap7.s[h]' 2>/dev/null; sleep 1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'
source /home/ubuntu/venv.env; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'" 2>/dev/null; done
echo "### queue55 stopped: PP candidates (235 MB ARs) hang on the degraded fabric -> queue56 (pool minus PP, 20-min cap)" >> /home/ubuntu/queue.log
cd /home/ubuntu; nohup setsid bash /home/ubuntu/ap7/queue56_ap7.sh > /home/ubuntu/queue56_nohup.log 2>&1 < /dev/null & disown
sleep 1; pgrep -fa 'queue5[56]_ap[7]' | grep -v 'bash -c' | cut -c1-50
