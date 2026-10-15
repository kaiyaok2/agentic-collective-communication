#!/bin/bash
pkill -f 'queue56_ap[7]' 2>/dev/null; pkill -f 'sweep_candidates.s[h]' 2>/dev/null; pkill -f 'run_ap7.s[h]' 2>/dev/null; sleep 1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'
source /home/ubuntu/venv.env; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'bin/python[0-9]* .*training/train_[a-z]'" 2>/dev/null; done
echo "### 03:40 queue56 stopped (ring_kv joint hung too; fabric degradation spreading) - cluster left idle; 26/50 of repeat 2 collected" >> /home/ubuntu/queue.log
echo stopped
