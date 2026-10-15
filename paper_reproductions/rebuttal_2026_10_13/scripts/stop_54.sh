#!/bin/bash
pkill -f 'queue54_ap[7]' 2>/dev/null; pkill -f 'run_llama7b.s[h]' 2>/dev/null; sleep 1
pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_llama_e2e_7[b]'
source /home/ubuntu/venv.env; for ip in $WORKERS; do ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /home/ubuntu/.ssh/Kaiyao.pem ubuntu@$ip "pkill -9 -f 'torchru[n]'; pkill -9 -f 'train_llama_e2e_7[b]'" 2>/dev/null; done
echo "### 02:00 queue54 stopped: 3-node Llama stuck in CCOM bootstrap (connection refused to root) on cluster 2; cluster left idle" >> /home/ubuntu/queue.log
pgrep -fa 'queue5[0-9]_ap[7]' | grep -v 'bash -c' | wc -l
