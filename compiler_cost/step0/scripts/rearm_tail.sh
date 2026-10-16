#!/bin/bash
pkill -f "scripts/pairs_stag[e]"; pkill -f "scripts/reprofil[e]"; pkill -f "scripts/post_nois[e]"; sleep 1
cd /home/ubuntu && nohup bash /home/ubuntu/scripts/tail_chain.sh > /home/ubuntu/results/tail_chain.out 2>&1 < /dev/null & disown
sleep 1; echo "waiters:"; pgrep -fa "scripts/(pairs_stag[e]|reprofil[e]|post_nois[e]|tail_chai[n])" | cut -c1-70
