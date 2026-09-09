#!/bin/bash
while ! grep -q 'R23I ALL DONE' /home/ubuntu/r23/HEARTBEAT_I 2>/dev/null; do
  sleep 60
done
echo 'R23i done at $(date -u); launching R23j' >> /home/ubuntu/r23/CHAIN_J_LOG
nohup /home/ubuntu/r23j_fix_perprob.sh > /home/ubuntu/r23j.log 2>&1 &
echo 'R23j PID $!' >> /home/ubuntu/r23/CHAIN_J_LOG
