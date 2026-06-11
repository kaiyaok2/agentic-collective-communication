#!/bin/bash
# Wait for R23 OLMoE rerun to finish, then launch post driver.
while ! grep -q 'R23 OLMoE DONE Tue Jun  2 2' /home/ubuntu/r23/HEARTBEAT 2>/dev/null; do
  sleep 30
done
echo 'OLMoE rerun done at $(date -u); launching post driver' >> /home/ubuntu/r23/CHAIN_LOG
nohup /home/ubuntu/r23c_post.sh > /home/ubuntu/r23c_post.log 2>&1 &
echo 'Post driver started PID $!' >> /home/ubuntu/r23/CHAIN_LOG
