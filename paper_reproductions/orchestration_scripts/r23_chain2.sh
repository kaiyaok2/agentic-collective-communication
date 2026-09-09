#!/bin/bash
# Wait for post driver (R23 POST DONE) then launch OLMoE rerun without grad_ar.
while ! grep -q 'R23 POST DONE' /home/ubuntu/r23/HEARTBEAT_POST 2>/dev/null; do
  sleep 60
done
echo 'Post driver done at $(date -u); launching OLMoE rerun' >> /home/ubuntu/r23/CHAIN2_LOG
nohup /home/ubuntu/r23d_olmoe_rerun.sh > /home/ubuntu/r23d_olmoe.log 2>&1 &
echo 'OLMoE rerun PID $!' >> /home/ubuntu/r23/CHAIN2_LOG
