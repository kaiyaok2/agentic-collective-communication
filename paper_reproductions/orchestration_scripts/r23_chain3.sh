#!/bin/bash
# Wait for OLMoE rerun (R23 OLMOE2 DONE) then launch extra benches.
while ! grep -q 'R23 OLMOE2 DONE' /home/ubuntu/r23/HEARTBEAT_OLMOE2 2>/dev/null; do
  sleep 60
done
echo 'OLMoE rerun done at $(date -u); launching extra bench' >> /home/ubuntu/r23/CHAIN3_LOG
nohup /home/ubuntu/r23e_extra_bench.sh > /home/ubuntu/r23e_extra.log 2>&1 &
echo 'Extra bench PID $!' >> /home/ubuntu/r23/CHAIN3_LOG
