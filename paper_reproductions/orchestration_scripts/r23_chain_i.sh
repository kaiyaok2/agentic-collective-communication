#!/bin/bash
while ! grep -q 'R23H LLAMA-REPROBE DONE' /home/ubuntu/r23/HEARTBEAT_H 2>/dev/null; do
  sleep 60
done
echo 'R23h done at $(date -u); launching R23i bench finalize' >> /home/ubuntu/r23/CHAIN_I_LOG
nohup /home/ubuntu/r23i_bench_finalize.sh > /home/ubuntu/r23i.log 2>&1 &
echo 'R23i PID $!' >> /home/ubuntu/r23/CHAIN_I_LOG
