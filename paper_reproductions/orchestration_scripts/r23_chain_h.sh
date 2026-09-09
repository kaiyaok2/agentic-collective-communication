#!/bin/bash
while ! grep -q 'R23G ALL DONE' /home/ubuntu/r23/HEARTBEAT_G 2>/dev/null; do
  sleep 60
done
echo 'R23g done at $(date -u); launching R23h Llama reprobe' >> /home/ubuntu/r23/CHAIN_H_LOG
nohup /home/ubuntu/r23h_llama_reprobe.sh > /home/ubuntu/r23h.log 2>&1 &
echo 'R23h PID $!' >> /home/ubuntu/r23/CHAIN_H_LOG
