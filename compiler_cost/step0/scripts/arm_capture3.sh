#!/bin/bash
cd /home/ubuntu && nohup bash /home/ubuntu/scripts/run_capture3.sh > /home/ubuntu/results/capture3.out 2>&1 < /dev/null & disown
sleep 1; pgrep -fa "scripts/run_captur[e]3" | cut -c1-80
