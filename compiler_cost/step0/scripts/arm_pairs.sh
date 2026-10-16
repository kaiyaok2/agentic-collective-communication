#!/bin/bash
cd /home/ubuntu && nohup bash /home/ubuntu/scripts/pairs_stage.sh > /home/ubuntu/results/pairs_stage.out 2>&1 < /dev/null & disown
sleep 1; pgrep -fa "scripts/pairs_stag[e]" | cut -c1-60
