#!/bin/bash
cd /home/ubuntu && nohup bash /home/ubuntu/scripts/post_noise.sh > /home/ubuntu/results/post_noise.out 2>&1 < /dev/null & disown
sleep 1; pgrep -fa "scripts/post_nois[e]" | cut -c1-60
