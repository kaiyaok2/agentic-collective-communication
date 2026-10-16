#!/bin/bash
cd /home/ubuntu && nohup bash /home/ubuntu/scripts/run_post_capture.sh > /home/ubuntu/results/post_capture.out 2>&1 < /dev/null & disown
sleep 1; pgrep -fa "scripts/run_[p]ost_capture" | cut -c1-80
