#!/bin/bash
cd /home/ubuntu && nohup bash /home/ubuntu/scripts/reprofile.sh > /home/ubuntu/results/reprofile.out 2>&1 < /dev/null & disown
sleep 1; pgrep -fa "scripts/reprofil[e]" | cut -c1-60
