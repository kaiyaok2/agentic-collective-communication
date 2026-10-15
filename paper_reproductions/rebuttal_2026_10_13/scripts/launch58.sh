#!/bin/bash
cd /home/ubuntu; nohup setsid bash /home/ubuntu/ap7/queue58_ap7.sh > /home/ubuntu/queue58_nohup.log 2>&1 < /dev/null & disown
sleep 2; pgrep -fa "queue58_ap[7]" | grep -v "bash -c" | cut -c1-50
