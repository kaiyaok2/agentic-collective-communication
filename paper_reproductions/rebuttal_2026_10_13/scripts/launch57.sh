#!/bin/bash
cd /home/ubuntu; nohup setsid bash /home/ubuntu/ap7/queue57_ap7.sh > /home/ubuntu/queue57_nohup.log 2>&1 < /dev/null & disown
sleep 2; pgrep -fa "queue57_ap[7]" | grep -v "bash -c" | cut -c1-50; tail -1 /home/ubuntu/queue.log | cut -c1-70
