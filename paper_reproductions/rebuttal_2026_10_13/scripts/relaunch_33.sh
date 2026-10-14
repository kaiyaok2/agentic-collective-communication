#!/bin/bash
pkill -f 'queue30_ap[7]' 2>/dev/null; sleep 1; cd /home/ubuntu
nohup setsid bash /home/ubuntu/ap7/queue30_ap7.sh > /home/ubuntu/queue30_nohup.log 2>&1 < /dev/null & disown
nohup setsid bash /home/ubuntu/ap7/queue33_ap7.sh > /home/ubuntu/queue33_nohup.log 2>&1 < /dev/null & disown
sleep 1; pgrep -fa 'queue3[0-3]_ap[7]|watch' | grep -v 'bash -c' | cut -c1-50
