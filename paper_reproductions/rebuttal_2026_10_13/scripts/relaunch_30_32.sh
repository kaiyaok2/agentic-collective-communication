#!/bin/bash
pkill -f 'queue30_ap[7]' 2>/dev/null; sleep 1
cd /home/ubuntu
pgrep -f 'queue30_ap[7]' >/dev/null || { nohup setsid bash /home/ubuntu/ap7/queue30_ap7.sh > /home/ubuntu/queue30_nohup.log 2>&1 < /dev/null & disown; }
pgrep -f 'queue32_ap[7]' >/dev/null || { nohup setsid bash /home/ubuntu/ap7/queue32_ap7.sh > /home/ubuntu/queue32_nohup.log 2>&1 < /dev/null & disown; }
sleep 1; pgrep -fa 'queue3[0-2]_ap[7]' | grep -v 'bash -c' | cut -c1-50
