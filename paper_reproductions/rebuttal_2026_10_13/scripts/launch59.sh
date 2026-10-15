#!/bin/bash
cd /home/ubuntu && nohup setsid bash /home/ubuntu/ap7/queue59_ap7.sh > /home/ubuntu/queue59.out 2>&1 < /dev/null & disown; echo launched59
