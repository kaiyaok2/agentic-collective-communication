#!/bin/bash
# copy files to master's ~/ap7: bash /tmp/ap7/push.sh file1 file2 ...
source /tmp/ap7/cluster.env
exec scp -q -i /Users/kekaiyao/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@" ubuntu@$MASTER_PUB:/home/ubuntu/ap7/
