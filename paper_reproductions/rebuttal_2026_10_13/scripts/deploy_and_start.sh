#!/bin/bash
# Local (laptop) side: wait for the cluster to be launched, then bootstrap it and start the queue.
set -uo pipefail
ENVF=/tmp/ap7/cluster.env
for i in $(seq 1 240); do [ -f $ENVF ] && break; sleep 15; done
[ -f $ENVF ] || { echo "cluster.env never appeared"; exit 1; }
source $ENVF
KEY=/Users/kekaiyao/.ssh/Kaiyao.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
echo "master $MASTER_PUB ($MASTER_PRIV) workers: $WORKER_PRIVS"
for i in $(seq 1 60); do $SSH ubuntu@$MASTER_PUB "echo up" </dev/null 2>/dev/null && break; sleep 15; done
$SSH ubuntu@$MASTER_PUB "echo master reachable: \$(hostname)" </dev/null || { echo "master ssh failed"; exit 1; }
# workers reachable from master? (they come up a bit later) - wait for ssh on each via master
for W in $WORKER_PRIVS; do
  for i in $(seq 1 60); do $SSH ubuntu@$MASTER_PUB "timeout 3 bash -c 'cat < /dev/null > /dev/tcp/$W/22'" </dev/null 2>/dev/null && break; sleep 10; done
done
echo "copying bundle..."
scp -q -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $KEY ubuntu@$MASTER_PUB:/home/ubuntu/.ssh/Kaiyao.pem
scp -q -r -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /tmp/ap7 ubuntu@$MASTER_PUB:/home/ubuntu/ap7
$SSH ubuntu@$MASTER_PUB "chmod 600 ~/.ssh/Kaiyao.pem; echo MASTER_PRIV=$MASTER_PRIV > ~/cluster.env; echo 'WORKER_PRIVS=\"$WORKER_PRIVS\"' >> ~/cluster.env" </dev/null
echo "bootstrapping..."
$SSH ubuntu@$MASTER_PUB "bash /home/ubuntu/ap7/bootstrap_ap7.sh '$WORKER_PRIVS'" </dev/null 2>&1 | tee /tmp/ap7/bootstrap.log
grep -q "bootstrap complete" /tmp/ap7/bootstrap.log || { echo "BOOTSTRAP FAILED"; exit 2; }
echo "starting queue..."
$SSH ubuntu@$MASTER_PUB "export MASTER_PRIV=$MASTER_PRIV; nohup bash /home/ubuntu/ap7/queue_ap7.sh > /home/ubuntu/queue_nohup.log 2>&1 & echo queue pid \$!" </dev/null
echo "=== deploy complete $(date -u +%FT%TZ) ==="
