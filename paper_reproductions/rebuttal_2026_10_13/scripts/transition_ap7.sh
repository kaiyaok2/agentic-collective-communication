#!/bin/bash
# transition_ap7.sh <cr-id> : launch 7 instances into the follow-on Capacity Block, rebind EIP, bootstrap, restore the
# patched OverlayCCL tree (repo_state/), install ap7, and start queue30 (B>1 OLMoE runs). Run on the laptop.
set -uo pipefail
CR=${1:?cr-id}; KEY=/Users/kekaiyao/.ssh/Kaiyao.pem
cd /tmp/ap7 && cp cluster.env cluster.env.prev.$(date -u +%H%M)
bash launch_ap7.sh $CR || { echo "LAUNCH FAILED (run the run-instances yourself via ! if the classifier blocked it)"; exit 1; }
source cluster.env
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
MYIP=$(curl -s --max-time 10 checkip.amazonaws.com); aws ec2 authorize-security-group-ingress --region ap-southeast-2 --group-id sg-041ed0ec1d7764349 --protocol tcp --port 22 --cidr $MYIP/32 >/dev/null 2>&1 || true
for i in $(seq 1 80); do $SSH ubuntu@$MASTER_PUB "echo up" </dev/null 2>/dev/null && break; sleep 15; done
for W in $WORKER_PRIVS; do for i in $(seq 1 60); do $SSH ubuntu@$MASTER_PUB "timeout 3 bash -c 'cat < /dev/null > /dev/tcp/$W/22'" </dev/null 2>/dev/null && break; sleep 10; done; done
scp -q -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $KEY ubuntu@$MASTER_PUB:/home/ubuntu/.ssh/Kaiyao.pem
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude pull --exclude '*.log' /tmp/ap7/ ubuntu@$MASTER_PUB:/home/ubuntu/ap7/
# live-probe table from the previous cluster (a fresh probe is re-run by queue31) + LLM credentials for the pipeline runs
$SSH ubuntu@$MASTER_PUB "mkdir -p /home/ubuntu/hwprobe" </dev/null
scp -q -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /tmp/ap7/pull/hwprobe/hw_probe_measured.json ubuntu@$MASTER_PUB:/home/ubuntu/hwprobe/hw_probe_measured_prevcluster.json
$SSH ubuntu@$MASTER_PUB "umask 077; echo \"export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY\" > /home/ubuntu/llm.env" </dev/null
$SSH ubuntu@$MASTER_PUB "chmod 600 ~/.ssh/Kaiyao.pem; bash /home/ubuntu/ap7/bootstrap_ap7.sh '$WORKER_PRIVS'" </dev/null 2>&1 | tee /tmp/ap7/bootstrap_new.log | tail -3
grep -q "bootstrap complete" /tmp/ap7/bootstrap_new.log || { echo "BOOTSTRAP FAILED"; exit 2; }
# restore the patched tree from the previous cluster (paper runtime .bak files, pp harness patch, ua2a deploy targets, probe, harness variants)
$SSH ubuntu@$MASTER_PUB "cd /home/ubuntu/OverlayCCL && git apply /home/ubuntu/ap7/repo_state/overlayccl_master.diff && tar xzf /home/ubuntu/ap7/repo_state/untracked_files.tgz --exclude='*/__pycache__/*' --exclude='training/results/*' 2>/dev/null; cp /home/ubuntu/ap7/cands/train_olmoe10b_B*fix*.py training/; cp /home/ubuntu/ap7/train_hwprobe_7node.py training/; git status --porcelain | wc -l; source /home/ubuntu/venv.env; for ip in \$WORKERS; do rsync -az --delete -e 'ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' --exclude=.git --exclude=__pycache__ --exclude=training/results /home/ubuntu/OverlayCCL/ ubuntu@\$ip:/home/ubuntu/OverlayCCL/; done; echo tree restored" </dev/null
$SSH ubuntu@$MASTER_PUB "cd /home/ubuntu && nohup setsid bash /home/ubuntu/ap7/queue30_ap7.sh > /home/ubuntu/queue30_nohup.log 2>&1 < /dev/null & disown; nohup setsid bash /home/ubuntu/ap7/queue31_ap7.sh > /home/ubuntu/queue31_nohup.log 2>&1 < /dev/null & disown; sleep 1; pgrep -fa 'queue30_ap[7]' | grep -v 'bash -c' | cut -c1-60" </dev/null
echo "=== transition complete $(date -u +%FT%TZ): master $MASTER_PUB ==="
