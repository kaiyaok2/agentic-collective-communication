#!/bin/bash
# Bootstrap the rebuttal cluster. Run ON the master (after scp of Kaiyao.pem + this file).
# Usage: bash bootstrap_ap7.sh "<worker_priv_1> ... <worker_priv_6>"
set -euo pipefail
WORKERS="$1"
REPO=/home/ubuntu/OverlayCCL
KEY=/home/ubuntu/.ssh/Kaiyao.pem
chmod 600 $KEY
SSHK="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

echo "[master] clone public artifact repo"
[ -d $REPO/.git ] || git clone -q https://github.com/OverlayCCL/OverlayCCL.git $REPO
mkdir -p $REPO/training/results/7node/logs

echo "[master] detect Neuron venv"
VENV=""
for v in /opt/aws_neuronx_venv_pytorch_2_9 /opt/aws_neuronx_venv_pytorch_2_8 /opt/aws_neuronx_venv_pytorch_2_10; do
  [ -x $v/bin/torchrun ] && { VENV=$v; break; }
done
[ -n "$VENV" ] || { echo "no neuron venv found:"; ls /opt | grep -i venv; exit 1; }
echo "VENV=$VENV" | tee /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH
$VENV/bin/python -c "import torch, torch_xla; print('torch', torch.__version__, 'torch_xla', torch_xla.__version__)" || echo "WARN: torch_xla import check failed (continuing)"
{ neuron-ls 2>/dev/null || /opt/aws/neuron/bin/neuron-ls; } | grep -c "trn1\|^| *[0-9]" | sed 's/^/neuron-ls rows: /' || true
echo "compiler: $(ls /var/tmp/neuron-compile-cache 2>/dev/null | head -1) $($VENV/bin/neuronx-cc --version 2>&1 | head -1)"

echo "[master] passwordless ssh master->workers"
[ -f ~/.ssh/id_rsa ] || ssh-keygen -q -t rsa -f ~/.ssh/id_rsa -N ""
PUB=$(cat ~/.ssh/id_rsa.pub)
for W in $WORKERS; do
  $SSHK ubuntu@$W "grep -qF '$PUB' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> ~/.ssh/authorized_keys; mkdir -p $REPO /tmp/neuron_cache" </dev/null
  rsync -az --delete -e "$SSHK" --exclude='.git' --exclude='__pycache__' --exclude='training/results' $REPO/ ubuntu@$W:$REPO/
  echo "  worker $W synced: $($SSHK ubuntu@$W "ls $REPO/runtime | wc -l" </dev/null) runtime files; neuron devices: $($SSHK ubuntu@$W "neuron-ls 2>/dev/null | grep -c trn" </dev/null)"
done
echo "WORKERS=\"$WORKERS\"" >> /home/ubuntu/venv.env
echo "=== bootstrap complete ==="
