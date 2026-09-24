#!/bin/bash
# Bootstrap all 7 nodes for the warm-RT sweep. Run ON the master after SSH in.
# Prereqs on master: ~/.ssh/Kaiyao.pem present (scp'd from laptop), GH_PAT exported.
# Usage: bootstrap_master.sh "<WORKER_PRIV_1 ... WORKER_PRIV_6>"
set -euo pipefail
WORKERS="$1"
read -r -a WARR <<< "$WORKERS"
REPO=/home/ubuntu/agentic-collective-communication
REPO_URL=https://github.com/kaiyaok2/agentic-collective-communication.git
B64=$(printf 'x-access-token:%s' "${GH_PAT}" | base64 | tr -d '\n')
SSHK="ssh -i /home/ubuntu/.ssh/Kaiyao.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

clone_or_update () {  # runs in the current shell (master) or piped to a worker
  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" -c http.extraHeader="Authorization: Basic ${B64}" fetch origin main -q
    git -C "$REPO" reset --hard origin/main -q
  else
    git -c http.extraHeader="Authorization: Basic ${B64}" clone -q "$REPO_URL" "$REPO"
    git -C "$REPO" checkout main -q
  fi
}

echo "[master] repo sync..."; clone_or_update

# master -> worker passwordless SSH: push master's id_rsa.pub via Kaiyao.pem
[ -f ~/.ssh/id_rsa ] || ssh-keygen -q -t rsa -f ~/.ssh/id_rsa -N ""
PUB=$(cat ~/.ssh/id_rsa.pub)
for W in "${WARR[@]}"; do
  echo "[master] auth + repo sync on worker $W ..."
  $SSHK ubuntu@"$W" "grep -qF '$PUB' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> ~/.ssh/authorized_keys"
  # clone/update repo on the worker (carry GH_PAT + helpers over the ssh env)
  $SSHK ubuntu@"$W" "GH_PAT='${GH_PAT}' B64='${B64}' REPO='${REPO}' REPO_URL='${REPO_URL}' bash -s" <<'WEOF'
set -e
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" -c http.extraHeader="Authorization: Basic ${B64}" fetch origin main -q
  git -C "$REPO" reset --hard origin/main -q
else
  git -c http.extraHeader="Authorization: Basic ${B64}" clone -q "$REPO_URL" "$REPO"
  git -C "$REPO" checkout main -q
fi
echo "  worker repo at $(git -C "$REPO" rev-parse --short HEAD)"
WEOF
done

echo "[master] torch_xla sanity..."
NEURON_VENV=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_2_8}
PJRT_DEVICE=NEURON $NEURON_VENV/bin/python -c "import torch; print('torch', torch.__version__)"
echo "[master] repo HEAD: $(git -C "$REPO" rev-parse --short HEAD)"
echo "=== bootstrap complete on master + ${#WARR[@]} workers ==="
