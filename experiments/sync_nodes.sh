#!/bin/bash
# Sync project code to all worker nodes.
#
# Usage:
#   ./experiments/sync_nodes.sh <worker1_ip> [worker2_ip ...]
#
# Example:
#   ./experiments/sync_nodes.sh 172.31.55.245

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <worker1_ip> [worker2_ip ...]"
    exit 1
fi

for node in "$@"; do
    echo "Syncing to $node..."
    rsync -az --delete \
        --exclude '.kiss.artifacts' \
        --exclude '*.pyc' \
        --exclude '__pycache__' \
        --exclude 'log-neuron-cc.txt' \
        --exclude 'global_metric_store.json' \
        --exclude 'PostSPMDPassesExecutionDuration.txt' \
        "$PROJECT_DIR/" "ubuntu@${node}:${PROJECT_DIR}/"
    echo "  Done: $node"
done

echo "All nodes synced."
