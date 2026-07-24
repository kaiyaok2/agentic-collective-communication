#!/bin/bash
exec > /home/ubuntu/kiss_hwtest.log 2>&1
export PATH=/home/ubuntu/venv/bin:$PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1
export USE_BEDROCK=1 AWS_REGION=us-east-1 BEDROCK_REGION=us-east-1
export NEURON_PY=/home/ubuntu/venv/bin/python
export MASTER_IP=172.31.17.9 WORKER_IP=172.31.26.232
export HW_GATE=1
export HW_GATE_SCRIPT=/home/ubuntu/hw_gate_run.py
export KISS_MODEL=claude-opus-4-8
unset ANTHROPIC_API_KEY

for P in row_id_grid_bcast col_id_grid_bcast scaled_arange_bcast pair_max_bcast; do
  echo "==== $(date -u -Iseconds) $P ===="
  rm -rf /tmp/hwtest_$P
  mkdir -p /tmp/hwtest_$P
  timeout 600 /home/ubuntu/kiss/.venv/bin/python \
    /home/ubuntu/cb2_verify/repo/experiments/ablation_kiss_vs_cc/kiss_phase3.py \
    --problem $P --pattern moe --output-dir /tmp/hwtest_$P \
    --max-budget 8.0 --max-steps 25 --num-nodes 2 > /tmp/hwtest_$P/run.log 2>&1
  cat /tmp/hwtest_$P/kiss_summary.json 2>&1 | tail -10
done
echo "DONE $(date -u -Iseconds)"
touch /tmp/kiss_hwtest_done
aws s3 sync /tmp/hwtest_row_id_grid_bcast /tmp/hwtest_col_id_grid_bcast /tmp/hwtest_scaled_arange_bcast /tmp/hwtest_pair_max_bcast s3://overlayccl-kaiyao-artifacts/2026-07-24-cb-v4/hwtest/ 2>&1 | tail -3
