#!/bin/bash
exec > /home/ubuntu/novel_test.log 2>&1
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

PROBLEMS="mod_sq_bcast xor_grid_bcast popcount_bcast triangle_num_bcast sign_alt_bcast bimodal_dist_bcast"

# Kiss v3 on all 6
for P in $PROBLEMS; do
  echo "==== $(date -u -Iseconds) kiss $P ===="
  rm -rf /tmp/novel_kiss_$P
  mkdir -p /tmp/novel_kiss_$P
  timeout 900 /home/ubuntu/kiss/.venv/bin/python \
    /home/ubuntu/cb2_verify/repo/experiments/ablation_kiss_vs_cc/kiss_phase3.py \
    --problem $P --pattern moe --output-dir /tmp/novel_kiss_$P \
    --max-budget 12.0 --max-steps 40 --num-nodes 2 > /tmp/novel_kiss_$P/run.log 2>&1
  cat /tmp/novel_kiss_$P/kiss_summary.json 2>&1 | head
done

# Strat on all 6
export STRAT_HW_GATE=1
for P in $PROBLEMS; do
  echo "==== $(date -u -Iseconds) strat $P ===="
  rm -rf /tmp/novel_strat_$P
  mkdir -p /tmp/novel_strat_$P
  timeout 600 /home/ubuntu/venv/bin/python \
    /home/ubuntu/cb2_verify/repo/experiments/run_search.py \
    --problem $P --pattern moe --phase3-style strategy-enumerate --llm-model opus \
    --num-nodes 2 --worker-addrs $WORKER_IP --master-addr $MASTER_IP \
    --output-dir /tmp/novel_strat_$P > /tmp/novel_strat_$P.log 2>&1
  grep -E "Winner|SimTime|HW_GATE_FAIL" /tmp/novel_strat_$P.log | tail -5
  # copy strat runtime
  if [ -f /home/ubuntu/runtime/trainium_${P}_2node.py ]; then
    mkdir -p /home/ubuntu/runtime_novel_strat_${P}
    cp /home/ubuntu/runtime/trainium_${P}_2node.py /home/ubuntu/runtime_novel_strat_${P}/
  fi
done
unset STRAT_HW_GATE

echo "NOVEL_TEST DONE $(date -u -Iseconds)"
touch /tmp/novel_test_done

# snapshot
cd /tmp
tar czf novel_snapshot.tar.gz novel_kiss_* novel_strat_* 2>&1 | tail
aws s3 cp novel_snapshot.tar.gz s3://overlayccl-kaiyao-artifacts/2026-07-24-cb-v4/novel_snapshot.tar.gz 2>&1 | tail
