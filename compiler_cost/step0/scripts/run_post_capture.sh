#!/bin/bash
# after the main capture: collective-workload capture (2 ranks) -> rebuild manifest -> bench campaign
R=/home/ubuntu/results; S=/home/ubuntu/scripts
until grep -q CAPTURE_ALL_DONE $R/capture_all.log 2>/dev/null; do sleep 30; done
echo "### [$(date -u +%T)] post-capture: acc workloads" >> $R/bench_all.log
bash $S/capture_acc.sh > $R/capture_acc.log 2>&1
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; python $S/build_manifest.py --corpus /home/ubuntu/corpus --out $R/manifest.jsonl 2>&1 | grep -vE "Warning|warn" >> $R/bench_all.log
bash $S/run_bench_all.sh
