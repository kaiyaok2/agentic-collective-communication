#!/bin/bash
# after the tail chain: re-profile the subset NEFFs built with O1 and O3 (their first profiles were lost to the summary-json format bug)
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu
until grep -q "BENCH_ALL_DONE (tail chain)" $R/bench_all.log 2>/dev/null; do sleep 60; done
python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_profile.jsonl --cc-tag 2.21p --flagsets O1,O3 --repeats 1 --timeout 1200 --profile --num-exec 10 >> $R/bench_profile.log 2>&1
echo "### [$(date -u +%T)] O1/O3 re-profile done: $(wc -l < $R/bench_profile.jsonl) rows" >> $R/bench_all.log
