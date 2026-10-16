#!/bin/bash
# after the cold stage: re-run the subset with profiling (the first pass lost the profile JSON to a parser bug); separate cc tag forces re-run
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu
until grep -q "cold done" $R/bench_all.log 2>/dev/null; do sleep 30; done
nice -n 5 python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_profile.jsonl --cc-tag 2.21p --flagsets default --repeats 1 --timeout 1200 --profile --num-exec 10 >> $R/bench_profile.log 2>&1
echo "### [$(date -u +%T)] subset re-profile done: $(wc -l < $R/bench_profile.jsonl) rows" >> $R/bench_all.log
