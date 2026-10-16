#!/bin/bash
# stop the flags arm after O1 and O3 (48 rows) so the version arms start before the user is back; 'transformer' flagset is deferred
R=/home/ubuntu/results
until [ "$(wc -l < $R/bench_flags.jsonl 2>/dev/null)" -ge 48 ]; do sleep 20; done
pkill -f "ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_flags.jsonl"
echo "### [$(date -u +%T)] flags arm cut after O1,O3 (transformer deferred)" >> $R/bench_all.log
