#!/bin/bash
set -uo pipefail
for P in a2av ua2a rkv dxe; do
  PORT=$((32500 + RANDOM % 200))
  PORT=$PORT bash /tmp/h7_bench/run_bench.sh $P 2>&1 | tee -a /tmp/h7_bench/run_all.log
done
echo "[h7-bench] ALL DONE $(date -u)"
