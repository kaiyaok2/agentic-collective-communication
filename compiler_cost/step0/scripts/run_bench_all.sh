#!/bin/bash
# bench campaign, chained after capture: waits for CAPTURE_ALL_DONE, then
#   1. default flags, repeat 0, all modules, with neuron-profile execution   -> bench_default.jsonl
#   2. family-balanced subset: cold-cache x2, flag variants, compiler versions -> bench_cold / bench_flags / bench_versions
#   3. default flags repeats 1,2 for all modules (noise)                        -> bench_default.jsonl (appended)
# every stage is resumable (ncc_bench skips done keys).
set -u; cd /home/ubuntu; R=/home/ubuntu/results; S=/home/ubuntu/scripts; L=$R/bench_all.log
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
stage () { echo "### [$(date -u +%T)] $1" | tee -a $L; }
until grep -q CAPTURE_ALL_DONE $R/capture_all.log 2>/dev/null; do sleep 30; done
stage "BENCH start; manifest rows: $(wc -l < $R/manifest.jsonl)"
TO=${TIMEOUT:-1200}
# 1. family-balanced subset first (chosen from capture-time compile times): 3 warm repeats (+profile on r0), 2 cold, flags, versions
python $S/select_subset.py --bench /nonexistent --manifest $R/manifest.jsonl --out $R/manifest_subset.jsonl --k 3 --lo 1.5 --hi 240 2>&1 | tee -a $L
python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout $TO --profile --num-exec 10 >> $R/bench_default.log 2>&1
stage "subset r0 done: $(wc -l < $R/bench_default.jsonl) rows"
python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 3 --timeout $TO >> $R/bench_default.log 2>&1
stage "subset r1,r2 done: $(wc -l < $R/bench_default.jsonl) rows"
python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_cold.jsonl --flagsets default --repeats 1 --cold --timeout $TO --limit 24 >> $R/bench_cold.log 2>&1
stage "cold done: $(wc -l < $R/bench_cold.jsonl) rows"
python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_flags.jsonl --flagsets O1,O3,transformer --repeats 1 --timeout $TO --profile --num-exec 10 --limit 24 >> $R/bench_flags.log 2>&1
stage "flags done: $(wc -l < $R/bench_flags.jsonl) rows"
for v in 2.24.8799.0 2.19.8089.0 2.22.12471.0 2.20.9961.0; do
  python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_versions.jsonl --cc /home/ubuntu/venv_cc_$v/bin/neuronx-cc --cc-tag cc$v --flagsets default --repeats 1 --timeout $TO --limit 24 >> $R/bench_versions.log 2>&1
  stage "version $v done: $(wc -l < $R/bench_versions.jsonl) rows"
done
# 2. everything else: default flags, repeat 0, step graphs first (profiled), then init graphs
python - <<EOF
import json
rows=[json.loads(l) for l in open("$R/manifest.jsonl") if l.strip()]
open("$R/manifest_new.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows if r.get("module_is_new")))
print("manifest_new", sum(1 for r in rows if r.get("module_is_new")), "of", len(rows))
EOF
python $S/ncc_bench.py --manifest $R/manifest_new.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout $TO --profile --num-exec 10 >> $R/bench_default.log 2>&1
stage "all step graphs r0 done: $(wc -l < $R/bench_default.jsonl) rows"
python $S/ncc_bench.py --manifest $R/manifest.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout $TO >> $R/bench_default.log 2>&1
stage "init graphs r0 done: $(wc -l < $R/bench_default.jsonl) rows"
python $S/ncc_bench.py --manifest $R/manifest_new.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 2 --timeout $TO >> $R/bench_default.log 2>&1
stage "all step graphs r1 done: $(wc -l < $R/bench_default.jsonl) rows"
stage "BENCH_ALL_DONE"
