#!/bin/bash
# Step 1 campaign: capture the new grids (two single-core drivers, then the 2-rank collective candidates), rebuild the
# manifest, then bench: default r0 (+profile) for new modules, all compiler versions and flag sets on every step graph
# (two concurrent workers, note=par2), and a second default repeat for the new modules. Every stage is resumable.
set -u; cd /home/ubuntu; S=/home/ubuntu/scripts; R=/home/ubuntu/results; C=/home/ubuntu/corpus; L=$R/step1.log
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
stage () { echo "### [$(date -u +%FT%TZ)] $1" | tee -a $L; }
TO=${TIMEOUT:-1500}
stage "STEP1 start"
if ! grep -q "CAPTURE_SWEEPS_DONE" $L; then
  python $S/capture.py --grids sweeps,synth2 --corpus $C --timeout 2400 --min-free-gb 9 > $R/capture_step1_a.log 2>&1 &
  NEURON_RT_VISIBLE_CORES=1 python $S/capture.py --grids sweeps,synth2 --corpus $C --timeout 2400 --reverse --min-free-gb 9 > $R/capture_step1_b.log 2>&1 &
  wait; stage "CAPTURE_SWEEPS_DONE"
fi
if ! grep -q "CAPTURE_ACC_DONE" $L; then
  python $S/capture.py --grids acc_cand --corpus $C --nproc 2 --timeout 900 > $R/capture_step1_acc.log 2>&1; stage "CAPTURE_ACC_DONE"
fi
python $S/build_manifest.py --corpus $C --out $R/manifest_step1.jsonl 2>&1 | tee -a $L
python - <<PY
import json
rows=[json.loads(l) for l in open("$R/manifest_step1.jsonl") if l.strip()]
new=[r for r in rows if r.get("module_is_new")]
open("$R/manifest_step1_new.jsonl","w").write("".join(json.dumps(r)+"\n" for r in new))
for i in range(2): open(f"$R/manifest_step1_new_s{i}.jsonl","w").write("".join(json.dumps(r)+"\n" for r in new[i::2]))
old=set(json.loads(l)["hlo_id"] for l in open("$R/manifest.jsonl") if l.strip())
add=[r for r in rows if r["hlo_id"] not in old]
open("$R/manifest_step1_added.jsonl","w").write("".join(json.dumps(r)+"\n" for r in add))
for i in range(2): open(f"$R/manifest_step1_added_s{i}.jsonl","w").write("".join(json.dumps(r)+"\n" for r in add[i::2]))
print("manifest_step1", len(rows), "step graphs", len(new), "added vs step0", len(add))
PY
stage "manifest built"
export NCC_BENCH_NOTE=par2
# default r0 with profiling for every module added in Step 1 (two workers)
for i in 0 1; do python $S/ncc_bench.py --manifest $R/manifest_step1_added_s$i.jsonl --out $R/bench_step1_default_s$i.jsonl --flagsets default --repeats 1 --timeout $TO --profile --num-exec 10 --workroot /home/ubuntu/ncc_work_s$i > $R/bench_step1_default_s$i.log 2>&1 & done; wait
stage "step1 default r0 done: $(cat $R/bench_step1_default_s*.jsonl | wc -l) rows"
# compiler versions on all step graphs
for v in 2.24.8799.0 2.19.8089.0 2.22.12471.0 2.20.9961.0; do
  for i in 0 1; do python $S/ncc_bench.py --manifest $R/manifest_step1_new_s$i.jsonl --out $R/bench_versions_full_s$i.jsonl --cc /home/ubuntu/venv_cc_$v/bin/neuronx-cc --cc-tag cc$v --flagsets default --repeats 1 --timeout $TO --workroot /home/ubuntu/ncc_work_s$i > $R/bench_versions_full_$v_s$i.log 2>&1 & done; wait
  stage "version $v done: $(cat $R/bench_versions_full_s*.jsonl | wc -l) rows"
done
# flag sets on all step graphs
for i in 0 1; do python $S/ncc_bench.py --manifest $R/manifest_step1_new_s$i.jsonl --out $R/bench_flags_full_s$i.jsonl --flagsets O1,O3,transformer --repeats 1 --timeout $TO --workroot /home/ubuntu/ncc_work_s$i > $R/bench_flags_full_s$i.log 2>&1 & done; wait
stage "flags done: $(cat $R/bench_flags_full_s*.jsonl | wc -l) rows"
# second default repeat for the added modules (noise)
for i in 0 1; do python $S/ncc_bench.py --manifest $R/manifest_step1_added_s$i.jsonl --out $R/bench_step1_default_s$i.jsonl --flagsets default --repeats 2 --timeout $TO --workroot /home/ubuntu/ncc_work_s$i > $R/bench_step1_default_r1_s$i.log 2>&1 & done; wait
stage "STEP1_DONE"
