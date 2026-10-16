#!/bin/bash
# after the 2.19 version stage: stop the generic chain, bench the near-identical pairs (2 warm repeats), then resume the rest
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu
until grep -q "version 2.19.8089.0 done" $R/bench_all.log 2>/dev/null; do sleep 30; done
pkill -f "scripts/run_bench_al[l]"; sleep 1; pkill -f "scripts/ncc_bench.p[y]"; sleep 2
echo "### [$(date -u +%T)] PAIRS stage: chain paused" >> $R/bench_all.log
python - <<PY
import json
rows=[json.loads(l) for l in open("$R/manifest.jsonl") if l.strip()]
keep=[]
for r in rows:
    if not r.get("module_is_new"): continue
    pid=r["prog_id"]; mut=r.get("mutation") or ""
    base = pid in ("synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_train1","synth_attn__dim1024_heads8_layers4_seq1024_batch4_train1",
                   "hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_train1","hf__archllama_layers4_hidden512_heads8_batch1_seq512_train1")
    pair = r.get("source")=="mutation" and r.get("family")!="synth_conv" and any(mut.endswith(s) for s in ("plus1","minus1","half","x2","dtype_bf16","train_off"))
    if base or pair: keep.append(r)
open("$R/manifest_pairs.jsonl","w").write("".join(json.dumps(r)+"\n" for r in keep)); print("pairs manifest", len(keep))
PY
python $S/ncc_bench.py --manifest $R/manifest_pairs.jsonl --out $R/bench_pairs.jsonl --flagsets default --repeats 2 --timeout 900 >> $R/bench_pairs.log 2>&1
echo "### [$(date -u +%T)] PAIRS done: $(wc -l < $R/bench_pairs.jsonl) rows" >> $R/bench_all.log
# resume: remaining versions, then all step graphs r0 (profiled), init graphs, r1
for v in 2.22.12471.0 2.20.9961.0; do
  python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_versions.jsonl --cc /home/ubuntu/venv_cc_$v/bin/neuronx-cc --cc-tag cc$v --flagsets default --repeats 1 --timeout 1200 --limit 24 >> $R/bench_versions.log 2>&1
  echo "### [$(date -u +%T)] version $v done: $(wc -l < $R/bench_versions.jsonl) rows" >> $R/bench_all.log
done
python - <<PY
import json
rows=[json.loads(l) for l in open("$R/manifest.jsonl") if l.strip()]
open("$R/manifest_new.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows if r.get("module_is_new")))
PY
python $S/ncc_bench.py --manifest $R/manifest_new.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout 1200 --profile --num-exec 10 >> $R/bench_default.log 2>&1
echo "### [$(date -u +%T)] all step graphs r0 done: $(wc -l < $R/bench_default.jsonl) rows" >> $R/bench_all.log
python $S/ncc_bench.py --manifest $R/manifest.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout 1200 >> $R/bench_default.log 2>&1
python $S/ncc_bench.py --manifest $R/manifest_new.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 2 --timeout 1200 >> $R/bench_default.log 2>&1
echo "### [$(date -u +%T)] BENCH_ALL_DONE (resumed chain)" >> $R/bench_all.log
