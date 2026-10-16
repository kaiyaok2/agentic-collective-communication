#!/bin/bash
# single sequential tail: after the 2.19 version arm, pause the generic chain and run (in order)
# pairs bench -> known-fail bench -> vit/t5 re-capture + manifest rebuild -> subset re-profile -> versions 2.22/2.20 -> all step graphs r0 (profiled) -> init graphs -> r1
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu; export PJRT_DEVICE=NEURON
mark () { echo "### [$(date -u +%T)] $1" >> $R/bench_all.log; }
until grep -q "version 2.19.8089.0 done" $R/bench_all.log 2>/dev/null; do sleep 30; done
pkill -f "scripts/run_bench_al[l]"; sleep 1; pkill -f "scripts/ncc_bench.p[y]"; sleep 2; mark "TAIL: generic chain paused"
python - <<PY
import json
rows=[json.loads(l) for l in open("$R/manifest.jsonl") if l.strip()]
bases=("synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_train1","synth_attn__dim1024_heads8_layers4_seq1024_batch4_train1","hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_train1","hf__archllama_layers4_hidden512_heads8_batch1_seq512_train1")
keep=[r for r in rows if r.get("module_is_new") and (r["prog_id"] in bases or (r.get("source")=="mutation" and r.get("family")!="synth_conv" and any((r.get("mutation") or "").endswith(s) for s in ("plus1","minus1","half","x2","dtype_bf16","train_off"))))]
open("$R/manifest_pairs.jsonl","w").write("".join(json.dumps(r)+"\n" for r in keep)); print("pairs manifest", len(keep))
PY
python $S/ncc_bench.py --manifest $R/manifest_pairs.jsonl --out $R/bench_pairs.jsonl --flagsets default --repeats 2 --timeout 900 >> $R/bench_pairs.log 2>&1; mark "PAIRS done: $(wc -l < $R/bench_pairs.jsonl) rows"
python - <<PY
import json, os, glob
rows=[]
for pid in ("mut__synth_conv__ch128_layers8_img64_batch8_train1__batch_plus1", "synth_gather__kindgather_vocab32768_dim256_n_idx8192"):
    pbs=sorted(glob.glob(f"/home/ubuntu/corpus/{pid}/cache/neuronxcc-*/MODULE_*/model.hlo_module.pb"), key=os.path.getsize)
    for pb in pbs:
        mh=os.path.basename(os.path.dirname(pb)).split("+")[0].replace("MODULE_","")
        rows.append(dict(hlo_id=f"{pid}__M{mh[-10:]}", hlo_pb=pb, prog_id=pid, family=("synth_conv" if "conv" in pid else "synth_gather"), source="known_fail", module_is_new=(pb==pbs[-1])))
open("$R/manifest_fail.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows)); print("manifest_fail rows", len(rows))
PY
python $S/ncc_bench.py --manifest $R/manifest_fail.jsonl --out $R/bench_fail.jsonl --flagsets default --repeats 1 --timeout 1500 >> $R/bench_fail.log 2>&1; mark "known-fail bench done"
NEURON_RT_VISIBLE_CORES=0 python $S/capture.py --grids real --filter "archvit|archt5" --corpus /home/ubuntu/corpus --timeout 1800 --min-free-gb 6 > $R/capture_vit_t5.log 2>&1; mark "vit/t5 re-capture done"
python $S/build_manifest.py --corpus /home/ubuntu/corpus --out $R/manifest.jsonl 2>&1 | grep -vE "Warning|warn" >> $R/bench_all.log
nice -n 5 python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_profile.jsonl --cc-tag 2.21p --flagsets default --repeats 1 --timeout 1200 --profile --num-exec 10 >> $R/bench_profile.log 2>&1; mark "subset re-profile done: $(wc -l < $R/bench_profile.jsonl) rows"
for v in 2.22.12471.0 2.20.9961.0; do
  python $S/ncc_bench.py --manifest $R/manifest_subset.jsonl --out $R/bench_versions.jsonl --cc /home/ubuntu/venv_cc_$v/bin/neuronx-cc --cc-tag cc$v --flagsets default --repeats 1 --timeout 1200 --limit 24 >> $R/bench_versions.log 2>&1; mark "version $v done: $(wc -l < $R/bench_versions.jsonl) rows"
done
python - <<PY
import json
rows=[json.loads(l) for l in open("$R/manifest.jsonl") if l.strip()]
open("$R/manifest_new.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows if r.get("module_is_new")))
PY
python $S/ncc_bench.py --manifest $R/manifest_new.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout 1200 --profile --num-exec 10 >> $R/bench_default.log 2>&1; mark "all step graphs r0 done: $(wc -l < $R/bench_default.jsonl) rows"
python $S/ncc_bench.py --manifest $R/manifest.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 1 --timeout 1200 >> $R/bench_default.log 2>&1; mark "init graphs r0 done"
python $S/ncc_bench.py --manifest $R/manifest_new.jsonl --out $R/bench_default.jsonl --flagsets default --repeats 2 --timeout 1200 >> $R/bench_default.log 2>&1; mark "BENCH_ALL_DONE (tail chain)"
