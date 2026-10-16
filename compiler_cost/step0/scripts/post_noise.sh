#!/bin/bash
# after the noise stage: bench the two known compiler failures, then re-capture the fixed vit/t5 programs on core 1
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu; export PJRT_DEVICE=NEURON
until grep -q "subset r1,r2 done" $R/bench_all.log 2>/dev/null; do sleep 30; done
python - <<PY
import json, os, glob
rows=[]
for pid in ("mut__synth_conv__ch128_layers8_img64_batch8_train1__batch_plus1", "synth_gather__kindgather_vocab32768_dim256_n_idx8192"):
    pbs=sorted(glob.glob(f"/home/ubuntu/corpus/{pid}/cache/neuronxcc-*/MODULE_*/model.hlo_module.pb"), key=os.path.getsize)
    for pb in pbs:
        mh=os.path.basename(os.path.dirname(pb)).split("+")[0].replace("MODULE_","")
        rows.append(dict(hlo_id=f"{pid}__M{mh[-10:]}", hlo_pb=pb, prog_id=pid, family=pid.split("__")[0] if not pid.startswith("mut__") else pid.split("__")[1], source="known_fail", module_is_new=(pb==pbs[-1])))
open("$R/manifest_fail.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows)); print("manifest_fail rows", len(rows))
PY
python $S/ncc_bench.py --manifest $R/manifest_fail.jsonl --out $R/bench_fail.jsonl --flagsets default --repeats 1 --timeout 1500 >> $R/bench_fail.log 2>&1
echo "### [$(date -u +%T)] known-fail bench done" >> $R/bench_all.log
NEURON_RT_VISIBLE_CORES=1 python $S/capture.py --grids real --filter "archvit|archt5" --corpus /home/ubuntu/corpus --timeout 1800 --min-free-gb 6 > $R/capture_vit_t5.log 2>&1
echo "### [$(date -u +%T)] vit/t5 re-capture done" >> $R/bench_all.log
