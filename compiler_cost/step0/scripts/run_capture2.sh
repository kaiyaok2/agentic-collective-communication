#!/bin/bash
# recovery chain: wait for the first chain's marker, then re-capture synth/real/mutation with two drivers (one Neuron core each),
# then collective grid (2 ranks), acc workloads (2 ranks), manifest, bench campaign.
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu; export PJRT_DEVICE=NEURON
until grep -q CAPTURE_ALL_DONE $R/capture_all.log 2>/dev/null; do sleep 20; done
echo "### [$(date -u +%T)] capture2: two drivers" | tee -a $R/capture2.log
NEURON_RT_VISIBLE_CORES=0 python $S/capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus --timeout 2400 > $R/capture_d1.log 2>&1 &
P1=$!
NEURON_RT_VISIBLE_CORES=1 python $S/capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus --timeout 2400 --reverse > $R/capture_d2.log 2>&1 &
P2=$!
wait $P1 $P2
echo "### [$(date -u +%T)] capture2: drivers done; collective + acc (2 ranks)" | tee -a $R/capture2.log
python $S/capture.py --grids collective --corpus /home/ubuntu/corpus --timeout 1200 --nproc 2 > $R/capture_coll2.log 2>&1
bash $S/capture_acc.sh > $R/capture_acc.log 2>&1
python $S/build_manifest.py --corpus /home/ubuntu/corpus --out $R/manifest.jsonl 2>&1 | grep -vE "Warning|warn" | tee -a $R/capture2.log
echo "### [$(date -u +%T)] CAPTURE2_DONE" | tee -a $R/capture2.log
bash $S/run_bench_all.sh
