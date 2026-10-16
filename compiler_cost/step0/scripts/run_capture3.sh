#!/bin/bash
# post-reboot chain: two capture drivers (one Neuron core each, memory-guarded) -> collective grid -> acc workloads -> manifest -> bench
R=/home/ubuntu/results; S=/home/ubuntu/scripts; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu; export PJRT_DEVICE=NEURON
echo "### [$(date -u +%T)] capture3: two drivers" | tee -a $R/capture3.log
NEURON_RT_VISIBLE_CORES=0 python $S/capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus --timeout 2400 > $R/capture_d1.log 2>&1 &
P1=$!
sleep 20
NEURON_RT_VISIBLE_CORES=1 python $S/capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus --timeout 2400 --reverse > $R/capture_d2.log 2>&1 &
P2=$!
wait $P1 $P2
echo "### [$(date -u +%T)] capture3: drivers done; collective + acc (2 ranks)" | tee -a $R/capture3.log
python $S/capture.py --grids collective --corpus /home/ubuntu/corpus --timeout 1200 --nproc 2 > $R/capture_coll3.log 2>&1
bash $S/capture_acc.sh > $R/capture_acc.log 2>&1
python $S/build_manifest.py --corpus /home/ubuntu/corpus --out $R/manifest.jsonl 2>&1 | grep -vE "Warning|warn" | tee -a $R/capture3.log
echo "### CAPTURE_ALL_DONE $(date -u +%T)" | tee -a $R/capture3.log $R/capture_all.log
bash $S/run_bench_all.sh
