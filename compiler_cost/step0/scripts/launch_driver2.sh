#!/bin/bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu; export PJRT_DEVICE=NEURON NEURON_RT_VISIBLE_CORES=1
nohup python scripts/capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus --timeout 2400 --reverse > /home/ubuntu/results/capture_driver2.log 2>&1 < /dev/null & disown
echo driver2 started
