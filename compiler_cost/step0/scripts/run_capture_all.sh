#!/bin/bash
# full corpus capture: synth -> real -> mutation -> collective (torchrun 2 ranks)
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu; export PJRT_DEVICE=NEURON
echo "### capture start $(date -u +%T)"
python scripts/capture.py --grids synth,real,mutation --corpus /home/ubuntu/corpus --timeout 2400 2>&1 | grep -E "^\[capture"
echo "### collective start $(date -u +%T)"
python scripts/capture.py --grids collective --corpus /home/ubuntu/corpus --timeout 1200 --nproc 2 2>&1 | grep -E "^\[capture"
echo "### capture done $(date -u +%T)"
python scripts/build_manifest.py --corpus /home/ubuntu/corpus --out /home/ubuntu/results/manifest.jsonl 2>&1 | grep -vE "Warning|warn"
echo "### CAPTURE_ALL_DONE $(date -u +%T)"
