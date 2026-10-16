#!/bin/bash
# one-time node setup: transformers in the torch venv; standalone venvs for alternate neuronx-cc versions
set -u; cd /home/ubuntu; mkdir -p logs corpus results
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
pip install -q transformers==4.46.3 datasets==3.1.0 2>&1 | tail -1; python -c "import transformers; print('transformers', transformers.__version__)"
deactivate
for v in 2.20.9961.0 2.22.12471.0 2.24.8799.0 2.19.8089.0; do
  d=/home/ubuntu/venv_cc_$v; [ -x $d/bin/neuronx-cc ] && { echo "$v present"; continue; }
  python3 -m venv $d && $d/bin/pip install -q --upgrade pip >/dev/null 2>&1
  $d/bin/pip install -q "neuronx-cc==$v" --extra-index-url https://pip.repos.neuron.amazonaws.com > logs/venv_cc_$v.log 2>&1 && echo "$v installed: $($d/bin/neuronx-cc --version 2>&1 | tail -1)" || echo "$v FAILED (see logs/venv_cc_$v.log)"
done
echo SETUP_DONE
