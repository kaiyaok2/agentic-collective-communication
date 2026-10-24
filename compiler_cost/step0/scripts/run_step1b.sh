#!/bin/bash
# Step 1b: once the collective-candidate captures are done, stop the Step 1 chain before it starts benching, capture the
# real-workload grid (two drivers), install the production NxD inference graphs, then re-enter run_step1.sh (its capture
# stages are marked done, so it rebuilds the manifest and benches everything that is new).
set -u; cd /home/ubuntu; S=/home/ubuntu/scripts; R=/home/ubuntu/results; C=/home/ubuntu/corpus; L=$R/step1.log
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
stage () { echo "### [$(date -u +%FT%TZ)] $1" | tee -a $L; }
until grep -q CAPTURE_ACC_DONE $L; do sleep 30; done
pkill -f "scripts/run_step1\.s[h]"; pkill -f "scripts/ncc_bench\.p[y]"; sleep 3
stage "STEP1B start (real workloads)"
if ! grep -q CAPTURE_REAL2_DONE $L; then
  python $S/capture.py --grids real2 --corpus $C --timeout 2400 --min-free-gb 9 > $R/capture_step1_real2_a.log 2>&1 &
  NEURON_RT_VISIBLE_CORES=1 python $S/capture.py --grids real2 --corpus $C --timeout 2400 --reverse --min-free-gb 9 > $R/capture_step1_real2_b.log 2>&1 &
  wait; stage "CAPTURE_REAL2_DONE"
fi
if [ ! -d $C/nxd33b__* ] 2>/dev/null; then
  mkdir -p /home/ubuntu/nxd33b && tar xzf /home/ubuntu/nxd33b_hlo.tgz -C /home/ubuntu/nxd33b && python $S/import_nxd.py /home/ubuntu/nxd33b $C nxd33b 2>&1 | tee -a $L
fi
stage "re-entering run_step1.sh for manifest + bench"
exec bash $S/run_step1.sh
