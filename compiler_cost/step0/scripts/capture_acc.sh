#!/bin/bash
# capture the collective-optimization workloads (OverlayCCL per-problem training harnesses) at 2 ranks on one chip:
# baseline and agent backends for each problem -> corpus/acc__<problem>__<backend>/cache
set -u; source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate; cd /home/ubuntu/acc; export PJRT_DEVICE=NEURON
C=/home/ubuntu/corpus; S=/home/ubuntu/scripts; STEPS=${STEPS:-8}
for p in tp_mlp fsdp_prefetch llama_block_ar pp_send_recv dxe ring_kv alltoallv uniform_a2a; do
  for b in baseline agent; do
    id="acc__${p}__${b}"; d=$C/$id; [ -f $d/meta.json ] && continue; mkdir -p $d/cache $d/results
    spec="{\"id\": \"$id\", \"family\": \"acc_$p\", \"source\": \"collective_workload\", \"params\": {\"problem\": \"$p\", \"backend\": \"$b\", \"world_size\": 2, \"steps\": $STEPS}}"
    echo "$spec" > $d/spec.json; t0=$(date +%s)
    NEURON_COMPILE_CACHE_URL=$d/cache NEURON_CC_FLAGS="--cache_dir=$d/cache" RESULTS_DIR=$d/results WORLD_SIZE=2 \
      timeout -k 20 1500 torchrun --nproc_per_node=2 training/train_${p}_7node.py --backend $b --steps $STEPS --warmup 2 > $d/capture.log 2>&1; rc=$?
    pkill -9 -f "train_${p}_7node" >/dev/null 2>&1
    python $S/write_meta.py $d "$spec" $rc $(( $(date +%s) - t0 )) $d/capture.log
  done
done
echo "### ACC_CAPTURE_DONE $(date -u +%T)"
