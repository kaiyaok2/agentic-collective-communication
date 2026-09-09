#!/bin/bash
# 8-slot compile semaphore: caps concurrent walrus_driver memory.
SLOTS=8
for i in $(seq 0 $((SLOTS-1))); do touch /tmp/nccslot_$i.lock; done
while true; do
  for i in $(seq 0 $((SLOTS-1))); do
    exec 9>/tmp/nccslot_$i.lock
    if flock -n 9; then
      exec /opt/aws_neuronx_venv_pytorch_2_8/bin/neuronx-cc.real "$@"
    fi
    exec 9>&-
  done
  sleep 2
done
