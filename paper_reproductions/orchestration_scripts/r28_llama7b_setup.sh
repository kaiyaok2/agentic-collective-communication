#!/bin/bash
# Create Llama-7B-equivalent training script as a variant of amp2.
# Llama-7B reference shape: DM=4096, HID=11008, N_LAYERS=32, S=2048.
# Use HID=14336 (64*224) and VOCAB=32256 (144*224) for clean sharding.
set +u
[ -f /home/ubuntu/.profile ] && . /home/ubuntu/.profile
set -u
REPO=/home/ubuntu/agentic-collective-communication
SRC=$REPO/experiments/model_extension/train_llama_e2e_amp2.py
DST=$REPO/experiments/model_extension/train_llama_e2e_7b.py
cp $SRC $DST
# Edit the shape constants
sed -i.bak '
  s/^DM = 2048/DM = 4096/;
  s/^HID = 5376/HID = 14336/;
  s/^N_LAYERS_PER_STAGE = 1/N_LAYERS_PER_STAGE = 16/;
  s/^VOCAB = 224 \* 32.*$/VOCAB = 224 * 144  # 32256 tokens-per-rank shard/;
' $DST
echo "===  shape constants:"
grep -E "^DM|^HID|^N_LAYERS|^VOCAB|^N_MB|^B |^S " $DST | head -10
rm -f $DST.bak
