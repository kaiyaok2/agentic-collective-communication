#!/bin/bash
set -u
echo "##### DEEP HEADLINE N_MB=32 STEPS=24 (llama then gpt) $(date -u) #####"
STEPS=24 bash /home/ubuntu/e2e_headline.sh llama 32
echo E2E_DEEP32_LLAMA_DONE
STEPS=24 bash /home/ubuntu/e2e_headline.sh gpt 32
echo E2E_DEEP32_GPT_DONE
