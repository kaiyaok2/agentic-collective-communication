#!/bin/bash
set -u
echo "##### TOP-LINE HEADLINE N_MB=48 STEPS=24 (llama then gpt) $(date -u) #####"
STEPS=24 bash /home/ubuntu/e2e_headline.sh llama 48
echo E2E_DEEP48_LLAMA_DONE
STEPS=24 bash /home/ubuntu/e2e_headline.sh gpt 48
echo E2E_DEEP48_GPT_DONE
