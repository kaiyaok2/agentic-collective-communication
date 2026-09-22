#!/bin/bash
# Chain: wait for family-2 (r43fam2) to finish, then run family-1 (task b) and
# family-3 (task c) sequentially, all at 9 seeds. Keeps cloud concurrency bounded
# (one round of MAX_PAR=10 at a time).
set -u
cd /private/tmp/fair_diverge
source /private/tmp/ablation_venv/bin/activate
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 CLAUDE_CODE_USE_BEDROCK=1 MAX_PAR=10 KISS_SRC=/private/tmp/kiss_ai/src

# 1) wait for family-2 to finish
while ps -p 25987 >/dev/null 2>&1; do sleep 30; done
echo "[chain] family-2 done; starting family-1 (task b) @ 9 seeds"

# 2) family-1: 19 problems @ 9 seeds
FAM1="r1_fold_s256,r1_fold_s1024,r1_fold_s4096,r1_fold_s256_deep,r2_deep4,r2_deep8,r2_deep6_big,r9_deep7,r9_deep8_big,r16_deepdoc,r20_su8_narr,r21_su8_countonly,r21_su8_proconly,r22_su8_count16,r22_su8_count8,r23_deep8_count8,r26_perm_count8,r31_lin5_count8,r33_permscale8_count8"
python3 confirm_only.py --round r44fam1 --problems "$FAM1" --confirm-seeds 9 > r44fam1.stdout 2>&1
echo "[chain] family-1 done; starting family-3 (task c) @ 9 seeds"

# 3) family-3: off-diagonal coupling probe, 8 problems @ 9 seeds
FAM3="r42_couple_d8_count8,r42_couple_d7_count8,r42_couple_d6_count8,r42_couple_d5_count8,r42_couple_d4_count8,r42_couple_d8_big,r42_couple_d8_res,r42_couple_d6_res"
python3 confirm_only.py --round r45fam3 --problems "$FAM3" --confirm-seeds 9 > r45fam3.stdout 2>&1
echo "[chain] family-3 done; ALL 9-seed runs complete"
