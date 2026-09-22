#!/bin/bash
# Archival 9-seed re-run of all 25 confirmed problems (3 families), sequential
# per-family batches; after each batch the results tree + ledger are committed
# and pushed so a reboot can never destroy trajectories again. Round names match
# curate_9seed.py's GROUPS so curation is turnkey.
set -u
source /private/tmp/ablation_venv/bin/activate
export PYTHONPATH=/private/tmp/acc_verify ACC_REPO=/private/tmp/acc_verify
export KISS_SRC=/private/tmp/kiss_ai/src
export CLAUDE_CODE_USE_BEDROCK=1 AWS_PROFILE=kaiyao GATE_MODE=fp32 MAX_PAR=8
FD=/private/tmp/fair_diverge
REPO=/private/tmp/acc_verify
cd "$FD"

FAM1="r21_su8_countonly,r23_deep8_count8,r22_su8_count16,r26_perm_count8,r31_lin5_count8,r20_su8_narr,r16_deepdoc,r1_fold_s4096,r1_fold_s1024,r2_deep4,r1_fold_s256"
FAM2="r43_route_B11_d8_L3_count8,r43_route_B10_strided_count8,r43_route_B9_d8_L3_count8,r40_route_d8_L3_count8,r40_route_d6_L3_count8,r43_route_B10_p384_count8,r43_route_B12_d8_L3_res"
FAM3A="r47_xc_r4,r47_dd_relu_d8,r47_dd_meanabs_d7,r47_dd_meanabs_d8"
FAM3B="r48_dd_meansq_d8,r48_dd_square_d8,r48_dd_relu_d6"

commit_batch () {
  local rnd="$1"
  mkdir -p "$REPO/fair_divergence_campaign/archival_9seed"
  rsync -a --exclude='*.pyc' "$FD/results_${rnd}" "$REPO/fair_divergence_campaign/archival_9seed/" 2>/dev/null
  cp -f "$FD/campaign_ledger.json" "$REPO/fair_divergence_campaign/archival_9seed/" 2>/dev/null
  cd "$REPO"
  git add fair_divergence_campaign/archival_9seed >/dev/null 2>&1
  git -c user.name="kaiyaok2" -c user.email="kaiyaoke@berkeley.edu" \
      commit --author="Kaiyao Ke <kaiyaoke@berkeley.edu>" -q \
      -m "Archival 9-seed trajectories: ${rnd}" && git push origin main 2>&1 | tail -1
  cd "$FD"
}

echo "=== archival re-run start $(date) ==="
for pair in "r44fam1:$FAM1" "r43fam2:$FAM2" "r47fam3:$FAM3A" "r48fam3:$FAM3B"; do
  rnd="${pair%%:*}"; probs="${pair#*:}"
  echo "=== batch $rnd start $(date) ==="
  python3 confirm_only.py --round "$rnd" --problems "$probs" --confirm-seeds 9
  echo "=== batch $rnd done $(date); committing ==="
  commit_batch "$rnd"
done

echo "=== curating ==="
python3 curate_9seed.py
rsync -a "$FD/CONFIRMED_WINS_ARTIFACTS" "$REPO/fair_divergence_campaign/" 2>/dev/null
cd "$REPO"
git add fair_divergence_campaign/CONFIRMED_WINS_ARTIFACTS >/dev/null 2>&1
git -c user.name="kaiyaok2" -c user.email="kaiyaoke@berkeley.edu" \
    commit --author="Kaiyao Ke <kaiyaoke@berkeley.edu>" -q \
    -m "Archival 9-seed curated artifacts: 3 families, 25 confirmed problems" && git push origin main 2>&1 | tail -1
echo "=== archival re-run COMPLETE $(date) ==="
