#!/usr/bin/env bash
# Resume the fair-gate divergence campaign after the Anthropic workspace API usage
# cap lifts (2026-10-01 00:00 UTC). No re-derivation needed: r29/r31 problem defs
# are already staged in /private/tmp/acc_verify/search/ and pass the fp32 gate.
#
# What these rounds test (generality of the 17 robust confirmed wins):
#   r29_pow2_count8 / r29_pow2_res  : 4-level powers-of-2 per-shard scale/unscale
#                                     (a[r]=2**((r%4)-1)) -> is the trapped regime
#                                     tied to the 3-level {1,1.5,2} constants, or is
#                                     it a property of "contiguous rank-heterogeneous
#                                     multiplicative depth"? Predict: count8 CONFIRMS.
#   r31_lin5_count8 / r31_lin5_res  : 5-level additive-linear (a[r]=1+0.25*(r%5)).
#                                     Same generality question, richer constant set.
#
# Prediction from the r22/r23/r26 causal chain: the *_count8 variants CONFIRM robustly
# (truthful high count over deep rank-heterogeneous contiguous code); the *_res
# variants trap the MEDIAN but may escape best-of-N (result-only doc, per L12).
#
# Both outcomes are informative:
#   - CONFIRM  -> the win generalizes across constant sets (kills a single-constant-
#                 artifact objection to the 17).
#   - Screen fold -> the win is constant-specific (narrows the honest scope claim).
set -euo pipefail
cd /private/tmp/fair_diverge

PY=/private/tmp/ablation_venv/bin/python

# 1. Gate on the cap: bail cleanly if still capped (safe to run repeatedly / via cron).
if ! "$PY" -c "
import anthropic
c=anthropic.Anthropic()
try:
    c.messages.create(model='claude-sonnet-4-5-20250929',max_tokens=4,
        messages=[{'role':'user','content':'ping'}])
    print('API_OK')
except Exception as e:
    if 'workspace API usage' in str(e):
        print('STILL_CAPPED'); raise SystemExit(3)
    raise
" ; then
  echo "[resume] API still capped (regains 2026-10-01 00:00 UTC) — not launching."
  exit 0
fi

# 2. Run the two generality rounds through the SAME symmetric best-of-8 screen+confirm.
#    campaign.py auto-discovers r29/r31 via problems_all_catalogs.
"$PY" campaign.py --round r29 --problems r29_pow2_count8,r29_pow2_res --confirm-seeds 8
"$PY" campaign.py --round r31 --problems r31_lin5_count8,r31_lin5_res --confirm-seeds 8

# 3. Recheck any best-of-8 confirms at best-of-16 (the robustness bar the 17 met).
#    Edit the problem list below to whatever step 2 confirmed.
echo "[resume] Now recheck confirmed problems at best-of-16 via confirm_only.py, e.g.:"
echo "  $PY confirm_only.py --round r29b --problems <confirmed> --confirm-seeds 16"
echo "[resume] Then update CAMPAIGN_LOG.md, FINAL_CONCLUSION_*.md, and the memory file."
