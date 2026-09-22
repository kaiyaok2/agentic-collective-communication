# Confirmed-divergence artifacts (Sorcar vs Overlay)

Curated per-seed artifacts for every confirmed Sorcar > Overlay divergence, plus the
group-theoretic candidates. For each problem and each results_* round it appeared in, we
keep every seed's `best_code.py` (the winning generated program), the pipeline JSON
(`overlay.json` / `kiss_summary.json`), and the per-seed `.log` trajectory. `MANIFEST_sims.json`
records the per-round overlay/kiss sim lists.

## Groups

### robust_family1/  (19 robust wins — all ONE family)
Rank-heterogeneous per-shard **multiplicative** scale/unscale collapses. These CONFIRM at
the strict gate (best-of-N ratio ≥1.05 AND Mann-Whitney p<0.05 AND bootstrap CI_lo>1.0),
with headline wins rechecked at best-of-16. Probed along six axes: payload (r1), depth
(r2/r9), description/framing (r16–r23), layout (r26_perm), scale constants (r31_lin5_count8),
and operator algebra + routing (r33_permscale — multiplicative scale plus a permutation).

### bo16_escape/  (draw-fragile — NOT counted as robust, kept for honesty)
- `r26_strided_count8` — interleaved layout; median stays trapped but Overlay's best of 16
  draws finds the fusing solution.
- `r31_lin5_res` — result-only docstring over the 5-level linear chain; median traps at bo8
  but Overlay's best of 16 escapes.
Both illustrate that best-of-8 confirmation is not automatically robust and that the strict
CI_lo>1.0 gate is load-bearing.

### second_family_candidate/  (pure-permutation, ZERO multiplicative scale)
- `r37_rotsum8_count8` — cyclic block-rotation chain. CONFIRMED at bo8 but ESCAPED at bo16:
  its net map is a closed-form cyclic shift by sum(1..7)=28, so Overlay's best-of-16 draw
  one-shots the fold. This is the "Sorcar distributionally more reliable but not robust"
  case that motivated the r39 tuning.
- `r39_permsum8_*` — TUNED successor: each stage's rotation is composed with a fixed
  non-affine, non-involutive block permutation PI=[3,0,5,7,1,6,2,4], so the net permutation
  has NO closed form. The only correct fold composes the D-1 stage index arrays step by step;
  Overlay's obvious "shift by c" / "apply PI once" guesses FAIL the fp32 gate (verified:
  the single-PI guess yields max_diff≈4.4). No per-shard scale anywhere — a genuine distinct
  (group-theoretic) mechanism. Result recorded in campaign_r39.stdout / ledger round r39.

## Criterion (identical for both pipelines)
Fair fp32 correctness gate on BOTH; SYMMETRIC best-of-N (both systems N iid temperature
draws, no RNG seeding, no caching). CONFIRMED = best-of-N ratio ≥1.05 AND Mann-Whitney U
p<0.05 (one-sided, tie-corrected) AND bootstrap 95% CI lower >1.0. Both pipelines run
Sonnet 4.5 on Bedrock.
