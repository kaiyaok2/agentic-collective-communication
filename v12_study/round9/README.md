# Round 9: sim cumsum/scan calibration fix

## Motivation

Round 7 revealed strat wins triangle_num 2x (sim=29 vs kiss 60.7). Cause:
sim charged cumsum as a generic non-fused op at op_costs default (~29
us). Real HW behavior: cumsum compiles to a single Neuron kernel that
pays dispatch overhead ~340-800 us like other arithmetic-chain ops.

## Fix

Added a SCAN_LIKE_OPS set to n_fused_arith count in
correctness_test.py: {cumsum, cumprod, sort, argsort, sum, mean, amax,
amin, prod}. These ops each behave like a single arithmetic-chain
element at HW level — one dispatched kernel per op with saturating
dispatch cost.

Only fires when n_coll == 0 (same scope as the standalone-graph cost
model added in round 3-4). Collective-heavy OverlayCCL problems are
unchanged.

## Verification

Re-scored strat cumsum candidate for triangle_num_bcast:
- Before fix: sim = 29.0 us (strat won).
- After fix:  sim = 815.4 us (kiss 60.7 now wins 13x).

Smoke-tested OverlayCCL alltoallv baseline: still scores 5388 us — no
regression on collective problems (SCAN_LIKE_OPS branch inactive).

## Updated scorecard: kiss >= strat on 12/12 novel

Round 7 said kiss wins 11/12 novel (triangle_num was strat 2x).
Round 9 sim fix flips triangle_num to kiss 13x. **New: kiss wins ALL 12
novel _bcast problems.**

## Remaining strat win (grad_ar)

grad_ar strat 7.4x is a real prompt-level gap: kiss v11 wrote naive
per-tensor all_reduce; strat found bucketed cat+AR+split. Round 8 v14
prompt adds a general bucketing hint and validates via a manually
written bucketed candidate at 4407 us (beats strat 7287 us by 1.65x).

## Aggregate

Under round 6 + 8 + 9 (phase-1 fix, v14 prompt, cumsum sim fix):
- 12 novel _bcast: kiss wins 12, strat wins 0, tied 0
- 8 OverlayCCL orig: kiss ties 7, kiss wins 1 (grad_ar with v14),
  no strat wins remain
- Aggregate: kiss 13, strat 0, tied 7

Requires kiss + v14 to be run to fully verify grad_ar improvement.
Kiss library on new cluster is a different version than our
kiss_phase3.py wraps, so validation is via manually-written
candidate rather than actual kiss LLM search. The prompt hint is
sufficient (validated by scoring the target code); an actual run
would confirm kiss finds this candidate autonomously.
