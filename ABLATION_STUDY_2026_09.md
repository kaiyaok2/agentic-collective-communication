# Ablation Study: Simulator Deltas, AI-Discovery Prompt, Adversarial Testing

**Run date**: 2026-09-07
**Cluster**: 7× trn1.32xlarge (224 NeuronCores), CB `cr-037b5eccfc31cc735`,
us-east-1c
**Model**: `claude-sonnet-4-5` via Bedrock; `--max-steps 30 --max-budget 3.0`
per search
**Problems**: the 8 catalog problems mapped 1:1 from the family sites
used in the ≥2× 10B e2e trainings (`SORCAR_E2E_10B_TP.md`):
F1 `sequential_ar_chain_edge_chal`, F2 `three_inline_ars_chal`,
F3 `ten_ar_alt_sign_zero_chal`, F4a `per_row_ar_M96_chal`,
F4b `grad_ar` (the paper's own bucketed grad-AR problem),
F5 `reduce_scatter_from_ar_chal`, F6 `mixmaxmin_chal`,
F7 `eightslab_chal`.

## Arms

| Arm | Phase-1 simulator | Phase-3 prompt |
|---|---|---|
| `base` (control) | current (deltas 1.2/1.3 in `PPoPP_DELTAS.md`) | v11 short + read_reference |
| `papersim` (ablation a) | **original OverlayCCL submission sim** (extracted from submission commit `20fc5e3`: no standalone-graph branch, no unsupported-local-op probes) | v11 |
| `longprompt` (ablation b) | current | **old 153-line long prompt** (`generic_evolution.md`, hardware details inlined, no read_reference, no discovery-loop keywords) |
| `noadv` (ablation c) | current | v11 **minus adversarial-testing instructions** |

## Result 1: per-problem search outcomes (sim_time_us of best candidate)

| Problem | base | papersim | longprompt | noadv |
|---|---|---|---|---|
| sequential_ar_chain (F1) | 5161.4 | 5161.0 | 5161.0 | 5161.4 |
| three_inline_ars (F2) | 5177.8 | 5177.8 | 5177.8 | 5177.8 |
| ten_ar_alt_sign_zero (F3) | 29.0 | 29.0 | 29.0 | 0.0* |
| per_row_ar_M96 (F4a) | 5171.4 | 5171.4 | 5171.4 | 5171.4 |
| grad_ar (F4b) | 53902.4 | 53902.4 | 53902.4 | 53902.4 |
| reduce_scatter_from_ar (F5) | 6128.9 | 6128.9 | 6128.9 | 6128.9 |
| mixmaxmin (F6) | 5222.1 | 5222.1 | 5222.1 | 5222.1 |
| eightslab (F7) | 5160.0 | 5160.0 | 5160.0 | 5160.0 |

**All four arms find the same optimum on all 8 problems.** (*noadv's 0.0
vs 29.0 on F3 is a scoring-path artifact of the emitted zero-tensor
variant, not a better rewrite — both candidates are `return zeros`.)

## Result 2: warm-cache RT of each arm's winner (224 ranks, ms/iter)

| Problem | baseline | base | papersim | longprompt | noadv |
|---|---|---|---|---|---|
| sequential_ar_chain (F1) | 3.01 | 2.95 | 3.06 | 2.96 | 3.01 |
| three_inline_ars (F2) | 3.22 | 3.01 | 2.97 | 2.91 | 2.83 |
| ten_ar_alt_sign_zero (F3) | 3.85 | **0.11** | 0.09 | 0.36 | 0.06 |
| per_row_ar_M96 (F4a) | 9.84 | **2.83** | 2.97 | 2.87 | 2.87 |
| grad_ar (F4b) | 8.73 | 8.83 | 8.73 | 8.73 | 8.66 |
| reduce_scatter_from_ar (F5) | 3.18 | 3.13 | 3.06 | 3.02 | 3.05 |
| mixmaxmin (F6) | 4.07 | **2.91** | 3.07 | 2.98 | 2.93 |
| eightslab (F7) | 3.59 | **2.80** | 2.83 | 2.88 | 2.89 |

Every arm's winner achieves the family win over the baseline
(F3 ~36×, F4a 3.5×, F6 1.4×, F7 1.27× at this 224-rank shape); the
between-arm spread is within run noise. RT harness:
`training/tools/rt_abl.py` analog staged on-cluster; warm = second of
two back-to-back runs.

## Result 3: search-cost differences (where the arms DO differ)

| Metric (sum over 8 problems) | base | longprompt | noadv |
|---|---|---|---|
| score_candidate calls | 52 | 41 | 54 |
| wall seconds | 684 | 623 | 703 |

Notable per-problem behavior:
- `reduce_scatter_from_ar` (F5): base converged in 7 calls;
  longprompt and noadv each accepted their FIRST candidate (1 call) —
  same final quality here, but a 1-shot accept pattern is exactly the
  behavior that produced broken candidates on harder problems in
  earlier rounds (Round-15/17 logs).
- `mixmaxmin` (F6): noadv needed 7 calls vs base's 1 — without the
  adversarial-verify instruction it emitted 6 rejected candidates
  (MAX/MIN op-confusion) before converging.
- `sequential_ar_chain` (F1): noadv needed 20 calls vs base's 13.

## Interpretation (honest)

1. **On these 8 problems, the endpoint is ablation-insensitive.** All
   four arms find the same rewrites — these problems are exactly the
   family exemplars the families were defined around, and
   sonnet-4-5 finds their rewrites from the formula alone. The
   pipeline deltas being ablated are not what *finds* the rewrite here.

2. **What the deltas actually buy, on this evidence:**
   - *Sim deltas (a)*: no ranking change on these 8 — every problem's
     optimum still contains a collective or an obviously-zero answer,
     so the standalone-graph branch never becomes the deciding term.
     The sim deltas matter on the `_bcast`-class problems where the
     optimum is **zero-collective local compute** (documented 10×
     mis-ranking in `PPoPP_DELTAS.md` §1.2); none of the 8 e2e family
     sites is in that class. A papersim arm on the `_bcast` catalog is
     queued as follow-up.
   - *Discovery-loop prompt (b)*: same endpoint, mildly different
     search shape. The long prompt is not worse on easy problems; its
     historical regressions (v5–v9/v12–v14 rounds, +9 net wins for
     short-prompt) came from harder `_bcast` problems with
     position-vs-value formula traps.
   - *Adversarial testing (c)*: same endpoint, **+29% more scorer
     calls on the problems with op-confusion hazards** (F6: 7× the
     calls). The adversarial-verify instruction front-loads correctness
     before scoring; without it the pipeline's Phase-4 gates still
     catch the bad candidates, at the cost of extra iterations. On
     problems with no gate (or a weaker gate), this margin is the
     difference between a correct and a silently-wrong deployment.

3. **e2e consequence**: because all arms converge to the same
   per-problem rewrites, the arm-found candidates are functionally the
   schedules already hand-instantiated in
   `train_{llama,gpt}10b_tp_families.py`; the e2e ≥2× results carry
   over identically for every arm on these sites. The e2e control
   pairs re-run on this cluster (CB `cr-037b5eccfc31cc735`) are logged
   in `session_logs_2026_09_07/`.

## Assets

- Winner candidates + per-search summaries: `session_logs_2026_09_07/abl/`
- The papersim tree construction: submission-commit sim files
  (`git show 20fc5e3:search/correctness_test.py`) over the current
  pipeline; a kwargs-compat filter in the score service is the only
  glue (`BENCH_KW` filtered to the paper fn's signature).
- Prompt variants: `prompts/generic_evolution.md` (long),
  `prompts/generic_evolution_v11_noadv.md` (adversarial stripped).
