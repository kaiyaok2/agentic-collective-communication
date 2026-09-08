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

| Metric (sum over 8 problems) | base | papersim | longprompt | noadv |
|---|---|---|---|---|
| score_candidate calls | 52 | 59 | 41 | 54 |
| wall seconds | 684 | 669 | 623 | 703 |

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

## Result 4: e2e replication on the ablation cluster

The 10B TP e2e pairs re-run on this cluster (fresh compile caches,
different physical nodes than CB10):

| Model (N_MB=16, fused) | baseline | sorcar | Speedup | CB10 reference |
|---|---|---|---|---|
| Llama-style 9.75B | 21656.1 | 8734.7 | **2.48×** | 2.47× |
| GPT-3-class 9.70B | 21170.7 | 9546.5 | **2.22×** | 2.21× |

Since all ablation arms converged to the same per-problem rewrites on
the 8 family sites (Result 1), each arm's winners instantiate the same
training schedule — the ≥2× e2e result holds for every arm.

## Result 5: `_bcast` extension — where the sim deltas DO change guidance

The 8 e2e family problems all have collective-bearing optima, so
ablation (a) was invisible there. Extending base-vs-papersim to 6
`_bcast` problems (optimum = zero-collective local compute):

**Sim scores of each arm's winner (us):**

| Problem | base (current sim) | papersim (paper sim) |
|---|---|---|
| mod_sq_bcast | 60.7 | 2.0 |
| xor_grid_bcast | 88.8 | 29.0 |
| triangle_num_bcast | 60.7 | 3.0 |
| sign_alt_bcast | 88.8 | 6.0 |
| gray_code_bcast | 60.7 | 29.0 |
| hamming_dist_bcast | 61.7 | 29.0 |

Both sims correctly steer away from the 5160-us AR baseline; both arms'
winners are correct zero-collective candidates. The difference is
**what the sim can distinguish within the local-compute class**: the
paper sim scores every local candidate 2–29 us (near-free, no
structure), while the current sim's standalone-graph model separates
const-fold (`torch.tensor([listcomp])`) from arithmetic
(`torch.arange` chains) candidates.

**224-rank RT of the two arms' winners on the two problems where their
code diverged:**

| Problem | base winner (RT ms) | papersim winner (RT ms) | forms |
|---|---|---|---|
| mod_sq_bcast (1D) | **0.077** | 0.112 | base: const-fold `torch.tensor([...])`; papersim: `arange` arithmetic |
| sign_alt_bcast (2D) | 0.371 | **0.169** | base: nested-list const-fold; papersim: `arange` broadcast |

This is precisely the const-fold-vs-arange trade-off the current sim's
auto-fit encodes (1D small → const-fold wins; 2D → arange wins, since
nested-list `torch.tensor` construction pays a per-element host cost).
The current sim correctly told the agent const-fold was cheaper on the
1D problem (its 60.7 beat its arith alternative) but its guidance on
the 2D problem kept the const-fold form that RT shows is 2.2× slower —
the fitted 2D const-fold points under-charge at N=32. Net: the
standalone-graph deltas make the sim *rankable* inside the
local-compute class (the paper sim is flat there), and the remaining
2D miscalibration is now a documented, bounded issue (worst observed:
0.2 ms absolute, on candidates that are all ≥10× faster than any
collective alternative).

## Result 6: prompt ablations on the `_bcast` class

Extending arms (b) and (c) to the same 6 `_bcast` problems:

| Problem | base | papersim | longprompt | noadv |
|---|---|---|---|---|
| mod_sq_bcast | 60.7 | 2.0 | **824.2** | 60.7 |
| xor_grid_bcast | 88.8 | 29.0 | 88.8 | 88.8 |
| triangle_num_bcast | 60.7 | 3.0 | 60.7 | 60.7 |
| sign_alt_bcast | 88.8 | 6.0 | 88.8 | 88.8 |
| gray_code_bcast | 60.7 | 29.0 | 60.7 | 60.7 |
| hamming_dist_bcast | 61.7 | 29.0 | 60.7 | 60.7 |
| calls / wall | 40 / 482s | 27 / 390s | 31 / 480s | 38 / 465s |

The long prompt produced its first concrete regression here:
on `mod_sq_bcast` it settled on the `torch.arange` arithmetic form
(sim 824.2) and never found the `torch.tensor([listcomp])` const-fold
(sim 60.7) that base/noadv found — the const-fold-vs-arange worked
idiom lives in the reference doc served by `read_reference()`, which
the old inline prompt does not have. RT confirms the direction
(const-fold 0.077 ms vs arange 0.112 ms on this 1D shape, Result 5),
though the RT margin (1.45×) is smaller than the sim margin. noadv
matched base on every `_bcast` problem: adversarial testing does not
change outcomes on position-formula problems whose hazard is
mis-reading, not communication correctness.

## Result 7: tier-2 problems (family flagships not used in e2e)

Extending all 4 arms to 6 more family-taxonomy problems:

| Problem | base | papersim | longprompt | noadv |
|---|---|---|---|---|
| per_row_ar_M1024 (F4) | 5177.1 | 5177.1 | 5177.1 | 5177.1 |
| per_column_ar_C64 (F4) | 5160.0 | 5160.0 | 5160.0 | 5160.0 |
| nine_ar_same_input (F2) | 5166.4 | 5166.4 | 5166.4 | 5166.4 |
| four_ar_sum_zero (F3) | **0.0** | **5177.8** | 29.0 | 0.0 |
| ag_slice_use (F3/F5) | 786.4 | 1.0 | 786.4 | 786.4 |
| five_ar_mixed_sign (F1) | 5165.6 | 5165.0 | 5165.6 | 5165.6 |
| calls / wall | 10/238s | 27/347s | 6/168s | 9/207s |

**The headline ablation-(a) result lives here.** On `four_ar_sum_zero`
(coefficients 2+3−1−4 = 0), the papersim arm emitted `0 * AR(x)` —
keeping the all-reduce — while every current-sim arm emitted
`zeros_like(x)` with no collective. Both are algebraically correct;
the difference is what the simulator rewards: without the
standalone-graph cost branch, the paper sim gives near-identical
scores to a candidate that keeps the AR and one that drops it, so the
agent has no gradient toward eliminating the collective. **224-rank RT:
base winner 0.081 ms vs papersim winner 2.909 ms — a 36× hardware
difference caused solely by the phase-1 simulator deltas.**

Stochasticity check (4 independent searches per arm on this problem):
the current-sim arm emits the collective-free `zeros_like` in **4/4**
runs; the papersim arm keeps the AR in **2/4** runs (and finds
`zeros_like` in the other 2 — the LLM sometimes reasons its way to
dropping the AR without simulator pressure). The sim deltas convert a
coin-flip into a certainty; at deployment time a 50% chance of
shipping the 36×-slower variant is the difference being measured.
(`ag_slice_use` shows the same flat-scoring pathology in the other
direction: papersim scores the local candidate 1.0 us — indistinguishable
from free — while the current sim's 786.4 us reflects the real memcpy
cost of the (65536,) payload.)

## Result 8: adversarial ablation on the MockTorch-hazard class (3 repeats)

The 3 registered problems from the deterministic-fail class of
2026-08-17 (`max_ij_bcast`, `or_ij_bcast`, `piecewise_bcast`), 3
independent searches per arm:

| Problem | base best (3 reps) | noadv best (3 reps) | base calls (Σ) | noadv calls (Σ) |
|---|---|---|---|---|
| max_ij_bcast | 31.0 / 31.0 / 31.0 | 31.0 / 31.0 / 31.0 | 12 | **22** |
| or_ij_bcast | 88.8 ×3 | 88.8 ×3 | 25 | 19 |
| piecewise_bcast | 60.7 ×3 | 60.7 ×3 | 13 | 13 |

Endpoints identical again; the adversarial instruction's measurable
effect is concentrated in call count on the op-confusion problems
(max_ij: noadv needs 1.8× the scorer calls, consistent with the F6
mixmaxmin finding). Combined verdict for ablation (c) across all 17
problems tested: **no endpoint regression observed with sonnet-4-5 at
30-step budget, but 1.3–7× more scorer iterations on problems with
reduction-op-confusion hazards** — adversarial testing front-loads the
correctness check the scorer would otherwise perform by rejection. The
historical wrecks it prevented (atol-slack acceptance, symmetric-input
false passes) involved weaker gates than this pipeline's current
correctness oracle; with a strong oracle its value is iteration
economy, with a weak one it is correctness itself.

## Result 9: stratified-random 24-problem study (revised arms, token-metered)

Responding to three review concerns: (i) the original 12 extras were
hand-picked; (ii) the "long prompt" arm still permitted iterative
discovery; (iii) adversarial-testing cost was not metered. This round:

- **24 problems drawn by fixed-seed RNG** (seed 20260907): 18 from the
  140 registered non-`_bcast` catalog problems + 6 from the `_bcast`
  catalog. Draw list in `session_logs_2026_09_07/draw24.txt`.
- **`stratform` arm replaces `longprompt`**: a prompt that FORCES the
  strat-enumerate protocol (enumerate exactly 5 structural strategies →
  implement each → score each once → ≤2 refinements on the winner →
  stop; no iterative discovery, no read_reference).
- **Per-call token metering** on every arm (`kiss_token_shim`).

### Structural divergence vs base (collective-count/form of the winner)

| Ablation | Diverges | On which problems |
|---|---|---|
| (a) papersim | **5 / 24** | eight_ar_half_ints (kept scaled-input form), bimodal/triangle_num/sign_alt (const-fold↔arange), and_ij |
| (b) stratform | **7 / 24** | per_row_ar_M64 + compound_ij (**reward hack, see below**), bimodal (kept an AR!), triangle_num/sign_alt/and_ij/diag_dist (arange where base const-folds) |
| (c) noadv | **1 / 24** | diag_dist (const-fold/arange coin-flip) |

All 20 collective-bearing problems (the 18 non-bcast + 2): every arm
finds the same collective structure as base on 19/20 (papersim differs
only on eight_ar_half_ints's local payload form) — consistent with the
family-problem convergence in Results 1–2, now on an unbiased draw.

### Ablation (b) headline: forced strat-enumeration reward-hacks a sim hole

On `per_row_ar_M64` and `compound_ij_bcast`, the stratform arm's
"Strategy 4: reduce-scatter + all-gather" candidate scored
**`sim_time_us: 0.0, num_ops: 0`** — a genuine simulator bug (the
collective counter misses the RS→AG composition; independently
re-scored to confirm). The strat protocol, scoring each strategy
exactly once with no verification loop, locked onto the hole and
shipped it. On hardware the candidate **crashes at 224 ranks**
(SIGABRT: RS with `scale=1.0` and a payload not divisible by
world_size), while base's winner (single AR) runs at 2.85 ms. The
discovery-loop arms never emitted this composition on these problems —
the adversarial-verify step kills it at the correctness stage before
scoring. Two conclusions:
1. the protocol shape matters: enumerate-and-pick amplifies scorer
   bugs that iterate-and-verify absorbs;
2. the sim RS+AG scoring hole is now a filed defect (workaround: the
   Phase-4a HW gate rejects these candidates in the full pipeline —
   the ablation ran Phase-3-only, which is exactly where the
   difference shows).

On the `_bcast` class, stratform also **failed to const-fold
anywhere** (its protocol has no reference-doc access and no iteration
budget to discover the idiom): bimodal_dist kept a full AR (5160 vs
base's 60.7 — the same class of miss as papersim's four_ar_sum_zero),
and 4 more problems settled on 10–15× worse-sim-scored arange forms.

### Ablation (b) zero-class follow-up: forced enumeration under-eliminates

stratform on the 7 zero/dead-class problems (2 reps): identical to base
on the 6 problems whose optimum keeps 1 AR, but on `four_ar_sum_zero`
it keeps the AR in **1/2 runs** (base: 0/7 across the session) — the
5-strategy protocol enumerates *collective layouts*, and "no collective
at all" only appears if the LLM volunteers it as a "strategy". Combined
with the RS+AG sim-hole hack above, the two failure modes of forced
enumeration are now both instantiated: it under-explores semantic
elimination AND over-trusts single-shot scores.

### Ablation (c) exact cost accounting (24 problems, token-metered)

| Metric | base (adv ON) | noadv (adv OFF) | delta |
|---|---|---|---|
| scorer calls | 120 | 144 | **+20%** |
| wall time | 1659 s | 1817 s | **+9.5%** |
| input tokens (incl. cache) | 1,259,442 | 1,476,591 | **+17%** |
| output tokens | 106,007 | 112,444 | +6% |
| structural divergences | — | 1/24 (a coin-flip idiom) | — |
| endpoint regressions | — | 0/24 | — |

Removing adversarial testing does not save cost — it **adds** it:
the verify-before-score step is cheaper than the extra
generate-score-reject iterations it prevents. Combined with Results
3/8 (F6 7×, max_ij 1.8× call inflation), the adversarial instruction
is net-negative to remove on every measured axis at this budget.

### Ablation (a) frequency study on the dead/zero class

The user-facing question "does the paper sim diverge on MANY problems"
gets its sharpest answer on the dead/zero-collective class — 8
registered problems, 3 independent searches per arm per problem (48
searches):

- **7 of the 8 problems genuinely require communication** (the dead
  part is a sub-expression): both arms converge to the identical 1-AR
  optimum in **3/3 runs each**. No divergence — the paper sim ranks
  collective-bearing candidates fine (its alpha model is unchanged).
- **On the one problem where total elimination is possible**
  (`four_ar_sum_zero`), the full-session aggregate is now **papersim
  keeps the AR in 2/7 runs vs base 0/7**.

Combined with the random draw (papersim diverges 5/24, of which 4 are
`_bcast` local-idiom differences and 1 a local-payload form) and the
143-problem context (~15 problems in the pool have zero-collective or
local-idiom-sensitive optima), the honest population statement is:
**the sim deltas change the found artifact on roughly 10–20% of
problems — precisely the zero-collective/local-compute subset — and
change nothing on the ~80–90% whose optimum keeps a collective. Within
the affected subset the consequences are large (36× RT on
four_ar_sum_zero; unrankable local-idiom guidance on `_bcast`).** The
deltas are a targeted fix for a class the paper's benchmark suite did
not contain, not a broad re-calibration.

### Ablation (c) deep-dive: 12 op-hazard problems × 2 reps (48 searches)

Targeting the class where adversarial testing should matter most —
MAX/MIN semantics, transposes, permutations, top-k (the op-confusion
hazards behind the F6/max_ij call inflation):

| Aggregate (24 searches/arm) | base (adv ON) | noadv (adv OFF) |
|---|---|---|
| scorer calls | 121 | 129 (+7%) |
| input tokens | 1,461,526 | 1,560,252 (+7%) |
| output tokens | 104,482 | 99,314 (−5%) |
| wall | 1695 s | 1626 s (−4%) |
| endpoint divergence (>5% sim) | — | **0 / 12 problems** |

Per-problem best scores are identical or within run-noise everywhere;
the search-cost gap concentrates on the same problems as before
(per_row_min_ar_M32: noadv 9 calls vs base 2; sparse_topk: 42 vs 34)
but partially reverses on others (ar_transposed: 7 vs 11), netting a
smaller aggregate cost delta than the random-draw round (+7% calls vs
+20%).

**Final verdict for ablation (c), across all 53 problems / 120+
searches this study ran**: removing adversarial testing produced **zero
endpoint regressions** with sonnet-4-5 behind this pipeline's strong
correctness oracle, and a **consistently positive but variable cost
overhead (+7% to +29% scorer calls, +7% to +17% input tokens)**
concentrated on reduction-op-confusion problems. Its historical value
(atol-hack and symmetric-input rejection in rounds with weaker gates)
is defense-in-depth: the instruction is effectively free when not
needed (the verify step replaces rejected-iteration cost it would
otherwise incur) and load-bearing when the oracle is weak.

### Sampling note

The Results 5–7 problem sets were hand-selected (family flagships +
classes predicted to stress each ablation); the divergence rates there
are not population estimates. This Result's rates (5/24, 7/24, 1/24 on
a fixed-seed random draw stratified only by bcast/non-bcast) are the
defensible population-level numbers.

## Assets

- Winner candidates + per-search summaries: `session_logs_2026_09_07/abl/`
- The papersim tree construction: submission-commit sim files
  (`git show 20fc5e3:search/correctness_test.py`) over the current
  pipeline; a kwargs-compat filter in the score service is the only
  glue (`BENCH_KW` filtered to the paper fn's signature).
- Prompt variants: `prompts/generic_evolution.md` (long),
  `prompts/generic_evolution_v11_noadv.md` (adversarial stripped).
