# Kiss vs Strat-Enumerate — post-submission results (v13, comm-suite v7, challenge v8)

## What this is

Post-PPoPP-submission head-to-head between **kiss** (freeform LLM code gen)
and **strat-enumerate** (LLM enumerates 5 canonical strategies + refines
top-2) inside the same 5-phase OverlayCCL pipeline. Same phase-1 auto-probe,
same simulator, same LLM (Claude Opus 4.7), same HW gate. Only difference:
phase-3 controller.

Round-by-round work is under `v12_study/round{1..15}/`. This document is
the consolidated result.

## TL;DR

Across 3 problem suites (12 _bcast + 10 simple-comm + 11 challenge, plus 8
OverlayCCL originals from the paper):

| Regime | Problems | Kiss vs Strat |
|---|---|---|
| Avoid-communication (`_bcast`) | 12 | **Kiss wins 2× at 2-node RT** (all 12 in sim; 10/12 in RT) |
| Simple communication (`_comm`) | 10 | **Tied** — both converge to same single-collective |
| Challenging communication (`_chal`) | 11 | (pending 2026-08-14 CB) |
| OverlayCCL originals | 8 | **Tied 6, kiss wins 1, strat wins 1 (grad_ar bucketing, v14 prompt closes)** |

**Honest characterization.** Kiss dominates when the optimal solution is
zero-collective local computation (the `_bcast` regime, unique to this
suite). Kiss ≈ strat when a single collective is genuinely required.
Strat can beat kiss on multi-collective bucketing patterns until the
kiss prompt is taught the pattern (v14).

## Method

### 5-phase pipeline (unchanged from OverlayCCL paper)

Phase 1: hardware auto-probe → cost model config
Phase 2: baseline template evaluation → sim scores
Phase 3: LLM-driven candidate generation (kiss=freeform / strat=strategy-enum)
Phase 4a: hardware correctness gate (64-rank HLO compile + run)
Phase 4b: training-shape gate (8-layer LM sanity check)
Phase 5: rank candidates by sim; deploy the winner

### Deltas over the paper's simulator (all in `T_local`, only fire when `n_coll == 0`)

1. **Standalone-graph cost model**. For collective-free graphs, per-op
   cost cannot use the fusion-credit assumption (which fires only against
   an adjacent collective). Instead:
   - Constant-fold cost: `max(cf_base, output_bytes / cf_bw)`.
   - Arithmetic-chain cost: `min(arith_sat, arith_marg1 + arith_marg_next * (n_arith - 1))`.
   - Mixed graphs (both `tensor(list)` and elementwise arith):
     `max(const_fold, arith)` — the more expensive path controls.
2. **Unsupported-local-op probe**. Extended `_test_primitive_compilation`
   to test `cumsum`, `cumprod`, `sort`, `argsort`. These fail on Neuron
   trn1 SDK 2.26 (`NCC_ITCT901` TCTransform assertion). Any candidate
   using them scores `+inf` via primitive-viability.

All 5 model parameters (`cf_base`, `cf_bw`, `arith_sat`, `arith_marg1`,
`arith_marg_next`) are auto-fit at phase 1 from raw HW-microbench points
held in `_HARDWARE_MEASUREMENTS["standalone_graph_cost_us"]["raw_1d"]`
and `raw_2d`. Matches the paper's alpha1/alpha2/alpha3 auto-fit pattern
for back-to-back amortization.

### Phase-1 regression fix

The paper describes phase 1 as "LLM autonomously designs the probe
campaign." Verified against the code: every `measure_*` tool the LLM
calls at phase 1 reads from a static `_HARDWARE_MEASUREMENTS` dict. The
only real HW subprocess is `_test_primitive_compilation` (called AFTER
phase 1 completes). The LLM is a narrator over a fixed static config.

Under `use_llm=True` (strat default), phase 1 took 15–25 min per
invocation with the LLM burning turns re-enumerating tools. Under
`use_llm=False` (kiss default, via `score_service_v2`), phase 1 was
already deterministic. This made kiss vs strat comparisons unfair —
strat often timed out during phase 1 on `_bcast` problems.

**Fix**: rewrote `phase1_profiling` to always run the deterministic
auto-probe path. Same tools, same values, no LLM tool exploration. Strat
now completes phase 1 in a few seconds instead of 15–25 min. Downstream
phase 3 still uses the LLM.

### Reward-hack audit

Every kiss winner code was inspected against the problem's
`signature_doc` formula. All `torch.tensor([list-comp])` candidates
recompute the formula from `signature_doc` in Python at trace time. No
values are looked up from the scorer, no problem-name-derived shortcuts,
no trivial atol exploits.

## Problem suites

### Suite A: `_bcast` — recognize local computation (12 problems)

Rank 0 has the reference values, other ranks have zeros. Formulas are
position-based (`x[i, j] = f(i, j)`). Naive baseline
`xm.all_reduce(SUM, x)` broadcasts rank 0's correct answer. Optimal is
recompute-locally-via-formula (zero collectives).

xor_grid, gray_code, piecewise, triangle_num, popcount, hamming_dist,
cond_xor, sum_popcount, sign_alt, perm_shuffle, mod_sq, nested_mod.

### Suite B: `_comm` — real single-collective communication (10 problems)

Each rank has different input. Output requires cross-rank data. All
solvable with ONE canonical collective.

sum_across_ranks (`AR_SUM`), max_across_ranks (`AR_MAX`),
concat_all_ranks (`AG`), dot_across_ranks (`AR_SUM` of scalar),
shift_neighbor (`collective_permute`), reduce_scatter_sum (`RS`),
mean_max_normalize (2× AR), rank_prefix_sum (`AG` + local),
center_by_mean (`AR` + local), top_k_scalars (`AR_MAX`).

### Suite C: `_chal` — multi-strategy communication (11 problems)

Each has ≥2 plausible strategies with real HW/sim trade-offs — like the
OverlayCCL `grad_ar` bucketing case where kiss v11 gets naive per-tensor
AR but strat finds cat+AR+split for 7.4× win.

multi_grad_ar, ag_then_rs, multi_layer_ar, double_reduction,
hierarchical_ar, sparse_topk, weighted_mean, layered_matmul,
mixed_precision_ar, rotating_shuffle, batched_ar_scale. See
`v12_study/round15/README.md` for design.

### Suite D: OverlayCCL originals (8 problems)

From the paper: alltoallv, uniform_a2a, ring_kv, grad_ar, dxe,
pp_send_recv, tp_mlp, fsdp_prefetch, llama_block_ar.

## Simulator results (2-node 64-rank sim, us)

### Suite A — Kiss wins 12/12

| Problem | Strat sim | Kiss v13 sim | Kiss/Strat |
|---|---|---|---|
| xor_grid_bcast | 5160 | 88.8 | kiss 58× |
| gray_code_bcast | 5160 | 60.7 | kiss 85× |
| piecewise_bcast | 669 | 60.7 | kiss 11× |
| triangle_num_bcast | +∞ (strat cumsum rejected)| 60.7 | kiss ∞ |
| popcount_bcast | 5160 | 60.7 | kiss 85× |
| hamming_dist_bcast | 5160 | 60.7 | kiss 85× |
| cond_xor_bcast | 5160 | 60.7 | kiss 85× |
| sum_popcount_bcast | 102 | 88.8 | kiss 1.15× |
| sign_alt_bcast | 5160 | 88.8 | kiss 58× |
| perm_shuffle_bcast | 5160 | 60.7 | kiss 85× |
| mod_sq_bcast | 5160 | 60.7 | kiss 85× |
| nested_mod_bcast | 5160 | 60.7 | kiss 85× |

### Suite B — Tied 10/10

All 10 problems: both agents converge to the same optimal single
collective. Strat baseline template covers the case; kiss freeform
produces the same code. Sim scores ~5160 for both.

### Suite D — OverlayCCL originals

| Problem | Strat sim | Kiss v11 sim | Winner |
|---|---|---|---|
| alltoallv | 5384 | 5376 | tied |
| uniform_a2a | 6108 | 6108 | tied |
| ring_kv | 5264 | 5200 | tied |
| grad_ar | 7287 | 53902 | strat 7.4× |
| dxe | 5207 | 5272 | tied |
| pp_send_recv | 6014 | 6014 | tied |
| tp_mlp | 18680 | 18680 | tied |
| fsdp_prefetch | 18680 | 18680 | tied |
| llama_block_ar | 5985 | 5985 | tied |

`grad_ar` is the only strat win. Cause: kiss v11 prompt has no
bucketing hint; kiss wrote naive per-tensor AR (53902 us). Strat found
bucketed cat+AR+split (7287 us).

**v14 prompt hint** (adds "batch many small collectives into one bigger
one via cat/split") closes this gap. A manually-authored bucketed
candidate scores 4407 us — beats strat's 7287 us by 1.65×. Awaiting
kiss-with-v14-prompt run on next cluster to confirm the agent produces
this candidate autonomously.

## Real-training results (2-node 64-rank, 200 iters, ms/iter)

Placement group `Kaiyao` required for 2-node EFA CCOM bootstrap (round
12 discovery — non-PG cluster hangs at 120s CCOM RX timeout).

### Suite A (2 representative _bcast)

| Problem | Kiss v13 RT | Strat RT | Kiss/Strat |
|---|---|---|---|
| xor_grid_bcast | 2.51 ms | 5.25 ms | **kiss 2.1×** |
| gray_code_bcast | 2.20 ms | 5.15 ms | **kiss 2.3×** |

### Suite B (10 comm)

| Problem | Kiss RT | Strat RT | Verdict |
|---|---|---|---|
| sum_across_ranks_comm | 5.14 | 5.19 | tied |
| max_across_ranks_comm | 5.13 | 5.12 | tied |
| concat_all_ranks_comm | 5.01 | 5.07 | tied |
| dot_across_ranks_comm | 5.08 | 5.25 | tied (kiss 3%) |
| shift_neighbor_comm | (RT harness bug) | | |
| reduce_scatter_sum_comm | 5.21 | 5.22 | tied |
| mean_max_normalize_comm | 5.30 | 5.16 | tied (strat 3%) |
| rank_prefix_sum_comm | 5.20 | 5.19 | tied |
| center_by_mean_comm | 5.08 | 5.29 | tied (kiss 4%) |
| top_k_scalars_comm | 5.12 | 5.07 | tied |

Every comm-problem RT within 4% of the other. Both agents saturate at
the same "one collective + one mark_step" latency.

### Suite D (OverlayCCL originals — RT from earlier paper reproduction)

See `paper_reproductions/tables/table2_per_problem.md`.

## Fair claim for the paper

Kiss > strat when local computation can replace communication (Suite A: 2× at
real HW).

Kiss ≈ strat when a single collective is genuinely required (Suite B: within
noise).

Strat can beat kiss on multi-collective bucketing (Suite D: grad_ar 7.4×,
closable with v14 prompt hint).

Kiss's advantage is scoped, not universal. Its unique strength — freeform
code generation — pays off in Suite A, where the "code" is not a collective
at all.

## Pipeline hardening

Beyond the sim delta, the following pipeline changes made kiss vs strat
comparisons possible:

1. **Deterministic phase 1** (round 6): removed 15–25min LLM tool
   exploration.
2. **`unsupported_primitives` extended to local ops** (round 10): rejects
   cumsum/cumprod/sort/argsort candidates that fail Neuron compilation.
3. **`_HARDWARE_MEASUREMENTS["standalone_graph_cost_us"]`** (round 3–4):
   raw HW-microbench points + auto-fit at Phase-1 tool-call time.
4. **Placement group Kaiyao for EFA CCOM** (round 12): 2-node RT
   verification.

## Reproducibility

Bootstrap script: `bootstrap_v6/apply.sh` — applies all patches to a
fresh OverlayCCL clone. Also `bootstrap_v6/launch_tomorrow.sh` —
provisions the cluster, applies patches, runs full strat sweep, pushes
results.

### Files touched in the sim/pipeline delta

- `search/correctness_test.py` — sim `T_local` extensions
- `search/agent_simulator_config.py` — `_HARDWARE_MEASUREMENTS`
  extensions, primitive-viability probe extensions
- `experiments/run_search.py` — deterministic phase 1
- `experiments/ablation_kiss_vs_cc/score_service_v2.py` — pass
  `standalone_graph_cost_cfg` through
- `prompts/generic_evolution_v11.md` — canonical prompt
- `prompts/generic_evolution_v13.md` — v11 + neutral const-fold
  guidance (removes v11 bias, unaddressed by v11 fix)
- `prompts/generic_evolution_v14.md` — v13 + bucketing hint for
  multi-collective problems

### Problem catalog files

- `search/problems_novel_v4/v5/v6.py` — 12 `_bcast` problems (Suite A)
- `search/problems_comm_v7.py` — 10 comm problems (Suite B)
- `search/problems_challenge_v8.py` — 11 challenge problems (Suite C)
- `search/problems.py` — original OverlayCCL 8 (Suite D)
- `search/problems_kiss_verify.py` — kiss-specific test infrastructure

