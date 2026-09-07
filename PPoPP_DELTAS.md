# Deltas over the PPoPP paper

This document tracks the substantive changes to the pipeline described
in the PPoPP paper (`paper.pdf`) — design changes that alter what the
system does or how it should be evaluated. Mechanical bug fixes and
plumbing (prompt-placeholder population, Bedrock auth shims,
compatibility patches) are excluded; they live in commit history.

## Phase 1 — Hardware auto-probe / cost-model config

### 1.1 Deterministic Phase-1: cheap AND still performant

The paper describes Phase 1 as "LLM autonomously designs the probe
campaign" via `measure_*` tools (Table 2). Under the paper's default
`use_llm=True`, Phase 1 took 15–25 min per invocation because the LLM
burned turns re-enumerating tools; downstream strat runs frequently
timed out here on `_bcast` problems.

**Why removing the LLM is cheap**: verified against the code, every
`measure_*` tool the LLM calls at Phase 1 reads from a static
`_HARDWARE_MEASUREMENTS` dict populated by the probe harness; the only
real HW subprocess (`_test_primitive_compilation`) runs after Phase 1
completes. The LLM's Phase-1 loop was a narrator over fixed data. The
deterministic path runs the same probes and the same alpha1/2/3 +
standalone-graph fits in seconds.

**Why it stays performant (no drift, no overfit)**: the concern with
removing the adaptive LLM is that a fixed probe schedule might overfit
one input distribution or drift as the deployment changes. Neither
happens, for a structural reason: **every probe measures a property of
the machine + library, never of the training task.**

- The probe inputs are synthetic tensors swept over *sizes* (byte
  ladders for collective latency, element ladders for standalone-graph
  cost, run-length ladders for back-to-back amortization) — not
  training data. Collective latency on trn1 is a function of payload
  bytes, rank count, and dispatch pipelining; it does not depend on
  the values inside the tensors or on which model produced them. The
  fitted model is therefore input- and task-agnostic by construction:
  the same fitted constants scored MoE a2av problems, dense grad-AR
  problems, and 130+ synthetic `_bcast`/family problems with no
  per-task refitting.
- Drift across deployments is handled where it actually occurs — at
  probe time, not at fit-design time. The auto-probe reruns on the
  target device/SDK whenever either changes, so a new compiler version
  or instance type re-derives all constants from fresh measurements
  (the paper's original motivation, preserved). What the LLM added was
  variance in *which* probes ran per invocation — and since the probe
  outcome space is machine-determined, that variance only added noise
  and wall-time, not information.
- Empirically, sim rankings under the deterministic Phase 1 were
  validated against warm-cache hardware RT on 143 problems spanning 7
  optimization families plus the 8 paper originals (see
  `SORCAR_FAMILY_TAXONOMY.md`): 0 cases where a sim-picked winner lost
  on hardware, across problem classes the probes never saw.

Code: `experiments/run_search.py::phase1_profiling` (deterministic
path); the same auto-fit machinery as the paper
(`measure_back_to_back_amortization` → alpha1/2/3, and the
deterministic-fallback fit that populates the alphas even when probe
reasoning is skipped).

### 1.2 Standalone-graph cost model auto-fit

The paper's Eq. 1 for `T_local` assumes a "fusion credit" against an
adjacent collective. This is correct for the 8 OverlayCCL problems (all
collective-heavy) but breaks on the post-paper `_bcast`/family problems
where the optimal candidate has zero collectives. The paper's model
under-charges these graphs by ~10×, wrongly favoring naive-AR candidates.

**Delta**: added a standalone-graph cost path for `n_coll == 0`:
- Constant-fold cost: `max(cf_base, output_bytes / cf_bw)`
- Arithmetic-chain cost: `min(arith_sat, arith_marg1 + arith_marg_next * (n_arith - 1))`
- Mixed graphs (both const-fold and arith): `max(const_fold, arith)`

All 5 model parameters (`cf_base`, `cf_bw`, `arith_sat`, `arith_marg1`,
`arith_marg_next`) are auto-fit at Phase 1 from raw HW-microbench points
held in `_HARDWARE_MEASUREMENTS["standalone_graph_cost_us"]["raw_1d"]`
and `["raw_2d"]` — same size-ladder, task-agnostic probe pattern as 1.1.

Code: `search/correctness_test.py::_local_cost_us` (standalone-graph
branch), `search/agent_simulator_config.py::measure_standalone_graph_cost`.

### 1.3 Unsupported-local-op probe extension

The paper's `_test_primitive_compilation` probes only collectives
(`all_gather`, `reduce_scatter`, `all_reduce`, `collective_permute`,
`all_to_all`). We extended it to test `cumsum`, `cumprod`, `sort`,
`argsort`. On Neuron trn1 SDK 2.26 these fail with `NCC_ITCT901`
TCTransform assertion, so any candidate using them scores `+inf` via
the primitive-viability term.

Code: `search/agent_simulator_config.py::_test_primitive_compilation`
+ the `for prim in [...]` loop in `experiments/run_search.py`.

### 1.4 Structural graph analysis at scoring time (post-paper additions)

Two AST-level analyses were added to the simulator's scoring path
beyond the paper's op-count features:

- **Graph-inducing collective count** (loop-structure analysis around
  `_count_collectives`): distinguishes M collectives issued inside a
  Python loop (M separate dispatch/launch events at runtime) from M
  collectives fused into one stacked payload. This is the feature that
  lets the sim price the dispatch-collapse family (F4) correctly —
  a flat op count scores both shapes identically.
- **Bucket-cap detection** (`_ast_detect_bucket_cap`): detects a
  hardcoded byte cap (e.g. `bucket_bytes = 32 * 1024 * 1024`) in a
  candidate and models its peak-intermediate clamp. At training scale
  this is the structural property separating a paper-quality bucketed
  reduce from a naive cat-all-then-reduce — the latter wins in a
  microbenchmark sim but OOMs device HBM in real training
  (independently re-confirmed at 10B: `SORCAR_E2E_10B_TP.md`,
  finding 5).

Code: `search/correctness_test.py::_ast_detect_bucket_cap`,
`::_count_collectives` and the surrounding loop-structure analysis.

### 1.5 Known accepted drift: amortization constants

Warm-cache RT probes measured the marginal per-added-AR cost at 47–135
us versus the fitted table's 30 us (alpha1). Direction-consistent (sim
still ranks correctly), so the constant was not re-pinned; documented
so nobody mistakes it for a calibration bug.

## Phase 2 — Baseline template evaluation

No functional deltas from the paper — the paper's `evaluate_template`
loop still runs. Only the input has grown: post-paper problem catalogs
(`problems_novel_v4/5/6.py`, `problems_comm_v7.py`,
`problems_challenge_v8.py`, `problems_round17.py` through
`problems_round26.py`) are registered before Phase 2 so their
baselines feed into the sim.

## Phase 3 — LLM-driven candidate generation

This is where the biggest change lives.

### 3.1 Phase-3 controller: Sorcar replaces the paper's kiss / multi-island / cc-react as the primary evaluator

The paper compares three Phase-3 shapes: multi-island GA, cc-react
(collective-communication ReAct), and Sorcar (freeform LLM code gen).
Post-paper, the **Sorcar ReAct agent** (from
`github.com/ksenxx/kiss`) replaces the paper's kiss. Sorcar is
strictly more capable than the paper's kiss: it exposes a
`score_candidate` tool plus a stateful ReAct loop with token/budget
enforcement, and integrates cleanly with a hosted reference doc via a
`read_reference` tool.

Head-to-head results (Round 28; per-problem numbers now consolidated
in `SORCAR_FAMILY_TAXONOMY.md`): Sorcar
wins 54 of 92 problems (58.7%), strat wins 29 (31.5%), tied 9 — where
"strat" is the paper's `strategy-enumerate` Phase-3 shape run through
the same 5-phase pipeline.

Code: `experiments/ablation_kiss_vs_cc/kiss_phase3.py` (Sorcar driver
entry point — retains the historical filename).

### 3.2 How Sorcar's AI-discovery loop + adversarial testing drive the CCL composition search

Sorcar's system framework (kiss `SYSTEM.md`) mandates two disciplines
that map directly onto this problem, and the Phase-3 prompt invokes
both by keyword (`AI discovery`, `adversarial testing`).

**The AI-discovery loop.** SYSTEM.md prescribes a fixed
explore-implement-evaluate cycle for any optimization task:

1. read + profile the data / tests / baseline; record baseline metrics
2. search for SOTA approaches / prior art
3. write candidate ideas and rationale to a scratch file
4. pairwise-judge the ideas to pick a winner
5. implement → run a real end-to-end evaluation → log the idea, the
   aspect it improves, and its metrics; if better, keep it and try
   composing it with prior winners on *different* aspects; if worse,
   mark it failed so it is never retried
6. search again for fresh ideas, excluding everything already
   explored; repeat
7. stop when the metric goal is met, **with a held-out /
   generalization check to prove the result is not overfit**

Instantiated for CCL composition, the loop's slots fill in as: the
"baseline metric" is the baseline template's `sim_time_us` from
`score_candidate`; "SOTA search" is reading the reference doc
(`read_reference`) plus the problem's own formula/signature; "ideas"
are candidate collective compositions (fuse the K ARs via linearity,
convert AR+slice to reduce_scatter, hoist the redundant calls, …);
"implement → evaluate" is emitting the candidate function and scoring
it in the real 5-phase pipeline (simulator ranking → Phase-4a hardware
compile-and-run gate → Phase-4b training-shape gate); "composing with
prior winners" is exactly how compound rewrites emerge (e.g.
perrow_mixed_bigM = F4 dispatch collapse composed with F6
mixed-reduce extraction); and the "held-out generalization check" is
the warm-cache RT measurement on hardware the simulator never saw —
which is why sim-only wins were never counted (12+ sim wins that
RT-tied are excluded from every results table).

The explored-ideas ledger (step 5's "never retry a failure") matters
specifically because the CCL search space is full of attractive
dead-ends the compiler already handles (auto-fusable AR pairs) or the
gate rejects (zero-collective candidates on problems whose
correctness oracle requires communication); logging these per problem
keeps the ReAct trajectory from cycling.

**Adversarial testing.** SYSTEM.md defines this as a two-role
discipline: one subtask tries to *break* the system with adversarial
tests/workloads, another fixes what breaks. In the CCL pipeline this
shows up at two levels:

- *Against each candidate*: before a rewrite is accepted, it is
  attacked with inputs designed to falsify the algebraic identity it
  claims — rank-asymmetric values (each rank's tensor scaled by
  rank+1, so any dropped or double-counted communication changes the
  answer), non-uniform shapes, and reduction-op corner cases
  (MAX/MIN idempotence vs SUM linearity). A candidate that computes
  the right answer only on symmetric inputs — the classic failure of
  "clever" collective elimination — dies here rather than in
  deployment.
- *Against the evaluation itself*: the same adversarial stance is
  turned on the harness — no-leak problem statements (the winning
  rewrite is never named in the prompt), atol-slack and
  trivial-input reward hacks reclassified as failures, and the
  cold-compile-cache artifact (10–16× inflated first-run timings)
  eliminated by warm-cache discipline. Several early "wins" did not
  survive this and were retracted (documented in the round logs).

The division of labor is what makes the combination work: the
discovery loop generates aggressive semantic rewrites, and adversarial
testing is the reason aggressive rewrites can be trusted — each
accepted candidate has survived an explicit attempt to break it on
real hardware.

Prompts: `prompts/generic_evolution_v11.md` (37-line primary prompt
carrying the `AI discovery` / `adversarial testing` keywords and the
hard rules on the reward), `prompts/reference_trainium_details.md`
(197-line domain reference served via the `read_reference()` tool).
Short-prompt + tool split outperforms inlining the reference into the
prompt by +9 net Sorcar wins (Rounds 19–26 → 28); v5–v9 and v12–v14
prompt variants regressed and v11 is canonical.

## Phase 4a — Hardware correctness gate

No functional delta. The paper's 64-rank HLO compile-and-run remains
authoritative for correctness. Sorcar candidates flow through the same
gate.

## Phase 4b — Training-shape gate

No functional delta. The paper's 8-layer LM sanity check remains the
gate. Every candidate that passes 4a is scored under 4b.

## Phase 5 — Rank candidates and deploy

No functional delta. The candidate with the lowest final `sim_time_us`
that passes 4a/4b is emitted as `runtime/trainium_<problem>_2node.py`.
The generated runtime file preserves the paper's structure (init
function + evolved kernel).

## Non-phase deltas (infrastructure)

### Cluster topology + EFA setup

- 2-node `trn1.32xlarge` in `us-east-1c` under Capacity Block
  `cr-0af8b7ceec0cb3154` (later 7-node CBs for scale verification).
  Placement group `Kaiyao` (cluster-strategy)
  required for cross-node CCOM bootstrap — non-PG launches hang at
  the 120s CCOM RX timeout (root-cause from prior sessions).
- Security-group egress requires an explicit self-referencing `-1/all`
  rule alongside `0.0.0.0/0` to unblock intra-cluster EFA peer=self
  RX (documented in `memory/efa_peer_self_root_cause.md`).
- Bedrock/internet access requires the EIP to sit on the
  `DeviceIndex=0` primary NIC, and the us-east-1c subnet must be
  associated with the VPC route table that carries the IGW default
  route. Both are one-time setup gotchas we hit when switching from
  the on-demand `us-east-1d` cluster to the CB `us-east-1c` cluster.

### RT warm-cache methodology

Every RT number reported in this project's results docs is the
**second** measurement of a candidate. The first run pays cold Neuron
compile-cache cost (~10-16× the steady-state ms/iter), which
masqueraded as sim-vs-RT divergence in earlier rounds until we
identified the artifact (see `memory/rt_warm_cache_pitfall.md`).

### Anchor tag

- `anchor-round28-Sorcar-2026-08-15` on `main` — reproducibility
  checkpoint for the Round-28 results (2-node era; superseded by the
  7-node numbers in `SORCAR_FAMILY_TAXONOMY.md`).
