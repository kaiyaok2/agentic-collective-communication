# Deltas over the PPoPP paper

This document tracks every change made to the pipeline described in the
PPoPP paper (`paper.pdf`). Each delta is scoped to one of the 5 phases,
with rationale and code paths.

## Phase 1 — Hardware auto-probe / cost-model config

### 1.1 Deterministic phase-1 (no LLM tool exploration)

The paper describes Phase 1 as "LLM autonomously designs the probe
campaign" via `measure_*` tools (Table 2). Under the paper's default
`use_llm=True`, phase 1 took 15–25 min per invocation because the LLM
burned turns re-enumerating tools; downstream strat runs frequently
timed out here on `_bcast` problems.

Verified against the code: every `measure_*` tool the LLM calls at
phase 1 reads from a static `_HARDWARE_MEASUREMENTS` dict. The only
real HW subprocess is `_test_primitive_compilation` (called AFTER
phase 1 completes). So the LLM's phase-1 loop is a narrator over a
fixed static config — not real hardware discovery.

**Delta**: rewrote `phase1_profiling` (in `experiments/run_search.py`)
to always run the deterministic auto-probe path with `use_llm=False`.
Same tools, same values, no LLM tool exploration. Strat now completes
phase 1 in a few seconds instead of 15–25 min. Downstream phase 3
still uses the LLM.

Code: `experiments/run_search.py::phase1_profiling` (deterministic
path) and the `run_with_route_patch.py` cc-react wrapper that pins
`use_llm=False`.

### 1.2 Standalone-graph cost model auto-fit

The paper's Eq. 1 for `T_local` assumes a "fusion credit" against an
adjacent collective. This is correct for the 8 OverlayCCL problems (all
collective-heavy) but breaks on the 12 post-paper `_bcast` problems
where the optimal candidate has zero collectives. The paper's model
under-charges these graphs by ~10×, wrongly favoring naive-AR candidates.

**Delta**: added a standalone-graph cost path for `n_coll == 0`:
- Constant-fold cost: `max(cf_base, output_bytes / cf_bw)`
- Arithmetic-chain cost: `min(arith_sat, arith_marg1 + arith_marg_next * (n_arith - 1))`
- Mixed graphs (both const-fold and arith): `max(const_fold, arith)`

All 5 model parameters (`cf_base`, `cf_bw`, `arith_sat`, `arith_marg1`,
`arith_marg_next`) are auto-fit at phase 1 from raw HW-microbench points
held in `_HARDWARE_MEASUREMENTS["standalone_graph_cost_us"]["raw_1d"]`
and `["raw_2d"]`. Fit pattern matches the paper's alpha1/alpha2/alpha3
auto-fit for back-to-back amortization.

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

## Phase 2 — Baseline template evaluation

No functional deltas from the paper — the paper's `evaluate_template`
loop still runs. Only the input has grown: post-paper problem catalogs
(`problems_novel_v4/v5/v6.py`, `problems_comm_v7.py`,
`problems_challenge_v8.py`, `problems_round17.py` through
`problems_round26.py`) are registered before phase 2 so their
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

Head-to-head results (`SORCAR_VS_STRAT_RESULTS.md` Round 28): Sorcar
wins 54 of 92 problems (58.7%), strat wins 29 (31.5%), tied 9 — where
"strat" is the paper's `strategy-enumerate` Phase-3 shape run through
the same 5-phase pipeline.

Code: `experiments/ablation_kiss_vs_cc/kiss_phase3.py` (Sorcar driver
entry point — retains the historical filename), `search/_anthropic_route.py`
(Bedrock/direct-Anthropic route hot-swap).

### 3.2 Sorcar prompt: minimal primary + reference doc via tool

Per kiss developer feedback, Sorcar favors a very short primary
prompt (trigger keywords `AI discovery`, `adversarial testing`, hard
rules on the reward) with domain knowledge in a separate document
served through a tool call.

- `prompts/generic_evolution_v11.md` — 37-line Sorcar prompt.
- `prompts/reference_trainium_details.md` — 197-line Trainium
  reference (XLA collectives, unsupported primitives, sim cost model,
  worked idioms including `torch.tensor([list-comp])` const-fold vs
  `torch.arange` arithmetic trade-offs, cat+AR+narrow bucketing
  pattern for many-small-collectives cases).
- `read_reference()` — new tool in `kiss_phase3.py` that returns the
  reference doc content on demand.

Prior long-prompt variants (v11 through v14) inlined all of this in
the prompt. The Sorcar short-prompt + `read_reference` split
outperforms the long-prompt equivalent by +9 net Sorcar wins and
+9.8pp win rate (Round 19-26 = 45 wins, Round 28 = 54 wins).

### 3.3 Signature-doc placeholder plumbing

Discovered post-paper that `kiss_phase3.py` never populated
`{signature_doc}`, `{signature}`, `{evolved_fn_name}`, or
`{display_name}` in the prompt — v11+ prompts rely on these to
describe the target tensor. `get_baseline_code` was returning only
`{name, code}` from the problem definition.

**Delta**: `get_baseline_code` now returns the full problem metadata
(`signature`, `signature_doc`, `display_name`, `evolved_fn_name`) and
the prompt population step fills all four placeholders before the
Sorcar agent starts.

Code: `experiments/ablation_kiss_vs_cc/kiss_phase3.py::get_baseline_code`
and the prompt-population block in `main`.

### 3.4 Model / Bedrock integration

The paper's kiss used the Anthropic direct API. Post-paper, we route
through Bedrock (`AnthropicBedrock`) so the cluster IAM role can
satisfy provider auth without a static `ANTHROPIC_API_KEY`.

- `kiss_bedrock_shim.py` monkey-patches
  `kiss.core.models.anthropic_model.AnthropicModel.Anthropic` →
  `AnthropicBedrock`, recursively strips `cache_control` (Bedrock
  rejects the field), and maps `claude-*` model IDs to their
  Bedrock inference-profile IDs.
- Default model: `claude-sonnet-4-5-20250929` (via
  `us.anthropic.claude-sonnet-4-5-20250929-v1:0`). Opus 4.1 and 4.7
  are IAM-blocked on the current cluster.

### 3.5 Strat-enumerate compatibility fixes

Some strat-enumerate strategies (post-paper) propose local recompute
in problems where the paper's kiss template set only produces
collective candidates. These are LLM-stochastic — the same problem
can produce either a 5160us AR-based candidate or a 29-us const-fold
candidate depending on the run. This is documented in
`SORCAR_VS_STRAT_RESULTS.md` (§ "Sorcar wins breakdown"): 27 of 29
strat wins are of this form and evaluate to identical NEFFs at RT.

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
  `cr-0af8b7ceec0cb3154`. Placement group `Kaiyao` (cluster-strategy)
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

Every RT number reported in `SORCAR_VS_STRAT_RESULTS.md` is the
**second** measurement of a candidate. The first run pays cold Neuron
compile-cache cost (~10-16× the steady-state ms/iter), which
masqueraded as sim-vs-RT divergence in earlier rounds until we
identified the artifact (see `memory/rt_warm_cache_pitfall.md`).

### Anchor tag

- `anchor-round28-Sorcar-2026-08-15` on `main` — reproducibility
  checkpoint for the results in `SORCAR_VS_STRAT_RESULTS.md`.
