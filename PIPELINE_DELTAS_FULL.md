# Complete Pipeline Deltas over the OverlayCCL PPoPP Submission

This document is the exhaustive change log of everything in this
repository's pipeline that differs from the system described in the
OverlayCCL PPoPP submission (`paper.pdf`). It supersedes and extends
`PPoPP_DELTAS.md` (which froze at the 2026-08-15 Round-28 anchor); all
of that document's content is folded in here, updated through the
142-problem 7-node verification of 2026-08-24.

Organization: paper's 5 phases first, then simulator, then LLM-search
controllers, then benchmark/problem catalog, then RT methodology and
infrastructure. Each delta says **what changed, why, and where the code
lives**.

---

## 1. Phase 1 — Hardware auto-probe / cost-model config

### 1.1 Deterministic Phase-1 (LLM removed from probing)

Paper: Phase 1 is "LLM autonomously designs the probe campaign" via
`measure_*` tools (paper Table 2). In practice every `measure_*` tool
reads from a static `_HARDWARE_MEASUREMENTS` dict; the only real HW
subprocess (`_test_primitive_compilation`) runs after Phase 1. The LLM
loop was a 15–25 min narrator over fixed data and caused strat timeouts
on `_bcast` problems.

**Delta**: `phase1_profiling` always runs the deterministic auto-probe
path (`use_llm=False`). Same probes, same fitted constants, seconds
instead of minutes. Code: `experiments/run_search.py::phase1_profiling`,
`run_with_route_patch.py`.

### 1.2 Standalone-graph cost model (new `n_coll == 0` branch)

Paper Eq. 1 charges `T_local` with a fusion credit against an adjacent
collective — correct for the paper's 8 collective-heavy problems,
~10× under-charging for zero-collective candidates (all `_bcast` and
F3-family problems whose optimum is local recompute).

**Delta**: a standalone-graph branch:
- const-fold cost `max(cf_base, output_bytes / cf_bw)`
- arithmetic-chain cost `min(arith_sat, arith_marg1 + arith_marg_next·(n_arith−1))`
- mixed graphs: `max(const_fold, arith)`

All 5 parameters are auto-fit at Phase 1 from raw microbench points
(`_HARDWARE_MEASUREMENTS["standalone_graph_cost_us"]["raw_1d"/"raw_2d"]`),
mirroring the paper's alpha1/2/3 fit. **No hardcoded sim constants** —
same auto-probe discipline as the paper. Code:
`search/correctness_test.py::_local_cost_us`,
`search/agent_simulator_config.py::measure_standalone_graph_cost`.

### 1.3 Unsupported-local-op probe extension

Paper probes only collectives for compilability. Extended to `cumsum`,
`cumprod`, `sort`, `argsort` — all fail on Neuron SDK 2.26
(`NCC_ITCT901`); candidates using them now score `+inf` instead of
producing false sim wins. Code:
`search/agent_simulator_config.py::_test_primitive_compilation`.

### 1.4 Simulator dependency-graph fix (`_ast_detect_ar_dep_flags`)

Post-paper (2026-08-19, CB5): the paper's simulator scored *chained*
ARs (output of one feeding the next) identically to *independent* ARs,
missing the pipeline-fill amortization distinction its own alpha-model
supports. An AST pass now detects whether each collective's input
depends on a prior collective's output and sets per-collective
dependency flags consumed by the back-to-back amortization term.

Status note: this fix was validated on the CB5 cluster copy of
`search/correctness_test.py`; the AST helper must be re-applied when
rebuilding a cluster from this repo (it is included in the bootstrap
notes below). Reference: session notes 2026-08-19.

### 1.5 Amortization-constant drift (known, accepted)

Warm-cache RT probes on CB5 measured the marginal per-added-AR cost at
47–135 us versus the static table's 30 us (alpha1). Direction-consistent
(sim still ranks correctly), so the constant was not re-pinned;
documented so nobody mistakes it for a calibration bug. Reference:
Round-17 amort probe, 2026-08-13.

## 2. Phase 2 — Baseline template evaluation

No functional delta. Input catalog grew: `problems_novel_v4/5/6.py`,
`problems_comm_v7.py`, `problems_challenge_v8.py`,
`problems_round17..26.py`, `problems_realcomm_edge_v2..13.py`,
`problems_realcomm_diverse_v1..25.py` (40 problem-definition modules in
`search/` at last count) all register before Phase 2.

## 3. Phase 3 — LLM-driven candidate generation (largest delta area)

### 3.1 Sorcar controller replaces the paper's kiss as the primary Phase-3 shape

The freeform ReAct agent (`KISSAgent` from `github.com/ksenxx/kiss`,
renamed Sorcar in all writeups) is the primary controller. Versus the
paper's kiss it adds: a stateful ReAct loop with budget enforcement, a
`score_candidate(code) → sim_time_us` tool, and a `read_reference()`
tool serving a hosted domain doc. Entry point:
`experiments/ablation_kiss_vs_cc/kiss_phase3.py` (historical filename).

Installation deltas: kiss `pyproject.toml` patched `>=3.13` → `>=3.12`
for the cluster Python; `boto3`/`anthropic` added to its venv.

### 3.2 Prompt architecture: 37-line primary + reference doc via tool

- `prompts/generic_evolution_v11.md` — minimal primary prompt.
- `prompts/reference_trainium_details.md` — 197-line Trainium reference
  (XLA collectives, unsupported primitives, sim cost model, worked
  idioms: `torch.tensor([list-comp])` const-fold vs `torch.arange`
  arithmetic, cat+AR+narrow bucketing).
- `read_reference()` tool returns the doc on demand.

Short-prompt + tool split outperforms inlined long prompts by +9 net
wins (Rounds 19–26 → 28). Prompt versions v5–v9 and v12–v14 were tried
and regressed; v11 is canonical.

### 3.3 Signature-doc placeholder plumbing (bug fix that changed results)

`kiss_phase3.py` originally never populated `{signature_doc}` /
`{signature}` / `{evolved_fn_name}` / `{display_name}`. Sorcar was
solving problems *without the formula description*, which produced the
early "kiss losses" that round-1 analysis over-attributed to the search
shape. After the fix, prior strat RT wins flipped or tied
(2026-07-24 session). Any comparison against pre-fix numbers is invalid.

### 3.4 Bedrock routing shim

Paper's kiss used the direct Anthropic API. The cluster IAM role speaks
Bedrock: `kiss_bedrock_shim.py` monkey-patches
`AnthropicModel.Anthropic → AnthropicBedrock` (including the `stream`
method), strips `cache_control` recursively (Bedrock rejects it), and
maps model IDs to Bedrock inference profiles. All three parts are
load-bearing; missing any one makes the agent fail silently. Default
model: `claude-sonnet-4-5-20250929`.

### 3.5 Strat-enumerate (comparison arm) — unchanged shape, documented caveats

`search/strategy_enumerate_phase3.py` is the paper's strategy-enumerate
Phase 3: 1 enumeration call + K=5 implementations + R=3×2 refinement
calls, simulator-ranked, same correctness gate, same Phase 4/5.

Two fairness caveats discovered post-paper and now documented:

1. **Reference-doc asymmetry.** Sorcar can call `read_reference()`;
   strat's prompts receive `{optimization_hints}` (empty for the
   realcomm/bcast catalogs) and do **not** include
   `reference_trainium_details.md`. Mitigating factors: strat *does*
   receive the seed library of correct reference implementations inline
   (Sorcar's `{reference_implementations}` is blanked), and the doc
   content is correctness-oriented rather than strategy-revealing. The
   clean control — appending the reference doc to strat's
   `optimization_hints` — has not yet been run and is on the queue.
2. **MockTorch gate asymmetry in practice.** Strat's zero-collective
   candidates are rejected by the MockTorch correctness gate on 7
   deterministic-fail problems (root-caused 2026-08-17); Sorcar recovers
   on the same problems by iterating against `score_candidate` errors.
   The gate is the same code for both arms, but only the iterative
   controller can route around it, which inflates the win margin on F3
   (algebraic-zero) problems.

### 3.6 Round budget

Sorcar runs `--max-steps 40 --max-budget 5.0` per problem versus strat's
~12 LLM calls. This is an intentional design-point difference (iterative
vs enumerative), not an accident, but any per-call-normalized comparison
should note it.

## 4. Phase 4a/4b — Correctness gates

No functional delta from the paper. HLO compile-and-run at target world
size (4a) and 8-layer LM training-shape sanity (4b) both retained;
Sorcar candidates pass through the identical gates.

## 5. Phase 5 — Ranking and deployment

No functional delta. Lowest `sim_time_us` passing 4a/4b is emitted as
`runtime/trainium_<problem>_<n>node.py`.

## 6. Benchmark / problem catalog deltas

The paper evaluates 8 problems (a2av, ua2a, ring-KV, grad-AR, PP,
TP-MLP, FSDP, LBAR). Post-paper the catalog grew to a 142-problem
RT-verified pool spanning 7 optimization families — see
`SORCAR_FAMILY_TAXONOMY.md` for the taxonomy and per-family analysis,
`SORCAR_VS_STRAT_RESULTS.md` (2-node) and
`SORCAR_VS_STRAT_7NODE_RESULTS.md` (7-node) for raw numbers.

Key methodology rules added with the catalog:
- **No leak / no reward hack**: problem docs never name the winning
  rewrite; prior wins that depended on atol slack or trivial inputs were
  reclassified as reward hacks and removed (2026-07-20 scorecard).
- **Every claimed win is RT-verified warm-cache**; sim-only wins are
  not counted (12+ sim wins that RT-tied are excluded and documented).
- **Sim-inversion class documented**: on 11 always-local problems the
  sim ranks strat's Python-loop candidates ahead 3×, but RT inverts
  (Sorcar wins 9/11, avg 14.5×) because scalar loops do not vectorize
  on Neuron. Sim is trusted for collective-count ranking, not for
  local-op microstructure.

## 7. RT measurement methodology deltas

- **Warm-cache discipline**: every reported number is the second of two
  back-to-back runs of the same NEFF; cold-compile first runs are
  10–16× inflated and are discarded (`rt_warm_cache_pitfall`).
- **PERCALL_PROBE bypass** for training-shape primitives whose autograd
  wrapper collapses baseline and agent into one graph (tp_mlp, fsdp,
  lbar per-problem scripts) — 2026-08-02 methodology.
- **7-node harness**: `rt_2node.sh` generalized to `rt_7node.sh`
  (workers file argument, node_rank 1..6, env forwarding, 900 s default
  timeout) plus an `rt_7node_xlong.sh` 5400 s variant for M≥1536
  dispatch-collapse baselines whose *baseline* compile alone takes
  30–60 min at 224 ranks. `rt_run_v12.py` gained per-problem setup
  blocks for all 142 problems and an `N_OVERRIDE` env knob.
- **Compile-cache hygiene**: killed torchruns leave Neuron cores held
  and `MODULE_*/*.lock` files stale; the runbook (kill neuron-ls PIDs on
  every node, delete the stale MODULE dir entirely, relaunch) is
  documented in the 7-node results doc. Deleting only the `.lock` file
  mid-compile crashes the compiler (`FileNotFoundError` on the pb.lock).

## 8. Infrastructure deltas

- **Clusters**: paper ran a static 7-node on-demand cluster. Post-paper
  work runs on Capacity Blocks: CB2 `cr-0af8b7ceec0cb3154` (2-node,
  Rounds 28–29), CB4/CB5 (2-node, catalog build-out), CB7
  `cr-0f2c701080c291ea8` (7-node, 142-problem verification), CB8
  `cr-096d448add2938404` (7-node, e2e family-training validation).
- **Placement group** `Kaiyao` (cluster strategy) required in
  us-east-1c; non-PG launches hang at the 120 s CCOM RX timeout.
- **EFA peer=self**: security group needs an explicit self-referencing
  `-1/all` egress rule alongside `0.0.0.0/0` (root cause of the paper
  era's intermittent peer=self failures; fixed 2026-07-14; cross-node
  26.6 GB/s avg / 61.9 GB/s peak busbw after fix).
- **8 EFA NICs per instance** via `--network-interfaces file://nic.json`;
  EIP must sit on the DeviceIndex-0 ENI for Bedrock/internet egress, and
  the subnet must carry the IGW default route.
- **Worker file distribution**: every worker has an independent
  filesystem; `rt_run_v12.py` and candidate files are scp'd to all
  workers after every edit (a recurring source of stale-code bugs).

## 9. Naming

`kiss` (the paper's Phase-3 freeform agent) is renamed **Sorcar**
in all post-paper writeups; code paths retain historical `kiss_*`
filenames. `strategy-enumerate` is abbreviated **strat**.
