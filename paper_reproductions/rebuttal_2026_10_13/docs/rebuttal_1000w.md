# PPoPP'27 #1429 — author response (≈1000 words; numbers from the 2026-10-13/15 7-node re-measurements)

We thank the reviewers. We answer the explicit questions first, then the comments. All new numbers come from two fresh
7-node trn1.32xlarge clusters (224 ranks, SDK 2.22/2.29 vs 2.24 in the submission); every artifact, script and log is in
the revision's repository.

**Correction we owe (A-Q3, B).** Re-running Phase 1 we found that the released pipeline — and the commit behind the paper's
runs — answered every `measure_*` tool from a *snapshot* of an earlier probe campaign; the submitted runs were not
live-calibrated, so Reviewer A's "one-time hand-fit" reading was correct for them. Re-running the probes (four runs, two
clusters): volume terms are stable (±3–13%), latency floors are not (up to 4.2×), and the snapshot was off by 2–15×.

**A-Q2 / C-Q1 — simulator accuracy.** With live constants alone the two payload-changing problems still mis-ranked (dxe
ρ −0.72, AllToAllV −0.32) because the sandbox traced toy shapes and charged local ops a flat ~29 µs. We corrected the
model structurally (real harness shapes; bytes actually moved per collective with measured latency curves; local ops
charged by volume at the measured HBM rate, in forward and backward; fusion-aware dispatch runs; XLA dead-collective
elimination; scope-aware graph boundaries), with no constant fitted to the hardware pool. On a 74-candidate pool
(seed templates, grammar variants, archived LLM strategies, deployed runtimes, baselines) measured 3× at 224-rank
training scope, within-problem Spearman between simulator score and measured step time is AllToAllV 0.90, dxe 0.73,
FSDP 0.81, Layer-block 0.90, PP 0.89, Ring KV 0.75, TP MLP 0.90, Uniform A2A 0.73 (pooled 0.69; direction agreement
31/33; top-1 correct on 6/8, within 0.3%/3.7% on the other two). The same simulator, unchanged, gives ρ 0.56–0.90 on an
independent second cluster (pooled 0.75). Prediction errors: ratio-MAPE 8–27% (120% on Uniform A2A: ranking right,
gap over-predicted); rescaled step-time error 0.1–2% on bundling problems, 13–15% on payload-changing ones — coarse enough
that we keep the hardware gate. The cross-scope inversion is *reproduced* by the model: AllToAllV at
per-primitive payload ranks AG+T+RS first (hardware 30.0 vs 24.2/38.0 ms); at OLMoE payload pack-and-gather first
(hardware 3127 vs 4338 ms). A leave-one-out ablation shows the structure is load-bearing: removing the measured collective
curves inverts five problems (pooled 0.69→0.27); removing bytes-moved, volume-charging or fusion-aware runs drops dxe to
≤0.27; removing real shapes drops FSDP to 0.52.

**A-Q3 — why per-deployment calibration.** Scaling the network curves by 4× or 0.25× leaves the five bundling problems'
rankings unchanged (Spearman 1.00) — the invariance behind our ρ=1.00 TPU swap — but flips AllToAllV (agreement −0.21) and
moves dxe; reverting only the local terms to the snapshot moves dxe and Ring KV. We also ran the full pipeline (Phase
1→3→4→5) three ways: snapshot table, live table, live table + corrected structure. On dxe the first two deploy the
full-vocab all-gather (13.0–13.6 ms); the third ranks it last and deploys the two-all-reduce recomposition (8.6 ms). On
Ring KV all three deploy the same per-slot all-gather (5.1 ms). Calibration is necessary where payload changes and
irrelevant where only dispatch count changes; constants alone are not sufficient — the structural terms they feed must be
in the model. We will rewrite §5 accordingly and claim nothing for TPU (D).

**A-Q1 — realistic configuration.** We measured the step with AllToAllV replaced by a local stub: at the paper shape the
collective is 92% of the step. Raising per-token compute lowers it — expert dim 8192: 86%; top-k 4: 89/84%; top-k 2:
85/82%; top-k 2 + expert dim 8192: 83% (baseline) / 79% (agent), where OverlayCCL still gives 2161→1727 ms (1.25×, 3 runs,
2 seeds; top-k 4 + 8192: 1.43×). Token count cannot change the share: payload and expert compute are both linear in
tokens, and B>1 / longer sequences fail at load or in the runtime (`NRT_EXEC_OOB`, compiler error `NCC_ITEN404`) for both
backends; model width 4096 OOMs or hits a stuck-collective hardware error. We will state the regime explicitly and report
the collective-component speedup (4016→2795 ms, 1.44×) as the transferable number.

**A-Q4 / C-Q3 — baselines.** None is shape-tuned; all are the generic recipes with fixed loop structure (listed
call-by-call in the revision). Concession: Megatron/NxD's vocab-parallel cross-entropy uses three small all-reduces, not an
all-gather; at 224 ranks it runs 8.25–8.31 ms vs 8.3–8.6 for our 2-AR strategy, so the dxe gain against the library
baseline is ≈1.0× and Table 5 will say so. The explicit bundling/pack-and-gather templates Reviewer C asks for are the
grammar variants in the pool above, measured under identical settings.

**A-Q5 — strategies.** Delivered: 4 mechanisms (AG+RS→AG+slice; stack/cat bundling; the dxe recomposition;
graph-boundary lifting) plus the masked-AR substitute. Attempted (archived Phase-3 logs, 135 LLM implementations, ~65
sketches): 10 classes; six never deployed — hierarchical two-level, permutation schedules and P2P substitution all died at
the correctness oracle; other recompositions mostly at the oracle, the rest on the simulator or the 224-rank gate;
bucketing and async orderings passed the oracle but lost in the simulator; index_select extraction failed the index gate.

**C-Q2 / B — is the LLM necessary.** A generator with no collective knowledge (generic AST edits; random, GA, beam,
compiler passes; same oracle; budgets 60–400) finds nothing strategy-level against the corrected simulator on seven problems; on Ring KV it finds a parameter
change of the baseline (fewer parts per chunk: 5.18 ms vs 8.8 baseline, 4.96 deployed), so there a non-LLM search
recovers most of the gain — we will say so. Against the *shipped*
simulator it "found" 60× wins that were a graph-boundary detector exploit (hardware 1.00×) — which we fixed. The archived
non-LLM Phase-3 arms reproduce OLMoE (pack-and-gather is a seed template) but deploy the baseline on all Llama problems: the
bundling behind 3.24× is LLM-authored. A better microbenchmark does not remove the inversion (our 224-rank per-primitive
harness still inverts); regression suffices for constants — the LLM's value is Phase 3.

**C-Q4 — accounting.** "$19/h" is LLM token cost (cluster ≈$150/h); Table 5 and Table 6 came from different harnesses —
we now give both from one run (stub decomposition); "8 cluster-hours" includes probes, gates and validation, "56 min" is
Phase-3 search only; measured vs illustrative figures will be labeled.

**C-Q5 / C7 / D — wall-clock and repeats.** Llama-7B 6000 steps from a cold cache: per_mb 593/596 s vs bundled 525/513 s,
1.145× net including compile (break-even ≈4,100 steps); steady 71.9→21–22 ms reproduces on both clusters (73.2→22.5 on
the second). OLMoE: n=7/6 archived repeats plus 3 new seeds (4349 vs 3127 ms, 1.39×). 3-node Llama with the model fixed:
65.7 vs 21.0 ms; the bundled time is flat across cluster sizes, so the ratio compresses because the baseline speeds up.
We will remove the H100 remark and the XLA phrasing, expand jargon, and fix §6.

**Artifact.** We will fix the defects this exercise exposed (stray line in the AllToAllV runtime; snapshot probes;
Uniform-A2A runtime path; codegen helper not routing to the deployed algorithm; hand-written OLMoE helper) and release
probes, simulator, pools and logs.
