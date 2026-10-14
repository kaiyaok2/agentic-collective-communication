# C-Q2 — Non-LLM generator: rule-based rewriting + beam/random search from the developer baseline

Setup (ran on the master CPU, `rewriter_search.py`): start from each problem's developer-baseline program;
apply four *generic*, problem-agnostic source rewrites; search with beam (width 3, depth ≤4) and a random
walk under a 40-evaluation budget; objective = the pipeline's simulator (default calibration, 224 ranks);
gate = the pipeline's sandbox correctness oracle, with a real-tensor multi-rank oracle (threads + true
collectives) as fallback. Rules: **R1** hoist a collective out of a loop by stacking/concatenating its
operands (dependency-sliced); **R2** merge two independent same-op all-reduces into one on a stacked tensor
(moving independent operand prep); **R3** all_gather→permute→reduce_scatter(SUM,1/ws) ⇒ all_gather→own
column (routing identity); **R4** push row reductions (max / shifted exp-sum / one-hot gather) through an
all_gather(dim=1) into local reduce + all_reduce (log-sum-exp decomposition).

| Problem | Baseline → deployed strategy | Rules that reach it | Reached? | Selected by default-sim beam? | Sim evals |
|---|---|---|---|---|---|
| AllToAllV | AG+T+RS → AG+slice (pack-and-gather) | R3 | yes | yes (578.7 → 516.0) | 2 |
| Uniform A2A | AG+T+RS → AG+slice | R3 | yes | yes (1208.5 → 205.0) | 2 |
| Ring KV | per-part AG (16/slot) → per-slot AG | R1 | yes | yes (951 → 684) | 2 |
| Distributed CE | 1-AG + softmax → MAX-AR + stacked SUM-AR | R4 then R2 | yes (2-AR structure reached, correct) | **no** — default sim ranks 1-AG best (247 vs 510) | 3 |
| PP cross-stage | M masked ARs → one stacked masked AR | R1 | yes | yes (40256 → 2056) | 2 |
| TP MLP | M·L ARs → one AR | R1 ×2 (inner, outer) | yes | yes (52060 → 2056) | 3 |
| FSDP prefetch | M·L AGs → one AG (the sim optimum; HW gate rejects at 224 ranks, as in the paper) | R1 ×2 | yes | yes | 3 |
| Layer-block AR | two ARs → one stacked AR | R2 | yes | yes (260 → 200) | 2 |

**Reading.** With four generic rewrite rules, non-LLM search starting from the baseline *reaches* the
deployed strategy structure on 8/8 problems and *selects* it on 7/8, in ≤3 simulator evaluations each. The
only miss is selection, not generation: for distributed CE the simulator (default constants; the archived
calibrated run also scored the 1-AG strategy lowest, 5184 vs 5429 µs) ranks the 1-AG baseline ahead of the
AR recomposition, while hardware measures 14.9 ms vs 8.2–8.9 ms — the hardware gate, not the simulator,
would be the arbiter there. Honest consequences for the paper: (i) the strategies OverlayCCL found are
reachable by a small hand-written rule library, so "LLM necessity" should not be claimed; (ii) what the
LLM supplied in the paper is the *rule set itself* (it proposed stacking, the AG+RS identity, and the
log-sum-exp decomposition without being given them) plus harness-compatible code; (iii) the load-bearing
components are the calibrated simulator and the hardware/training gates, which is what the no-simulator
ablation already showed. Two sandbox findings surfaced along the way: the registered ring_kv
`per_chunk_per_part_allgather` template reassembles parts in part-major order and fails the problem's own
reference (both oracles) — we started from a corrected rank-major per-part baseline; and the sandbox
oracle's coverage, not correctness, bounds which rewrites can be accepted (hence the real-tensor fallback).
