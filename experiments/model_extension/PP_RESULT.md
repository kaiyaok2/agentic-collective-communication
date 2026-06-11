# 5th-problem candidate: Pipeline-parallel cross-stage transfer bundling

## TL;DR
**A 5th problem exists**: pipeline-parallel cross-stage send/recv via
masked all_reduce. Naive PP code (per-microbatch mark_step) issues M
separate AR calls. The agent can bundle them into 1 AR by stacking M
activations into one buffer. On a 7-node Trainium (224 ranks):

| M (microbatches) | per_mb (baseline) | bundled (agent) | speedup |
|---|---|---|---|
| 4 | 21.17 ms | 14.15 ms | **1.50x** |
| 8 | 34.78 ms | 25.97 ms | **1.34x** |

The speedup is robustly above the 1.3x threshold across M. Per-microbatch
overhead in the baseline is ~3.4 ms/microbatch (the dispatch + per-call
latency floor that bundling eliminates).

## Setup
- Llama-style 2-stage PP: each stage has 2 LayerNorm+SwiGLU blocks
- DM=2048, HID=5504 (Llama-7B ratio), B=1, S=512
- Cross-stage activation = 2 MB per rank per microbatch
- Cross-stage transfer implemented as masked all_reduce in a
  (half_ws, M, B, S, DM) buffer where each rank owns its pair_id slot
- ws=224 ranks (7 nodes x 32 cores), half=112 ranks per stage
- Timing taken on first stage-1 rank (rank 112) which calls `.item()`
  on the final stage's output — ensuring the timer blocks on the actual
  cross-stage AR + stage-1 forward execution, not just graph dispatch

## Why this is structurally different from rounds 1-2
- Round 1's FSDP test (8 AGs vs 2 bundled AGs) failed because the
  AGs lived in the SAME mark_step graph and XLA fused them.
- Round 2's gradient accumulation with per-mb mark_step ALSO failed
  because the gradient ARs were inside each microbatch graph and XLA
  fused them inside that graph.
- PP cross-stage transfer is fundamentally different: each microbatch
  *requires* its own mark_step (memory + correctness), so the M ARs
  live in M *different* mark_step graphs. XLA cannot fuse across
  mark_steps. This is the structurally unfused dispatch the agent can
  collapse.

The agent's bundled version moves all M microbatches' work into ONE
mark_step graph (since Trainium has no async, there's no pipelining
benefit lost). One graph with M stage-0 forwards + 1 large AR carrying
all M activations + M stage-1 forwards.

## Files
- `/tmp/tp_search/train_pp_llama.py` — the test
- `/tmp/tp_search/run_pp.sh` — launcher

## Next steps for paper integration
- Add backward pass (autograd through masked AR) — should preserve the
  speedup since backward also has per-mb dispatch in baseline.
- Formalize as a problem definition (input/output schema) for agent search.
- Build a simulator cost model.
- Run agent search to verify the agent discovers bundling independently.
