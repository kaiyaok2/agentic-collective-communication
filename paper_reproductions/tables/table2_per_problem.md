# Table 2 (tab:perproblem) — per-problem 1-node bench / 7-node bench / 7-node training

**Tables referenced:** Table 2 (paper.tex `\label{tab:perproblem}`),
Table 7 (`\label{tab:perprob3style}`) for the strategy-enum reference column.

**Archive(s):**
- `archives/r28_main_paper_3styles.tar.gz` — the strategy-enum subdir is
  the bench + training source for AllToAllV, Uniform A2A, Ring KV, dxe,
  PP cross-stage, TP MLP, FSDP prefetch, Layer-block AR
- `archives/r28_main_artifacts_runtimes.tar.gz` — the
  `trainium_*_7node.py` files Phase 5 deployed; the agent column in
  every row reads its algorithm from these.
- `archives/r28_ua2a_debug.tar.gz` — the 2158 ms Uniform A2A footnote
  number (the rare ua2a_agent that completed at 224 ranks).
- `archives/r28d_rkv_retry.tar.gz` — Ring KV retries.
- `archives/r28e_lbar_extra.tar.gz` — Layer-block AR extras.
- `archives/r44_dxe_1ag_bench.tar.gz` — the **refreshed** dxe bench
  numbers in Table 2's dxe row (the bench_dxe_7node + bench_dxe_1node
  subdirs). The dxe agent_fn was patched to call the deployed
  strategy-enum runtime's `dxe_loss` instead of an inline
  2-AR-no-max-shift hand roll.

**Reproduction:**
1. `orchestration_scripts/r28_rerun.sh` — full re-run of the
   strategy-enum / cc-react / multi-island Phase-3 search +
   bench + training measurement for all 8 problems.
2. `orchestration_scripts/r44_dxe_iterate_bench.sh` — refreshes only
   the dxe bench row using the patched `bench_dxe.py` (calls deployed
   runtime).

**Spot-check:**
```bash
tar -xzf archives/r44_dxe_1ag_bench.tar.gz -C /tmp
grep -aE "bench\\] (baseline|agent)" /tmp/r44_dxe_iterate_bench/bench_dxe_7node/node_0.log
grep -aE "bench\\] (baseline|agent)" /tmp/r44_dxe_iterate_bench/bench_dxe_1node/node_0.log
```
