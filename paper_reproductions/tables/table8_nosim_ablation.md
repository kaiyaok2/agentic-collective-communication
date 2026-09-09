# Table 8 (tab:ablation-nosim) — no-simulator ablation (§7.2)

**Archive (current version):** `archives/r33_nosim_faithful.tar.gz` +
`archives/r33b_retries.tar.gz`.

**Archive (in-progress refresh after iterative no-sim + wrapper fix):**
- `archives/r42_nosim_full_iterative.tar.gz` — iterative no-sim search
  for all 8 problems + the OLMoE end-to-end measurement made BEFORE the
  alltoallv wrapper-template fix (so the OLMoE row in this run still
  reads 1.41×, attributable to the hardcoded AG+slice+cat wrapper).
- `archives/r43_wrapper_fix_olmoe.tar.gz` — same iterative no-sim
  search but with the alltoallv `alltoallv()` wrapper template patched
  to AG+T+RS; OLMoE collapses to 1.00× as expected (the wrapper was
  the load-bearing source of the previous 1.41×).
- `archives/r44_dxe_1ag_bench.tar.gz` — iterates the no-sim dxe search
  with a fixed `full_vocab_ag` template (no longer trips the
  TrackedTensor sandbox), converges to 1-AG, then re-runs OLMoE to
  confirm the OLMoE end-to-end is honestly 1.00×.

**Open follow-on (in the queue):** task #223 / #224 — refactor
`train_llama_e2e_7b.py`'s bundled path to actually call the deployed
`evolved_tp_mlp` / `evolved_fsdp_prefetch` / `evolved_llama_block`
runtimes (currently they're imported but never invoked, so the
3.58× Llama row in `tab:ablation-nosim` does not reflect the no-sim
runtimes). Once #223 lands, the Llama row gets refreshed to
the honest ratio.

**Reproduction (current acc-orphan-r35 version):**
`orchestration_scripts/r33_nosim_full_judge.sh`.

**Reproduction (next revision):**
`orchestration_scripts/r43_alltoallv_dxe_baseline_match.sh` +
`orchestration_scripts/r44_dxe_iterate_bench.sh`.

**Spot-check (most recent OLMoE numbers):**
```bash
tar -xzf archives/r44_dxe_1ag_bench.tar.gz -C /tmp
grep -aE "Steady step" /tmp/r44_dxe_iterate_bench/olmoe_baseline/node_0.log
grep -aE "Steady step" /tmp/r44_dxe_iterate_bench/olmoe_agent/node_0.log
```
