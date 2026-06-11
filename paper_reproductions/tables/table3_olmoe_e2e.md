# Table 3 (tab:e2e) — OLMoE-10B 7-node end-to-end training step time

**Archive:** `archives/r28_main_paper_3styles.tar.gz` — the
strategy-enum/olmoe_baseline + olmoe_agent subdirs contain the
250-step run logs whose steady median feeds Table 3's
"baseline" and "agent" rows.

**Reproduction:** `orchestration_scripts/r28_rerun.sh` (the
`olmoe_baseline` + `olmoe_agent` stages).

**Spot-check:**
```bash
tar -xzf archives/r28_main_paper_3styles.tar.gz -C /tmp
grep -aE "Steady step" /tmp/r28/strategy-enumerate/olmoe_baseline/node_0.log
grep -aE "Steady step" /tmp/r28/strategy-enumerate/olmoe_agent/node_0.log
```
