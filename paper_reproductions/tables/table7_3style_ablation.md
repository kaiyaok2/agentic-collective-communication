# Table 7 (tab:perprob3style) — 3-style ablation: strategy-enum vs cc-react vs multi-island

**Archive:** `archives/r28_main_paper_3styles.tar.gz` —
- `r28/strategy-enumerate/` for the strat-enum column,
- `r28/cc-react/` for the cc-react column,
- `r28/multi-island/` for the multi-island column.

Each subdir holds the Phase-3 search log, the deployed runtime, and
the bench + training measurement runs (one set per problem).

**Reproduction:** `orchestration_scripts/r28_rerun.sh` runs all
three styles end-to-end.

**Spot-check:**
```bash
tar -xzf archives/r28_main_paper_3styles.tar.gz -C /tmp
for style in strategy-enumerate cc-react multi-island; do
  echo "=== $style ==="
  grep -aE "Steady step" /tmp/r28/$style/olmoe_agent/node_0.log
done
```
