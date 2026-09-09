# Table 5 (tab:llama7b) — Llama-7B-equivalent 7-node end-to-end step time

**Archive:** `archives/r28_llama7b.tar.gz` — the per_mb and bundled
1000-step training runs whose steady median feeds Table 5.

**Reproduction:** `orchestration_scripts/r28_llama7b_run.sh`.

**Spot-check:**
```bash
tar -xzf archives/r28_llama7b.tar.gz -C /tmp
ssh ubuntu@<bench_rank_node> "python3 -c 'import json; print(json.load(open(\"/tmp/tp_search/llama_e2e_amp2_per_mb.json\"))[\"steady_median_ms\"])'"
```
