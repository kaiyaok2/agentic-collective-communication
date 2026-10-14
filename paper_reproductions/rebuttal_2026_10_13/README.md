# PPoPP'27 #1429 rebuttal experiments (7× trn1.32xlarge, ap-southeast-2c, 2026-10-13/14)

All artifacts of the rebuttal cluster run. Start with `docs/results_summary.md` (numbers), `docs/rebuttal_draft.md`
(response text), `docs/sim_v4_changes.md` + `docs/sim_correlation_final.md` (simulator fixes and correlation), and
`docs/taxonomy_AQ5_attempted.md` (strategies attempted vs deployed, from the archived Phase-3 logs).

Layout
- `scripts/`  — everything that ran on the cluster master (`/home/ubuntu/ap7`): `run_ap7.sh` (N-node torchrun launcher),
  `sweep_candidates.sh` + `deploy_candidate.py` + `manifest*.json` (candidate pools; `cands*/` hold the candidate sources),
  `train_hwprobe_7node.py` (the real Phase-1 probes), `score_v3.py`/`score_v4.py` (simulator arms), `analyze_v2.py`
  (within-problem Spearman/top-1/regret), `generic_search.py` (leak-free non-LLM search), `queue*_ap7.sh` (the exact run
  order), `derive_table.py`/`compare_table.py` (probe table vs shipped snapshot). Cluster addresses are redacted.
- `sim/correctness_test_v4.py` — the faithful simulator fork (local; not pushed to the public OverlayCCL repo).
- `results/` — raw per-run JSON/logs: `sweep_v2/rep{1,2,3}` (57-candidate pool, 3 reps, 224 ranks), `sweep_v2/sim_*.json`
  (simulator scores per arm; final = `sim_v4_v11c.json`), `sweep_a2av*` (AllToAllV seed-library helpers, 4 reps),
  `sweep_found*` (hardware checks of search finds), `hwprobe/` (probe JSON + logs), `olmoe_results/` (OLMoE seeds, EXDIM,
  TOPK, stub runs), `llama7b_results/`, `gs_v4/` (search vs fixed simulator), `queue.log` (full run log).

Reproduce: bootstrap a 7-node trn1 cluster (DLAMI Neuron, venv `/opt/aws_neuronx_venv_pytorch_2_9`), clone the public
OverlayCCL artifact to `/home/ubuntu/OverlayCCL`, put `scripts/` at `/home/ubuntu/ap7`, write `/home/ubuntu/venv.env`
(`VENV=...`, `WORKERS="<private IPs>"`), then run the `queue*_ap7.sh` scripts in numeric order (each is self-describing);
`train_hwprobe_7node.py` runs single-node via `run_ap7.sh` with `WORKERS_OVERRIDE=' '`. Simulator scoring is CPU-only:
`TABLE=results/hwprobe/hw_probe_measured.json MANIFEST=scripts/manifest_all.json TABLES=0 SCOPE=harness RESET=fusion
MULTS='{"ring_kv":2,"alltoallv":32,"uniform_a2a":384}' python scripts/score_v4.py` then `SIM=<out> python scripts/analyze_v2.py`.
