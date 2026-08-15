# Composing Collectives Above a Black-Box Vendor Library

> **Post-submission update (2026-08-13):** For Sorcar vs strat-enumerate
> head-to-head results on 30+ problems (12 `_bcast` + 10 `_comm` +
> 11 `_chal` + 8 OverlayCCL originals), see
> [`SORCAR_VS_STRAT_RESULTS.md`](SORCAR_VS_STRAT_RESULTS.md). Round-by-round
> methodology under `v12_study/round{1..15}/`. The `bootstrap_v6/` folder
> contains all post-submission sim/pipeline patches and a self-installing
> `apply.sh` for a fresh OverlayCCL clone.

---
**What this is.** A workflow that finds faster collective-communication
strategies above the AWS Trainium Neuron stack (a closed-source vendor
library). The Neuron runtime is treated as a fixed API — we never modify
it and never see inside it. We only call its primitives
(`xm.all_gather`, `xm.reduce_scatter`, `xm.all_reduce`,
`xm.collective_permute`, `xm.all_to_all`) and reshape, pad, or slice the
input/output tensors around each call. An LLM agent proposes strategies at
this layer; a workload-calibrated cost-model simulator scores them; the
top candidates are validated on real Trainium hardware before deployment.

**Headline results.** Against an internal-AWS-optimized production
baseline maintained by five experienced AWS developers, the deployed
strategies deliver:

- **1.40×** end-to-end on OLMoE-10B (224 ranks, 7×trn1.32xlarge), with
  matched descent on a 2500-step real-OpenWebText AdamW run.
- **3.24×** end-to-end on Llama-7B
  (PP + TP + FSDP + layer-block AR).
- $4$–$12×$ per-primitive per-step speedups, above the end-to-end ratio
  because constant non-collective compute dilutes the headline.

Removing the simulator from the loop (and letting the LLM judge
strategies by its own microbenchmarks instead) collapses the OLMoE
speedup to $1.00×$ — confirming that the on-device-calibrated simulator,
not the LLM, is what picks the strategy that wins at real training runs.

---

## The eight collective problems

The agent runs on eight collective-communication problems that arise in
real training (full descriptions in the paper, §1, Table 1):

**MoE-side:** `AllToAllV` (variable-count token dispatch),
`Uniform AllToAll` (equal-count token dispatch under expert-choice),
`Ring KV` (key/value shard rotation for ring-attention),
`Distributed CE` (cross-entropy over rank-sharded vocabulary).

**Llama-side:** `PP cross-stage` (activation/gradient transfer between
pipeline-parallel stages), `TP MLP` (all-reduce after the
tensor-parallel MLP block), `FSDP weight prefetch` (gather sharded
parameters before each layer), `Layer-block AR` (all-reduce of
replicated gradients at layer-block boundaries).

---

## Setup

Tested on a 7-node cluster of AWS `trn1.32xlarge` instances
(224 NeuronCores, NeuronLink intra-node, EFA cross-node).

```bash
# Neuron stack + PyTorch/XLA venv.
sudo apt-get install -y aws-neuronx-tools aws-neuronx-collectives \
                       aws-neuronx-runtime-lib aws-neuronx-dkms
python -m venv /opt/aws_neuronx_venv_pytorch_2_9
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com/ \
    torch-neuronx==2.9.* torch-xla==2.9.* \
    neuronx-distributed neuronx-distributed-training \
    transformers datasets
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH

# LLM proposer (Sonnet 4.5 in the paper; export the relevant API key).
export ANTHROPIC_API_KEY=...

# Clone and enter.
git clone https://github.com/OverlayCCL/OverlayCCL.git
cd OverlayCCL
```

For multi-node runs you'll additionally need rsync/SSH between the
master node and the workers (the orchestration scripts in
`paper_reproductions/orchestration_scripts/` assume key-based SSH).

---

## Running the agent

The agent entry point is `experiments/run_search.py`. It runs all five
phases (calibrate → seed pool → propose & refine → HW validation → emit)
and writes the deployed strategy to `runtime/trainium_<problem>.py` (and
`runtime/trainium_<problem>_7node.py` for the 7-node variant).

### Smoke test (1-node, no LLM)

```bash
python experiments/run_search.py \
  --problem alltoallv \
  --pattern moe \
  --no-llm \
  --phase3-style strategy-enumerate
```

Runs the 1-node strategy-enumerate loop for the MoE AllToAllV problem
using a heuristic mutation oracle (no LLM API calls). Writes
`runtime/trainium_alltoallv.py`. About 5 minutes on a single
trn1.32xlarge.

### Full 7-node run (LLM, all eight problems, paper headline configuration)

```bash
for prob in alltoallv uniform_a2a ring_kv dxe \
         pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
  python experiments/run_search.py \
    --problem $prob \
    --hw-eval \
    --num-nodes 7 \
    --master-addr <master-private-ip> \
    --worker-addrs <ip1>,<ip2>,...,<ip6> \
    --phase3-style strategy-enumerate \
    --llm-model sonnet
done
```

Runs Phase 1 calibration on the 7-node cluster, then strategy-enumerate
search over all eight problems with the LLM as mutation oracle, then
Phase 4 hardware validation, then Phase 5 codegen. Writes the deployed
runtime modules to `runtime/trainium_<problem>_7node.py`. Takes ≈56
wall-minutes on the 7-node cluster and ≈$19 of Sonnet 4.5 tokens.

### Search-style ablations

```bash
# cc-react (single-trajectory ReAct on Claude Code)
for prob in alltoallv uniform_a2a ring_kv dxe \
         pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
  python experiments/run_search.py --problem $prob --hw-eval --num-nodes 7 \
         --master-addr ... --worker-addrs ... \
         --phase3-style cc-react --llm-model sonnet
done

# multi-island (3-island parallel GA)
for prob in alltoallv uniform_a2a ring_kv dxe \
         pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
  python experiments/run_search.py --problem $prob --hw-eval --num-nodes 7 \
         --master-addr ... --worker-addrs ... \
         --phase3-style multi-island --llm-model sonnet
done
```

Both ablations converge to within $\sim$10% of strategy-enumerate per
problem but take $\sim$25% longer in wall-time and $\sim$20% more LLM
tokens; the paper deploys strategy-enumerate. See Tables 7 and 14.

### No-simulator ablation

```bash
for prob in alltoallv uniform_a2a ring_kv dxe \
         pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
  python experiments/run_search.py --problem $prob --hw-eval --num-nodes 7 \
         --master-addr ... --worker-addrs ... \
         --phase3-style strategy-enumerate --no-simulator --llm-model sonnet
done
```

Same search, but the LLM cannot query the cost model. It picks by its
own microbenchmarks instead. End-to-end OLMoE-10B speedup collapses to
$1.00×$ (Table 8) — this is the ablation that confirms the simulator,
not the LLM, is what surfaces the training-scope winner.

---

## Training with the deployed strategies

The agent's output is a set of drop-in `runtime/trainium_<problem>_7node.py`
modules. The training harnesses import them via a
`--backend {baseline,agent}` toggle.

### OLMoE-10B end-to-end (paper's 1.40× headline)

```bash
torchrun --nnodes=7 --node_rank=0 --nproc_per_node=32 \
  --master_addr=$MASTER --master_port=29701 \
  training/train_olmoe10b.py \
  --backend agent --ce agent --grad-sync baseline \
  --steps 2500 --realtok
```

`--backend agent` selects the agent's AllToAllV (and DXE / grad-AR when
their flags are also set). Add `--realtok` for the real-OpenWebText
loader used in Figure 3. Compare with `--backend baseline --ce baseline`
to reproduce the 1.40× ratio.

### Llama-7B end-to-end (paper's 3.24×)

```bash
torchrun --nnodes=7 --node_rank=0 --nproc_per_node=32 \
  --master_addr=$MASTER --master_port=29702 \
  experiments/model_extension/train_llama_e2e_7b.py \
  bundled 200
```

Replace `bundled` with `per_mb` for the per-microbatch developer baseline
(the 3.24× ratio is `per_mb` / `bundled`). Worker nodes run with
matching `--node_rank` 1..6. See
`paper_reproductions/orchestration_scripts/r28_llama7b_run.sh` for the
full multi-node launch.

---

## Where each search style's outputs live

| Style | Deployed runtime files | Per-problem logs & JSON |
|---|---|---|
| **strategy-enumerate** (deployed, paper headline) | `runtime/trainium_<problem>_7node.py` (checked into the repo) | `paper_reproductions/archives/r28_main_paper_3styles.tar.gz` → `r28/strategy-enumerate/` |
| **cc-react** (ablation) | not deployed; extract from archive | `paper_reproductions/archives/r28_main_paper_3styles.tar.gz` → `r28/cc-react/` |
| **multi-island** (ablation) | not deployed; extract from archive | `paper_reproductions/archives/r28_main_paper_3styles.tar.gz` → `r28/multi-island/` |
| **no-simulator** (ablation) | not deployed | `paper_reproductions/archives/r33_nosim_faithful.tar.gz` |

To inspect a cc-react / multi-island runtime, extract the relevant
archive:

```bash
mkdir -p /tmp/inspect && tar -xzf paper_reproductions/archives/r28_main_paper_3styles.tar.gz -C /tmp/inspect
ls /tmp/inspect/r28/cc-react/
ls /tmp/inspect/r28/multi-island/
```

The deployed strategy-enumerate runtime files (also present in
`runtime/`) are additionally archived in
`paper_reproductions/archives/r28_main_artifacts_runtimes.tar.gz`.

---

## Reproducing each table and figure in the paper

All scripts are versioned in `paper_reproductions/orchestration_scripts/`
and their result tarballs in `paper_reproductions/archives/`.
`paper_reproductions/archives_index.md` lists each archive's SHA-256 and
which paper artifact it supports.

| Paper artifact | What it is | How to reproduce |
|---|---|---|
| **Table 1** (Phase-1 probe tools) | Description of the 7 measurement tools the agent uses in calibration. No script — see `search/profiling.py` for the tool implementations. | — |
| **Table 2** (cost calculus) | Cost-calculus argument (E2E candidate trials vs. our loop). The numbers in the table are derived; the underlying wall-time / cost data comes from the strategy-enumerate runs (≈56 min, ≈$19 across 8 problems). | Run any `--phase3-style strategy-enumerate` search and read the printed wall-time/cost lines. |
| **Table 3** (eval setup) | Setup description of OLMoE / Llama harnesses. No reproduction needed. | — |
| **Table 4** (`tab:perproblem`) | Per-call 1n-bench, 7n-bench, and 7n-training latencies per problem (strategy-enumerate / deployed). | Numbers shipped in `paper_reproductions/archives/r28_main_paper_3styles.tar.gz` (look in `r28/strategy-enumerate/h7_bench_v6/` for the bench rows and `r28/strategy-enumerate/training/<problem>/` for the training rows). To rerun: loop the agent over all 8 problems (see "Full 7-node run" above), then run `experiments/h7_bench/bench_<problem>.py` for the 1n/7n bench columns. |
| **Table 5** (`tab:e2e`) | OLMoE-10B 1.40× and Llama-7B 3.24× end-to-end. | `bash paper_reproductions/orchestration_scripts/r28_llama7b_run.sh` (Llama-7B row); 2500-step `train_olmoe10b.py --realtok` baseline vs agent for OLMoE row (commands above). |
| **Table 6** (`tab:llama-sweep`) | Llama-block amp1–amp4 sweep at varying $S$ and $M$. | Use `experiments/model_extension/train_llama_e2e_amp{1,2,3,4}.py` with `per_mb` then `bundled` modes. |
| **Table 7** (`tab:ablation-cost`) | Per-style wall-time and LLM cost. | Read from the heartbeat / log files of each `--phase3-style {strategy-enumerate, cc-react, multi-island}` run. |
| **Table 8** (`tab:ablation-nosim`) | OLMoE 1.40× → 1.00× under `--no-simulator`. | `bash paper_reproductions/orchestration_scripts/r33_nosim_full_judge.sh`. |
| **Table 9** (`tab:ablation-3node`) | Cluster-size generalization at 3 and 5 nodes. | `training/train_olmoe10b_3node.py` and `experiments/model_extension/train_llama_e2e_7b_3node.py`. |
| **Table 14** (`tab:ablation-perproblem`, appendix) | Per-problem cells across the three search styles; baseline column shared (canonical, sourced from Table 4's strategy-enumerate reference) — see canonical-baseline note in `REPRODUCE.md`. | Run the same 8-problem agent loop above three times, once each for `--phase3-style {strategy-enumerate, cc-react, multi-island}`. Baseline cells are then pulled from the strategy-enumerate reference only and shared across the three style rows. |
| **Table 15** (`tab:ablation-nosim-perproblem`, appendix) | No-simulator per-problem rows. | `bash paper_reproductions/orchestration_scripts/r33_nosim_full_judge.sh`. |
| **Figure 1** (`fig:workflow`) | Workflow overview diagram. | `python figures/gen_paper_figures.py` → `figures/workflow.pdf`. |
| **Figure 2** (`fig:disagree`) | Cross-scope inversion bar chart. Bars sourced from strategy-enumerate reference numbers in Table 4. | `python figures/gen_paper_figures.py` → `figures/disagreement.pdf`. |
| **Figure 3** (`fig:loss-curve`, Appendix A) | OLMoE-10B real-OpenWebText 2500-step descent. | `bash paper_reproductions/orchestration_scripts/r45_figure2_olmoe_realowt.sh`, then `python figures/plot_loss_curve.py --baseline <json> --agent <json>`. |
| **Figure 4** (`fig:llama-descent`) | Llama-block descent overlay at $M{=}16$ on the NXD-primitive harness. | `bash paper_reproductions/orchestration_scripts/r49_llama_nxd_descent.sh` then `python figures/plot_loss_curve.py`. |
| **Figure 5** (`fig:llama-msweep`) | Step time and speedup vs $M$ on the NXD-primitive harness. | `bash paper_reproductions/orchestration_scripts/r49b_nxd_msweep.sh` then the M-sweep plotting block in `gen_paper_figures.py`. |

For per-row archive mappings, see `REPRODUCE.md` and
`paper_reproductions/archives_index.md`.

---

## Requirements

- 7×AWS `trn1.32xlarge` (224 NeuronCores total) for the headline numbers;
  smaller clusters (3 or 5 nodes) work for the cluster-size ablation
  (Table 9).
- Neuron SDK 2.x, PyTorch/XLA 2.9, NeuronX-Distributed 0.18+,
  NeuronX-Distributed-Training 1.7+.
- Anthropic API access for the LLM proposer (Claude Sonnet 4.5 in the
  paper; `--no-llm` is available for heuristic-only runs).

---

## License

See `LICENSE`.
