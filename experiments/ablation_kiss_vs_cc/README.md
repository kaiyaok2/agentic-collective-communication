# Phase-3 controller ablation: cc-react vs kiss-sorcar-react

This directory contains a self-contained harness that ablates the Phase-3
controller of the closed-stack search loop while holding every other
variable fixed. It answers: *given the same evolution prompt, the same
LLM (Claude Sonnet 4.5), and the same cost-model / simulator, how do
cc-react (the hand-rolled single-trajectory ReAct in
`search/template_evolution.py`) and kiss-sorcar-react (kiss's native
`KISSAgent.run` ReAct controller) compare on wall time, LLM cost, and
final simulator score?*

## What is being compared

| | cc-react (`_phase3_cc_react`) | kiss-sorcar-react (`KISSAgent.run`) |
|---|---|---|
| Loop implementation | `TemplateEvolution.evolve` in the repo | [kiss](https://github.com/ksenxx/kiss) framework's built-in ReAct |
| Prompt template | `prompts/generic_evolution.md` | `prompts/generic_evolution.md` (same file) |
| Prompt filling | Re-filled every round with updated `current_code`, `current_bench`, growing `history`; sent as a fresh single-turn user message | Filled once at the start with the baseline state; subsequent turns are kiss's own ReAct conversation |
| Extra prompt trailer | none | Appended: `Use the score_candidate(code) tool ... Call finish when you cannot improve further.` |
| Termination | Fixed budget `len(islands) * max_rounds` (24–48 calls per problem for `--max-rounds 8`); no plateau detection | Kiss's native `finish` tool exits early when the agent judges the score has plateaued |
| Prompt caching | Not enabled | Enabled by kiss (Anthropic prompt-cache) |
| LLM | `claude-sonnet-4-5-20250929` | `claude-sonnet-4-5-20250929` |
| Scoring | `benchmark_xla_candidate_generic` with cc-react's Phase-2 kwargs | Same function, same kwargs, called over a stdio pipe from the long-running `score_service.py` |

The pipelines call the *identical* scoring function; we verified this by
scoring the `ag_slice_cat` baseline on `alltoallv` on both sides — both
return `sim_time_us = 5432.57 us` byte-for-byte.

## What is held fixed

* Phase 1 is forced into no-LLM mode (`phase1_profiling(use_llm=False, ...)`).
  Cost-model constants are the deterministic defaults; Phase 1 costs
  zero tokens on both sides.
* Phase 2 (baseline evaluation) runs unchanged inside cc-react.
  `score_service.py` (used by kiss) replays cc-react's Phase-2 setup at
  service startup, so both agents see the identical starting state.
* Phase 4 (hardware validation) is skipped on both sides — this is a
  Phase-3-only comparison. The paper's headline evaluation runs Phase 4.

## Files

| File | Language | What it does |
|---|---|---|
| `run_with_route_patch.py`  | Python 3.12 (Neuron venv) | cc-react wrapper: routes Bedrock→Anthropic, forces `phase1_profiling(use_llm=False)`, then invokes `experiments.run_search.main()` in-process |
| `kiss_phase3.py`           | Python 3.13+ (kiss venv)  | kiss-sorcar-react driver: spawns `score_service.py`, builds the prompt from `prompts/generic_evolution.md`, runs `KISSAgent.run` with a `score_candidate(code)` tool |
| `kiss_token_shim.py`       | Python 3.13+ (kiss venv)  | Monkey-patches `kiss.core.models.anthropic_model.AnthropicModel._create_message` to append per-call usage records to `$ABLATION_TOKEN_LOG` |
| `score_service.py`         | Python 3.12 (Neuron venv) | Long-running scorer: replicates cc-react's Phase 1+2 setup exactly, then reads JSON `{"code": "..."}` requests on stdin and writes JSON score responses on stdout |
| `run_cc_baseline.sh`       | bash                      | Iterates the 8 collective problems, running cc-react on each with token/cost accounting |
| `run_kiss_baseline.sh`     | bash                      | Iterates the 8 problems, reading each problem's cc-react `results_$P.json` for a ±5% target and calling `kiss_phase3.py --target-score` |
| `run_llama_retry.sh`       | bash                      | Re-runs the 4 Llama-side problems (`pp_send_recv`, `tp_mlp`, `fsdp_prefetch`, `llama_block_ar`) with `--pattern moe` (the original attempt used `--pattern llama`, which is not a valid choice) |

## How to run

```bash
# One-time setup (Neuron venv exists; kiss installed).
export ANTHROPIC_API_KEY=sk-ant-api03-...

# From the repo root:
cd experiments/ablation_kiss_vs_cc/

# 1. Baseline: 8 problems, cc-react.  ~1.5 hours on the 1-node master.
./run_cc_baseline.sh

# 2. Kiss: 8 problems, targeted at cc-react's per-problem best +5%.
./run_kiss_baseline.sh
```

Environment knobs (all optional; each script documents its defaults in a
header comment):

| Variable | Purpose | Default |
|---|---|---|
| `ACC_REPO`      | Path to this repo (auto-detected as two dirs up from the scripts) | `$(cd "$SCRIPT_DIR/../.." && pwd)` |
| `ABLATION_WORK` | Output root (per-problem `cc_react/$P/` and `kiss/$P/` land here) | `$SCRIPT_DIR/outputs` |
| `NEURON_VENV`   | Neuron venv root (has the acc-repo dependencies) | `/opt/aws_neuronx_venv_pytorch_2_9` |
| `KISS_PY`       | Path to the kiss venv's `python` binary (kiss requires Python 3.13+) | `/home/ubuntu/kiss/.venv/bin/python` |
| `MAX_ROUNDS`    | cc-react `--max-rounds`. cc-react's total turn budget is `len(islands) * max_rounds` | `8` |
| `MAX_BUDGET`    | kiss `KISSAgent.run(max_budget=...)` in dollars | `5.0` |
| `MAX_STEPS`     | kiss `KISSAgent.run(max_steps=...)` | `30` |

## Outputs

Each script writes to `$ABLATION_WORK/{cc_react,kiss}/$PROBLEM/`:

* `run.log`                — full stdout of the run.
* `tokens.jsonl`           — one JSON object per LLM API call, with
                             `input_tokens`, `output_tokens`,
                             `cache_read_input_tokens`,
                             `cache_creation_input_tokens`, `model`, `ts`.
* `results_$PROBLEM.json`  (cc-react only) — the full Phase-3 winner and
                                             candidate ranking, including
                                             `sim_time_us` per candidate.
* `kiss_summary.json`      (kiss only) — problem name, wall seconds,
                                         baseline sim, best sim, number
                                         of score calls, whether the ±5%
                                         target was hit.

Aggregate results and prose analysis are on the paper author's Desktop at
`kiss_vs_cc_react_preliminary.md` (not checked in).

## Reproducing the paper table

The comparison runs single-node (1 x `trn1.32xlarge` master, 32
NeuronCores) because the seven worker nodes were unavailable at the time
of the ablation. The paper's Table 8 numbers (75 min / $22 for cc-react)
were measured on the full 7-node cluster with HW eval + Phase-1 LLM
calibration; the numbers this harness reproduces are Phase-3-only /
no-HW-eval / no-Phase-1-LLM. Both sides pay the same infrastructure cost
in this restricted setup — the harness is designed to isolate the
Phase-3 controller as the *only* variable, not to reproduce the paper
headline.

## Caveats

* **Non-determinism.** LLM responses are stochastic; kiss's `finish`
  decisions are prompt-conditioned; expect ±10% wall / cost variance
  between reruns.
* **Kiss cache-write costs.** Kiss's aggressive prompt caching means
  the *first* run for a fresh problem pays cache-creation cost
  (\$3.75/M) whereas subsequent runs pay cache-read (\$0.30/M). Rerun
  cost is lower than the numbers reported.
* **`shell-heredoc` inside `bash -c` on remote SSH.** The bash scripts
  use `<<PY ... PY` heredocs for the per-problem cost computation;
  these render fine locally but can break under `ssh host "..."`
  wrappers if quoted incorrectly. Run the scripts directly on the
  machine (or through `bash script.sh`) rather than piped through
  `ssh <cmd>`.
