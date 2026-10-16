# Step 0 / Step 1 — Is Neuron compile cost worth modeling? (2026-10-16, trn1.2xlarge on demand)

Everything below was measured on 2026-10-16 on one trn1.2xlarge (8 vCPU, 30 GB host RAM, 2 NeuronCores; neuronx-cc 2.21.33363,
torch-xla 2.8.1, runtime 2.28). Raw data: `results/*.jsonl`; the full generated analysis: `results/analysis_latest.md`;
compiler inputs for every module: `results/corpus_hlo.tar.gz` (HLO protos + capture metadata, 6.8 MB). The bench chain was
still running when this report was written (see §5); every stage is resumable and the report regenerates from `analyze.py`.

## 1. Pipeline (Step 0)

- **Exact compiler input.** Each torch-xla program runs with a private Neuron compile cache, so the `model.hlo_module.pb` the
  framework hands to neuronx-cc is kept, with the framework's flags, the NEFF, the HLO text, framework-side compile time,
  step wall and XLA ExecuteTime. neuronx-cc accepts only the proto (text HLO is rejected), so this is the only faithful input.
- **Standalone re-compilation** under `/usr/bin/time -v` + timeout: wall / user / sys, peak RSS, exit status, PASS / FAIL /
  TIMEOUT / OOM, NCC_* code, NEFF size. The compiler's info log is parsed into per-pass seconds (Tensorizer passes), per-job
  seconds (HLOToTensorizer, Frontend, StaticIOTranspose, WalrusDriver, BIRLinker, Kelper, NeffWrapper) and the frontend's
  MAC / traffic counters, so every compile carries a phase breakdown for free.
- **Execution** of each NEFF with `neuron-profile capture` → `view --output-format summary-json` (~190 metrics per NEFF:
  `total_time`, MFU, per-engine instruction counts and times, DMA bytes, spills). Note: the summary is keyed by NEFF hash.
- **Knobs**: flag sets (O1, O3, model-type transformer, llm-training, auto-cast) and five compiler versions in separate venvs
  (2.19, 2.20, 2.21 installed, 2.22, 2.24); cold-page-cache mode (`drop_caches` between compiles).
- **Structural features** from the HLO proto via protoc-generated openxla bindings (instruction / computation counts, opcode
  histogram, tensor bytes, dtype mix, dot / conv FLOPs, distinct shapes, control flow, collectives, proto size).

## 2. Corpus (Step 1)

| source | families | programs captured | notes |
|---|---|---|---|
| synthetic | mlp, attention, elementwise chains, reductions, conv stacks, gather/scatter, sort/topk, many-small-matmuls, reshape chains, random DAGs | 146 specs (137 ok) | fwd and training-step variants |
| real (HF configs, random init) | GPT-2, Llama, BERT, T5, ViT, Mixtral | 36 specs (16 ok; T5/ViT fixed and re-capturing, Mixtral router fails on XLA) | 1–12 layers, inference + training |
| mutations | one parameter changed from 5 bases | 65 specs (62 ok) | ±1, ×2, ½ on a dimension; +1 layer; activation; bf16; train off; norm / causal off |
| collective programs (2 ranks) | all_reduce / all_gather / reduce_scatter / AG+RS / stacked AR | 17 run (10 deferred) | |
| OverlayCCL workloads (2 ranks) | 8 problems × {baseline, agent} | 16 runs (12 ok; PP and Ring-KV harnesses need >2 ranks but still emitted graphs) | |

262 programs → **529 HLO modules** (306 step graphs + parameter-init graphs), one JSONL row each with features.

## 3. Results

### 3.1 Compile cost is large, heavy-tailed and shape-driven

Framework-side compile of every step graph (n = 214, one compile each): median 11.9 s, p90 140 s, max 308 s; 28% take
more than 60 s, 12% more than 120 s; max/min 164×. Standalone CLI wall agrees with the framework-side number within
~4% (median ratio 1.038, p10 0.97, p90 1.11, n = 30), so the capture-time numbers are valid measurements.

Per family (median / p90 / max, seconds): conv training 100 / 161 / 308; attention training 68 / 212 / 291; HF models
57 / 120 / 229; many-small-matmuls 9 / 128 / 265; gather 2.2 / 7.8 / **195**; MLP 6.5 / 12 / 23; reductions 3.3 / 4 / 4.3.

Peak host RSS spans 0.24–10.6 GB on the installed compiler at default flags (23.5 GB for one module on 2.24, see 3.4), i.e. a
single compile can take a third of this host's memory. NEFF size spans 0.02–37.7 MB.

Pathological cases found without looking for them:
- A **16-instruction gather graph** (1 KB proto; `torch.gather` of 8192 rows from a 1024×256 table) compiles in **195 s**
  with 10.6 GB RSS and a 10 MB NEFF; its index_select / embedding siblings compile in 2 s.
- **`randn` of 10M elements** (parameter-init graph of the 10M-element elementwise chains) compiles for **~20 min** with a
  37 MB NEFF, while the 8-op step graph on the same tensors compiles in 2.7 s. The 64- and 512-op variants were deferred.
- Two compiler **failures** from innocuous inputs: a conv training step at batch 9 (batch 8 compiles in 161 s) and
  `torch.gather` with a 32768-row table (1024 rows compiles). Both are reproducible from the saved protos.

### 3.2 Measurements are deterministic enough to resolve small effects

- Warm repeats (30 modules × 3): CV median **0.7%**, p90 2.4%, max 2.8%; max/min within a module median 1.016, max 1.06.
- Cold page cache vs warm (24 modules): ratio median **1.54×**, p90 2.9×, max 2.9×; absolute extra median 3.9 s — cold-start
  overhead is a constant few seconds that dominates small compiles and is negligible for large ones.
- Hence any difference above ~5% between two compiles is real, not noise.

### 3.3 Near-identical graphs have very different compile cost

62 one-parameter mutation pairs (capture-time compiles; the pair arm with 2 clean repeats is running):
42 pairs change compile time by more than 20%, 6 by more than 2×; |log ratio| median 0.30 (≈1.35×).
- **±1 on one dimension** (23 pairs): ratio median **1.26×**, max 1.84× — hidden 2048→2049 takes an MLP step from 6.7 to 12.2 s;
  seq 1024→1025 takes attention from 117 to 212 s.
- **Halving the batch** from 2 to 1 makes GPT-2 and Llama training steps compile **3.4–3.9× slower** (54→185 s, 42→166 s).
- Doubling a dimension costs 1.4–2.5×; removing the backward pass saves 30–60%; bf16 vs fp32 changes nothing (0.9–1.0×).
- Among modules with **identical instruction counts** (22 groups of ≥3), compile time differs **2.9× median, up to 80×**.

### 3.4 Flags and compiler versions move cost in module-specific, non-monotone ways (24-module subset)

| arm | median ratio to default / 2.21 | p10 | p90 | extremes |
|---|---|---|---|---|
| `--optlevel 1` | 0.998 | 0.32 | 1.02 | GPT-2 12-layer train 224 → 32 s (**0.14×**); conv bf16 train 143 → 669 s (**4.7× slower**) |
| `--optlevel 3` | 0.995 | 0.96 | 1.01 | none |
| neuronx-cc 2.24 | **0.64** (36% faster) | 0.29 | 0.87 | all-gather 4 KB graph 2.0 → 58 s (**29×**); conv bf16 train 143 → **1071 s at 23.5 GB RSS** (7.5×) |
| neuronx-cc 2.19 (partial, 8 of 24) | 0.92 | 0.80 | — | same all-gather graph 2.0 → 22 s (11×) |

No arm changed PASS/FAIL status on the subset. The per-pass log shows where the time goes in each outlier
(`PGLayoutTilingPipeline`, `PGTiling`, `InlineNativeKernels`, `AGOrderingAnalysisPass`, `InsertIOTransposes`).

### 3.5 Cheap features: good interpolation, poor extrapolation

Targets log(compile wall) and log(peak RSS) from 35 cheap HLO features (counts, bytes, FLOPs, dtype mix, proto size).

| set | model | CV | R² | median APE | within 2× | Spearman |
|---|---|---|---|---|---|---|
| all step graphs (n=214, capture-time) | GBDT | random 5-fold | **0.89** | 12% | 91% | 0.94 |
| all step graphs (n=214) | GBDT | leave-one-family-out | **0.59** | 41% | 72% | 0.76 |
| all step graphs (n=214) | ridge | leave-one-family-out | −0.01 | 70% | 50% | 0.66 |
| balanced 30-module bench subset | GBDT | leave-one-family-out | 0.58 | 44% | 70% | 0.81 |
| peak RSS, 30-module subset | GBDT | leave-one-family-out | 0.46 | 15% | 73% | 0.78 |

Single features: Spearman(log wall, n_instructions) 0.84, proto_bytes 0.83, total_tensor_bytes 0.72, dot_flops 0.62. Top
GBDT features: proto size, instruction counts, tensor bytes. A pass/fail classifier is not yet trainable (2 failures).

## 4. Verdict on the premise

1. **Expensive**: a quarter of ordinary training-step graphs compile for over a minute on this host, individual graphs for
   3–20 minutes, and a single compile can take 10–24 GB of host RAM. Compile time per step graph is 1–3 orders of magnitude
   above execution time (ms).
2. **Variable in ways size features do not explain**: ±1 on a dimension changes cost by 26% median, batch 2→1 by 3.4–3.9×,
   identical instruction counts hide 2.9× (median) to 80× spreads, and tiny graphs (16 instructions, 1 KB) can be the slowest.
   Flags and versions add module-specific swings of 0.14× to 29×. Measurement noise is 0.7%, so all of this is signal.
3. **Failure-prone**: two reproducible compiler failures from one-step perturbations of passing graphs, plus the known host-OOM
   regime at larger models (not reachable on 30 GB).
4. **Cheap baselines are not enough**: within-distribution GBDT reaches R² 0.89, but on an unseen graph family it drops to
   R² 0.59 with 41% median error and 28% of predictions off by more than 2×; ridge fails entirely (R² ≈ 0). The residuals
   are exactly the shape- and pass-driven effects above, which argues for Step 3 (active discovery of cost drivers) rather
   than more features alone — and the per-pass timings already collected are the natural supervision for it.

So the premise holds, with one refinement: the interesting target is not "compile time of typical graphs" (well predicted by
size) but the heavy tail and the discontinuities (shape thresholds, version/flag regressions, failures), which is where
an active, hypothesis-driven search over graph variants is needed.

## 5. State of the campaign and gaps

- Running at report time (chain continues unattended, resumable): compiler 2.19 arm, then the mutation-pair arm with 2 warm
  repeats, the two known-failure protos (to record NCC codes), T5/ViT re-capture, NEFF re-profiling of the subset (the first
  pass lost the profile JSON to a parser bug fixed since), versions 2.22 / 2.20, then default-flag compiles of all 306
  step graphs with profiling and a second repeat.
- Deferred: 10M-element elementwise chains with 64/512 ops (≥20-min init compiles each); `--model-type transformer` arm;
  Mixtral programs (HF router `one_hot` fails on XLA int dtype); 2-rank collective graphs with 16M elements.
- Not reachable on this host: the host-OOM regime (OLMoE DM=4096 on trn1.32xlarge, 512 GB), 224-rank collective graphs.
- Lessons: two capture drivers must not share `pkill` patterns; only rank 0 writes metadata under torchrun; the node went
  unreachable once under two drivers + torchrun (rebooted; memory guard added); laptop IP rotation needs the SG re-added.
