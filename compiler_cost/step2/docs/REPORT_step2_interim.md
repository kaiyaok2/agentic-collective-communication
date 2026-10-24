# Step 1 → Step 2 — benchmark status and cheap-baseline results (interim, 2026-10-24)

Measurements on one trn1.2xlarge (8 vCPU, 30 GB; neuronx-cc 2.21.33363 unless noted). Code: `compiler_cost/step0/scripts`
(capture, bench, grids) and `compiler_cost/step2` (features2.py, build_dataset.py, baselines.py, sweep_series.py).
Generated tables: `step2/results/baselines_latest.md`, `step2/results/sweep_series_latest.md`.

## 1. Is the benchmark faithful enough?

What it now contains (manifest_step1: **2308 HLO modules from 1315 programs**, every module with its exact `HloModuleProto`):

| source | programs | notes |
|---|---|---|
| real model graphs | HF configs: GPT-2, Llama, BERT, T5, ViT, Mixtral (Step 0) + Mistral, Qwen2, Phi-3, Gemma 2, OPT, BLOOM, Falcon, GPT-NeoX, GPT-J, RoBERTa, DistilBERT, ALBERT, ELECTRA, DeBERTa-v2, XLM-R, BART, Whisper, wav2vec2, CLIP, DINOv2, ConvNeXt, Swin, ResNet, MobileNetV2, SegFormer; 13 torchvision models; realistic training steps (AdamW/SGD/Adagrad, clipping, bf16 autocast, label smoothing, grad accumulation); DLRM, GNN, MoE, UNet stage, contrastive | inference + training, fp32/bf16 |
| production inference graphs | 63 modules of the deepseek-coder-33B TP=32 continuous-batching server (NeuronX Distributed Inference), with the framework's flag set | the only >10B graphs; host-memory bound |
| collective workloads + candidates | 8 OverlayCCL training harnesses × {baseline, agent}; 143 Sorcar problems × {baseline template, Sorcar candidate, strat candidate} = 286 programs (0 capture failures) | 2 ranks |
| synthetic | 20 operator families (MLP, attention, conv, elementwise, reductions, gather/scatter, sort, many-small-matmul, reshape, random DAG, normalization, layout, slicing, masking, scatter variants, conv variants, einsum, recurrence, logits/CE, casts) | fwd and training |
| near-identical series | 62 one-parameter mutations (Step 0) + **379 one-dimensional sweeps** through 128/512/1024/2048/4096 boundaries, batch 1–2048, heads, layers, vocab, kv-heads, conv batch/img/channels, gather table/index/dim, randn size, op counts | 29 series with 4–28 points |

Measurement properties established: warm-repeat CV 0.7 %; standalone CLI wall = framework compile wall within ~4 %; cold cache adds a
constant ~4 s; two concurrent compiles cost 0–14 % wall on small graphs (recorded as `note=par2`); every compile carries exit
status / NCC code / peak RSS / NEFF size / per-pass and per-job seconds, and PASS NEFFs are executed with neuron-profile.
Labels now include compiler failures (framework and CLI), host-OOM kills (memory watchdog) and timeouts.

Known limitations (documented, not fixed): single host size (30 GB) sets the OOM boundary; timeouts truncate the tail at
1500–2400 s; family imbalance (286 tiny collective-candidate graphs); version/flag arms exist for a 24-module subset only until
the full-corpus arms finish (running). Verdict: faithful enough to start Step 2 on; the full-corpus bench (default r0+profile →
4 versions → 3 flag sets → repeats) continues and the baselines re-run automatically on the final table.

New compiler behaviour found while building it: a reproducible neuronx-cc 2.21 crash (`MhloToPythonPrinter: Value is used
before definition`) on plain inference graphs of ResNet-18/50, MobileNetV2, ConvNeXt, SegFormer (torchvision and HF); embedding_bag,
depthwise/k7-stride-2 conv, conv+avg-pool and a 1 M-element integer chain fail; GPT-2 2-layer training at hidden 2048 (32 heads)
exceeds 41 min while hidden 1536 takes 87 s; gather graphs need >30 GB host RAM at 16 K+ indices.

## 2. Dataset (build_dataset.py)

One row per module: 37 Step-0 features + ~130 Step-2 features (fixed opcode vocabulary, dim-alignment statistics (multiples
of 128/32/8, odd dims, dims ≥1024), dot M/N/K and conv kernel/channel/stride statistics, literal bytes, entry-graph depth and
fan-out) and targets: bench wall (median of warm repeats, default flags, 2.21), peak RSS, status, NEFF bytes, exec latency,
per-pass seconds; `capture_compile_s` (framework-side compile time of the capture run) is used as the target where the bench
has not reached a module yet (validated ±4 %). At report time: 731 bench targets, 1283 capture-time targets.

## 3. Cheap baselines (baselines.py)

Metrics on log10(wall): R², median absolute percentage error (APE), fraction within 2×, Spearman. Splits: random 5-fold;
leave-one-family-out (34 families); leave-one-source-out; **sweep extrapolation** (train on all non-sweep programs, predict every
point of every sweep series). Step graphs only (1192 modules), the arm most relevant to users:

| features | model | random | leave-one-family-out | leave-one-source-out | sweep extrapolation |
|---|---|---|---|---|---|
| instructions only | power law | R² 0.32 / medAPE 86 % | 0.28 / 90 % | 0.21 / 90 % | <0 / 82 % |
| cheap-4 (instr, proto bytes, tensor bytes, FLOPs) | GBDT | **0.86 / 10 %** | 0.70 / 26 % | 0.70 / 21 % | 0.78 / 23 % |
| Step-0 35 | GBDT | 0.90 / 8 % | 0.76 / 27 % | 0.78 / 21 % | 0.81 / 21 % |
| full (~170) | GBDT | **0.94 / 6.6 %** | **0.80 / 18 %** | **0.84 / 15 %** | **0.89 / 18 %** |
| full | ridge | 0.59 / 49 % | 0.17 / 74 % | 0.03 / 81 % | <0 / 69 % |

Along the 29 sweep series (153 consecutive pairs with >20 % actual change), full+GBDT gets the **direction right 84 %** of the
time (ridge 90 %, instructions-only 23 %) with Spearman 0.58 between predicted and actual log-ratios — but on the series that
are dominated by shape thresholds it is near chance: conv channels (dir-acc 0.44, Spearman 0.00), attention heads (0.57), matmul
size 64 vs 65 vs 128 vs 129 (0.33), MLP batch (0.52). Those series contain the largest, non-monotone effects in the corpus
(`sweep_series_latest.md`): elementwise 6 M → 7 M elements 2.6 s → 61 s (23×) and 9 M → 10 M 81 s → 3 s; conv channels 256 → 257
734 s → 83 s and 31 → 32 419 s → 84 s; MLP hidden 2047 → 2048 → 2049 = 11 → 6.6 → 12 s; attention seq 128 → 129 14 → 28 s;
gather dim 8 → 32 → 64 3 → 11 → 28 s. Peak RSS: full+GBDT R² 0.93 random / 0.60 family (median APE 0.4 % / 3 %).

Compile-failure classifier (73 labelled failures in 2301 modules): full-feature GBDT reaches AUC 0.97 / PR-AUC 0.76 in-distribution,
but **AUC 0.84 / PR-AUC 0.23** on unseen families — flagging 10 % of modules catches 41 % of failures.

## 4. Reading

- In distribution, cheap features + GBDT are **good enough for scheduling-type use** (median error 7 %, 95 % within 2×): the Auler
  & Borin-style baseline is strong and should be kept as the reference.
- They are **not good enough** for (a) new graph families (median error 18–27 %, worst families 40–70 %), (b) the shape-threshold
  regime that produces the 2–20× swings, which the alignment features only partly capture, and (c) predicting failures outside
  the training families. These are exactly the targets of Steps 3–4, so the plan continues: on the completed bench table, add
  (i) version/flag as inputs (full-corpus arms), (ii) per-pass-time targets (which pass explodes), (iii) a graph-level model
  (message passing over the HLO DAG with per-instruction shape/alignment features) and (iv) failure/OOM/timeout classifiers
  with per-class labels; compare all against the GBDT reference under the same splits.
