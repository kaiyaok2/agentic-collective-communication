# Step 0/1 — Neuron compiler cost: measurement pipeline and benchmark corpus (2026-10-16)

Goal: establish whether neuronx-cc compilation cost (wall time, peak host memory, failures/timeouts, NEFF size) is variable,
expensive and failure-prone enough to justify a learned predictor, and build the first (HLO, flags, compiler version) ->
(compile cost, success, NEFF latency/profile) corpus.

Pipeline (scripts/):
- programs.py — registry of graph-producing programs: synthetic families (MLP, attention, elementwise chains, reductions, conv,
  gather/scatter, sort/topk, many-small-ops, reshape chains, random DAGs), real models from HF configs (GPT-2, Llama, BERT, T5,
  ViT, Mixtral; random init), near-identical mutation pairs (one parameter changed), 2-rank collective programs.
- capture.py / capture_one.py — run each program on the device with a private Neuron compile cache so the exact HloModuleProto
  the framework hands to neuronx-cc is kept, plus framework compile time, step wall, XLA ExecuteTime, HLO text.
- capture_acc.sh / write_meta.py — same for the OverlayCCL per-problem training harnesses (baseline + agent) at 2 ranks.
- features.py — structural features from the HLO proto (protoc-generated bindings from openxla hlo.proto).
- build_manifest.py — one JSONL row per captured module.
- ncc_bench.py — standalone compiles under /usr/bin/time -v with timeout: wall/user/sys, peak RSS, PASS/FAIL/TIMEOUT/OOM, NCC_*
  code, NEFF size; optional neuron-profile execution of the NEFF (latency, MFU, DMA bytes, spills, ...). Flag sets and
  alternate compiler versions (separate venvs: 2.19, 2.20, 2.22, 2.24 alongside installed 2.21). Cold-cache mode drops page caches.
- select_subset.py — family-balanced subset for flag/version/cold arms.
- run_capture_all.sh, run_post_capture.sh, run_bench_all.sh — the campaign chain; analyze.py — the Step-0 report.

Node: trn1.2xlarge on demand (us-east-1f), Neuron DLAMI PyTorch 2.8 (neuronx-cc 2.21.33363, torch-xla 2.8.1, runtime 2.28).
