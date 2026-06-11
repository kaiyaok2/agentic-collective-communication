# 7-Node HW Microbench

Isolated per-call HW microbench for the 4 collective problems at 7-node 224-rank scale.
Complements the 1-node 32-rank microbench (`experiments/real_alltoallv_bench.py`) by
measuring at the full cluster, where EFA cross-node latency dominates over NeuronLink
intra-node latency.

## Usage

```bash
bash experiments/h7_bench/run_all.sh   # runs all 4 problems
# or one at a time:
PORT=32500 bash experiments/h7_bench/run_bench.sh a2av
PORT=32502 bash experiments/h7_bench/run_bench.sh ua2a
PORT=32504 bash experiments/h7_bench/run_bench.sh rkv
PORT=32506 bash experiments/h7_bench/run_bench.sh dxe
```

## Known limitations

- The AllToAllV and Uniform AllToAll `baseline_fn` (canonical AG+RS) currently
  hits an XLA trace-time shape-inference issue when called in isolation outside
  the OLMoE training graph: `xm.all_gather(x.unsqueeze(0), dim=0)` is reported
  with shape `(1, N)` at trace time even though it runtime-materialises to
  `(ws, N)`. The subsequent `.view(ws, ws, mc)` fails the size check. The
  same baseline runs cleanly inside `train_olmoe10b.py` / `train_uniform_a2a_7node.py`,
  so this is a bench-infrastructure quirk, not a baseline correctness issue.
- Per-call numbers from this isolated bench are 5-25\u00d7 smaller than the
  per-call numbers measured inside `train_olmoe10b.py` (via `.item()` probes
  inside the autograd `Function`), because there's no surrounding model
  graph, no NEFF cache eviction, and no `mark_step` between calls.
