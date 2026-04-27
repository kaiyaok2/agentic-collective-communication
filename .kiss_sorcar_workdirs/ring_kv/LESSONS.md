# Lessons

- The benchmark latency model in `CollectiveProfiler.estimate_latency` is: `local_ops * 29us + num_collectives * 200us` (each collective counts 2× dispatch overhead for fwd+bwd). Minimize total op count.
- `counter.count` in benchmark output includes collective ops. The actual local_ops used for cost calculation excludes collective ops.
- For all_gather-based problems, calling `xm.all_gather(tensor, dim=0)` directly on 1D tensors avoids unnecessary unsqueeze/view ops, saving 58us.
- Hierarchical approaches (multiple collectives) are always worse in this latency model since each collective adds 200us minimum.
- Always run the benchmark first to understand the baseline before optimizing.
- The theoretical minimum cost for any problem requiring at least 1 collective is 200us (0 local ops + 1 collective).
- For ring_kv, the simplest optimal solution is `xm.all_gather(kv_chunk, dim=0)` — a direct 1D all_gather with no local ops.
