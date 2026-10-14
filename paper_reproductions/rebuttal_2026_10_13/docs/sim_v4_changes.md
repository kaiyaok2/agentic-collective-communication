# Simulator fixes applied locally for the rebuttal (correctness_test_v4.py; NOT pushed to the public repo)

Context: the shipped simulator (public repo and private commit 20fc5e3 alike) consumed a static `_HARDWARE_MEASUREMENTS`
table — a stale snapshot of an earlier probe campaign — and every Phase-1 `measure_*` tool returned values from it.
Re-running the probes on the rebuttal cluster (train_hwprobe_7node.py, 32 ranks / 1 node, cold cache; plus a 224-rank
run for the collective curves) and re-deriving the model from them gives the arms below. Everything is measured or
structural; no constant was tuned against the hardware pool.

| # | Change | Why (what the stale model got wrong) |
|---|---|---|
| 1 | Constants from the live probe: dispatch 1674 us (was 100), mark_step 150 us (was 1500), dense copy 67 GB/s (200), sub-region copy charged on the full-tensor basis the model uses at 297 GB/s (20 — the stale value over-charged `[:, rank, :].contiguous()` ~15x), per-op in-graph marginal costs 10-70 us for view ops (0.1) and 200-1200 us isolated for gather/scatter/index_select (29); compile first-exec 2-37 s | snapshot vs. this SDK |
| 2 | Trace candidates at the per-problem harness's REAL shapes (dxe T=256, V=32256; tp/fsdp/pp/lbar M=4, L=1, S=1024/512, DM=2048, bf16; a2av CAP*DM per pair) instead of toy shapes x a hand multiplier | dxe traced a 4 KB all-gather (x10) vs 147 KB in / 33 MB out on hardware; a2av 14 KB vs ~2 MB |
| 3 | Bytes actually moved per collective: AG receives (ws-1)*input, RS (ws-1)/ws*input, AR 2(ws-1)/ws*bytes | input bytes alone cannot see that 1-AG dxe moves 224x more data than the AR recomposition |
| 4 | Local elementwise / reduction / memset ops charged by volume at the measured HBM rate (124 GB/s) with the measured per-op marginal as floor | flat 29 us per op regardless of tensor size |
| 5 | Measured collective latency tables (bytes -> us, log-log interpolation, bandwidth-slope extrapolation) per primitive, at the scoring world size | topology constants (link GB/s) never matched the observed AG/AR/RS curves |
| 6 | Back-to-back dispatch amortization only for latency-bound collectives (payload <= 256 KB, where the 4 KB depth probe measured it); MB-size collectives pay their full measured time, and a run survives XLA-fusible local ops (elementwise/reductions/views) but breaks on materializing ops (cat/stack/zeros/copies) | the model amortized MB-size ARs to near zero, so bundling could never win; and any local op between two tiny ARs re-charged the full dispatch (3-AR dxe over-charged) |
| 7 | Scope-aware graph boundaries: `microbatch_loops_induce_graphs=False` when scoring against the per-problem harness (one graph per step), True for e2e training; and the microbatch-loop detector accepts any iterable that mentions a microbatch quantity or the per-microbatch argument (`range(0, M, 2)`, `range(len(partials))`, `for p in partials`) | the old detector only matched `range(M)`: unrolling the loop by two hid the per-microbatch graph boundary and the simulator scored an identical program 60x better (the leak-free search found exactly this exploit; hardware: 1.00x) |
| 8 | `index_select` charged at its measured random-access gather rate (probe: 1M indices -> 4 MB in 17.9 ms, ~0.2 GB/s) instead of the strided-memcpy rate | the AG+index_select extraction variants (Uniform A2A 5.7 s/step on hardware) were scored like plain slices |

**Second cluster, same SDK (2026-10-14 11:40 UTC probe):** volume terms reproduce (HBM copy 292 vs 297 GB/s strided basis,
74 vs 67 GB/s dense; index_select 1M = 18.5 vs 17.9 ms; 4 MB all-gather 5.07 vs 5.18 ms; gather/index_select marginals
within 3%), while latency-floor terms move ±2× (dispatch 968 vs 1674 µs; 4 KB all-reduce 406 vs 1511 µs; back-to-back
alphas (0.33, 0.0) vs (0.011, 0.027)). This is the per-deployment variance the Phase-1 probes exist to absorb, and it lands
on exactly the terms that decide small-payload bundling decisions.

Correlation against the 224-rank hardware pool (within-problem Spearman, step time; 3 repeats per candidate) is in
rebuttal_draft.md / results_summary.md. Residual limits we state explicitly: (i) compiler-fusion-dependent costs of
hundreds of 1-D slices (a2av helper variants) are not separable by a per-op model; (ii) small-collective latencies vary
+-2x between probe runs on this SDK, so Phase-1 constants carry that uncertainty.
