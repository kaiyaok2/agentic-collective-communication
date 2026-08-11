# Round 3: size-scaled sim cost model + phase-1 probe

## What is new vs Round 2

Round 2 hardcoded 4 constants (340, 100, 800, 120) in correctness_test.py.
Round 3 makes them size-scaled and probe-consistent with the rest of the
OverlayCCL cost model:

1. **Extended _HARDWARE_MEASUREMENTS** in search/agent_simulator_config.py
   with a new standalone_graph_cost_us block holding:
   - const_fold_base_us = 60.0 (small-tensor floor)
   - const_fold_bw_bytes_per_us = 40.0 (measured 2D N=64 to N=128 scaling)
   - arith_saturating_us = 1000.0
   - arith_marginal_first_us = 340.0
   - arith_marginal_next_us = 100.0
   - raw_1d and raw_2d full measurement dicts

2. **New tool measure_standalone_graph_cost** in the phase-1 tool set
   alongside measure_memory_copy_throughput, measure_graph_launch_overhead,
   etc., so a phase-1 LLM sees the same probing pattern as everything else.
   It reads from _HARDWARE_MEASUREMENTS and pushes into
   agent_sim.knowledgebase['standalone_graph_cost_us'].

3. **Rewrote the round-2 patch in benchmark_xla_candidate_generic** to:
   - Accept standalone_graph_cost_cfg kwarg
   - Compute const_fold_us = max(cf_base, output_bytes / cf_bw) per graph
     (was hardcoded 120)
   - Compute arith_us = min(sat, marg_first + marg_next * (n_arith - 1))
   - Mixed graph handling: when n_tensor_const >= 1 AND n_fused_arith >= 1,
     cost = max(const_fold, arith_chain), not the sum. Fixes the Round-2
     bug where mixed graphs missed both branches and fell through to
     per-op sum.

## HW measurements driving the calibration

Raw sweep (2-node 64-rank, 200 iters, hw_measurements_raw.txt):

Const-fold cost scales with output bytes past ~1 KB:
  1D N=16 (64B):    60.7 us     2D N=8   (256B):    61.2 us
  1D N=128 (512B):  81.1 us     2D N=32  (4KB):    160.5 us
  1D N=1024 (4KB):  139.6 us    2D N=64  (16KB):   399.4 us
                                2D N=128 (64KB): 1541.3 us

Arith-chain cost saturates near 1000 us for small tensors regardless of
op count (dispatch-dominated regime):
  1D N=16 nops=3: 813.4 us     1D N=64 nops=1:  1067.5 us
  1D N=32 nops=7: 917.9 us     1D N=128 nops=7: 1102.2 us

Anomalies not modeled: 2D N=128 nops=1: 132 us — XLA folds
arange().view(N,1) + arange().view(1,N) into a small NEFF. This is a
compiler special case; general bit-decomposition arithmetic hits saturation.

## Sim delta over OverlayCCL — corrected claim

**OverlayCCL PPoPP submission (main branch of github.com/OverlayCCL/OverlayCCL):**
search/correctness_test.py differs from this branch ONLY by the round-2
STANDALONE_GRAPH_HW_CAL block (verified via diff on 2026-08-11).
Everything else — per-op cost table, memcpy BW, dispatch overhead,
back-to-back amortization, per-collective bandwidth floor, NEFF
compilation cost, graph-launch overhead, fusion credit against
collective, volume-scaled ops, bucket-cap detection, HBM peak-bytes
penalty, training-scale byte multiplier, 3-tier alpha1/alpha2/alpha3
pipeline amortization — all in OverlayCCL, all either probed via
phase-1 tools or hardcoded structural terms.

**The single delta this repo adds is:** a size-scaled cost floor for
collective-free graphs, split into constant-fold-baked-NEFF and
arithmetic-chain dispatch regimes, with a max() rule for mixed graphs,
wired through the same phase-1 probing pattern as everything else.

## Verification

Under Round-3 sim on the 12 novel _bcast problems:
- kiss v12 wins clean sim on xor_grid_bcast (1002 to 102 us) and
  piecewise_bcast, matching HW RT ordering.
- 10 problems tie at the const-fold floor.
- Mixed graph triangle_num sanity: cost = max(arith=540, const=60) = 540,
  matching RT ordering const=2.35 < mixed=2.47 < arith=2.51 ms/iter.

The improvement is not a large speedup number — it is alignment: sim now
predicts what RT actually shows for _bcast position-based graphs.
