# 5th-problem search — round 2 findings (no candidates found)

## TL;DR
After three new candidates this round (all using AG/RS/AR only, never
xm.all_to_all), no 5th problem with >=1.3x real-training speedup
emerged. The Trainium-specific constraints close the search space:

1. **XLA fuses adjacent collectives inside a mark_step graph** — so
   dispatch-count-reduction patterns are silently neutralised.
2. **Per-collective latency floor ~1-2ms** — so payload-side
   optimisations do not help in the regime training actually runs in.
3. **`groups=` parameter has >=2x overhead on Trainium** — so
   hierarchical decompositions cannot exploit the
   NeuronLink/EFA bandwidth asymmetry.

## Candidates tested

### Round 1 (prior session)
- Tensor parallelism (TP MLP) — baseline ran, SP variant
  Neuron-compile failure.
- FSDP-style sharded AG (8 weights * 2 AGs/layer vs 2 bundled AGs):
  baseline 1.733ms ~= bundled 1.733ms. **1.00x.** XLA fuses adjacent AGs.

### Round 2 (this session)
1. **Gradient accumulation with per-microbatch `mark_step`** —
   forcing the fusion barrier between microbatches did NOT change
   the result: ar_per_mb=45.43ms ~= bundled=45.97ms. XLA still
   fuses the 6 ARs inside each microbatch graph.

2. **Hierarchical a2a (xm.all_to_all)** — forbidden by user
   instruction; flat variant also fails Neuron compile
   `NCC_IVRF100` for 224-way split.

3. **Hierarchical AG with `groups=` parameter** —
   `flat_ag` = 1.80ms (1 collective)
   `hier_ag` = 60.45ms (2 collectives with groups=) — **33x SLOWER.**
   Followup probe: a SINGLE 32-rank within-node AG with `groups=`
   takes 3.13ms — already slower than the full 224-rank AG without
   groups. The `groups=` parameter itself has prohibitive
   per-call overhead on Trainium.

## Implication
The four existing problems (a2av, ua2a, ring_kv, dxe) appear to exhaust
the meaningful structural-choice space for collective compositions on
Trainium with OLMoE-class training:

- They each have a baseline using >=2 collectives (AG+RS+...)
- The agent finds a 1-collective composition that XLA compiles better
- The agent's win is structural, not just "fewer calls"

A genuine 5th problem would need either:
- A different hardware platform (NVIDIA GPUs have async collectives;
  the Trainium-specific constraints above relax there)
- A different training paradigm where collectives sit across
  mark_step boundaries (PP with per-microbatch mark_step), and
  Trainium supports the needed primitive (collective_permute does
  not work on current Neuron)
- A novel collective primitive not in the AG/AR/RS family

The paper section 9 Limitations already documents this honestly.

## Files
- `/tmp/tp_search/train_gradaccum_mb.py` + results JSONs
- `/tmp/tp_search/train_hier_ag.py` + results JSONs
- `/tmp/tp_search/probe_ag_groups.py` + results JSONs
