# Round 10: cumsum properly rejected as unsupported primitive

## Root cause of round-9 fix being wrong

Round 9 lumped cumsum into SCAN_LIKE_OPS and charged it saturating arith
cost (815 us). That gave sim/HW ordering coincidentally right on
triangle_num but was WRONG about cause: torch.cumsum does not compile
on Neuron trn1 (SDK 2.26.6360). neuronx-cc raises TCTransform
assertion NCC_ITCT901 during hlo lowering. Verified by direct HW test.

Consequence: strats triangle_num candidate (arange + cumsum) is not a
real HW solution — it fails compilation. Paper phase-4 gate should
reject it. Sim should mark it +inf.

## Round-10 fix (proper)

1. Extended _test_primitive_compilation in agent_simulator_config.py to
   test local ops: cumsum, cumprod, sort, argsort.
2. Extended phase-1 auto-probe list in run_search.py to include these.
3. Added unsupported-local-op check in benchmark_xla_candidate_generic:
   if any op in counter.events is in unsupported_primitives (or a
   dedicated unsupported_local_ops kwarg), return +inf.
4. Reverted the SCAN_LIKE_OPS synthetic charge — no longer needed
   because the sim rejects cumsum candidates outright.

## Effect

- Strats triangle_num arange+cumsum candidate: sim +inf (was 29 or 815
  depending on prior round). Strat is forced back to baseline_ar_bcast
  or must find a non-cumsum solution.
- No regression on cumsum-free candidates.
- Auto-fitness: no hardcoded cost. Detection driven by real compiler
  test at phase 1 — matches papers primitive-viability abstraction.

## Follow-up

- Rerun phase 1 to populate agent_sim.config.unsupported_primitives
  with cumsum/etc.
- Rerun 12 novel + 8 OverlayCCL kiss vs strat sweep.
- Redo audit + RT verification.
