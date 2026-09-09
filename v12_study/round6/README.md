# Round 6: phase-1 regression fix

## Problem

Phase-1 profiling took 15-25 min per invocation when use_llm=True (strat
default) because the LLM ran through 25 turns of tool exploration
(measure_xla_op_overhead, measure_collective_latency, etc.). On some
runs it timed out entirely without emitting any downstream signal, so
strat could not produce candidates.

Under use_llm=False (kiss default via score_service_v2), phase 1 was
already deterministic and fast (round 4 added the auto-probe path).

Effect: kiss vs strat comparison was unfair. Strat kept timing out
during phase 1, before any code generation could happen.

## Root cause analysis

Audited search/agent_simulator_config.py: all the measure_* tools that
the LLM calls at phase 1 read from a static dict _HARDWARE_MEASUREMENTS.
None of them run actual HW measurement during the search. The ONE
subprocess call in the file is _test_primitive_compilation, which runs
a subprocess to check if a primitive compiles on HW (5 primitives x 30s
timeout each, called only once after phase 1).

Consequence: the LLM does not actually design a probe campaign in the
implemented codebase. It selects which pre-computed values to read and
in what order, but the values themselves do not change based on the
LLM strategy. Functionally, calling each tool once yields the same
information the LLM would gather over many turns.

The paper describes phase 1 as an LLM-driven probe campaign (system_model
section, Table 2). In the implementation, the LLM is a narrator over a
fixed static config — same values, ~25 min slower, and non-deterministic
across runs (LLM chooses different tool orders and sometimes fails to
converge).

## Fix

Rewrote phase1_profiling in experiments/run_search.py: removed the
use_llm branch, always run auto-probe deterministically. Same tools,
same values, no LLM tool exploration.

Downstream phase 3 still uses LLM (kiss=freeform code gen, strat=strategy
enumerate) — this fix only affects phase 1.

## Effect

- Kiss phase 1: unchanged (already used auto-probe).
- Strat phase 1: 15-25 min -> few seconds. Strat can now reach phase 3
  on all problems including _bcast where it previously timed out.
- Deterministic across runs. Reproducibility improved.
- No information loss vs the LLM-driven path (all tools read the same
  dict).

## Files patched

- experiments/run_search.py: phase1_profiling rewritten.
- All other files carry forward from rounds 3-5.
