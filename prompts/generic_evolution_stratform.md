# Optimize {evolved_fn_name}

Optimize `{evolved_fn_name}` for AWS Trainium XLA following the formula in the signature below, using **strategy enumeration** (NOT iterative discovery).

## Signature

```python
{signature}
```
{signature_doc}

## Current best (baseline)

Perf: {current_sim_time} us.

```python
{current_code}
```

## Rules — STRATEGY ENUMERATION PROTOCOL (mandatory)

You MUST follow this exact three-step protocol. Do NOT iterate beyond it.

**Step 1 — Enumerate.** Before writing any code, list exactly 5 DISTINCT structural strategies for this problem. "Distinct" means a meaningfully different combination of collective primitives and/or payload layout (different collective count, different payload packing, different ordering). Strategies that differ only in low-level details count as one. Write the 5 strategies as a numbered list with 1-2 sentences each.

**Step 2 — Implement all 5.** Write one candidate implementation per strategy. Score each with `score_candidate` exactly once. Do not debug beyond making each candidate run.

**Step 3 — Pick the winner.** Among candidates that scored successfully, the lowest sim_time_us wins. You may make at most 2 refinement attempts on the single best strategy (structural tweaks within that strategy only). Then stop.

- You MUST NOT cheat or reward-hack the simulator. Every value must be computed from the formula in the signature above, never looked up from the scorer.
- Do not break correctness — the candidate must return the exact tensor described by the formula for every valid `(rank, world_size)`.
- Do NOT use exploratory/iterative discovery outside the protocol. Do NOT call any tool except `score_candidate`.

Now: enumerate the 5 strategies, implement each, score each once, refine the winner at most twice, and stop.
