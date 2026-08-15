# Optimize {evolved_fn_name}

Optimize `{evolved_fn_name}` for AWS Trainium XLA following the formula in the signature below, using **AI discovery**.

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

## Reference implementations
{reference_implementations}

## History
{history}

## Rules

- You MUST NOT cheat or reward-hack the simulator. Every value must be computed from the formula in the signature above, never looked up from the scorer.
- Do not break correctness — the candidate must return the exact tensor described by the formula for every valid `(rank, world_size)`.
- Use **adversarial testing** to fix bugs before scoring.
- You may add temporary diagnostic prints (running time, op counts) at finer granularity to guide optimization, but **remove them before the final candidate is scored**.
- Strictly use the `run_parallel` tool for each independent subtask (candidate generation, diagnostics, verification).
- Thoroughly check whether the model has missed any requirement or introduced a bug.

Call the `read_reference()` tool for Trainium quirks (XLA collectives, unsupported primitives like cumsum/sort, `T_local` sim cost model, and worked idioms including `torch.tensor([list-comp])` const-fold vs `torch.arange` arithmetic trade-offs).

Now: paraphrase the formula, implement a candidate, verify correctness adversarially, then call `score_candidate`.
