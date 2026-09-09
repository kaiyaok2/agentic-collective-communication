# Discover an optimized `{evolved_fn_name}` on Trainium.

Your task: compute the tensor described in the specification below,
minimizing `sim_time_us` and passing a real-hardware correctness gate
at world size {world_size}.

## STOP AND READ THE SPECIFICATION FIRST

Below in the `## Signature` section you will find `{signature_doc}`
which contains a `Formula:` line. **The formula tells you exactly what
tensor to return.** Before writing any code, re-read the formula and
paraphrase it in your own words.

Example: if the formula says `x[i] = 1 if i == 0 else 0`, that
describes an impulse vector — one 1 at position 0, zeros elsewhere.
You can construct this locally on each rank with:
`(torch.arange(N, device=x.device) == 0).to(x.dtype)`
No collective needed. No probing needed. Just read the spec.

## Common misreadings — watch for these

**Operator confusion.** `==` and `<` are DIFFERENT. `>=` and `>` are
DIFFERENT. `%` is the modulo operator (remainder), not division.
The formula's operators must be preserved EXACTLY. Do not "helpfully"
substitute a similar operator.

**Position-vs-value confusion.** When a formula reads `t[i,j] = f(i, j)`
where `i, j` are positional indices (typical for `_bcast` problems),
this is a **function of position, not input tensor values**. Even if `f`
looks like arithmetic that could apply to values (`i - j`, `|i - j|`,
`max(i, j)`), the arguments here are the INDICES `i, j` — position
integers running `0, 1, ..., N-1`. Compute them locally as:

```python
idx = torch.arange(N, device=x.device)
ii = idx.view(N, 1)   # row index broadcast
jj = idx.view(1, N)   # col index broadcast
# then apply the formula's arithmetic to ii, jj
```

No collective is required.

**Shape confusion.** If the formula's structure is `t[i,j] = g(i)` (only
depends on row index — same across all `j`), the output shape is still
`(N, N)`. Broadcast a 1D result via `.unsqueeze(1).expand(N, N).contiguous()`.
Analogously `.unsqueeze(0).expand(N, N)` if the formula depends only on `j`.

## Discovery methodology

**Step 1 — Paraphrase the formula.** Write in a comment what tensor
the function must return. Copy operators VERBATIM from the formula.
Do NOT probe by returning zeros or the input unchanged — the correctness
signal returned by `score_candidate` is intentionally coarse (only
`n_wrong=X of Y elements`) and does NOT reveal the reference values.
You must derive the output from the spec.

**Step 2 — Is the formula position-based (`f(i, j, N, K)`) or
value-based (`sum`, `max`, `mean` over tensor entries)?**
- Position-based → local closed form, ZERO collectives. Most `_bcast`
  problems are position-based.
- Value-based → collective needed.

**Step 3 — Determine output shape** from signature_doc.

**Step 4 — Write the closed form.** Use `torch.arange`, comparisons
(`==`, `<`, `<=`, `>`, `>=`), arithmetic (`+`, `-`, `*`, `//`, `%`,
`.abs()`), `torch.max/min`, broadcasting via `.view`, `.unsqueeze`,
`.expand`, then `.contiguous().to(x.dtype)`.

**Step 5 — Score.** Call `score_candidate(code)`. Interpret errors:
- `shape (A,) != (B, C)` — output rank/shape wrong. Add
  unsqueeze/expand.
- `value_mismatch` — shape right, values wrong. Re-read the formula
  character-by-character. Check every operator, index order, constant.
- `HW_GATE_FAIL: value_mismatch (X% wrong)` — code structurally wrong
  at 64-rank scale.
- `CRASH` — you used an unsupported op (e.g. `bfloat16`, or a banned
  primitive).

## Rules (violations fail the gate)

- Code must handle every `world_size ∈ {1..64}` and every rank.
- No `if world_size in (4, 8): ...` — gate runs at 64.
- No `raise` in the algorithm.
- No `return torch.zeros_like(x)` probing — signal is coarse by design.
- No `manual_seed/rand/randn/bincount/scatter_add/bfloat16` casts.

## Signature

```python
{signature}
```
{signature_doc}

## Current best (baseline)

Perf: {current_sim_time} us · {current_num_permutes} CP · {current_num_gathers} AG · {current_local_ops} local ops.

```python
{current_code}
```

## Reference implementations
{reference_implementations}

## History
{history}

Now: paraphrase the formula in a comment, decide position-vs-value,
determine shape, implement the closed form, call `score_candidate`.
