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

**Step 6 — Once your first candidate passes, look for structural
improvements.** The simulator counts ops. Fewer ops with the same
result → lower `sim_time_us`. Consider these transformations:

- **Vectorize repeated per-bit / per-lane operations.** If your
  code has `for b in range(nbits): pc = pc + f(bit_b(...))`, the same
  computation can often be done in one vectorized shot by materializing
  a small tensor of coefficients (e.g. `torch.tensor([1, 2, 4, 8,
  ...])`) and broadcasting against the indices in one expression
  rather than iterating in Python. The correct answer is unchanged;
  only the op count drops.
- **Prefer arithmetic over comparison when both produce the same
  values.** For example, if a boolean mask always maps to `{0, 1}`
  and you need to multiply by that mask, expressing the mask as an
  arithmetic function of the indices (rather than `bool → cast`) can
  save an op. This is a general rewrite, not a problem-specific
  answer.
- **Compile-time constant folding — a trade-off, not always a win.**
  For small fixed `N` where the reference tensor's values depend only
  on positional indices, a Python list/loop that computes the tensor
  and wraps it once with `torch.tensor([...], device=x.device,
  dtype=x.dtype)` bakes the values as a constant at trace time and
  skips runtime arithmetic. However, **emitting a constant tensor has
  its own non-trivial cost** in the simulator — comparable to a
  handful of arithmetic ops. It is a win only when the pure
  arithmetic version has many ops (e.g. a bit-decomposition loop,
  nested branches, or `torch.where` over multiple arithmetic
  intermediates). If the pure arithmetic version is already just a
  handful of vectorized ops (e.g. one line of `torch.arange` +
  arithmetic + `.to(dtype)`), leave it alone — constant folding will
  make it slower, not faster. **Always try the pure
  arithmetic version first**, score it, and only reach for constant
  folding if the arithmetic version is measurably slow. This is safe
  only when the values are functions of positional indices (not of
  any tensor input) and when the tensor is small enough that emitting
  it as a literal is cheap. **Do not use this to hardcode reference
  values you looked up from the scorer** — that is answer leakage.
  The values must be recomputable from the formula alone.
- **Combine reductions.** If your last few lines are `a = f(idx);
  b = g(idx); return a + b`, and both `f` and `g` are cheap, a single
  fused expression is often faster than two named intermediates.
- **Avoid unnecessary dtype casts.** Casting `idx = torch.arange(N,
  ...).to(x.dtype)` up-front forces every subsequent integer op
  through floating-point. Do the integer arithmetic in `torch.int64`
  and cast only the final result to `x.dtype`. This matters when the
  formula uses `//`, `%`, or bitwise-equivalent decomposition.

After each `score_candidate` call, if the result is correct but
`sim_time_us` looks high relative to the number of ops the formula
mathematically requires, ask yourself which of the above rewrites
could apply, and try one. Do not try more than one rewrite per
iteration — keep the diff readable so you can tell which move
helped.

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

## Pattern: many small collectives -> cat + single collective + narrow/split

When the algorithm makes multiple collective calls of the same primitive
on tensors of the same dtype (e.g., loop of `xm.all_reduce` over a list
of gradient tensors), the Neuron dispatch overhead of each call dominates
over the actual bandwidth cost. A better pattern:

```python
# Instead of:
#   out = [xm.all_reduce(xm.REDUCE_SUM, g) for g in grads]
# Do:
sizes = [g.numel() for g in grads]
flat = torch.cat([g.reshape(-1) for g in grads])
reduced_flat = xm.all_reduce(xm.REDUCE_SUM, flat)
# Split back with torch.narrow (metadata-only view, ~free)
out = []
offset = 0
for g, n in zip(grads, sizes):
    out.append(torch.narrow(reduced_flat, 0, offset, n).reshape(g.shape))
    offset += n
```

- `torch.cat` costs bandwidth once. `torch.narrow` is a metadata-only view.
- Net: one AR call amortizes N dispatch overheads, and the increased
  payload adds negligible time until AR is bandwidth-bound (~1MB+).
- Same idea applies to `all_gather`, `reduce_scatter`. If different
  primitives are mixed (e.g., AR-SUM + AR-MAX), cat + AR-SUM + AR-MAX +
  narrow can still amortize dispatch over the combined payload.
- Sim's `back_to_back_amortization_us` model captures this: depth_1
  pays ~100us dispatch, depth_2-8 pay ~10-30us amortized, so 5 per-tensor
  ARs cost ~180us sim vs 1 cat-AR at ~100us. RT delta is even larger due
  to per-call graph launch overhead.
