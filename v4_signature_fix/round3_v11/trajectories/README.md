# v10/v11 kiss trajectories (candidates.jsonl per problem)

Full LLM search trajectory for every problem tested during the v10/v11 prompt refinement round.

Each `<round>/<problem>/candidates.jsonl` has one JSON object per line:
```json
{"n": <candidate #>, "ts": <unix time>, "code": "<python>",
 "result": {"ok": <bool>, "sim_time_us": <float>, "num_ops": <int>,
            "num_all_gather": <int>, "num_all_reduce": <int>,
            "num_collective_permute": <int>, "error": "<if not ok>"}}
```

`kiss_summary.json` has the final winner + wall time + n_score_calls.
`run.log` has kiss's full LLM trace (prompts, tool calls, reasoning).
`results.json` has the full score service payload.

## Layout

- **`v10_noregress/`** — 5 problems where kiss beat or tied strat under prompt v3, retested under prompt v10 to confirm no regression: xor_grid, gray_code, piecewise, triangle_num, popcount.
- **`v10_targets/`** — 6 problems where strat had won or led in sim (the ones v10 was intended to fix): hamming_dist, cond_xor, sum_popcount, sign_alt, perm_shuffle, nested_mod.
- **`v11_check/`** — 7 problems retested under prompt v11 (after v10 caused a nested_mod regression via aggressive constant-folding — v11 narrows that hint): hamming_dist, cond_xor, mod_sq, sign_alt, perm_shuffle, triangle_num, nested_mod.

The v11 versions are the final results reported in `PROMPT_V11_RESULTS.md`. See `../rt_summary.csv` for real-training numbers on the v11 winners.

## Notable trajectories

- **`v10_noregress/xor_grid_bcast/candidates.jsonl`** — 5 candidates. n=1-2 crash (mock env doesn't have bitwise_xor or `^` on TrackedTensor). n=3 achieves 71us with bit-decomposition. n=4-5 improve to 28us via vectorized coefficient tensor `torch.tensor([1,2,4,8,...])` — the v10 hint at work.
- **`v10_targets/cond_xor_bcast/candidates.jsonl`** — 6 candidates. n=1-3 crash. n=4 gets 38us via bit-level. n=5-6 get 29us via `torch.tensor(list_comp)` constant folding — the v10 hint applied correctly here (this problem's arithmetic path is complex enough that constant folding wins).
- **`v11_check/nested_mod_bcast/candidates.jsonl`** — the problem where v10's constant folding hint made kiss regress. v11 kiss tries pure arithmetic first (n=1-4), but the arithmetic path fails HW gate at 64-rank (Neuron compiler VALUE_MISMATCH on `%` over floats). n=5 falls back to constant folding at 29us. Confirms the regression is a Neuron compiler issue, not a prompt issue.
- **`v11_check/mod_sq_bcast/candidates.jsonl`** — kiss v11 tries several approaches, finally lands `((idx * idx) % K)` at 2us matching strat. Improvement from v10's 7us via the "avoid unnecessary dtype casts" hint.
