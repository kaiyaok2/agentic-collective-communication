# Faithful Methodology Ablation: does the research-discovery loop earn its cost?

**Question.** The Sorcar search controller (kiss `KISSAgent`, Phase-3) runs
with a system prompt that mandates two methodology blocks:

1. **AI-discovery research loop** (`SYSTEM.md` "AI discovery, auto research,
   and optimization" — *Mandatory Instructions (MUST FOLLOW)*): profile the
   baseline → web-search SOTA → write `./tmp/ideas.md` → pairwise-judge →
   implement → run real eval → log `./tmp/explored-ideas.md` → loop until the
   metric goal is met with a held-out generalization check.
2. **Adversarial testing / adversarial training** (`SYSTEM.md` two adjacent
   sections): break the candidate with adversarial workloads in a subtask,
   fix in another.

This ablation removes **both** blocks and re-runs the identical search to
measure the cost (LLM calls, tokens, wall-time) they add, and whether they
buy any solution-quality.

## Setup — controlled, single-variable

- **Only variable**: `SYSTEM.md` handed to `KISSAgent`.
  - `full` = pristine `SYSTEM_FULL.md` (both methodology blocks present).
  - `ablated` = `SYSTEM_ABLATED.md`, identical except the two blocks are
    replaced by a one-line note. 206 → 187 lines; nothing else changed.
- **Everything else held fixed**: same 8 problems, same scorer service
  (Phase-1+2 no-LLM defaults, 7-node/224-rank sim), same model
  (`claude-sonnet-4-5-20250929`), same `--max-steps 30 --max-budget 5.0`,
  same thread pinning.
- **Problems** = the 8 family exemplars instantiated in the 10B E2E step
  (one per E2E family site): `sixtyfourinline` (F1), `eightyaltsum`
  (F1×F4b), `nine_ar_same_input_chal` (F2), `sequential_ar_chain_edge_chal`
  (F3 total-cancel), `perslice3dM96` (F4a), `perrowM64N4K` (F4b),
  `mixmaxmin` (F6), `eightslab` (F7).
- Per-call token usage logged by `kiss_token_shim` (patches
  `AnthropicModel._create_message`); score-calls and wall from
  `kiss_summary.json`.

## Result — cost

| Metric | full (methodology ON) | ablated (OFF) | ablated / full |
|---|---|---|---|
| LLM calls | 58 | 46 | **0.79×** |
| score_candidate calls | 50 | 38 | **0.76×** |
| wall-time (s, PAR=8) | 678 | 558 | 0.82× |
| output tokens | 42,297 | 32,184 | 0.76× |
| total tokens (in+cache+out) | 647,613 | 393,244 | **0.61×** |

**Disabling the methodology makes the search ~24% cheaper in LLM calls and
~39% cheaper in total tokens.** Framed the way the mandate asks — *does
enabling the methodology increase cost?* — **yes: the full research-loop +
adversarial methodology costs ≈1.65× the total tokens and ≈1.26× the LLM
calls of the stripped controller** (647,613 / 393,244 = 1.65×; 58 / 46 =
1.26×).

The cost is concentrated on the two genuinely hard problems. On the 6 "easy"
divergent problems (F1/F2/F4/F6) both arms converge in a single scored
candidate (2 LLM calls each) — the methodology adds nothing because there is
nothing to iterate on. All the extra spend is on the two problems that need
real search:

| Problem | full LLM calls / best us | ablated LLM calls / best us |
|---|---|---|
| `sequential_ar_chain_edge_chal` (F3) | 27 / **5161** | 13 / 5361 |
| `eightslab` (F7) | 19 / **5161** | 20 / 5163 |

## Result — quality (the caveat that matters)

Cheaper is not free. On `sequential_ar_chain_edge_chal` the full methodology
drove the score to **5161 us**; the ablated arm plateaued at **5361 us** —
it stopped ~200 us short after half the iterations. The research-loop's
"search again for fresh ideas … go to step 4; stop only when the metric goal
is met" is exactly what kept the full arm iterating past the local optimum
the ablated arm settled for. On `eightslab` both reached the floor (5161 vs
5163, tie).

So the honest characterization:

- **On simple divergent families** (F1/F2/F4/F6 — the bulk of the 55-anchor
  set): the methodology is pure overhead. One candidate solves them; the
  loop and adversarial passes add tokens and no quality.
- **On hard multi-step problems** (F3 cancellation, F7 slab fusion): the
  methodology earns its cost — it finds a better optimum (5161 vs 5361 on
  F3) by iterating instead of stopping early.

This is the expected shape of a discovery loop: it is insurance you pay on
every problem but only collect on the hard ones. For a benchmark dominated by
single-step family rewrites, running the controller *without* the mandated
loop is 1.65× cheaper at the cost of occasionally settling for a
few-percent-worse schedule on the hardest problems.

## Reproduce

```
bash /home/ubuntu/run_ablation_cost.sh          # runs both arms, restores SYSTEM.md
python /home/ubuntu/agg_ablation.py             # prints tables, writes summary json
# artifacts: /home/ubuntu/ablation_cost/{full,ablated}/<problem>/{kiss_summary.json,tokens.jsonl}
#            /home/ubuntu/ablation_cost/ablation_cost_summary.json
```

The ablated system prompt is `kiss/src/kiss/SYSTEM_ABLATED.md` (diff against
`SYSTEM_FULL.md` shows exactly the two removed blocks).
