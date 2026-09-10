# Sorcar Methodology Ablation — 5-Arm × 58 Divergent Problems

Controlled single-variable ablation of the Sorcar (kiss `KISSAgent`) search methodology. Same 58 divergent anchor problems, same scorer, same model (`claude-sonnet-4-5`), same budget (`--max-steps 30 --max-budget 5.0`, `--pattern moe`, 7-node sim). Each arm strips exactly one methodology group from the system prompt; everything else is byte-identical. We record BOTH search cost (LLM calls, score calls, tokens, wall) AND warm-cache real-HW runtime (ms/iter, 2-node/64-rank, 100 timed iters after 20 warmup).

## Arms

| Arm | What is disabled |
|-----|------------------|
| `full` | full methodology (all groups ON) — reference |
| `noadv` | adversarial testing + adversarial training OFF |
| `noA` | research/ideation OFF (Group A: read+profile baseline, web-search SOTA, write ideas.md) |
| `noB` | select/iterate loop OFF (Group B: pairwise-judge ideas, implement→eval→log explored-ideas.md, search-again loop) |
| `noC` | convergence-check OFF (Group C: stop-on-goal + held-out generalization check) |

The 7 research-loop bullets are grouped into 3: **A** = read+profile baseline / web-search SOTA / write ideas.md; **B** = pairwise-judge ideas / implement→eval→log explored-ideas.md / search-again loop; **C** = stop-on-goal + held-out check. Plus a separate `noadv` arm for the adversarial testing/training blocks.

## Search cost (all 58 problems, per arm)

| Arm | LLM calls | score calls | wall (s) | output tok | total tok | tok ×full |
|-----|-----------|-------------|----------|------------|-----------|-----------|
| `full` | 311 | 253 | 4310 | 252569 | 2871018 | 1.000 |
| `noadv` | 241 | 183 | 3638 | 212552 | 1838838 | 0.640 |
| `noA` | 291 | 183 | 3775 | 217859 | 2170311 | 0.756 |
| `noB` | 250 | 97 | 3043 | 177461 | 1528652 | 0.532 |
| `noC` | 250 | 192 | 3824 | 222628 | 2114016 | 0.736 |

Full methodology is the most expensive arm (311 LLM calls, 253 score calls, 2.87M tokens). Every ablation is *cheaper* — but the savings are not free, as the quality columns below show. **`noB` is cheapest (0.53× tokens, only 97 score calls)** and also the most damaging: with the select/iterate loop disabled the agent barely scores candidates at all.

## Quality: candidates never validated (`n_score_calls == 0`)

| Arm | # problems emitted with ZERO scored candidates |
|-----|-----|
| `full` | 0 |
| `noadv` | 0 |
| `noA` | 2 |
| `noB` | 20 |
| `noC` | 0 |

With the iterate loop OFF (`noB`), the agent emits un-vetted code on **20/58** problems — it writes a rewrite and never checks it against the scorer. `noA` skips scoring on 2. `full`, `noadv`, `noC` always validate.

## Quality: sim regressions and warm-cache RT vs full

For each arm, the problems whose emitted code is *worse* than full's (best sim > full by >1%), with the corresponding warm-cache HW runtime. `RT FAIL` = the ablated code did not run on Neuron HW (compile/runtime abort) where full's code runs cleanly — the hardest quality failure.

### `noadv` — 1 sim regressions

| problem | full sim (us) | arm sim (us) | sim ×worse | full RT (ms) | arm RT (ms) | RT ×worse |
|---------|--------------|--------------|-----------|--------------|-------------|-----------|
| chained_ar_nested_edge_chal | 5161 | 5563 | 1.1× | 0.5333 | 0.2165 | 0.4× |

### `noA` — 7 sim regressions

| problem | full sim (us) | arm sim (us) | sim ×worse | full RT (ms) | arm RT (ms) | RT ×worse |
|---------|--------------|--------------|-----------|--------------|-------------|-----------|
| thirtysixinline | 5178 | 13060 | 2.5× | 0.0930 | 0.8953 | 9.6× |
| mixmaxmin | 5362 | 8199 | 1.5× | 0.1113 | 0.4601 | 4.1× |
| six_ar_arith_edge_chal | 5167 | 6167 | 1.2× | 0.1900 | 0.2567 | 1.4× |
| mixedscaledseq | 5178 | 6169 | 1.2× | 0.1106 | 0.2149 | 1.9× |
| large_N_4ar | 5178 | 5911 | 1.1× | 0.0915 | 0.3055 | 3.3× |
| three_group_dead_verify_chal | 5656 | 5911 | 1.0× | 0.4140 | 0.3648 | 0.9× |
| sequential_ar_chain_edge_chal | 5165 | 5361 | 1.0× | 0.2967 | 0.1154 | 0.4× |

### `noB` — 21 sim regressions

| problem | full sim (us) | arm sim (us) | sim ×worse | full RT (ms) | arm RT (ms) | RT ×worse |
|---------|--------------|--------------|-----------|--------------|-------------|-----------|
| perslice3dM96 | 5417 | 24446 | 4.5× | 0.1110 | FAIL | FAIL |
| eightyaltsum | 5178 | 21960 | 4.2× | 0.1009 | FAIL | FAIL |
| fiftyinline | 5178 | 15910 | 3.1× | 0.1469 | 1.1832 | 8.1× |
| fortyinline | 5178 | 13875 | 2.7× | 0.0984 | 1.0062 | 10.2× |
| perbatchM32 | 5309 | 11538 | 2.2× | 0.0764 | 0.5665 | 7.4× |
| perrowM32N8K | 5309 | 11538 | 2.2× | 0.0995 | 0.5364 | 5.4× |
| sixteeninlin | 5178 | 8982 | 1.7× | 0.0933 | 0.5046 | 5.4× |
| perbatchM12 | 5189 | 7418 | 1.4× | 0.1342 | 0.4169 | 3.1× |
| nine_ar_same_input_chal | 5166 | 6948 | 1.3× | 0.1047 | 0.2810 | 2.7× |
| eightslab | 5163 | 6603 | 1.3× | 0.1754 | 0.5836 | 3.3× |
| six_ar_altsign_edge_chal | 5168 | 6168 | 1.2× | 0.1938 | 0.2060 | 1.1× |
| five_ar_scaled_same_input_chal | 5178 | 6169 | 1.2× | 0.0885 | 0.1927 | 2.2× |
| five_ar_indep_sumatend_chal | 5177 | 6166 | 1.2× | 0.3654 | 0.3434 | 0.9× |
| four_ar_indep_large_N_chal | 5200 | 6128 | 1.2× | 0.3973 | 0.3238 | 0.8× |
| five_ar_mixed_sign_edge_chal | 5166 | 5966 | 1.2× | 0.1712 | 0.1701 | 1.0× |
| seq_dep_chain5_edge_chal | 5166 | 5966 | 1.2× | 0.1909 | 0.1950 | 1.0× |
| four_ar_same_input_chal | 5178 | 5914 | 1.1× | 0.1134 | 0.1631 | 1.4× |
| four_ar_pow2_edge_chal | 5163 | 5764 | 1.1× | 0.1318 | 0.1725 | 1.3× |
| four_ar_N224_edge_chal | 5165 | 5765 | 1.1× | 0.1342 | 0.1508 | 1.1× |
| seq_dep_chain4_scaled_edge_chal | 5165 | 5765 | 1.1× | 0.1279 | 0.2790 | 2.2× |
| sequential_ar_chain_edge_chal | 5165 | 5361 | 1.0× | 0.2967 | 0.1094 | 0.4× |

### `noC` — 1 sim regressions

| problem | full sim (us) | arm sim (us) | sim ×worse | full RT (ms) | arm RT (ms) | RT ×worse |
|---------|--------------|--------------|-----------|--------------|-------------|-----------|
| three_group_dead_verify_chal | 5656 | 6419 | 1.1× | 0.4140 | 0.3738 | 0.9× |

## Takeaways

- **Every methodology group pays for itself in quality, and Group B (select/iterate) is load-bearing.** Disabling it is cheapest (0.53× tokens) but produces un-scored code on 20/58 problems and 21 sim regressions, several of which are 5–10× slower on real HW or fail to run at all.

- **Group A (research/ideation) matters on the hard many-AR problems**: 7 sim regressions, up to 3×, incl. HW slowdowns ~9× (e.g. `thirtysixinline`).

- **Group C (convergence check) and adversarial (`noadv`) are nearly free to remove on this anchor set** (1 and 1 sim regressions) — they buy robustness/generalization that these single-shot sim+RT metrics don't fully stress.

- The full methodology's extra cost concentrates on the many-collective / large-N problems where naive rewrites are wrong or slow; on simple single-step families all arms converge cheaply.
