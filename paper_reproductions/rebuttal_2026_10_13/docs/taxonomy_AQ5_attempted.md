# A-Q5 supplement: strategies ATTEMPTED during Phase 3 (archival audit of 20fc5e3 archives, 2026-10-13)

Caveats: no Phase-3 candidate source is archived (only strategy name/80-char description, oracle verdict, sim, HW
verdict); the with-simulator main-paper Phase-3 logs (r20/r21/r22) are NOT in the archives — the 27 archived LLM Phase-3
logs are the --no-simulator ablation runs (same enumeration prompts, K=5, R=2). Counts below are therefore the attempted
population of the same procedure. 5 strategies were silently dropped (bf16-oracle/bench errors, phase3 lines 363-369).

## Funnel (27 runs x K=5 = 135 LLM implementations; 19-21 seed templates, 13 live)
| Problem | Runs | Enum | Oracle pass | Distinct named strategies | HW-gated / pass | Deployed-as-LLM-strategy | Main-paper deployed class |
|---|---|---|---|---|---|---|---|
| alltoallv | 4 | 20 | 6 | 10 | 5 / 2 (both slower than seed) | 0 | A pack+1AG+slice/cat |
| uniform_a2a | 3 | 15 | 8 | 8 | 5 / 1 (AG+T+RS 1045 vs 709 ms) | 0 | A 1AG+per-source slice loop (= seed slice_loop) |
| ring_kv | 3 | 15 | 12 | 7 | 6 / 4 | 1 | I per-slot AG (= seed) |
| dxe | 5 | 25 | 16 | 9 | 7 / 6 | 2 | B/C MAX-AR + stacked (T,2) SUM-AR |
| pp_send_recv | 3 | 15 | 9 | 8 | 5 / 5 | 3 | B stacked masked single AR |
| tp_mlp | 3 | 15 | 14 | 9 | 7 / 7 | 3 | B flatten -> 1 AR -> reshape |
| fsdp_prefetch | 3 | 15 | 12 | 7 | 6 / 0 (harness TypeError bug) | 1 | I per-mb loop (mark_step lifted at integration) |
| llama_block_ar | 3 | 15 | 12 | 7 | 6 / 6 | 1 | I sequential 2xAR (= seed) |
| Total | 27 | 135 | 89 (66%) | ~65 names -> 10 classes | 47 / 31 | 11 | 8 runtimes, 4 shapes |

## Strategy classes attempted (A-K), with where each died
| # | Class | Attempted (problem x count) | Oracle | HW | Deployed? | Why not |
|---|---|---|---|---|---|---|
| A | Pack-and-gather (1 AG of packed buffer + local slice/cat) | a2av 4, ua2a 1 | 5/5 | 2 pass / 3 IndexError at 224 | YES (a2av, ua2a) | |
| B | Cross-call bundling stack/cat/flatten -> single collective | pp 3, tp 2, lbar 7, rkv 4, dxe 5, fsdp 3 | 21/24 | pp/tp/dxe/lbar pass; rkv stacked shape-fail | YES (pp, tp, dxe) | lbar stacked passes but main deployed sequential; rkv concat-AG slower in sim |
| C | Algebraic recomposition changing collective type/payload (AG+T+RS, RS->AG, AG-of-logits, RS-partial-loss) | a2av 7, ua2a 3, rkv 1, dxe 9, pp 3, lbar 3 | 10/26 | a2av/ua2a shape-fail at 224 or 4x slower | dxe only | mostly oracle failures (max_diff 13.99; MockTorch crashes on F.cross_entropy) |
| D | Hierarchical intra-/inter-node two-level | a2av 4, ua2a 3, rkv 3, dxe 2 | 0/12 | never | no | ZeroDivisionError (sandbox num_nodes=1) / bad shapes |
| E | Ring / butterfly / Bruck / recursive-halving permutation schedules | a2av 7, ua2a 2, pp 1 | 0/10 | never | no | wrong results or collective_permute invalid at ws>64 |
| F | Bucketing / chunking (sqrt-M groups, K-chunks, per-layer grouping) | tp 7, pp 3, fsdp 5, rkv 4, ua2a 1 | 19/20 | tp/pp pass; rkv TV fail; ua2a shape fail | no | sim always between full bundle and per-mb |
| G | Async / back-to-back / pipelined / sliding-window ordering | dxe 4, lbar 3, tp 2, fsdp 3, pp 1, rkv 1 | 11/14 | pass where gated | no | slower in sim; async APIs missing in MockXM |
| H | Substitute P2P send/recv or all_to_all primitive | pp 4 | 0/4 | never | no | no P2P in sandbox; all_to_all unsupported -> masked AR (B) is the deployed substitute |
| I | Baseline re-emission (per-slot AG, sequential 2AR, per-mb loop, three-AR) | rkv 3, lbar 3, tp 1, fsdp 3, dxe 3 | 13/13 | pass | YES (rkv, fsdp, lbar) | |
| J | AG + flat index_select extraction | ua2a 3 | 3/3 | IndexError at 224 | no | index OOB at 224 ranks; sim 10837 vs 6804 |
| K | Graph-boundary / mark_step lifting | 0 enumerated | | | implicit (fsdp integration) | never proposed by the LLM |

Answer: 10 qualitatively distinct classes attempted (~65 named sketches, 135 implementations); 4 shapes deployed.
6 classes never deployed: D, E, H died 100% at the correctness oracle (26/26); C mostly at the oracle (16/26), survivors
lost on sim or the 224-rank shape gate; F, G, J passed the oracle but were slower in sim (F, G) or failed the 224-rank
index/shape gate (J). LLM budget: 1 enumeration + 5 implementation calls per run (+ 70 refinement calls in iterative
runs) = ~232 LLM Phase-3 calls across the archives; 170 LLM-generated candidates vs 19-21 seed templates.

Discrepancies to handle carefully: appendix claims the fsdp stacked candidate failed the HBM gate — archived runs show a
harness TypeError for all fsdp candidates instead; r33 deployed a ring_kv candidate despite 3 failed training validations
(Phase 5 did not fall back); the a2av main runtime is behaviorally the seed ag_slice_cat (453 local ops), not the
index_select form the appendix describes; 3 main runtimes carry r28_nosim strategy docstrings verbatim.
