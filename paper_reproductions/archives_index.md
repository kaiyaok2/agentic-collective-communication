# Paper artifacts archive index

Generated: 2026-06-05T01:20:56Z

| Run dir | Archive | Size | SHA-256 | Used by |
|---------|---------|------|---------|---------|
| `r28` | `r28_main_paper_3styles.tar.gz` | 136K | `efb0a6499ca15847…` | Table 2, Table 3, Table 7 (3-style ablation: strategy-enum, cc-react, multi-island) |
| `r28_main_artifacts` | `r28_main_artifacts_runtimes.tar.gz` | 8.0K | `d669bfe81418fcbf…` | Strategy-enum deployed runtimes (the trainium_*_7node.py files referenced throughout the paper) |
| `r28_llama7b` | `r28_llama7b.tar.gz` | 12K | `8c6b4742ded6eacb…` | Table 5 (Llama-7B e2e) |
| `r28_ua2a_debug` | `r28_ua2a_debug.tar.gz` | 8.0K | `2de0d4213baac387…` | Table 2 Uniform A2A ‡ footnote (the rare ua2a_agent success at 2158 ms) |
| `r28d` | `r28d_rkv_retry.tar.gz` | 28K | `2759e52eafc260da…` | Ring KV retries supporting Table 2 ring_kv row |
| `r28e` | `r28e_lbar_extra.tar.gz` | 16K | `6ade4d3520bd30de…` | Layer-block AR extras supporting Table 2 lbar row |
| `r28_nosim` | `r28_nosim_early.tar.gz` | 24K | `68e6251d90ab09d3…` | Early no-sim run (superseded by r33; kept for provenance) |
| `r28_nosim_train` | `r28_nosim_train.tar.gz` | 308K | `1802aa58ae4bad26…` | Early no-sim training measurements (superseded by r33) |
| `r30_nosim_full` | `r30_nosim_full.tar.gz` | 56K | `bd3b6544b0d4e26e…` | Earlier no-sim full run, intermediate version of §7.2 |
| `r31_a2av_dxe` | `r31_a2av_dxe.tar.gz` | 16K | `ad70e218b76bcef0…` | alltoallv and dxe specific deployment retries |
| `r33_nosim_full_judge` | `r33_nosim_faithful.tar.gz` | 104K | `83547d46ec14893c…` | §7.2 no-sim ablation predecessor (acc-orphan-r33 commit version) |
| `r33b_retries` | `r33b_retries.tar.gz` | 12K | `48aa2c5bf562b7e0…` | r33 stage-2 SIGHUP/SIGABRT retries |
| `r35_h3_bench` | `r35_h3_bench_3node.tar.gz` | 20K | `f1d0f8499fc814eb…` | 3-node bench (acc-orphan-r34 section, removed in r35; kept for provenance) |
| `r36_nonllm_phase3` | `r36_nonllm_phase3.tar.gz` | 48K | `5b1c226fa4583f76…` | Non-LLM Phase-3 ablation (added then removed; kept for provenance) |
| `r36b_random` | `r36b_random.tar.gz` | 40K | `7d5ed5685408a58e…` | Non-LLM Phase-3 random style (same) |
| `r37_llama_ua2a` | `r37_llama_ua2a.tar.gz` | 16K | `27652fddac981d9d…` | Toy ua2a training + Llama-7B no-dxe retry |
| `r41_nosim_test` | `r41_nosim_iterative_ua2a_a2av.tar.gz` | 28K | `3e84f66b8480c04f…` | Iterative no-sim verification on ua2a + alltoallv (converged to AG+RS) |
| `r42_nosim_full` | `r42_nosim_full_iterative.tar.gz` | 96K | `0ed3a56b13ff9c6d…` | Iterative no-sim full 8-problem run + OLMoE e2e (pre-wrapper-fix) |
| `r43_alltoallv_dxe_baselinematch` | `r43_wrapper_fix_olmoe.tar.gz` | 32K | `dd226d323c7dd091…` | OLMoE e2e with AG+T+RS wrapper fix (alltoallv baseline-match → 1.0×) |
| `r44_dxe_iterate_bench` | `r44_dxe_1ag_bench.tar.gz` | 24K | `c828876fbb5e6311…` | Dxe full_vocab_ag convergence + bench_dxe 1n+7n refresh |

## Orchestration scripts

All r*.sh in /home/ubuntu/ copied to `paper_archives/orchestration_scripts/`.
(71 scripts archived.)
