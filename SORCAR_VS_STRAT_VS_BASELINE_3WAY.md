# Sorcar vs Strat vs Baseline — Canonical Three-Way Results

This is the single source of truth for the three-way comparison. Every
number below is measured on 7× trn1.32xlarge (224 NeuronCores),
us-east-1c, and traces to a committed data file.

## The three code paths

| Path | What it is | Collective schedule |
|---|---|---|
| **baseline** | naive textbook PyTorch-DDP source | one collective per logical op |
| **strat** | OverlayCCL 5-strategy enumeration output | *distinct source*, but == baseline's schedule on the divergent set |
| **sorcar** | searched family rewrites (F1–F7) | fused / eliminated collectives |

The key empirical finding: **strat emits different source code from
baseline** (accumulate loops, per-tensor AR loops, re-bucketed payloads)
**but on the problems where the families apply, it reaches baseline's
collective schedule** — its cost model scores collective structure, not
collective algebra, so it never proves the fusions. That is why
`strat_ms ≈ baseline_ms` everywhere below.

## Anchor set: 55 divergent problems (not the full 143 pool)

Of the 143-problem taxonomy pool, exactly **55** are *divergent* — the
sorcar rewrite beats strat by >5% in the calibrated simulator. The other
88 are ties (strat already reaches the optimal schedule, or both sit at
the dispatch floor) and are excluded from the anchor set. There are
**zero** problems where strat beats sorcar.

Family distribution of the 55 (see `SORCAR_FAMILY_TAXONOMY.md`):

| Family | Anchors | What diverges |
|---|---|---|
| F1 Sequential-AR linearity | 39 | k linearly-combined ARs → 1 |
| F2 CSE of same-input ARs | 7 | N identical ARs → 1 |
| F4 Per-row/col/batch dispatch collapse | 5 | M per-slice ARs → 1 |
| F3 Dead-collective / algebraic zero | 2 | telescoping sum → 0 collectives |
| F6 Mixed-reduction-op extraction | 1 | SUM + MAX/MIN de-duplicated |
| F7 Slab/chunk payload fusion | 1 | per-slab AR loop → 1 |

(F5, collective-type conversion / ZeRO-1 data-flow narrowing, is
exercised only in the E2E optimizer path, not as a standalone anchor.)

Warm-cache RT verification of the 55 (`taxonomy_3col_results/RT_THREE_COL_RESULTS.json`):
**45 RT-confirmed Sorcar wins ≥1.05×**, 8 at the RT dispatch floor
(sim divergence below RT noise), 2 F3 total-cancel HW-aborts (sorcar
sim-passes; reduce_scatter/224 edge on hardware). On all 55,
`strat_ms ≈ baseline_ms`.

## End-to-end 10B training: three-way (`ablation_results.jsonl`)

224 ranks, 48 layers, DM=4096, SEQ=128, N_MB=16, 12 steps, seed 42,
median warm ms/step:

| Model | baseline | strat | sorcar | **sorcar/strat** | sorcar/baseline | strat/baseline |
|---|---|---|---|---|---|---|
| **Llama-10B + F4b×F5 fuse** | 21905.9 | 22032.5 | 8966.7 | **2.457×** | 2.443× | 1.006× |
| **GPT-10B + F4b×F5 fuse** | 21541.3 | 21467.9 | 10006.6 | **2.145×** | 2.153× | 0.997× |
| Llama-10B (unfused) | 21864.8 | 21921.0 | 9717.5 | **2.256×** | 2.250× | 1.003× |
| Llama-10B L24 (fused) | 10181.4 | 10156.8 | 3908.5 | **2.599×** | 2.605× | 0.998× |

Both 10B architectures clear **2× over strat**, using multiple divergent
family sites (F1, F2, F3, F4, F6, F7) drawn from the 55-anchor set in a
single natural training step (see `SORCAR_E2E_10B_TP.md` for the
site-by-site family map).

**Loss confirms strat == baseline schedule**: strat's final loss is
bit-identical to baseline (llama 3.3128, gpt 6.2141, L24 2.6268) — its
distinct source compiles to baseline's exact reduction order. Sorcar's
loss tracks baseline within fp reduction-order noise.

## Reproduce

```
# three-way E2E (writes ablation_results.jsonl)
bash run_ablations.sh        # baseline + sorcar, 8 configs
bash run_3way_strat.sh       # strat, 4 configs (appends)
# --backend selects the path; all three share the same model + data
torchrun ... train_llama10b_tp_families.py --backend {baseline,strat,sorcar} --nmb 16 --fuse
```
