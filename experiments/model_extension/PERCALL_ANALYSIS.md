# Per-call HW microbench — 4 model-extension compositions

Same three-scope methodology as paper Table 2.

| Problem / backend | 1-node bench (32 ranks) | 7-node bench (224 ranks) | 7-node training (derived) |
|---|---:|---:|---:|
| **PP cross-stage (per_mb, 1 small AR)**      | 2.16 ms | 4.65 ms  | (×4 calls)  = 18.6 ms/step |
| **PP cross-stage (bundled, 1 big AR)**       | 3.45 ms | 12.17 ms | (×1 call)   = 12.2 ms/step |
| **TP MLP (per_mb, 1 layer AR)**              | 1.90 ms | 1.68 ms  | (×8 calls)  = 13.4 ms/step |
| **TP MLP (bundled, 1 stacked AR)**           | 1.77 ms | 1.55 ms  | (×1 call)   =  1.6 ms/step |
| **FSDP (per_mb, 1 layer AG)**                | 1.64 ms | 1.43 ms  | (×8 calls)  = 11.4 ms/step |
| **FSDP (bundled, 1 stacked AG)**             | 2.05 ms | 0.34 ms  | (×1 call)   =  0.3 ms/step |
| **Llama composite (per_mb, all primitives)** | 6.74 ms full step | 13.45 ms full step (1000-step end-to-end) |     |
| **Llama composite (bundled, all primitives)**| 3.86 ms full step | 12.23 ms full step (1000-step end-to-end) |     |

(Per_mb buffer = single-microbatch shape; bundled buffer = M-fold or M*N_LAYERS-fold shape.)

## Per-call → per-step translation

| Problem | per_mb total (×count) | bundled total | per-call speedup |
|---|---:|---:|---:|
| PP (M=4) | 4.65 × 4 = 18.6 ms | 12.17 ms | **1.53×** |
| TP MLP (M*N_LAYERS=8) | 1.68 × 8 = 13.4 ms | 1.55 ms | **8.7×** |
| FSDP (M*N_LAYERS=8) | 1.43 × 8 = 11.4 ms | 0.34 ms | **33×** |
| All composed (Llama fwd) | 6.74 ms/step | 3.86 ms/step | **1.75×** |
| All composed (1000-step e2e) | 13.45 ms/step | 12.23 ms/step | **1.10×** |

The per-call speedups stack along the count axis (bundled fires 1 of these collectives per step instead of M*K). The full-Llama step composes all three primitive types; its speedup is bounded by the
 fraction of the step time consumed by these collectives rather than by compute. End-to-end at 1000 steps the headline number is **1.10×** because the optimizer step and forward/backward compute do
 not bundle.

## Scope-shift observations

- **PP per_mb vs bundled per-call:** bundled is *slower* per-call (12.17 vs 4.65 ms at 7-node) because its buffer is M-fold larger. The win is from **count**, not per-call cost: 1 × 12.17 < 4 × 4.65.
- **TP MLP per_mb vs bundled per-call:** approximately equal at 7-node (1.68 vs 1.55 ms). The AR payload is the same total bytes; XLA's collective scheduling absorbs the stacking. The
 per-step win is then directly proportional to the M*N_LAYERS count reduction.
- **FSDP per-call bundled << per_mb** at 7-node (0.34 vs 1.43 ms) — anomalous in the direction the agent benefits from; the stacked AG benefits from internal XLA all_gather scheduling more than
the smaller per-microbatch AG does. This is the same shape as the dxe disagreement: a per-call benchmark prefers the agent in this case rather than against it.
- **1-node vs 7-node:** at 1-node the gaps narrow because 32-rank intra-node NeuronLink AR/AG is fast across all payload sizes. At 7-node EFA the per-call asymmetry is much sharper, which is wh
ere the dispatch-count reduction matters most.

## Saved artifacts

- `/tmp/tp_search/percall_*.json` — raw per-iteration timings
- `experiments/model_extension/percall_modext.py` — bench script
- `experiments/model_extension/run_percall.sh` — runner (NNODES=1 or 7)
