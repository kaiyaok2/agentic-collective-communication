# A-Q5: "Beyond pack-and-gather and mark_step lifting/bundling, how many qualitatively distinct strategies did the search find across the 8 problems?"

Source of truth: the deployed strategy-enumerate runtimes archived with the submission
(`r28_main_artifacts/runtime/trainium_<p>_7node.py`, byte-identical to the public artifact except
AllToAllV, see note 1) diffed against the developer baselines in `training/train_<p>_7node.py`
(`_baseline_*`) and the Llama e2e harness (`step_per_mb` vs `step_bundled`).

## Per-problem diff (what actually changed in the emitted code)

| Problem | Developer baseline (call level) | Deployed strategy (call level) | Collective dispatches / step | Mechanism(s) | Origin |
|---|---|---|---|---|---|
| AllToAllV | pack → `all_gather` (ws×ws×mc) → `permute(1,0,2).contiguous()` (strided HBM copy) → `reduce_scatter(SUM, scale=1/ws)` | pack → one `all_gather` → read own slot per source by offset math → `cat` | 2 → 1 | drop the RS (it is a routing no-op after AG: every rank holds identical copies, so RS∘AG = slice), drop the transpose copy, trace-time offset math | seed library (`ag_slice_cat`) |
| Uniform A2A | same AG+T+RS | one `all_gather` → contiguous 1-D slices `flat[src, rank*c:(rank+1)*c]` → `cat` | 2 → 1 | same as above; slices are contiguous so no strided-copy term | seed library (`slice_loop`) |
| Ring KV | one `all_gather` per (slot, head): KV×HEADS = 16 | one `all_gather` per slot: 2 | 16 → 2 | cross-head dispatch bundling (gather the whole slot tensor; layout = whole slot) | seed library (`per_slot_allgather`) |
| Distributed CE | `all_gather` of (T, V_local) logits along vocab → local `F.cross_entropy` | `all_reduce(MAX)` on (T,) → `all_reduce(SUM)` on a stacked (T, 2) = [Σexp(z−max), target-logit] | 1 AG (T×V) → 2 AR (T), (T×2) | algebraic recomposition (log-sum-exp with max-shift; in-shard target gather), collective TYPE change AG→AR, payload O(T·V) → O(T), 2-D stacking packs two SUM reductions into one dispatch | LLM (Phase 3); library's nearest seed (`async_back_to_back`) uses 3 ARs |
| PP cross-stage | per microbatch: zero (half, …) buffer, write own slot, masked `all_reduce` — M dispatches | one zero (half, M, …) buffer, write M slots, one masked `all_reduce`, unstack | M → 1 | cross-microbatch dispatch bundling by stacking; masked-AR as the only primitive that works where `collective_permute` fails at ws>64 | LLM (Phase 3) |
| TP MLP | per (m, L): `all_reduce(SUM)/ws` — M·L dispatches | `flatten`+`cat` all M·L partials → one `all_reduce` → slice/reshape back | M·L → 1 | cross-microbatch, cross-layer bundling by flatten+cat (heterogeneous shapes handled by offset bookkeeping) | LLM (Phase 3) |
| FSDP prefetch | per (m, L): `all_gather(dim=-1)`, per-microbatch `mark_step` | identical per-(m, L) `all_gather` loop (m-major); the stacked single-AG candidate failed the Phase-4 training-shape gate (HBM) | M·L → M·L (unchanged) | graph-boundary lifting only: per-microbatch `mark_step` removed → NEFF count M → 1; dispatch count unchanged | library fallback (`per_mb`) + harness-level `mark_step` placement |
| Layer-block AR | two `all_reduce(SUM)/ws` per layer (attn, mlp) | archived/deployed runtime is **byte-identical to the baseline** (two ARs); the measured amp3 gain comes from `step_bundled` (one graph for all microbatches) and, in the no-sim run, a stacked (2, …) single AR (`r28_nosim_train`, hash 8f7e002c) | 2 → 2 per call (per-problem harness); 2K → K (stacked variant) | `mark_step` lifting at e2e scope; stacking attn+mlp into one AR where used | library baseline (`sequential_2ar`); stacked variant LLM |

## Count of qualitatively distinct mechanisms

Across the 8 problems the emitted code uses **four** distinct mechanisms (five if the primitive
substitution is counted separately):

1. **Collective elimination by algebraic identity** — RS after AG is a routing no-op, so AG+RS → AG+slice
   (AllToAllV, Uniform A2A). *This is pack-and-gather.*
2. **Cross-call dispatch bundling** — stack / cat / flatten independent payloads into one tensor and issue
   one collective (PP: across microbatches; TP MLP: across microbatches × layers; Ring KV: across heads;
   Layer-block stacked variant: attn+mlp). Same mechanism, four instantiations.
3. **Algebraic recomposition that changes the collective type and payload volume** — distributed CE:
   replace the (T, V) all-gather + local softmax with max-shift log-sum-exp over two scalar-per-token
   all-reduces, with the two SUM reductions packed into one (T, 2) dispatch. The only problem where the
   *kind* of collective and the asymptotic payload change.
4. **Graph-boundary (`mark_step`) placement** — remove the per-microbatch graph cut so NeuronCC fuses
   M graphs into one (FSDP; Llama e2e `step_bundled`). Dispatch count unchanged; NEFF count M → 1.
   *This is mark_step lifting.*
5. (enabling choice, PP) **Primitive substitution under a hardware constraint** — masked `all_reduce`
   standing in for `collective_permute`, which does not work at ws>64; and avoidance of
   `xm.all_to_all`, which does not compile multi-node. Not a speed mechanism by itself.

So beyond pack-and-gather (#1) and mark_step lifting/bundling (#2, #4), there is exactly **one**
further qualitatively distinct strategy: the distributed-CE recomposition (#3). Mechanism #2 recurs
in four problems; the reviewer's framing is essentially right.

## What the emitted code does NOT do (checked)

- No changes to `xm.*` call parameters: `scale`, `groups`/replica groups, `pin_layout`, reduce op
  choice (other than MAX in dxe, which the baseline's `F.cross_entropy` implies anyway).
- No dtype or precision changes; no bucketing / chunk-size tuning (the `bucket_32mb` seed is a
  grad_ar template, not among the 8).
- No async/overlap or ordering tricks (Trainium exposes none above the API).
- No dead-code elimination beyond dropping the RS no-op; no caching of the `recv_counts` exchange
  (the `compute_recv_counts` all_gather is kept as a separate dispatch in AllToAllV).
- Pre/post-processing changes are limited to: removing the `permute().contiguous()` transpose copy
  (A2A pair), trace-time offset arithmetic, and zero-buffer + slot-write packing (PP). The vectorized
  `gathered[:, rank, :]` extraction exists only in the public artifact's drifted AllToAllV helper
  (note 1), not in what the paper measured.

## Origin: seed library vs LLM

| Winner came from | Problems |
|---|---|
| seed library template, selected by the simulator (no LLM needed in Phase 3; confirmed by the r36 "none" run: OLMoE 3125.8 ms vs paper 3139.8 ms) | AllToAllV, Uniform A2A, Ring KV, FSDP (fallback), Layer-block (deployed file) |
| LLM-authored in Phase 3 | Distributed CE (2-AR stacked), PP (stacked masked AR), TP MLP (flatten+cat single AR), Layer-block stacked variant (amp3 / no-sim run) |

Consequence for the rebuttal: the OLMoE headline (AllToAllV-dominated) is attributable to the
calibrated simulator ranking a known template correctly against the microbench ranking; the Llama
headline requires the LLM-authored bundling strategies (the no-LLM "none" run deploys the baseline on
all four Llama problems).

## Notes / artifact issues to fix

1. **Artifact drift (AllToAllV):** public `runtime/trainium_alltoallv_7node.py` replaces the measured
   loop+`cat` extraction in `alltoallv()` with a vectorized `view(ws, ws, mc)[:, rank, :]` (19-line
   diff vs the archived deployed runtime). Restore the measured version or label the change.
2. **Layer-block AR:** the deployed runtime file is the baseline; the paper's Layer-block numbers come
   from the amp3 e2e harness (Table 8 footnote already says so). State this explicitly and ship the
   stacked variant that was actually measured, or re-measure with the per-problem harness.
3. **FSDP:** dispatch count is identical to the baseline; the gain is graph-count (M → 1). Reviewer A's
   reading ("the baseline had a fixable inefficiency") is accurate and should be conceded.
4. TP MLP baseline divides by `ws` (mean) while the reference and the agent return the SUM; timing is
   unaffected (one elementwise op) but the harness baseline is not numerically the reference.
