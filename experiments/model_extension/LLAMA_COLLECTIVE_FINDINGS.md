# 2-hour Llama collective profile — findings

## Methodology
Profiled 10 Llama-training collective patterns on the 7-node 224-rank
trn1.32xlarge cluster. Each is a single isolated call (or call-group)
timed with WARMUP=5 and N_ITER=30 iterations, mean over the last 25.

## Results

| Primitive | mean (ms) |
|---|---:|
| tp_attn_ar (AR after attention out_proj) | 2.81 |
| tp_mlp_ar (AR after MLP down_proj)       | 2.71 |
| rmsnorm_ar (small per-token sum-of-squares AR) | 3.00 |
| gradnorm_ar (single AR of (100,) tensor) | 2.49 |
| gradnorm_per_param (100 separate ARs, same data) | 2.26 |
| lm_logits_ag (AG of vocab-parallel logits) | 1.15 |
| layer_block_ar2 (2 sequential ARs in one graph) | **3.13** |
| layer_block_fused (1 AR of stacked (2,B,S,DM)) | **2.38** |
| nlayer_per_layer_ar (8 ARs same data, in graph) | 1.95 |
| nlayer_bundled_ar (1 stacked AR, 8x bytes) | 2.87 |

## Findings

**1. Within-layer attention+MLP AR fusion (1.32x opportunity).** In a
   standard Llama transformer block, the row-parallel attention out_proj
   fires one AR (post-attention), and the row-parallel MLP down_proj
   fires another (post-MLP). Both live in the same `mark_step` graph
   when activation checkpointing is not used, but Neuron does NOT
   auto-fuse them despite being independent of each other's input:

   - 2 sequential ARs: 3.13 ms
   - 1 stacked AR over (2, B, S, DM): 2.38 ms
   - **1.32x speedup**

   **Catch:** the standard transformer block has a data dependency
   attention -> residual+norm -> MLP, so the MLP input depends on the
   attention AR output. To exploit the fusion the agent must discover
   the **parallel-attention-MLP** architectural rewrite (used in
   PaLM, GPT-J): attention and MLP both consume `x` directly and
   contribute additively to the residual, so their partials can be
   AR'd together. This is an architectural change with a
   well-known mathematical equivalence (PaLM showed comparable
   training quality), not a pure primitive composition.

**2. XLA already fuses identical-tensor ARs (CSE).** The
   `nlayer_per_layer_ar` test fires 8 ARs of the same tensor and runs
   in 1.95 ms — faster than 1 stacked AR over the same data (2.87 ms).
   The compiler recognises the duplicate inputs and dedups. This means
   for repeated ARs of the SAME tensor (e.g. broadcast scenarios) the
   agent has no headroom — XLA is already optimal.

**3. AR latency floor dominates small payloads.** RMSNorm (3.00 ms),
   gradnorm scalar (2.49 ms), and LM logits AG (1.15 ms) are all
   close to the ~1-3 ms per-call Trainium floor regardless of bytes.
   Bandwidth-side compositions have no headroom here.

**4. Per-param vs single grad-norm AR.** Multiple per-param ARs of
   the same scalar tensor (2.26 ms) is faster than one AR of (100,)
   (2.49 ms). This is the CSE pattern again: when params have the
   same scalar value, XLA collapses 100 ARs to 1. In real training
   the per-param scalars differ so CSE won't trigger; the practical
   pattern is one AR of the concatenated norm-squareds.

## Candidate problem for the agent

**Llama transformer block: 2-AR-per-layer fusion via parallel
attention+MLP.**

Reference implementation (developer-written baseline):
```python
def llama_block(x, attn_weights, mlp_weights):
    h_attn = attention(norm(x), attn_weights)        # partial
    attn = xm.all_reduce(REDUCE_SUM, h_attn)          # AR #1
    x = x + attn
    h_mlp = mlp(norm(x), mlp_weights)                # partial
    mlp = xm.all_reduce(REDUCE_SUM, h_mlp)            # AR #2
    return x + mlp
```

Agent's composition (parallel attention+MLP, single AR):
```python
def llama_block_parallel(x, attn_weights, mlp_weights):
    x_norm = norm(x)
    h_attn = attention(x_norm, attn_weights)         # partial
    h_mlp  = mlp(x_norm, mlp_weights)                # partial (same input!)
    stacked = torch.stack([h_attn, h_mlp], dim=0)    # (2, B, S, DM)
    ared = xm.all_reduce(REDUCE_SUM, stacked)        # ONE AR
    return x + ared[0] + ared[1]
```

Trade-off: mathematically different (skips inter-layer attn/MLP
sequential dependency), known to be acceptable training-quality wise
in PaLM-class models, hard requirement is the user's willingness to
adopt parallel-attention block architecture.

## Saved artifacts

- `/tmp/tp_search/profile_llama_collectives.py` (10-primitive bench)
- `/tmp/tp_search/llama_profile_*.json` (raw measurements)
- `experiments/model_extension/LLAMA_COLLECTIVE_FINDINGS.md`
   (this file, after copy)
