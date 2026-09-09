"""Profile Llama-style training collectives to find improvement candidates
beyond the dispatch-bundling pattern.

Per-call probes on each collective category in a Llama training step:
  - TP attention out_proj AR (row-parallel)
  - TP MLP down_proj AR (row-parallel)
  - Distributed gradient norm AR (scalar)
  - Distributed RMSNorm sum-of-squares AR (small per-token scalar)
  - LM head logits AR for sampling (large)
  - Cross-rank token routing (a2a-style for MoE-less Llama: just permute)

For each, we time:
  - standalone per-call cost (in isolation)
  - in-graph cost (within a typical step that has compute before/after)

Output: identifies which patterns have the largest absolute cost AND
which look like multi-call structures the agent could bundle.
"""
import os, sys, time, json, statistics
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2048
HID = 5376
NUM_HEADS = 16
HEAD_DIM = DM // NUM_HEADS
N_LAYERS = 4
N_PARAMS = 100   # number of "param" tensors for grad-norm test
B = 1
S = 512
WARMUP = 5
N_ITER = 30


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    assert HID % ws == 0
    shard_hid = HID // ws
    shard_head = DM // ws  # for col-parallel projections

    # Static tensors per pattern.
    # TP attention: row-parallel out_proj input is partial (shard_head per rank).
    attn_partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
    # TP MLP: row-parallel down_proj input partial is same shape per rank
    mlp_partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
    # RMSNorm sum-of-squares: (B, S) per rank from local D
    rmsnorm_partial = torch.randn(B, S, device=dev, dtype=torch.float32)
    # Grad norm scalar: one float per "param"
    grad_norms = torch.randn(N_PARAMS, device=dev, dtype=torch.float32)
    # Vocab-parallel LM head: this rank holds (DM, V/ws) shard
    V_SHARD = 144  # ~ what OLMoE uses; V_total = 144 * 224 = 32256
    lm_head_shard = torch.randn(DM, V_SHARD, device=dev, dtype=torch.bfloat16) * 0.01

    primitive = sys.argv[1] if len(sys.argv) > 1 else 'tp_attn_ar'

    if primitive == 'tp_attn_ar':
        def fn():
            # Single row-parallel AR after attention out_proj
            return xm.all_reduce(xm.REDUCE_SUM, attn_partial)
    elif primitive == 'tp_mlp_ar':
        def fn():
            return xm.all_reduce(xm.REDUCE_SUM, mlp_partial)
    elif primitive == 'rmsnorm_ar':
        def fn():
            # AR per RMSNorm call; tiny payload
            return xm.all_reduce(xm.REDUCE_SUM, rmsnorm_partial)
    elif primitive == 'gradnorm_ar':
        def fn():
            # Sum of squares across N_PARAMS shards' norms
            return xm.all_reduce(xm.REDUCE_SUM, grad_norms)
    elif primitive == 'gradnorm_per_param':
        def fn():
            # The naive baseline: AR each param's norm separately (N AR calls)
            out = []
            for i in range(N_PARAMS):
                out.append(xm.all_reduce(xm.REDUCE_SUM, grad_norms[i:i+1]))
            return out
    elif primitive == 'lm_logits_ag':
        # Vocab-parallel LM head: compute local logits, AG across ranks for full vocab
        x = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
        def fn():
            logits_local = x @ lm_head_shard            # (B, S, V_SHARD)
            return xm.all_gather(logits_local, dim=-1)  # (B, S, V_total)
    elif primitive == 'layer_block_ar2':
        # Realistic Llama layer: attention AR + MLP AR (2 ARs per layer)
        def fn():
            a = xm.all_reduce(xm.REDUCE_SUM, attn_partial)
            m = xm.all_reduce(xm.REDUCE_SUM, mlp_partial)
            return a + m
    elif primitive == 'layer_block_fused':
        # Concat attention+MLP partials, single AR, split back
        def fn():
            stacked = torch.stack([attn_partial, mlp_partial], dim=0)
            ared = xm.all_reduce(xm.REDUCE_SUM, stacked)
            return ared[0] + ared[1]
    elif primitive == 'nlayer_per_layer_ar':
        # N_LAYERS layer blocks each doing attn_AR + mlp_AR (2N ARs in one graph;
        # XLA may fuse all to ~1, but we time)
        def fn():
            x = attn_partial
            for _ in range(N_LAYERS):
                x = x + xm.all_reduce(xm.REDUCE_SUM, attn_partial)
                x = x + xm.all_reduce(xm.REDUCE_SUM, mlp_partial)
            return x
    elif primitive == 'nlayer_bundled_ar':
        # Collapse all N_LAYERS * 2 ARs into 1 by stacking everything
        def fn():
            stacked = torch.stack([attn_partial, mlp_partial] * N_LAYERS, dim=0)
            ared = xm.all_reduce(xm.REDUCE_SUM, stacked)
            return ared.sum(dim=0)
    else:
        raise ValueError(primitive)

    if rank == 0:
        try:
            y = fn()
            shape = tuple(y.shape) if hasattr(y, 'shape') else type(y).__name__
            print(f'[init] ws={ws} primitive={primitive} output={shape}', flush=True)
        except Exception as e:
            print(f'[init] err: {e}', flush=True)

    for _ in range(WARMUP):
        y = fn()
        if isinstance(y, list):
            _ = y[-1].sum().item()
        else:
            _ = y.sum().item()
    if rank == 0:
        print('[init] warmup done', flush=True)

    times = []
    for _ in range(N_ITER):
        xm.mark_step()
        t0 = time.time()
        y = fn()
        if isinstance(y, list):
            _ = y[-1].sum().item()
        else:
            _ = y.sum().item()
        times.append((time.time() - t0) * 1000)

    if rank == 0:
        steady = times[5:] if len(times) > 5 else times
        print(f'[bench] {primitive} mean={statistics.mean(steady):.3f}ms '
              f'median={statistics.median(steady):.3f}ms', flush=True)
        with open(f'/tmp/tp_search/llama_profile_{primitive}.json', 'w') as f:
            json.dump({'primitive': primitive, 'mean_ms': statistics.mean(steady),
                       'med_ms': statistics.median(steady),
                       'all': times}, f)


if __name__ == '__main__':
    main()
