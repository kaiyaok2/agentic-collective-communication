"""7-node HW microbench v5: Llama block AR (attn + mlp) — restores cross-scope inversion.

Baseline = 2 small per-block all_reduce calls (attn AR + mlp AR) in sequence.
Agent   = 1 large bundled all_reduce on the stacked (attn, mlp) buffer.

Shapes from Llama amp3: B=1, S=2048, DM=2048.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

B = 1
S = 2048
DM = 2048

def baseline_fn(attn_partial, mlp_partial):
    """2 small sequential ARs (the developer convention)."""
    a = xm.all_reduce(xm.REDUCE_SUM, attn_partial)
    m = xm.all_reduce(xm.REDUCE_SUM, mlp_partial)
    return a, m

def agent_fn_inline(attn_partial, mlp_partial):
    """1 big bundled AR on the stacked (2, B, S, DM) buffer."""
    big = torch.stack([attn_partial, mlp_partial], dim=0)
    big_red = xm.all_reduce(xm.REDUCE_SUM, big)
    return big_red[0], big_red[1]

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))

    attn_partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16)
    mlp_partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16)
    xm.mark_step(); _ = attn_partial.sum().item()

    agent_fn = None
    try:
        from runtime.trainium_llama_block_ar_7node import evolved_llama_block as evolved
        agent_fn = lambda: evolved(attn_partial, mlp_partial, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; using inline bundled AR for agent')
        agent_fn = lambda: agent_fn_inline(attn_partial, mlp_partial)

    if rank == 0:
        print(f'[init] ws={ws} S={S} DM={DM} (v5 cross-scope-inversion shapes)')

    cases = [('baseline', lambda: baseline_fn(attn_partial, mlp_partial)),
             ('agent',    agent_fn)]
    for label, fn in cases:
        try:
            for _ in range(WARMUP):
                y = fn(); _ = y[0].sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                med, mean = statistics.median(ts), statistics.mean(ts)
                print(f'[bench] llama_block_ar {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/llama_block_ar_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] llama_block_ar {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
