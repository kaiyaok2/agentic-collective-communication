"""7-node HW microbench: Llama block AR fusion (smaller shape)."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

def block_sequential_2ar(attn_partial, mlp_partial, rank, world_size, xm, torch):
    a = xm.all_reduce(xm.REDUCE_SUM, attn_partial)
    m = xm.all_reduce(xm.REDUCE_SUM, mlp_partial)
    return a, m

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    B = 1; S = 128; DM = 256

    attn_partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16)
    mlp_partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16)
    xm.mark_step(); _ = attn_partial.sum().item()

    agent_fn = None
    try:
        from runtime.trainium_llama_block_ar_7node import evolved_llama_block as agent_fn
    except Exception as e:
        if rank == 0: print(f'[init] no agent: {e}')

    if rank == 0:
        print(f'[init] ws={ws} S={S} DM={DM}')

    cases = [('baseline', lambda: block_sequential_2ar(attn_partial, mlp_partial, rank, ws, xm, torch))]
    if agent_fn is not None:
        cases.append(('agent', lambda: agent_fn(attn_partial, mlp_partial, rank, ws, 16, 2, xm, torch, num_nodes=7)))

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
