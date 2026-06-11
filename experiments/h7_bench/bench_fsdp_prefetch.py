"""7-node HW microbench: FSDP weight prefetch (smaller shape)."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

def fsdp_per_mb(shards, M, N_LAYERS, rank, world_size, xm, torch):
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_gather(shards[m][L], dim=-1))
        outs.append(per_layer)
    return outs

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    M = 4; N_LAYERS = 2
    shard_size = 64
    DM = 256

    shards = [[torch.randn(shard_size, DM, device=dev, dtype=torch.bfloat16) for L in range(N_LAYERS)] for m in range(M)]
    xm.mark_step(); _ = shards[0][0].sum().item()

    agent_fn = None
    try:
        from runtime.trainium_fsdp_prefetch_7node import evolved_fsdp_prefetch as agent_fn
    except Exception as e:
        if rank == 0: print(f'[init] no agent: {e}')

    if rank == 0:
        print(f'[init] ws={ws} M={M} N_LAYERS={N_LAYERS} shard={shard_size} DM={DM}')

    cases = [('baseline', lambda: fsdp_per_mb(shards, M, N_LAYERS, rank, ws, xm, torch))]
    if agent_fn is not None:
        cases.append(('agent', lambda: agent_fn(shards, M, N_LAYERS, rank, ws, 16, 2, xm, torch, num_nodes=7)))

    for label, fn in cases:
        try:
            for _ in range(WARMUP):
                y = fn(); _ = y[0][0].sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0][0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                med, mean = statistics.median(ts), statistics.mean(ts)
                print(f'[bench] fsdp_prefetch {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/fsdp_prefetch_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] fsdp_prefetch {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
