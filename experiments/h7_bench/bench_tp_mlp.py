"""7-node HW microbench: TP MLP (smaller shape: M=4 N_LAYERS=2 B=1 S=128 DM=256)."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

def tp_per_mb(partials, M, N_LAYERS, rank, world_size, xm, torch):
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_reduce(xm.REDUCE_SUM, partials[m][L]))
        outs.append(per_layer)
    return outs

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    M = 4; N_LAYERS = 2; B = 1; S = 128; DM = 256

    partials = [[torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) for L in range(N_LAYERS)] for m in range(M)]
    xm.mark_step(); _ = partials[0][0].sum().item()

    agent_fn = None
    try:
        from runtime.trainium_tp_mlp_7node import evolved_tp_mlp as agent_fn
    except Exception as e:
        if rank == 0: print(f'[init] no agent: {e}')

    if rank == 0:
        print(f'[init] ws={ws} M={M} N_LAYERS={N_LAYERS} S={S} DM={DM}')

    cases = [('baseline', lambda: tp_per_mb(partials, M, N_LAYERS, rank, ws, xm, torch))]
    if agent_fn is not None:
        cases.append(('agent', lambda: agent_fn(partials, M, N_LAYERS, rank, ws, 16, 2, xm, torch, num_nodes=7)))

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
                print(f'[bench] tp_mlp {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/tp_mlp_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] tp_mlp {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
