"""7-node + 1-node HW microbench v7: FSDP weight prefetch — developer-faithful with N_LAYERS=4.

Same cold-first-iter framework as v6 but N_LAYERS=4 (matches Llama amp1 N_LAYERS_PER_STAGE)
so baseline's M*N_LAYERS = 16 small calls accumulate more per-call overhead, making the
bundled agent's big-NEFF compile cost relatively more visible.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 3
WARMUP = 0

M = 4
N_LAYERS = 4
DM = 2048
HID = 5376
SHARD_SIZE = HID // 32

def baseline_fn(shards):
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_gather(shards[m][L], dim=-1))
        outs.append(per_layer)
    return outs

def agent_fn_inline(shards):
    big = torch.stack([torch.stack(layer_list, dim=0) for layer_list in shards], dim=0)
    big_ag = xm.all_gather(big, dim=-1)
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(big_ag[m, L])
        outs.append(per_layer)
    return outs

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    shards = [[torch.randn(SHARD_SIZE, DM, device=dev, dtype=torch.bfloat16) for L in range(N_LAYERS)] for m in range(M)]
    xm.mark_step()
    agent_fn = None
    try:
        from runtime.trainium_fsdp_prefetch_7node import evolved_fsdp_prefetch as evolved
        agent_fn = lambda: evolved(shards, M, N_LAYERS, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; inline bundled AG')
        agent_fn = lambda: agent_fn_inline(shards)
    if rank == 0: print(f'[init] fsdp_prefetch v7 ws={ws} M={M} N_LAYERS={N_LAYERS} SHARD={SHARD_SIZE} DM={DM} N_ITER={N_ITER} WARMUP={WARMUP}')
    cases = [('baseline', lambda: baseline_fn(shards)),
             ('agent',    agent_fn)]
    bench_dir = os.environ.get('BENCH_OUT', '/tmp/h7_bench')
    os.makedirs(bench_dir, exist_ok=True)
    for label, fn in cases:
        try:
            ts = []
            for i in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0][0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                cold = ts[0]
                warm_med = statistics.median(ts[1:]) if len(ts) > 1 else cold
                print(f'[bench] fsdp_prefetch {label:10s} cold={cold:.3f}ms warm_med={warm_med:.3f}ms')
                with open(f'{bench_dir}/fsdp_prefetch_{label}.json', 'w') as f:
                    json.dump({'label': label, 'cold_ms': cold, 'warm_med_ms': warm_med, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] fsdp_prefetch {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
