"""7-node HW microbench: Uniform AllToAll. Compares developer baseline vs agent runtime."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 13
DM = 2048
N_ITER = 30
WARMUP = 5

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    chunk_size = CAP * DM
    total = chunk_size * ws

    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = x.sum().item()

    # Baseline: AG + transpose + RS (matches train_uniform_a2a_7node.py baseline path)
    def baseline_fn(x):
        gathered = xm.all_gather(x.unsqueeze(0), dim=0).reshape(-1)
        gathered_3d = gathered.reshape(ws, ws, chunk_size)
        transposed = gathered_3d.permute(1, 0, 2).contiguous().reshape(-1)
        return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                                 scale=1.0/ws, scatter_dim=0, shard_count=ws)

    # Agent: from runtime/trainium_uniform_a2a.py
    from runtime.trainium_uniform_a2a import uniform_a2a as agent_fn, init_uniform_a2a
    init_uniform_a2a()
    def call_agent(x):
        return agent_fn(x, chunk_size)

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} DM={DM} chunk={chunk_size} total/rank={total}')

    for label, fn in [('baseline', baseline_fn), ('agent', call_agent)]:
        try:
            for _ in range(WARMUP):
                y = fn(x); _ = y.sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(x); _ = y.sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                print(f'[bench] {label:10s} n={N_ITER} med={statistics.median(ts):.3f}ms mean={statistics.mean(ts):.3f}ms')
                with open(f'/tmp/h7_bench/ua2a_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': statistics.median(ts), 'mean_ms': statistics.mean(ts), 'all': ts}, f)
        except Exception as e:
            if rank == 0: print(f'[bench] {label} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
