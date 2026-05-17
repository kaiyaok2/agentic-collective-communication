"""7-node HW microbench: AllToAllV (variable). Compares developer baseline vs agent runtime."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 10
DM = 2048
N_ITER = 30
WARMUP = 5

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    mc = CAP * 2  # 2x expansion; matches OLMoE's per-(src,dst) capacity at training
    total = mc * ws  # per-rank input/output size for canonical packed-layout a2av

    # Random input
    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = x.sum().item()

    # Baseline: canonical AG + reshape + RS (2 dispatches)
    def baseline_fn(x):
        # x is 1D of (mc*ws,) — input has data for each dst rank, packed [d0|d1|...|d_{ws-1}]
        gathered = xm.all_gather(x.unsqueeze(0), dim=0).reshape(-1)  # 1D of ws*(mc*ws)
        gathered_3d = gathered.reshape(ws, ws, mc)
        transposed = gathered_3d.permute(1, 0, 2).contiguous().reshape(-1)
        return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                                 scale=1.0/ws, scatter_dim=0, shard_count=ws)

    # Agent: pack+AG+slice (uses runtime/trainium_alltoallv_7node.py)
    from runtime.trainium_alltoallv_7node import alltoallv as agent_fn, init_alltoallv
    init_alltoallv()
    # alltoallv(x, world_size, max_chunk) is the uniform variant.

    def call_agent(x):
        return agent_fn(x, ws, mc)

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} mc={mc} total/rank={total}')

    for label, fn in [('baseline', baseline_fn), ('agent', call_agent)]:
        try:
            for _ in range(WARMUP):
                # Use a fresh input shape suitable for each
                y = fn(x); _ = y.sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(x); _ = y.sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                print(f'[bench] {label:10s} n={N_ITER} med={statistics.median(ts):.3f}ms mean={statistics.mean(ts):.3f}ms')
                with open(f'/tmp/h7_bench/a2av_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': statistics.median(ts), 'mean_ms': statistics.mean(ts), 'all': ts}, f)
        except Exception as e:
            if rank == 0: print(f'[bench] {label} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
