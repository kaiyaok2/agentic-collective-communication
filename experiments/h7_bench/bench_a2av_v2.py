"""7-node HW microbench v2: AllToAllV.

Fix for the v1 bench harness's `xm.all_gather(unsqueeze(0))` shape-inference
regression on torch_xla 2.9: we wrap the baseline + agent calls inside a
torch.autograd.Function so the call happens in the same XLA context that
the training script uses (which is known to work; see
training/train_olmoe10b.py:_A2AV).
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 10
DM = 2048
N_ITER = 30
WARMUP = 5


def baseline_fn_inner(x, ws, mc):
    """Canonical AG + transpose + RS — matches train_olmoe10b.py baseline."""
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)        # (ws, mc*ws)
    reshaped = gathered.view(ws, ws, mc)                   # (src, dst, chunk)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0/ws, scatter_dim=0, shard_count=ws)


class _Bench(torch.autograd.Function):
    """Same autograd-Function wrapping that training uses, so XLA gives the
    same shape-inference treatment to the baseline path."""
    @staticmethod
    def forward(ctx, x, fn, ws, mc):
        ctx.fn, ctx.ws, ctx.mc = fn, ws, mc
        xm.mark_step()
        out = fn(x, ws, mc)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    mc = CAP * 2
    total = mc * ws

    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = x.sum().item()

    # Agent: pack+AG+slice (uses runtime/trainium_alltoallv_7node.py)
    from runtime.trainium_alltoallv_7node import alltoallv as agent_inner, init_alltoallv
    init_alltoallv()
    def agent_fn_inner(x, ws_, mc_):
        return agent_inner(x, ws_, mc_)

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} mc={mc} total/rank={total}')

    for label, inner in [('baseline', baseline_fn_inner), ('agent', agent_fn_inner)]:
        try:
            for _ in range(WARMUP):
                y = _Bench.apply(x, inner, ws, mc)
                _ = y.sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = _Bench.apply(x, inner, ws, mc)
                _ = y.sum().item()
                ts.append((time.time() - t0) * 1000)
            if rank == 0:
                med, mean = statistics.median(ts), statistics.mean(ts)
                print(f'[bench] a2av {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/a2av_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] a2av {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
