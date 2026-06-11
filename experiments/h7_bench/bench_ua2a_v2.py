"""7-node HW microbench v2: Uniform AllToAll. autograd-Function-wrapped to dodge
the xm.all_gather(unsqueeze(0)) shape-inference regression seen in v1."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 13
DM = 2048
N_ITER = 30
WARMUP = 5


def baseline_inner(x, ws, chunk_size):
    """Mirrors training/train_uniform_a2a_7node.py:_ua2a_baseline (known good)."""
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, chunk_size)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0/ws, scatter_dim=0, shard_count=ws)


class _Bench(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn, ws, chunk):
        ctx.fn, ctx.ws, ctx.chunk = fn, ws, chunk
        xm.mark_step()
        out = fn(x, ws, chunk)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    chunk_size = CAP * DM
    total = chunk_size * ws

    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = x.sum().item()

    from runtime.trainium_uniform_a2a import uniform_a2a as agent_inner_raw, init_uniform_a2a
    init_uniform_a2a()
    def agent_inner(x, ws_, cs_):
        return agent_inner_raw(x, cs_)

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} DM={DM} chunk={chunk_size} total/rank={total}')

    for label, inner in [('baseline', baseline_inner), ('agent', agent_inner)]:
        try:
            for _ in range(WARMUP):
                y = _Bench.apply(x, inner, ws, chunk_size)
                _ = y.sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = _Bench.apply(x, inner, ws, chunk_size)
                _ = y.sum().item()
                ts.append((time.time() - t0) * 1000)
            if rank == 0:
                med, mean = statistics.median(ts), statistics.mean(ts)
                print(f'[bench] ua2a {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/ua2a_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] ua2a {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
