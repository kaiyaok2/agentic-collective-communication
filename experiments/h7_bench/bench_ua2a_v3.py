"""7-node HW microbench v3: Uniform AllToAll — uses torch.distributed primitives
to bypass the xm.all_gather shape-inference regression."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 13
DM = 2048
N_ITER = 30
WARMUP = 5


def baseline_fn(x, ws, chunk_size):
    """AG + transpose + RS using torch.distributed; matches the semantic of
    training/train_uniform_a2a_7node.py:_ua2a_baseline."""
    out_buf = torch.zeros(ws * chunk_size * ws, device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(out_buf, x)
    gathered_3d = out_buf.view(ws, ws, chunk_size)   # (src, dst, chunk)
    transposed = gathered_3d.permute(1, 0, 2).contiguous().view(-1)
    rs_out = torch.zeros(chunk_size * ws, device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(rs_out, transposed, op=dist.ReduceOp.SUM)
    return rs_out / ws


def main():
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    chunk_size = CAP * DM
    total = chunk_size * ws

    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = x.sum().item()

    from runtime.trainium_uniform_a2a import uniform_a2a as agent_fn_raw, init_uniform_a2a
    init_uniform_a2a()
    agent_fn = lambda x: agent_fn_raw(x, chunk_size)

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} DM={DM} chunk={chunk_size} total/rank={total}')

    for label, fn in [('baseline', lambda x: baseline_fn(x, ws, chunk_size)),
                      ('agent',    agent_fn)]:
        try:
            for _ in range(WARMUP):
                y = fn(x); _ = y.sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(x); _ = y.sum().item()
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
