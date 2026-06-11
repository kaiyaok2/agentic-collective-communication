"""7-node HW microbench v3: AllToAllV — uses torch.distributed.all_gather_into_tensor
to bypass the xm.all_gather(unsqueeze(0), dim=0) shape-inference regression on
torch_xla 2.9. The output tensor is allocated explicitly with the expected shape,
removing XLA's shape-inference DOF that caused the v1/v2 failures."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 10
DM = 2048
N_ITER = 30
WARMUP = 5


def baseline_fn(x, ws, mc):
    """Canonical AG + reshape + RS, but using torch.distributed.all_gather_into_tensor
    with an explicitly-allocated output."""
    # all_gather: concatenate along dim 0
    out_buf = torch.zeros(ws * mc * ws, device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(out_buf, x)
    gathered_3d = out_buf.view(ws, ws, mc)               # (src, dst, chunk)
    transposed = gathered_3d.permute(1, 0, 2).contiguous().view(-1)
    # reduce_scatter via torch.distributed
    rs_out = torch.zeros(mc * ws, device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(rs_out, transposed, op=dist.ReduceOp.SUM)
    return rs_out / ws


def main():
    # Ensure process group is initialized (torchrun usually does this; reinit if needed)
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    mc = CAP * 2
    total = mc * ws

    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = x.sum().item()

    from runtime.trainium_alltoallv_7node import alltoallv as agent_fn, init_alltoallv
    init_alltoallv()

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} mc={mc} total/rank={total}')

    for label, fn in [('baseline', lambda x: baseline_fn(x, ws, mc)),
                      ('agent',    lambda x: agent_fn(x, ws, mc))]:
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
                print(f'[bench] a2av {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/a2av_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] a2av {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
