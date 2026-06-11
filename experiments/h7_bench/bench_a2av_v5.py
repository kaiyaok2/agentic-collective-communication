"""7-node HW microbench v5: AllToAllV — per-microbatch AG+RS at per-mb payload size.

Baseline = M=4 small per-microbatch AG+RS calls (one per microbatch).
Agent   = 1 large bundled AG+RS on the M-stacked buffer.

Same total bytes moved as the prior R21 v3 (bundled). The new layout exposes the
M-small-NEFF vs 1-big-NEFF dispatch tax, restoring the cross-scope inversion shape.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 10
DM = 2048
M = 4
N_ITER = 30
WARMUP = 5


def baseline_fn(xs, ws, mc):
    """M small per-microbatch AG+RS dispatches in a Python loop."""
    outs = []
    for m in range(M):
        x = xs[m]
        out_buf = torch.zeros(ws * mc * ws, device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(out_buf, x)
        gathered_3d = out_buf.view(ws, ws, mc)
        transposed = gathered_3d.permute(1, 0, 2).contiguous().view(-1)
        rs_out = torch.zeros(mc * ws, device=x.device, dtype=x.dtype)
        dist.reduce_scatter_tensor(rs_out, transposed, op=dist.ReduceOp.SUM)
        outs.append(rs_out / ws)
    return outs


def agent_fn_inline(xs, ws, mc):
    """One big bundled AG+RS on the M-stacked buffer."""
    big = torch.cat(xs, dim=0)  # shape (M * mc * ws,)
    big_total = ws * (M * mc) * ws
    out_buf = torch.zeros(big_total, device=big.device, dtype=big.dtype)
    dist.all_gather_into_tensor(out_buf, big)
    gathered_3d = out_buf.view(ws, ws, M * mc)
    transposed = gathered_3d.permute(1, 0, 2).contiguous().view(-1)
    rs_out = torch.zeros(M * mc * ws, device=big.device, dtype=big.dtype)
    dist.reduce_scatter_tensor(rs_out, transposed, op=dist.ReduceOp.SUM)
    rs_out = rs_out / ws
    # Unstack back to M chunks of size (mc * ws,)
    outs = [rs_out[i * mc * ws:(i + 1) * mc * ws] for i in range(M)]
    return outs


def main():
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    mc = CAP * 2
    total = mc * ws

    xs = [(torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous() for _ in range(M)]
    xm.mark_step(); _ = xs[0].sum().item()

    agent_fn = None
    try:
        from runtime.trainium_alltoallv_7node import alltoallv as evolved, init_alltoallv
        init_alltoallv()
        # The evolved alltoallv operates on a single bundled buffer; we wrap by
        # stacking xs into the same big buffer the inline agent uses.
        def _agent():
            big = torch.cat(xs, dim=0)
            big_y = evolved(big, ws, M * mc)
            return [big_y[i * mc * ws:(i + 1) * mc * ws] for i in range(M)]
        agent_fn = _agent
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; using inline bundled AG+RS for agent')
        agent_fn = lambda: agent_fn_inline(xs, ws, mc)

    if rank == 0:
        print(f'[init] ws={ws} CAP={CAP} mc={mc} total/rank={total} M={M} (v5)')

    cases = [('baseline', lambda: baseline_fn(xs, ws, mc)),
             ('agent',    agent_fn)]
    for label, fn in cases:
        try:
            for _ in range(WARMUP):
                y = fn(); _ = y[0].sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0].sum().item()
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
