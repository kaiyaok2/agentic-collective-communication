"""7-node + 1-node HW microbench v6 (FIXED): AllToAllV — cold-NEFF developer-faithful.

Uses torch.distributed.all_gather_into_tensor + reduce_scatter_tensor for the
baseline (bypasses an xm.all_gather shape-inference regression that caused
the previous v6 to crash). Methodology unchanged: WARMUP=0, report ts[0] cold.
"""
import os, sys, time, json, statistics
sys.path.insert(0, "/home/ubuntu/agentic-collective-communication")
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

CAP = 10
DM = 2048
N_ITER = 3
WARMUP = 0

def baseline_fn(x, ws, mc):
    # torch.distributed-based AG + RS to bypass xm.all_gather regression
    gathered = torch.empty(ws * x.numel(), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(gathered, x)
    reshaped = gathered.view(ws, ws, mc).permute(1, 0, 2).contiguous().view(-1)
    out = torch.empty(reshaped.numel() // ws, dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, reshaped, op=dist.ReduceOp.SUM)
    return out / ws

def main():
    if not dist.is_initialized():
        dist.init_process_group("xla")
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
    mc = CAP * 2
    total = mc * ws
    x = (torch.randn(total, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step()
    agent_fn = None
    try:
        from runtime.trainium_alltoallv_7node import alltoallv, init_alltoallv
        init_alltoallv()
        agent_fn = lambda: alltoallv(x, ws, mc)
    except Exception as e:
        if rank == 0: print(f"[init] no evolved agent (7node): {e}; trying 1-node fallback")
        try:
            from runtime.trainium_alltoallv import alltoallv as a1, init_alltoallv as i1
            i1()
            agent_fn = lambda: a1(x, ws, mc)
        except Exception as e2:
            if rank == 0: print(f"[init] no agent at all: {e2}; using inline baseline as agent")
            agent_fn = lambda: baseline_fn(x, ws, mc)
    if rank == 0:
        print(f"[init] a2av v6 ws={ws} CAP={CAP} mc={mc} total/rank={total} N_ITER={N_ITER} WARMUP={WARMUP}")
    cases = [("baseline", lambda: baseline_fn(x, ws, mc)), ("agent", agent_fn)]
    bench_dir = os.environ.get("BENCH_OUT", "/tmp/h7_bench")
    os.makedirs(bench_dir, exist_ok=True)
    for label, fn in cases:
        try:
            ts = []
            for i in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y.sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                cold = ts[0]
                warm_med = statistics.median(ts[1:]) if len(ts) > 1 else cold
                print(f"[bench] a2av {label:10s} cold={cold:.3f}ms warm_med={warm_med:.3f}ms")
                with open(f"{bench_dir}/a2av_{label}.json", "w") as f:
                    json.dump({"label": label, "cold_ms": cold, "warm_med_ms": warm_med, "all": ts}, f)
        except Exception as e:
            if rank == 0:
                print(f"[bench] a2av {label} FAILED: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
