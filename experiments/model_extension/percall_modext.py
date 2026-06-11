"""Per-call HW microbench for the 4 model-extension compositions.

Mirrors the structure of experiments/h7_bench/bench_*.py: measures the
isolated per-call latency of ONE collective call from each composition.

For each problem we time:
  baseline (per_mb): one small AR/AG of the per-microbatch payload
  agent  (bundled) : one large AR/AG of the M-fold bundled payload

Both are isolated dispatches (one xm.* call wrapped in mark_step +
.item() probe), iterated N_ITER times with WARMUP-iter cache warmup.

The 4 problems:
  pp        : masked AR over (half, B, S, DM) buffer (per_mb) vs (half, M, B, S, DM) (bundled)
  tp_mlp    : AR over (B, S, DM) partial (per_mb) vs (M*N_LAYERS, B, S, DM) (bundled)
  fsdp      : AG of (DM, shard_hid) shard (per_mb) vs (M*N_LAYERS, DM, shard_hid) (bundled)
  llama_e2e : end-to-end placeholder — uses the cross-stage masked AR which is
              the dominant collective (same as pp).
"""
import os, sys, time, json, statistics
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2048
HID = 5376
N_LAYERS = 2
B = 1
S = 512
M = 4
N_ITER = 30
WARMUP = 5


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    half = ws // 2
    pair_id = rank if rank < half else rank - half
    shard_hid = HID // ws

    problem = sys.argv[1] if len(sys.argv) > 1 else 'pp'
    backend = sys.argv[2] if len(sys.argv) > 2 else 'per_mb'

    if problem == 'pp':
        # per_mb: 1 AR over (half, B, S, DM); bundled: 1 AR over (half, M, B, S, DM)
        if backend == 'per_mb':
            buf = torch.zeros(half, B, S, DM, device=dev, dtype=torch.bfloat16)
            act = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
            buf[pair_id] = act if rank < half else 0
            def fn():
                return xm.all_reduce(xm.REDUCE_SUM, buf)
        else:
            buf = torch.zeros(half, M, B, S, DM, device=dev, dtype=torch.bfloat16)
            act = torch.randn(M, B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
            buf[pair_id] = act if rank < half else 0
            def fn():
                return xm.all_reduce(xm.REDUCE_SUM, buf)
    elif problem == 'tp_mlp':
        if backend == 'per_mb':
            partial = torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
            def fn():
                return xm.all_reduce(xm.REDUCE_SUM, partial)
        else:
            partial = torch.randn(M * N_LAYERS, B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01
            def fn():
                return xm.all_reduce(xm.REDUCE_SUM, partial)
    elif problem == 'fsdp':
        if backend == 'per_mb':
            shard = torch.randn(DM, shard_hid, device=dev, dtype=torch.bfloat16) * 0.01
            def fn():
                return xm.all_gather(shard, dim=-1)
        else:
            shard_stack = torch.randn(M * N_LAYERS, DM, shard_hid, device=dev, dtype=torch.bfloat16) * 0.01
            def fn():
                return xm.all_gather(shard_stack, dim=-1)
    else:
        raise ValueError(f"unknown problem: {problem}")

    if rank == 0:
        try:
            x = fn()
            print(f'[init] ws={ws} half={half} problem={problem} backend={backend} '
                  f'output_shape={tuple(x.shape)}', flush=True)
        except Exception as e:
            print(f'[init] error: {e}', flush=True)

    # Warmup
    for _ in range(WARMUP):
        y = fn()
        _ = y.sum().item()
    if rank == 0:
        print('[init] warmup done', flush=True)

    times = []
    for _ in range(N_ITER):
        xm.mark_step()
        t0 = time.time()
        y = fn()
        _ = y.sum().item()
        times.append((time.time() - t0) * 1000)

    if rank == 0:
        steady = times[5:] if len(times) > 5 else times
        print(f'[bench] {problem}/{backend} mean={statistics.mean(steady):.3f}ms '
              f'median={statistics.median(steady):.3f}ms n={len(steady)}', flush=True)
        with open(f'/tmp/tp_search/percall_{problem}_{backend}.json', 'w') as f:
            json.dump({'problem': problem, 'backend': backend,
                       'mean_ms': statistics.mean(steady),
                       'med_ms': statistics.median(steady),
                       'all': times, 'ws': ws,
                       'DM': DM, 'HID': HID, 'B': B, 'S': S, 'M': M, 'N_LAYERS': N_LAYERS}, f)


if __name__ == '__main__':
    main()
