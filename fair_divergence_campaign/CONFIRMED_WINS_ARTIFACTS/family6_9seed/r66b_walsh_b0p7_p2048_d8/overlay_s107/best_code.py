def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    N = 16384
    
    # Precompute fixed sign vectors
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # First all_reduce to get initial s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pipeline: maintain 2-deep pipeline with overlapped communication and computation
    # We have 7 more all_reduce operations to perform (8 total, first one done)
    
    # For pipelining, we'll use a strategy where we:
    # 1. Start all_reduce on current buffer (non-blocking if possible)
    # 2. Compute next buffer while communication happens
    # However, xm.all_reduce is blocking in practice, so we focus on minimizing latency
    # by preparing the next buffer as soon as possible
    
    # Iteration 1
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    
    # Iteration 2
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    
    # Iteration 3
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    
    # Iteration 4
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    
    # Iteration 5
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    
    # Iteration 6
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    
    # Iteration 7 (final)
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s