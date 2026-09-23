def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    
    # Pre-compute u and v masks once (fused buffer preparation)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Pre-compute reciprocal for efficiency
    inv_W = 1.0 / W
    
    # Initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Unrolled 7 iterations with fused operations
    # Each iteration: compute buffer with fused mean, all-reduce, update s with fused mean
    
    # Iteration 1
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc * inv_W
    vacc_mean = (v * acc).mean()
    s = acc - BETA * u * vacc_mean
    
    # Iteration 2
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc * inv_W
    vacc_mean = (v * acc).mean()
    s = acc - BETA * u * vacc_mean
    
    # Iteration 3
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc * inv_W
    vacc_mean = (v * acc).mean()
    s = acc - BETA * u * vacc_mean
    
    # Iteration 4
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc * inv_W
    vacc_mean = (v * acc).mean()
    s = acc - BETA * u * vacc_mean
    
    # Iteration 5
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc * inv_W
    vacc_mean = (v * acc).mean()
    s = acc - BETA * u * vacc_mean
    
    # Iteration 6
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc * inv_W
    vacc_mean = (v * acc).mean()
    s = acc - BETA * u * vacc_mean
    
    # Iteration 7 (final, no s update needed after)
    vs_mean = (v * s).mean()
    buf = s + BETA * u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc * inv_W
    
    return s