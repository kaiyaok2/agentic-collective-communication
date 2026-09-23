def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    
    # Pre-compute u and v masks once (hoisted out of loop)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Hoist invariant multiplication outside loop
    BETA_u = BETA * u
    
    # Initial all-reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Unrolled loop with optimized computation ordering
    # Iteration 1
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA_u * (v * acc).mean()
    
    # Iteration 2
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA_u * (v * acc).mean()
    
    # Iteration 3
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA_u * (v * acc).mean()
    
    # Iteration 4
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA_u * (v * acc).mean()
    
    # Iteration 5
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA_u * (v * acc).mean()
    
    # Iteration 6
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA_u * (v * acc).mean()
    
    # Iteration 7 (final iteration, no backward step needed)
    mean_val = (v * s).mean()
    buf = s + BETA_u * mean_val
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s