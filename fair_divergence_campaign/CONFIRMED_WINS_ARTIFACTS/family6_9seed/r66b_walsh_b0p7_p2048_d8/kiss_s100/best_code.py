def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    N = 16384
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute Walsh vectors and BETA * u (optimization: reduce repeated operations)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    beta_u = BETA * u  # Pre-compute to save repeated scalar multiplications
    
    # Iterative refinement: 6 full iterations + 1 final
    for i in range(7):
        # Forward transform
        vs_mean = (v * s).mean()
        buf = s + beta_u * vs_mean
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc = acc / W
        
        # Inverse transform (skip on last iteration)
        if i < 6:
            vacc_mean = (v * acc).mean()
            s = acc - beta_u * vacc_mean
        else:
            s = acc
    
    return s