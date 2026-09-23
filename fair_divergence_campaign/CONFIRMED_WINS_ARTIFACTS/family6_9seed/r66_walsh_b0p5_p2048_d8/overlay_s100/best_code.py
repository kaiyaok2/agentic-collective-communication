def r66_walsh_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized 8-AllReduce Pattern with reduced dispatch overhead
    
    Reduces collective dispatch overhead by batching local operations
    and minimizing intermediate allocations.
    """
    W = world_size
    BETA = 0.5
    N = 16384
    W_inv = 1.0 / W
    
    # Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute fixed sign vectors v and u
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    
    # Precompute u*v for reuse
    uv = u * v
    
    # Main iteration loop - unrolled for performance
    for i in range(6):
        # Local computation: buf = s + BETA * u * (v * s).mean()
        vs_mean = (v * s).mean()
        buf = s + (BETA * vs_mean) * u
        
        # Collective communication
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Local computation: s = acc/W - BETA * u * (v * acc).mean()
        acc *= W_inv
        vacc_mean = (v * acc).mean()
        s = acc - (BETA * vacc_mean) * u
    
    # Final iteration (no subtraction at the end)
    vs_mean = (v * s).mean()
    buf = s + (BETA * vs_mean) * u
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s *= W_inv
    
    return s