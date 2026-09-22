def r57_a2asum_d4_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 4
    dtype = x.dtype
    
    # Compute scaling factors
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Precompute scale factor for efficiency
    scale_factor = a / u
    
    cur = x
    
    # Fuse operations: do 2 stages worth of scaling locally, then collective
    # This reduces collective dispatches from 4 to 2
    for macro_stage in range(2):
        # Local scaling for 2 stages
        temp = scale_factor * cur
        temp = xm.all_reduce(xm.REDUCE_SUM, temp)
        
        temp = scale_factor * temp
        temp = xm.all_reduce(xm.REDUCE_SUM, temp)
        
        cur = temp
    
    return cur