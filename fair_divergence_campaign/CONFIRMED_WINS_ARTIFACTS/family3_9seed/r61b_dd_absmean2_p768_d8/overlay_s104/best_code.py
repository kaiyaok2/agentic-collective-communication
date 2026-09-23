def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape once outside the loop
    s = s.view(B, S)
    
    # 6 iterations with scaling and unscaling
    for iteration in range(6):
        # Compute scaling factors for all blocks at once (vectorized)
        abs_means = s.abs().mean(dim=1)
        f = 1.0 + 2.0 * abs_means
        
        # Scale all blocks at once using broadcasting
        buf = s * f.unsqueeze(1)
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        
        # Unscale all blocks at once using broadcasting
        acc = acc.view(B, S)
        s = acc / (world_size * f.unsqueeze(1))
    
    # Final (7th) iteration - scale but don't unscale by f[b]
    abs_means = s.abs().mean(dim=1)
    f = 1.0 + 2.0 * abs_means
    
    buf = s * f.unsqueeze(1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    acc = acc / world_size
    
    return acc