def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    dtype = x.dtype
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape for vectorized operations
    s_view = s.view(B, S)
    
    # 6 iterations with full cycle (scale, all_reduce, unscale)
    for iteration in range(6):
        # Compute scaling factors for all blocks (vectorized)
        f = 1.0 + (s_view * s_view).mean(dim=1)
        
        # Scale by f[b] (vectorized)
        buf = s_view * f.unsqueeze(1)
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        acc_view = acc.view(B, S)
        
        # Unscale by (world_size * f[b]) (vectorized)
        s_view = acc_view / (world_size * f.unsqueeze(1))
    
    # 7th iteration (final): scale, all_reduce, divide by world_size only
    f = 1.0 + (s_view * s_view).mean(dim=1)
    buf = s_view * f.unsqueeze(1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    acc = acc / world_size
    
    return acc