def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Compute scaling factors a[r] = 1.0 + 0.35 * (r % 6) for each rank
    # Create as a tensor for vectorized operations
    a_tensor = torch.tensor([1.0 + 0.35 * (r % 6) for r in range(W)], 
                            dtype=dtype, device=x.device)
    
    # Repeat each scaling factor S times to match shard sizes
    a_full = a_tensor.repeat_interleave(S)
    
    # Stage 1: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stages 2-8: Apply scaling, all-reduce, then unscaling
    for stage in range(7):
        # Vectorized scaling: scale each shard by its factor and divide by W
        buf = (a_full * s) / W
        
        # All-reduce the scaled buffer
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Vectorized unscaling (except on the last stage)
        if stage < 6:
            s = s / torch.clamp(a_full, min=1e-9)
    
    return s