def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors for each rank
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Stage 1: First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local operations with more substantial computation
    buf0 = s1.clone()
    for r in range(W):
        # Apply combined scaling: a[r] / (W * W)
        scale_factor = a[r] / (W * W)
        buf0[r*S:(r+1)*S] = scale_factor * s1[r*S:(r+1)*S]
    
    # Stage 2: Second all-reduce that combines remaining reductions
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Additional local computation before final result
    # Apply per-rank transformations
    result = s2.clone()
    for r in range(W):
        result[r*S:(r+1)*S] = s2[r*S:(r+1)*S] * W
    
    return result