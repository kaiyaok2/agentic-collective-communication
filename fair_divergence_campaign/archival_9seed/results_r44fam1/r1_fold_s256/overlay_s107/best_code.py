def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Stage 1: all_reduce + fused scaling
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused vectorized scaling for stage 1: a[r] / W for each shard
    for r in range(W):
        s1[r*S:(r+1)*S] *= (a[r] / W)
    
    # Stage 2: all_reduce + fused scaling
    s2 = xm.all_reduce(xm.REDUCE_SUM, s1)
    
    # Fused vectorized scaling for stage 2: (1/a[r]) * (a[r]/W) = 1/W for each shard
    for r in range(W):
        s2[r*S:(r+1)*S] *= (1.0 / max(a[r], 1e-9)) * (a[r] / W)
    
    # Stage 3: all_reduce (final)
    out = xm.all_reduce(xm.REDUCE_SUM, s2)
    
    return out