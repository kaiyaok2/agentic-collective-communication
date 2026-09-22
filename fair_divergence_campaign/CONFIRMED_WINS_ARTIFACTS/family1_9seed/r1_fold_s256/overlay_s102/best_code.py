def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Stage 1: all-reduce -> reduce-scatter + all-gather
    # Reduce-scatter: sum and divide by world_size to get per-rank chunk
    s1_scattered = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0/W, scatter_dim=0, shard_count=W, groups=None)
    
    # Apply scaling factor to local chunk
    s1_scattered = a[rank] * s1_scattered
    
    # All-gather to reconstruct full tensor
    s1 = xm.all_gather(s1_scattered, dim=0)
    
    # Stage 2: all-reduce -> reduce-scatter + all-gather
    # Apply inverse scaling per shard before reduce-scatter
    buf1 = s1.clone()
    for r in range(W):
        buf1[r*S:(r+1)*S] = s1[r*S:(r+1)*S] / max(a[r], 1e-9)
    
    # Reduce-scatter
    s2_scattered = xm.reduce_scatter(xm.REDUCE_SUM, buf1, scale=1.0/W, scatter_dim=0, shard_count=W, groups=None)
    
    # Apply scaling factor to local chunk
    s2_scattered = a[rank] * s2_scattered
    
    # All-gather to reconstruct full tensor
    s2 = xm.all_gather(s2_scattered, dim=0)
    
    # Stage 3: all-reduce -> reduce-scatter + all-gather
    # Reduce-scatter (no pre-processing needed)
    s3_scattered = xm.reduce_scatter(xm.REDUCE_SUM, s2, scale=1.0, scatter_dim=0, shard_count=W, groups=None)
    
    # All-gather to get final result
    out = xm.all_gather(s3_scattered, dim=0)
    
    return out