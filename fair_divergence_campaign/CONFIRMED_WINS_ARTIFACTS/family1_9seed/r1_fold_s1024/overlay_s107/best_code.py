def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Compute per-rank scaling factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Stage 1: Reduce-Scatter + All-Gather
    # Reduce-scatter: sum across all ranks, each rank gets size S chunk
    s1_chunk = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=W, groups=None)
    
    # Apply scaling to local chunk: a[rank] * s1_chunk / W
    s1_chunk = a[rank] * s1_chunk / W
    
    # All-gather: collect all scaled chunks
    buf0 = xm.all_gather(s1_chunk, dim=0)
    
    # Stage 2: Reduce-Scatter + All-Gather
    # Reduce-scatter
    s2_chunk = xm.reduce_scatter(xm.REDUCE_SUM, buf0, scale=1.0, scatter_dim=0, shard_count=W, groups=None)
    
    # Apply inverse scaling to local chunk: s2_chunk / max(a[rank], 1e-9)
    s2_chunk = s2_chunk / max(a[rank], 1e-9)
    
    # All-gather
    s1 = xm.all_gather(s2_chunk, dim=0)
    
    # Stage 3: Reduce-Scatter + All-Gather
    # Prepare bufN by applying per-shard scaling
    bufN = s1.clone()
    for r in range(W):
        bufN[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W
    
    # Reduce-scatter
    s3_chunk = xm.reduce_scatter(xm.REDUCE_SUM, bufN, scale=1.0, scatter_dim=0, shard_count=W, groups=None)
    
    # All-gather
    out = xm.all_gather(s3_chunk, dim=0)
    
    return out