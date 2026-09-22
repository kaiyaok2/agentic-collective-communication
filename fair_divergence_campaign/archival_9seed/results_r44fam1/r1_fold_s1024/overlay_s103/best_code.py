def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                     cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized implementation using reduce-scatter and all-gather pattern.
    
    Strategy: Replace some all-reduce operations with more efficient 
    reduce-scatter + all-gather patterns to reduce collective dispatch overhead.
    """
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Define the per-rank scaling factors
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Stage 1: Use reduce-scatter instead of all-reduce
    # This reduces data movement by only giving each rank its portion
    local_result = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, 
                                      scatter_dim=0, shard_count=W)
    
    # Apply first scaling: a[rank]/W on local shard
    scale_factor = a[rank] / W
    scaled_local = local_result * scale_factor
    
    # Stage 2: All-gather to reconstruct full tensor
    buf0 = xm.all_gather(scaled_local, dim=0)
    
    # Second all-reduce with fused local scaling
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Fused local operations: divide by a[r] then multiply by a[r]/W
    # This simplifies to just dividing by W
    bufN = s2 / W
    
    # Stage 3: Final all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out