def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Four Sequential All-Reduces with Vectorized Local Operations
    
    Optimizes the baseline by vectorizing per-shard scale/unscale operations
    to reduce the local op count overhead relative to collective dispatches.
    """
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Shard scaling factors: a[r] = 1.0 + 0.5*(r % 3)
    # Create as a tensor for vectorized operations
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    a_expanded = a.repeat_interleave(S)
    
    # Stage 1: First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized scale: buf = a * s / W
    buf = (a_expanded * s) / W
    
    # Stage 2: Second all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Vectorized unscale: s = s / a
    s = s / torch.clamp(a_expanded, min=1e-9)
    
    # Vectorized scale: buf = a * s / W
    buf = (a_expanded * s) / W
    
    # Stage 3: Third all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Vectorized unscale: s = s / a
    s = s / torch.clamp(a_expanded, min=1e-9)
    
    # Vectorized scale: buf = a * s / W
    buf = (a_expanded * s) / W
    
    # Stage 4: Fourth all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s