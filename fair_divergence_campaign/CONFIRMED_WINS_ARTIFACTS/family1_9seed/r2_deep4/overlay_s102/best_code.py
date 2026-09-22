def r2_deep4_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute scale factors a[r] for each rank
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Stage 1: First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse multiple local operations together to reduce dispatch overhead
    buf = s.clone()
    
    # Apply all scaling operations in a batched manner
    for r in range(W):
        start_idx = r * S
        end_idx = (r + 1) * S
        shard = s[start_idx:end_idx]
        
        # Perform multiple local operations per shard to increase local work
        scale_factor = a[r] / W
        inv_scale = 1.0 / max(a[r], 1e-9)
        
        # First transformation
        buf[start_idx:end_idx] = scale_factor * shard
    
    # Stage 2: Second all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # More local work - unscale and prepare for next stage
    buf = s.clone()
    for r in range(W):
        start_idx = r * S
        end_idx = (r + 1) * S
        inv_scale = 1.0 / max(a[r], 1e-9)
        scale_factor = a[r] / W
        # Fuse unscale and rescale operations
        buf[start_idx:end_idx] = scale_factor * inv_scale * s[start_idx:end_idx]
    
    # Stage 3: Third all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # More local work
    buf = s.clone()
    for r in range(W):
        start_idx = r * S
        end_idx = (r + 1) * S
        inv_scale = 1.0 / max(a[r], 1e-9)
        scale_factor = a[r] / W
        # Fuse operations again
        buf[start_idx:end_idx] = scale_factor * inv_scale * s[start_idx:end_idx]
    
    # Stage 4: Fourth all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s