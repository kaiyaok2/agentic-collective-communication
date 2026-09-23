def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Pre-compute scale coefficients a[r] for each rank
    a = [1.0 + 0.35 * (r % 6) for r in range(W)]
    
    # Pre-compute cumulative scale coefficients for all 8 stages
    # Each stage does: scale by a[r]/W, all_reduce (sum), unscale by 1/a[r]
    # We need to track the compound effect across stages
    
    # Stage 1: Initial all_reduce (no scaling)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # For stages 2-8, we apply scale/unscale operations
    # Pattern per stage: scale by a[r]/W, all_reduce, unscale by 1/a[r]
    # We can pre-compute the compound scaling factors
    
    # Create scale tensors for efficient application
    # For each stage, we need to apply a[r]/W before all_reduce
    # and 1/a[r] after all_reduce
    
    scale_before = torch.zeros(W * S, dtype=dtype)
    scale_after = torch.zeros(W * S, dtype=dtype)
    
    for r in range(W):
        scale_before[r*S:(r+1)*S] = a[r] / W
        scale_after[r*S:(r+1)*S] = 1.0 / max(a[r], 1e-9)
    
    # Stage 2
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * scale_after
    
    # Stage 3
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * scale_after
    
    # Stage 4
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * scale_after
    
    # Stage 5
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * scale_after
    
    # Stage 6
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * scale_after
    
    # Stage 7
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * scale_after
    
    # Stage 8 (final)
    buf = s * scale_before
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s