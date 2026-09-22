def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Pre-compute scaling coefficients
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Stage 1: First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local operation 1: Apply per-shard scaling (a[r] / W) in one pass
    # Create a scaling tensor that covers all shards at once
    scale1 = torch.cat([torch.full((S,), a[r] / W, dtype=dtype, device=x.device) 
                        for r in range(W)])
    buf0 = s1 * scale1
    
    # Stage 2: Second all-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Fused local operation 2: Apply inverse scaling (1 / a[r]) in one pass
    scale2 = torch.cat([torch.full((S,), 1.0 / max(a[r], 1e-9), dtype=dtype, device=x.device) 
                        for r in range(W)])
    s2_scaled = s2 * scale2
    
    # Fused local operation 3: Apply per-shard scaling (a[r] / W) again in one pass
    scale3 = torch.cat([torch.full((S,), a[r] / W, dtype=dtype, device=x.device) 
                        for r in range(W)])
    bufN = s2_scaled * scale3
    
    # Stage 3: Final all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out