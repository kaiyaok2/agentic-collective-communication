def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    dtype = x.dtype
    
    # Convert a to tensor for vectorized operations
    a_tensor = torch.tensor(a, dtype=dtype, device=x.device)
    a_expanded = a_tensor.repeat_interleave(S)
    
    # Stage 1: All-reduce on full tensor (1 collective)
    s1 = xm.all_reduce(xm.REDUCE_SUM, x.clone())
    
    # Stage 2: Apply per-shard scale, then all-reduce (1 collective)
    # Vectorized scaling instead of loop
    buf0 = (a_expanded * s1) / W
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Stage 3: Inverse scale, apply final scale, then all-reduce (1 collective)
    # Vectorized operations
    a_expanded_safe = torch.clamp(a_expanded, min=1e-9)
    s2_scaled = s2 / a_expanded_safe
    bufN = (a_expanded * s2_scaled) / W
    s3 = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return s3