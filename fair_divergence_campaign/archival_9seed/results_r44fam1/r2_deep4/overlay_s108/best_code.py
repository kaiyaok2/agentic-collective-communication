def r2_deep4_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute scaling factors as a tensor for vectorized operations
    a_tensor = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W)], dtype=x.dtype, device=x.device)
    a_tensor = a_tensor.repeat_interleave(S)
    
    # Stage 1: All-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply per-shard scaling (vectorized)
    buf1 = (a_tensor * s1) / W
    
    # Stage 2: All-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    # Unscale (vectorized with epsilon for numerical stability)
    s2 = s2 / torch.clamp(a_tensor, min=1e-9)
    
    # Stage 3: Scale and all-reduce (vectorized)
    buf2 = (a_tensor * s2) / W
    s3 = xm.all_reduce(xm.REDUCE_SUM, buf2)
    
    # Unscale (vectorized)
    s3 = s3 / torch.clamp(a_tensor, min=1e-9)
    
    # Stage 4: Final scale and all-reduce (vectorized)
    buf3 = (a_tensor * s3) / W
    s4 = xm.all_reduce(xm.REDUCE_SUM, buf3)
    
    return s4