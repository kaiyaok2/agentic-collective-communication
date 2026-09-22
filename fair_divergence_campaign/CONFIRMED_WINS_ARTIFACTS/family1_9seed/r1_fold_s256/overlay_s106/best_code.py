def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute scale factors as a tensor for vectorized operations
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # Repeat each scale factor S times to match tensor shape
    scale_vector = a.repeat_interleave(S)
    
    # Stage 1: all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized operation: scale and divide
    s1 = (scale_vector / W) * s1
    
    # Stage 2: all_reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, s1)
    
    # Vectorized operation: divide by scale factor and world size
    # Equivalent to s2 / (a[r] * W) for each segment, then multiply by a[r]
    # This simplifies to s2 / W
    s2 = s2 / W
    
    # Stage 3: all_reduce
    out = xm.all_reduce(xm.REDUCE_SUM, s2)
    
    return out