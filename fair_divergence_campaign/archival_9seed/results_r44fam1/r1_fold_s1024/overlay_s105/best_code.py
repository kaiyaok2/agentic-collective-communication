def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Define per-shard scaling factors as a tensor for vectorized operations
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    a_expanded = a.repeat_interleave(S)
    
    # Stage 1: First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply a[r] / W scaling (vectorized)
    buf0 = s1 * a_expanded / W
    
    # Stage 2: Second all-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Apply 1/W scaling (vectorized)
    buf1 = s2 / W
    
    # Stage 3: Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf1)
    
    return out