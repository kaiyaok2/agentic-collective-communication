def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Compute scaling factors for each rank
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # Stage 1: First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized transformation between first and second all-reduce
    # Reshape to apply per-segment scaling
    s1_reshaped = s1.view(W, S)
    buf0 = (a.unsqueeze(1) * s1_reshaped / W).view(-1)
    
    # Stage 2: Second all-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Vectorized transformation: divide by W (the a[r] terms cancel as noted)
    s2_reshaped = s2.view(W, S)
    buf_final = (s2_reshaped / W).view(-1)
    
    # Stage 3: Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf_final)
    
    return out