def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W)], device=x.device)
    
    # First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized scaling: a[r]/W for each segment, then by 1/W (from step 4)
    # Combined: a[r]/W * 1/W = a[r]/W^2
    a_scaled = (a / (W * W)).repeat_interleave(S)
    s1 = s1 * a_scaled
    
    # Second all-reduce (combines what were previously the 2nd and 3rd all-reduces)
    s1 = xm.all_reduce(xm.REDUCE_SUM, s1)
    
    # Third all-reduce (still needed for correctness)
    out = xm.all_reduce(xm.REDUCE_SUM, s1)
    
    return out