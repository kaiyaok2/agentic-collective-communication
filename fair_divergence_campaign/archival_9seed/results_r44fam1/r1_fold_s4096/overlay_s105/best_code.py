def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors - simpler computation
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Scale: multiply by a/W (vectorized, minimal reshaping)
    buf0 = s1.view(W, S) * (a / W).view(W, 1)
    
    # Second all-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0.view(-1))
    
    # Divide by (W*a) and multiply by a = divide by W
    bufN = s2.view(W, S) / W
    
    # Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, bufN.view(-1))
    
    return out