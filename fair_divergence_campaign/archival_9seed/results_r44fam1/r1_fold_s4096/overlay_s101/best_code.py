def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    
    # Create scaling coefficients as a tensor
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=x.dtype, device=x.device)
    a_expanded = a.repeat_interleave(S)
    
    # Stage 1: First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized scaling: apply a[r] / W in one operation
    s1 = s1 * a_expanded / W
    
    # Stage 2: Second all-reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, s1)
    
    # Vectorized scaling: apply 1/W in one operation
    s2 = s2 / W
    
    # Stage 3: Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, s2)
    
    return out