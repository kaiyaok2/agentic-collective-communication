def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                     cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    dtype = x.dtype
    
    # Compute per-rank scaling factors once
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # First all-reduce: sum across all ranks
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized scaling: reshape and use broadcasting
    # Instead of loop, create scale vector and multiply directly
    scale_vector = (a / W).repeat_interleave(S)
    buf0 = s1 * scale_vector
    
    # Second all-reduce: sum the scaled shards
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Final scaling (vectorized, no per-element loop)
    bufN = s2 / W
    
    # Third all-reduce: final sum
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out