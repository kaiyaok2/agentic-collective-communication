def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors as a tensor (vectorized)
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # Phase 1: First all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized operations - combine multiple scalings
    s1_reshaped = s1.view(W, S)
    a_expanded = a.unsqueeze(1)
    
    # Combine the scaling operations from phases 1 and 2
    # Original: (a/W) then (1/W) = a/W^2
    # After s2 all_reduce: sum over ranks, so we get W * (a/W^2) = a/W per element
    # Then final scaling and all_reduce
    
    # Optimize: combine buf0 computation and next scaling
    buf0 = (a_expanded * s1_reshaped / (W * W)).view(-1)
    
    # Phase 2: Second all_reduce  
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Final phase: directly compute output without third all_reduce if possible
    # Since the pattern shows redundancy, the final all_reduce of s2/a * a/W = s2/W
    # simplifies to just s2 (already summed), so we might just need proper scaling
    
    # However, to maintain correctness with 3 collectives pattern,
    # merge the last local ops more efficiently
    out = xm.all_reduce(xm.REDUCE_SUM, s2)
    
    return out