def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized implementation using all_reduce to minimize collective dispatches.
    
    Replaces all_to_all + local sum + all_gather with a single all_reduce,
    reducing from 2 collective dispatches per stage to 1.
    """
    S = 256
    W = world_size
    D = 2
    
    # Compute scaling factor for this rank
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute total scaling sum across all ranks
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Preserve input dtype
    dtype = x.dtype
    
    cur = x
    
    # Two stages (depth = 2)
    for _t in range(D):
        # Stage operations:
        # 1. Scale by rank-specific factor
        y = a * cur
        
        # 2. All-reduce with SUM operation (single dispatch replaces all_to_all + all_gather)
        #    This sums contributions from all ranks and replicates the result
        cur = xm.all_reduce(xm.REDUCE_SUM, y)
        
        # 3. Divide by normalization factor
        cur = cur / u
    
    return cur