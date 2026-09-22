def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized two-stage implementation using all_reduce.
    Each stage performs: scale -> all_reduce -> divide.
    Total 2 collective dispatches (2 all_reduce), reduced from 4.
    """
    S = 256
    W = world_size
    D = 2
    
    dtype = x.dtype
    
    # Compute local scaling factor a[rank]
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute total sum of all a[k] for normalization
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    cur = x
    
    # Execute D=2 stages sequentially
    for stage in range(D):
        # Stage step 1: Scale by a
        scaled = a * cur
        
        # Stage step 2: All-reduce with SUM to aggregate across all ranks
        # This replaces: all_to_all -> local sum -> all_gather
        # The all_reduce with SUM operation effectively computes the sum
        # of all scaled values across all ranks and replicates the result
        summed = xm.all_reduce(xm.REDUCE_SUM, scaled)
        
        # Stage step 3: Divide by normalization factor
        cur = summed / u
    
    return cur