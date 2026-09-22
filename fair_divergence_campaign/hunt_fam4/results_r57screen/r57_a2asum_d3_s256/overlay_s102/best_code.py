def r57_a2asum_d3_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential three-stage baseline: Direct translation of reference with 3 iterations.
    Each iteration: all_to_all, reshape+sum, all_gather, scale.
    Total 9 collective dispatches (3×all_to_all + 3×all_gather).
    """
    S = 256
    W = world_size
    D = 3
    
    # Compute scaling factor for this rank
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute total sum of all ranks' scaling factors
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    # Preserve input dtype
    dtype = x.dtype
    
    # Start with input
    cur = x
    
    # Execute D iterations
    for _t in range(D):
        # Stage 1: Scale by a and perform all_to_all
        # Split input into W chunks of size S, transpose across ranks
        y = xm.all_to_all(a * cur, split_dimension=0, concat_dimension=0,
                          split_count=W)
        
        # Stage 2: Local reduction - reshape to (W, S) and sum along dimension 0
        z = torch.sum(y.reshape(W, S), dim=0)
        
        # Stage 3: All-gather the reduced shard and scale by 1/u
        cur = xm.all_gather(z, dim=0) / u
    
    return cur