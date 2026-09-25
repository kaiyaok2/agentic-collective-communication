def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Optimized version with fused local operations to reduce dispatch overhead.
    Combines tensor operations to minimize the number of kernel launches.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute normalization factors A[b] for each block
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 6
        for j in range(5):
            block_idx = (start + 1 * j) % 6
            A[block_idx] += w
    
    # Pre-compute this rank's window blocks and weight
    start = (rank + 0) % 6
    keep = set((start + 1 * j) % 6 for j in range(5))
    w = 0.5 + 0.02 * rank
    
    # Create reusable mask tensor (1.0 for kept blocks, 0.0 otherwise)
    mask = torch.zeros(6 * S, dtype=dtype, device=x.device)
    for b in range(6):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization tensor for blocks (except last iteration)
    norm_factors = torch.ones(6 * S, dtype=dtype, device=x.device)
    for b in range(6):
        norm_factors[b*S:(b+1)*S] = 1.0 / A[b]
    
    # Perform 7 weighted iterations
    for iteration in range(7):
        # Fused operation: mask * weight * s (single kernel dispatch)
        buf = mask * w * s
        
        # All-reduce to sum weighted contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Apply normalization (except on last iteration) - single operation
        if iteration < 6:
            s = acc * norm_factors
        else:
            s = acc
    
    return s