def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain.
    Performs 8 sequential all_reduce operations:
    1. Initial sum of x across all ranks
    2-8. Seven weighted iterations, each applying rank-dependent weights to 5 of 6 blocks
    
    Each iteration:
    - Applies rank weight to blocks in the rank's window (5 blocks)
    - Zeros out 1 block (the one not in window)
    - All-reduces the result
    - Normalizes by accumulated weights
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Step 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute accumulated weights A[b] for each block b
    # A[b] = sum over all ranks r of weight(r) if block b is in rank r's window
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 6
        for j in range(5):
            block_idx = (start + 1 * j) % 6
            A[block_idx] += w
    
    # Steps 2-8: Seven weighted iterations
    for iteration in range(7):
        # Determine which blocks this rank contributes to (5 out of 6)
        start = (rank + 0) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # Rank-dependent weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer: apply weight to kept blocks, zero others
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce sum
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by accumulated weights (except last iteration)
        if iteration < 6:
            for b in range(6):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        s = acc
    
    return s