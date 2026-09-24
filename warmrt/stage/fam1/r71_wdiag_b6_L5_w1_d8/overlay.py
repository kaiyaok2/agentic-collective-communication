def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain
    Performs 8 sequential all_reduce operations (1 initial sum + 7 weighted iterations)
    with intermediate local masking, scaling, and normalization operations.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Initial all-reduce: sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute normalization factors A[b] for each block
    # A[b] = sum over all ranks r of weight(r) if block b is in rank r's window
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 6
        for j in range(5):
            block_idx = (start + 1 * j) % 6
            A[block_idx] += w
    
    # Perform 7 weighted iterations
    for iteration in range(7):
        # Determine which blocks this rank keeps (its window)
        start = (rank + 0) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # This rank's weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer: mask and scale by weight
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce the weighted contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by total weight for each block (except last iteration)
        if iteration < 6:
            for b in range(6):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        s = acc
    
    return s