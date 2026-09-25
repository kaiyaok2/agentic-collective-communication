def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain:
    Performs 8 dependent all-reduce operations sequentially 
    (1 initial sum + 7 weighted iterations), each operating on the full 6-block tensor.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Step 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute the total weight A for each block across all ranks
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = (r + 0) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Perform 7 weighted iterations
    for iteration in range(7):
        # Determine which blocks this rank contributes to
        start = (rank + 0) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # Rank's weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer and apply weight to kept blocks
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce sum
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by total weights (except on the last iteration)
        if iteration < 6:
            for b in range(6):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        s = acc
    
    return s