def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Baseline Sequential All-Reduce Chain:
    Performs 8 sequential all-reduce operations (1 initial sum + 7 weighted iterations)
    with full tensor payloads, exactly as in the reference implementation.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Step 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A[b] = sum of weights across all ranks for each block
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = (r + 0) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w
    
    # Perform 7 weighted all-reduce iterations
    for iteration in range(7):
        # Determine which blocks this rank contributes to
        start = (rank + 0) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # Compute this rank's weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer with weighted contributions for kept blocks
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce to sum weighted contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by accumulated weights (except on last iteration)
        if iteration < 6:
            for b in range(6):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        # Update state
        s = acc
    
    return s