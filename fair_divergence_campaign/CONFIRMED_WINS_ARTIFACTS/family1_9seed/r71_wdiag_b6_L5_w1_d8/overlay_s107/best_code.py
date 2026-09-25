def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Naive Sequential All-Reduce Chain
    
    Performs 8 sequential all-reduce operations:
    1. Initial sum of x across all ranks
    2-7. Six iterations of weighted selective masking, all-reduce, and normalization
    8. Final weighted selective masking and all-reduce (no normalization)
    
    Each all-reduce uses full 12,288-element payload (6 blocks * 2048).
    Total: 8 all-reduce dispatches.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Step 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute normalization factors A[b] for each block b
    # A[b] = sum of weights w_r for all ranks r whose window covers block b
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        start = (r + 0) % 6
        for j in range(5):
            block_idx = (start + 1 * j) % 6
            A[block_idx] += w
    
    # Iterations 2-7: weighted selective masking with normalization
    for iteration in range(6):
        # Compute which blocks this rank contributes to
        start = (rank + 0) % 6
        keep = set((start + 1 * j) % 6 for j in range(5))
        
        # Rank-dependent weight
        w = 0.5 + 0.02 * rank
        
        # Create buffer with selective weighted blocks
        buf = torch.zeros_like(s)
        for b in range(6):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce sum
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize each block
        for b in range(6):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        s = acc
    
    # Step 8: Final weighted all-reduce without normalization
    start = (rank + 0) % 6
    keep = set((start + 1 * j) % 6 for j in range(5))
    w = 0.5 + 0.02 * rank
    
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc