def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    STRATEGY: Reference Sequential All-Reduce Chain
    Performs 8 all-reduce dispatches sequentially: 
    1 initial sum, then 7 weighted iterations each with selective masking, 
    weight application, all-reduce, and normalization.
    """
    S = 2048
    W = world_size
    dtype = x.dtype
    
    # Dispatch 1: Initial all-reduce sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute normalization factors A[b] for each block
    A = [0.0] * 5
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = (r + 0) % 5
        for j in range(3):
            A[(st + 1 * j) % 5] += w
    
    # Dispatches 2-8: 7 weighted iterations
    for iteration in range(7):
        # Determine which blocks this rank contributes to
        start = (rank + 0) % 5
        keep = set((start + 1 * j) % 5 for j in range(3))
        
        # This rank's weight
        w = 0.5 + 0.02 * rank
        
        # Create masked and weighted buffer
        buf = torch.zeros_like(s)
        for b in range(5):
            if b in keep:
                buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
        
        # All-reduce to sum contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize each block (except on last iteration where normalization is skipped)
        if iteration < 6:
            for b in range(5):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
        
        s = acc
    
    return s